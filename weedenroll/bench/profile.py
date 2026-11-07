"""Hardware profiling for DR2 (latency, power) and DR3 (footprint).

On a Jetson, ``tegrastats`` is sampled in a background thread while the
backbone runs; VDD_IN (total board power) is parsed and energy per frame is
latency × mean power.  Elsewhere, power is reported as NaN and the row is
labelled with the platform string so it is never mistaken for a Jetson
measurement.

Usage on the device::

    weedenroll profile --config configs/jetson.yaml

which writes ``outputs/hardware_results.json`` consumed by the report.
"""
from __future__ import annotations

import json
import os
import platform
import re
import shutil
import subprocess
import threading
import time
from dataclasses import dataclass, asdict
from typing import List, Optional

import numpy as np

from ..embed.base import Backbone
from ..store import PrototypeStore


def _peak_rss_mb() -> float:
    """Peak resident set size of this process in MB (NaN if it cannot be read).

    ``resource`` exists only on POSIX (ru_maxrss is KB on Linux, bytes on
    macOS); on Windows the value comes from the process memory counters.
    """
    try:
        import resource  # POSIX only
        import sys
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return rss / 1e6 if sys.platform == "darwin" else rss / 1024.0
    except ImportError:
        pass
    try:
        import ctypes
        from ctypes import wintypes

        class _PMC(ctypes.Structure):
            _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
                        ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                        ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t)]
        pmc = _PMC(); pmc.cb = ctypes.sizeof(_PMC)
        h = ctypes.windll.kernel32.GetCurrentProcess()
        if ctypes.windll.psapi.GetProcessMemoryInfo(h, ctypes.byref(pmc), pmc.cb):
            return pmc.PeakWorkingSetSize / 1e6
    except Exception:
        pass
    return float("nan")


@dataclass
class ProfileResult:
    device: str
    backbone: str
    precision: str
    image_size: int
    batch_size: int
    n_frames: int
    latency_ms_mean: float
    latency_ms_sd: float
    latency_ms_p50: float
    latency_ms_p95: float
    fps: float
    power_w_mean: float          # NaN when no tegrastats
    power_w_idle: float
    energy_mj_per_frame: float   # (power - idle) * latency
    model_mb: float
    store_kb: float
    rss_mb_peak: float
    enroll_ms_10shot: float
    simulated: bool = False
    notes: str = ""


class _Tegrastats(threading.Thread):
    """Sample `tegrastats` and collect VDD_IN power (mW)."""
    _pat = re.compile(r"VDD_IN\s+(\d+)mW")

    def __init__(self, interval_ms: int = 200):
        super().__init__(daemon=True)
        self.interval = interval_ms
        self.samples: List[float] = []
        self._proc = None
        self._stop = threading.Event()

    @staticmethod
    def available() -> bool:
        return shutil.which("tegrastats") is not None

    def run(self):
        self._proc = subprocess.Popen(["tegrastats", "--interval", str(self.interval)],
                                      stdout=subprocess.PIPE, text=True)
        for line in self._proc.stdout:
            if self._stop.is_set():
                break
            m = self._pat.search(line)
            if m:
                self.samples.append(int(m.group(1)) / 1000.0)

    def stop(self):
        self._stop.set()
        if self._proc:
            self._proc.terminate()

    def mean(self) -> float:
        return float(np.mean(self.samples)) if self.samples else float("nan")


class HardwareProfiler:
    def __init__(self, backbone: Backbone, image_size: int = 224, batch_size: int = 1,
                 n_warmup: int = 20, n_frames: int = 300, device_name: Optional[str] = None):
        self.backbone = backbone
        self.image_size = image_size
        self.batch_size = batch_size
        self.n_warmup = n_warmup
        self.n_frames = n_frames
        self.device_name = device_name or self._detect_device()

    @staticmethod
    def _detect_device() -> str:
        try:
            with open("/proc/device-tree/model") as f:
                return f.read().strip("\x00\n")
        except OSError:
            return f"{platform.machine()} CPU ({platform.processor() or platform.system()})"

    def _dummy_batch(self):
        if getattr(self.backbone, "name", "") == "synthetic":
            return np.random.standard_normal((self.batch_size, self.backbone.dim)).astype(np.float32)
        return np.random.randint(0, 255, (self.batch_size, self.image_size, self.image_size, 3),
                                 dtype=np.uint8)

    def run(self, store: Optional[PrototypeStore] = None, idle_seconds: float = 3.0) -> ProfileResult:
        store = store or PrototypeStore(self.backbone.dim)
        if not store.names:
            for i in range(8):
                store.enroll(f"sp{i}", np.random.standard_normal((10, self.backbone.dim)))
        use_tegra = _Tegrastats.available()
        idle_w = float("nan")
        if use_tegra:
            t = _Tegrastats(); t.start(); time.sleep(idle_seconds); t.stop(); idle_w = t.mean()
        batches = [self._dummy_batch() for _ in range(self.n_warmup + self.n_frames)]
        for b in batches[:self.n_warmup]:
            store.predict(self.backbone.embed(self.backbone.preprocess(b) if self.backbone.name != "synthetic" else b))
        lat = []
        mon = _Tegrastats() if use_tegra else None
        if mon: mon.start()
        for b in batches[self.n_warmup:]:
            t0 = time.perf_counter()
            x = self.backbone.preprocess(b) if self.backbone.name != "synthetic" else b
            z = self.backbone.embed(x)
            store.predict(z)
            lat.append((time.perf_counter() - t0) * 1e3 / self.batch_size)
        if mon: mon.stop()
        lat = np.asarray(lat)
        power = mon.mean() if mon else float("nan")
        energy = (power - idle_w) * lat.mean() if use_tegra else float("nan")
        # enrollment cost: 10 shots
        t0 = time.perf_counter()
        shots = [self._dummy_batch()[0] for _ in range(10)]
        z = self.backbone.embed_all(shots) if self.backbone.name != "synthetic" else self.backbone.embed(np.stack(shots))
        store.enroll("new_species", z)
        enroll_ms = (time.perf_counter() - t0) * 1e3
        rss = _peak_rss_mb()
        fb = self.backbone.footprint_bytes()
        return ProfileResult(
            device=self.device_name, backbone=getattr(self.backbone, "arch", self.backbone.name),
            precision=getattr(self.backbone, "precision", "fp32"), image_size=self.image_size,
            batch_size=self.batch_size, n_frames=len(lat), latency_ms_mean=float(lat.mean()),
            latency_ms_sd=float(lat.std(ddof=1)), latency_ms_p50=float(np.percentile(lat, 50)),
            latency_ms_p95=float(np.percentile(lat, 95)), fps=float(1000.0 / lat.mean()),
            power_w_mean=power, power_w_idle=idle_w, energy_mj_per_frame=float(energy),
            model_mb=(fb / 1e6) if fb else float("nan"), store_kb=store.nbytes() / 1e3,
            rss_mb_peak=rss, enroll_ms_10shot=enroll_ms, simulated=(self.backbone.name == "synthetic"),
            notes="tegrastats VDD_IN" if use_tegra else "no tegrastats: power/energy not measured")

    @staticmethod
    def save(results: List[ProfileResult], path: str) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump([asdict(r) for r in results], f, indent=2)
