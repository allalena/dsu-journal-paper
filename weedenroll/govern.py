"""Local governance layer.

Per-session, append-only JSONL log kept on the device.  Adapts ongoing
model-monitoring practice (SR 11-7 style) to an edge node: every session
records the confidence (distance) distribution, rejection rate, enrollment
events, and prototype drift relative to the previous session snapshot.
A reviewer later reads the log offline; nothing is transmitted.

Drift indicator: cosine distance between a species' mean prototype now and
at the last snapshot.  A session-level ``alert`` fires when rejection rate or
drift exceeds configured limits — the hook for a human-in-the-loop review.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional

import numpy as np


@dataclass
class SessionSummary:
    session_id: str
    started: float
    ended: float
    n_frames: int
    n_rejected: int
    rejection_rate: float
    dist_mean: float
    dist_p50: float
    dist_p95: float
    n_enrollments: int
    drift: Dict[str, float]
    max_drift: float
    alerts: List[str]


class GovernanceLogger:
    def __init__(self, log_dir: str, session_id: Optional[str] = None,
                 max_rejection_rate: float = 0.30, max_drift: float = 0.15):
        os.makedirs(log_dir, exist_ok=True)
        self.log_dir = log_dir
        self.session_id = session_id or time.strftime("%Y%m%d-%H%M%S")
        self.path = os.path.join(log_dir, f"session-{self.session_id}.jsonl")
        self.max_rejection_rate = max_rejection_rate
        self.max_drift = max_drift
        self.started = time.time()
        self._dists: List[float] = []
        self._rejected = 0
        self._frames = 0
        self._enrollments = 0
        self._baseline: Optional[Dict[str, np.ndarray]] = None
        self._write({"event": "session_start", "t": self.started, "session": self.session_id})

    def _write(self, rec: dict) -> None:
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")

    # ---- events --------------------------------------------------------
    def set_baseline(self, snapshot: Dict[str, np.ndarray]) -> None:
        self._baseline = {k: np.asarray(v, dtype=np.float32) for k, v in snapshot.items()}

    def log_inference(self, distances: np.ndarray, is_unknown: np.ndarray) -> None:
        d = np.asarray(distances, dtype=float).ravel()
        u = np.asarray(is_unknown, dtype=bool).ravel()
        self._dists.extend(d.tolist())
        self._rejected += int(u.sum())
        self._frames += len(d)

    def log_enrollment(self, report) -> None:
        self._enrollments += 1
        rec = asdict(report) if hasattr(report, "__dataclass_fields__") else dict(report)
        rec.update({"event": "enrollment", "t": time.time()})
        self._write(rec)

    # ---- close ----------------------------------------------------------
    def drift(self, snapshot: Dict[str, np.ndarray]) -> Dict[str, float]:
        if not self._baseline:
            return {}
        out = {}
        for k, v in snapshot.items():
            if k in self._baseline:
                a, b = self._baseline[k], np.asarray(v, dtype=np.float32)
                a = a / max(np.linalg.norm(a), 1e-8)
                b = b / max(np.linalg.norm(b), 1e-8)
                out[k] = float(1.0 - a @ b)
        return out

    def close(self, snapshot: Optional[Dict[str, np.ndarray]] = None) -> SessionSummary:
        d = np.asarray(self._dists) if self._dists else np.zeros(1)
        drift = self.drift(snapshot) if snapshot is not None else {}
        rate = self._rejected / self._frames if self._frames else 0.0
        max_drift = max(drift.values()) if drift else 0.0
        alerts = []
        if rate > self.max_rejection_rate:
            alerts.append(f"rejection_rate {rate:.2f} > {self.max_rejection_rate}")
        if max_drift > self.max_drift:
            alerts.append(f"max_drift {max_drift:.3f} > {self.max_drift}")
        s = SessionSummary(self.session_id, self.started, time.time(), self._frames,
                           self._rejected, rate, float(d.mean()), float(np.percentile(d, 50)),
                           float(np.percentile(d, 95)), self._enrollments, drift, max_drift, alerts)
        rec = asdict(s)
        rec["event"] = "session_end"
        self._write(rec)
        return s

    @staticmethod
    def read_sessions(log_dir: str) -> List[dict]:
        out = []
        for fn in sorted(os.listdir(log_dir)):
            if fn.endswith(".jsonl"):
                with open(os.path.join(log_dir, fn), encoding="utf-8") as f:
                    for line in f:
                        r = json.loads(line)
                        if r.get("event") == "session_end":
                            out.append(r)
        return out
