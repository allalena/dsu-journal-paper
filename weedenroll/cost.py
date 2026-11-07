"""Season-level cost model (DR8).

Three deployment options for one growing season, N nodes, F frames per node
per season, S newly encountered species per season:

A. **Offline edge node with field enrollment** (this paper)
   amortised hardware + energy + enrollment labour.  No cloud, no link.
B. **Cloud inference** — thin capture device + cellular link + per-image
   inference + data egress.  New species still require retraining (option C
   cost for S species) because the cloud model is closed-world.
C. **Closed-world edge model with periodic retraining** — same hardware as A,
   plus, per new species: data collection/labelling, GPU training, and
   redeployment to every node.

All assumptions live in :class:`CostAssumptions` with a source note; the
YAML config overrides them.  Nothing here depends on the benchmark results
except ``energy_mj_per_frame`` and ``latency`` which come from the profiler
when available.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict, replace
from typing import Dict, List

import numpy as np
import pandas as pd


@dataclass
class CostAssumptions:
    # --- fleet / workload ------------------------------------------------
    n_nodes: int = 10
    frames_per_node_season: float = 200_000      # e.g. 2 fps × 1 h/day × 28 d/season × ~1 pass
    new_species_per_season: int = 2
    season_years: float = 1.0                    # seasons per year (amortisation unit)

    # --- edge hardware (A and C) -------------------------------------------
    edge_hw_usd: float = 600.0                   # Orin Nano-class dev kit (USD 399, Oct 2026) + camera + enclosure + power
    edge_hw_life_years: float = 3.0
    edge_power_w: float = 10.0                   # measured VDD_IN under load (profiler overrides)
    edge_idle_power_w: float = 4.0
    edge_hours_per_season: float = 28.0 * 1.0    # powered hours (28 field days × 1 h)
    electricity_usd_kwh: float = 0.15
    enrollment_minutes_per_species: float = 10.0 # operator captures 5–10 images, confirms label
    operator_usd_hour: float = 25.0

    # --- cloud (B) ---------------------------------------------------------
    thin_device_usd: float = 150.0               # capture-only device (SBC + camera + modem)
    thin_device_life_years: float = 3.0
    cellular_usd_node_month: float = 20.0
    months_per_season: float = 2.0
    image_kb: float = 150.0                      # JPEG at 224–512 px
    egress_usd_gb: float = 0.09
    cloud_inference_usd_per_1000: float = 1.50   # managed vision endpoint, per 1000 images
    cloud_latency_penalty_usd: float = 0.0       # optional: cost of missed actuation; 0 = ignore

    # --- retraining (C, and B for new species) ------------------------------
    label_images_per_species: int = 300
    label_minutes_per_image: float = 0.5
    labeller_usd_hour: float = 25.0
    gpu_train_hours: float = 4.0
    gpu_usd_hour: float = 1.50
    ml_engineer_hours_per_cycle: float = 8.0     # data prep, validation, packaging
    ml_engineer_usd_hour: float = 75.0
    redeploy_minutes_per_node: float = 15.0      # flash / verify on each node
    retrain_cycles_per_season: int | None = None # default: one cycle per new species

    # --- from profiler (optional overrides) ----------------------------------
    energy_mj_per_frame: float | None = None     # if set, used instead of power × time

    source_notes: Dict[str, str] = field(default_factory=lambda: {
        "edge_hw_usd": "Orin Nano-class board (Jetson Orin Nano Super Developer Kit, USD 399 list, Oct 2026) + camera/enclosure/power",
        "cloud_inference_usd_per_1000": "typical managed image-classification endpoint pricing tier",
        "cellular_usd_node_month": "IoT data plan, 1–5 GB/month",
        "operator_usd_hour": "farm technician wage incl. overhead",
        "ml_engineer_usd_hour": "contract ML engineer, fully loaded",
        "egress_usd_gb": "major cloud provider internet egress, first 10 TB",
    })

    def override(self, **kw) -> "CostAssumptions":
        return replace(self, **kw)


@dataclass
class CostBreakdown:
    option: str
    hardware: float
    energy: float
    connectivity: float
    cloud_inference: float
    data_transfer: float
    enrollment_labour: float
    retraining: float
    redeployment: float

    @property
    def total(self) -> float:
        return (self.hardware + self.energy + self.connectivity + self.cloud_inference +
                self.data_transfer + self.enrollment_labour + self.retraining + self.redeployment)

    def as_row(self) -> dict:
        d = asdict(self)
        d["total"] = self.total
        return d


class SeasonCostModel:
    def __init__(self, a: CostAssumptions):
        self.a = a

    # ---- shared pieces -------------------------------------------------------
    def _edge_hw(self) -> float:
        a = self.a
        return a.n_nodes * a.edge_hw_usd / (a.edge_hw_life_years * a.season_years)

    def _edge_energy(self) -> float:
        a = self.a
        if a.energy_mj_per_frame is not None and not np.isnan(a.energy_mj_per_frame):
            kwh = a.n_nodes * a.frames_per_node_season * a.energy_mj_per_frame / 3.6e9
            kwh += a.n_nodes * a.edge_idle_power_w * a.edge_hours_per_season / 1000.0
        else:
            kwh = a.n_nodes * a.edge_power_w * a.edge_hours_per_season / 1000.0
        return kwh * a.electricity_usd_kwh

    def _retrain_cycle(self) -> Dict[str, float]:
        a = self.a
        labelling = a.label_images_per_species * a.label_minutes_per_image / 60 * a.labeller_usd_hour
        training = a.gpu_train_hours * a.gpu_usd_hour + a.ml_engineer_hours_per_cycle * a.ml_engineer_usd_hour
        redeploy = a.n_nodes * a.redeploy_minutes_per_node / 60 * a.operator_usd_hour
        return {"retraining": labelling + training, "redeployment": redeploy}

    def _n_cycles(self) -> int:
        a = self.a
        return a.retrain_cycles_per_season if a.retrain_cycles_per_season is not None else a.new_species_per_season

    # ---- options ---------------------------------------------------------------
    def option_edge_enroll(self) -> CostBreakdown:
        a = self.a
        labour = a.n_nodes * a.new_species_per_season * a.enrollment_minutes_per_species / 60 * a.operator_usd_hour
        return CostBreakdown("A: Offline edge + field enrollment", self._edge_hw(), self._edge_energy(),
                             0, 0, 0, labour, 0, 0)

    def option_cloud(self) -> CostBreakdown:
        a = self.a
        frames = a.n_nodes * a.frames_per_node_season
        hw = a.n_nodes * a.thin_device_usd / (a.thin_device_life_years * a.season_years)
        conn = a.n_nodes * a.cellular_usd_node_month * a.months_per_season
        infer = frames / 1000.0 * a.cloud_inference_usd_per_1000
        egress = frames * a.image_kb / 1e6 * a.egress_usd_gb
        # new species still need retraining of the cloud model (no per-node redeploy)
        rc = self._retrain_cycle()
        retrain = self._n_cycles() * rc["retraining"]
        return CostBreakdown("B: Cloud inference", hw, 0, conn, infer + a.cloud_latency_penalty_usd * frames,
                             egress, 0, retrain, 0)

    def option_edge_retrain(self) -> CostBreakdown:
        rc = self._retrain_cycle()
        n = self._n_cycles()
        return CostBreakdown("C: Edge, closed-world, periodic retraining", self._edge_hw(), self._edge_energy(),
                             0, 0, 0, 0, n * rc["retraining"], n * rc["redeployment"])

    def table(self) -> pd.DataFrame:
        rows = [o.as_row() for o in (self.option_edge_enroll(), self.option_cloud(), self.option_edge_retrain())]
        df = pd.DataFrame(rows)
        frames = self.a.n_nodes * self.a.frames_per_node_season
        df["usd_per_node_season"] = df["total"] / self.a.n_nodes
        df["usd_per_1000_decisions"] = df["total"] / frames * 1000.0
        return df

    # ---- analyses ----------------------------------------------------------------
    def breakeven_frames(self, lo: float = 1e3, hi: float = 5e6, n: int = 200) -> pd.DataFrame:
        """Cost of each option as frames/node/season varies (find the crossover)."""
        rows = []
        for f in np.geomspace(lo, hi, n):
            m = SeasonCostModel(self.a.override(frames_per_node_season=f))
            for o in (m.option_edge_enroll(), m.option_cloud(), m.option_edge_retrain()):
                rows.append({"frames_per_node_season": f, "option": o.option, "total": o.total})
        return pd.DataFrame(rows)

    def species_sweep(self, max_species: int = 10) -> pd.DataFrame:
        rows = []
        for s in range(0, max_species + 1):
            m = SeasonCostModel(self.a.override(new_species_per_season=s))
            for o in (m.option_edge_enroll(), m.option_cloud(), m.option_edge_retrain()):
                rows.append({"new_species_per_season": s, "option": o.option, "total": o.total})
        return pd.DataFrame(rows)

    def nodes_sweep(self, max_nodes: int = 50) -> pd.DataFrame:
        rows = []
        for n in range(1, max_nodes + 1):
            m = SeasonCostModel(self.a.override(n_nodes=n))
            for o in (m.option_edge_enroll(), m.option_cloud(), m.option_edge_retrain()):
                rows.append({"n_nodes": n, "option": o.option, "total": o.total, "per_node": o.total / n})
        return pd.DataFrame(rows)

    def sensitivity(self, params: List[str] | None = None, pct: float = 0.5,
                    versus: str = "cheapest") -> pd.DataFrame:
        """Tornado: effect of ±pct on each parameter on the saving of A versus
        ``versus`` = "cloud" (B), "retrain" (C) or "cheapest" (min of B, C)."""
        params = params or ["edge_hw_usd", "edge_hw_life_years", "cloud_inference_usd_per_1000",
                            "cellular_usd_node_month", "frames_per_node_season", "new_species_per_season",
                            "ml_engineer_usd_hour", "label_images_per_species", "enrollment_minutes_per_species",
                            "operator_usd_hour", "egress_usd_gb", "electricity_usd_kwh"]
        base = self._saving(self.a, versus)
        int_params = {"n_nodes", "new_species_per_season", "label_images_per_species", "retrain_cycles_per_season"}
        rows = []
        for p in params:
            v = float(getattr(self.a, p))
            cast = (lambda x: int(round(x))) if p in int_params else float
            lo = self._saving(self.a.override(**{p: cast(v * (1 - pct))}), versus)
            hi = self._saving(self.a.override(**{p: cast(v * (1 + pct))}), versus)
            rows.append({"parameter": p, "base_value": v, "saving_base": base, "saving_low": lo,
                         "saving_high": hi, "swing": abs(hi - lo)})
        return pd.DataFrame(rows).sort_values("swing", ascending=False).reset_index(drop=True)

    @staticmethod
    def _saving(a: CostAssumptions, versus: str = "cheapest") -> float:
        m = SeasonCostModel(a)
        A = m.option_edge_enroll().total
        B, C = m.option_cloud().total, m.option_edge_retrain().total
        alt = {"cloud": B, "retrain": C}.get(versus, min(B, C))
        return alt - A
