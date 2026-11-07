"""Distance-based open-set rejection.

A query whose distance to its nearest prototype exceeds a threshold is
flagged *unknown* and queued for enrollment.  Two threshold rules:

* ``global``   — one threshold, calibrated as a quantile of known-class
  nearest-prototype distances on held-out known images (e.g. 95th percentile
  → ~5 % false-rejection on known classes).
* ``per_class`` — a threshold per species, mean + z * std of that species'
  own distances; better when species have different intra-class spread.

``score`` returns an "unknown-ness" score (nearest distance) so AUROC can be
computed independently of the threshold.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np

from .store import PrototypeStore


@dataclass
class RejectionResult:
    pred_idx: np.ndarray        # nearest species index
    distance: np.ndarray        # distance to it
    is_unknown: np.ndarray      # bool mask
    threshold: np.ndarray       # threshold applied per query


class OpenSetRejector:
    def __init__(self, store: PrototypeStore, rule: str = "global",
                 quantile: float = 0.95, z: float = 2.0):
        self.store = store
        self.rule = rule
        self.quantile = quantile
        self.z = z
        self.global_thr: Optional[float] = None
        self.class_thr: Dict[str, float] = {}

    # ---- calibration -------------------------------------------------
    def calibrate(self, emb: np.ndarray, labels: np.ndarray) -> None:
        """Calibrate on held-out *known* images (labels are species names)."""
        idx, dist = self.store.predict(emb)
        names = np.asarray(self.store.names)
        pred_names = names[idx]
        correct = pred_names == np.asarray(labels)
        d_ok = dist[correct] if correct.any() else dist
        self.global_thr = float(np.quantile(d_ok, self.quantile))
        self.class_thr = {}
        for n in names:
            m = (np.asarray(labels) == n) & correct
            if m.sum() >= 3:
                self.class_thr[n] = float(dist[m].mean() + self.z * dist[m].std())
            else:
                self.class_thr[n] = self.global_thr

    def threshold_for(self, name: str) -> float:
        if self.rule == "per_class":
            return self.class_thr.get(name, self.global_thr or 0.5)
        return self.global_thr if self.global_thr is not None else 0.5

    def add_class_threshold(self, name: str, emb: np.ndarray) -> None:
        """After field enrollment, derive a per-class threshold from the
        enrollment images' own spread (falls back to global)."""
        d = self.store.distances(emb)[:, self.store.names.index(name)]
        if len(d) >= 3:
            self.class_thr[name] = float(max(d.mean() + self.z * d.std(), self.global_thr or 0))
        else:
            self.class_thr[name] = self.global_thr or 0.5

    # ---- inference -----------------------------------------------------
    def score(self, emb: np.ndarray) -> np.ndarray:
        """Unknown-ness score = nearest-prototype distance (higher = more unknown)."""
        _, dist = self.store.predict(emb)
        return dist

    def predict(self, emb: np.ndarray) -> RejectionResult:
        idx, dist = self.store.predict(emb)
        names = self.store.names
        thr = np.array([self.threshold_for(names[i]) for i in idx], dtype=np.float32)
        return RejectionResult(idx, dist, dist > thr, thr)

    def state(self) -> dict:
        return {"rule": self.rule, "quantile": self.quantile, "z": self.z,
                "global_thr": self.global_thr, "class_thr": dict(self.class_thr)}
