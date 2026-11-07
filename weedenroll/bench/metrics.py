from __future__ import annotations

from typing import Dict, Sequence

import numpy as np
from sklearn.metrics import f1_score, roc_auc_score, roc_curve, accuracy_score


def macro_f1(y_true: Sequence[str], y_pred: Sequence[str], labels: Sequence[str] | None = None) -> float:
    return float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0))


def accuracy(y_true, y_pred) -> float:
    return float(accuracy_score(y_true, y_pred))


def auroc_unknown(score_known: np.ndarray, score_unknown: np.ndarray) -> float:
    """AUROC for 'is unknown' with higher score = more unknown."""
    y = np.r_[np.zeros(len(score_known)), np.ones(len(score_unknown))]
    s = np.r_[score_known, score_unknown]
    return float(roc_auc_score(y, s))


def roc_points(score_known: np.ndarray, score_unknown: np.ndarray):
    y = np.r_[np.zeros(len(score_known)), np.ones(len(score_unknown))]
    s = np.r_[score_known, score_unknown]
    fpr, tpr, thr = roc_curve(y, s)
    return fpr, tpr, thr


def tpr_at_fpr(score_known, score_unknown, fpr_target: float = 0.05) -> float:
    fpr, tpr, _ = roc_points(score_known, score_unknown)
    return float(np.interp(fpr_target, fpr, tpr))


def summarize(values: Sequence[float]) -> Dict[str, float]:
    v = np.asarray(values, dtype=float)
    n = len(v)
    return {"mean": float(v.mean()), "sd": float(v.std(ddof=1)) if n > 1 else 0.0,
            "se": float(v.std(ddof=1) / np.sqrt(n)) if n > 1 else 0.0, "n": n,
            "min": float(v.min()), "max": float(v.max())}
