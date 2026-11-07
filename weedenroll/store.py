"""On-device prototype store.

Each species is represented by one or more L2-normalised prototype vectors
(centroids).  Classification is nearest-prototype under cosine distance.
Enrolling a species is a pure write: no gradients, no network.

The store is a flat JSON/NPZ file so it can live on the device's SD card and
be inspected or merged by hand.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple

import numpy as np

from .embed.base import L2Normalize

_norm = L2Normalize()


@dataclass
class SpeciesEntry:
    name: str
    prototypes: np.ndarray            # (P, D) L2-normalised
    n_shots: int                      # images used to build the prototypes
    source: str = "train"             # "train" | "field"
    enrolled_at: float = field(default_factory=time.time)
    platform: str = "ground"          # imaging platform of enrollment images

    def to_json(self) -> dict:
        d = asdict(self)
        d["prototypes"] = self.prototypes.tolist()
        return d

    @staticmethod
    def from_json(d: dict) -> "SpeciesEntry":
        d = dict(d)
        d["prototypes"] = np.asarray(d["prototypes"], dtype=np.float32)
        return SpeciesEntry(**d)


class PrototypeStore:
    def __init__(self, dim: int, prototypes_per_class: int = 1):
        self.dim = int(dim)
        self.k = int(prototypes_per_class)
        self.species: Dict[str, SpeciesEntry] = {}

    # ---- building prototypes -------------------------------------------
    def _make_prototypes(self, emb: np.ndarray) -> np.ndarray:
        emb = _norm(np.asarray(emb, dtype=np.float32))
        if self.k <= 1 or len(emb) <= self.k:
            return _norm(emb.mean(axis=0, keepdims=True))
        # k prototypes via a few Lloyd iterations (no sklearn dependency on device)
        rng = np.random.default_rng(0)
        c = emb[rng.choice(len(emb), self.k, replace=False)].copy()
        for _ in range(10):
            d = 1.0 - emb @ c.T
            a = d.argmin(axis=1)
            for j in range(self.k):
                m = emb[a == j]
                if len(m):
                    c[j] = m.mean(axis=0)
            c = _norm(c)
        return c.astype(np.float32)

    def enroll(self, name: str, emb: np.ndarray, source: str = "field",
               platform: str = "ground", replace: bool = False) -> SpeciesEntry:
        """Add (or replace) a species from its embeddings.  Pure write."""
        if name in self.species and not replace:
            # Merge = add a view: the new prototypes are appended to the
            # species, so nearest-prototype can match either platform.
            old = self.species[name]
            new_p = self._make_prototypes(emb)
            entry = SpeciesEntry(name, np.concatenate([old.prototypes, new_p]), old.n_shots + len(emb),
                                 source, time.time(),
                                 f"{old.platform}+{platform}" if old.platform != platform else platform)
        else:
            entry = SpeciesEntry(name, self._make_prototypes(emb), len(emb), source, time.time(), platform)
        self.species[name] = entry
        return entry

    def remove(self, name: str) -> None:
        self.species.pop(name, None)

    # ---- classification ---------------------------------------------------
    @property
    def names(self) -> List[str]:
        return list(self.species.keys())

    def _matrix(self) -> Tuple[np.ndarray, np.ndarray]:
        """Stack all prototypes -> (M, D) and owner index -> (M,)."""
        ps, owners = [], []
        for i, (n, e) in enumerate(self.species.items()):
            ps.append(e.prototypes)
            owners.append(np.full(len(e.prototypes), i))
        if not ps:
            return np.zeros((0, self.dim), np.float32), np.zeros(0, int)
        return np.concatenate(ps), np.concatenate(owners)

    def distances(self, emb: np.ndarray) -> np.ndarray:
        """Cosine distance from each query to the nearest prototype of each
        species.  Returns ``(N, C)`` ordered as ``self.names``."""
        emb = _norm(np.asarray(emb, dtype=np.float32))
        P, owners = self._matrix()
        if len(P) == 0:
            return np.zeros((len(emb), 0), np.float32)
        d = 1.0 - emb @ P.T                     # (N, M)
        C = len(self.species)
        out = np.full((len(emb), C), np.inf, np.float32)
        for j in range(C):
            m = owners == j
            out[:, j] = d[:, m].min(axis=1)
        return out

    def predict(self, emb: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Returns (predicted index, distance to that prototype)."""
        d = self.distances(emb)
        idx = d.argmin(axis=1)
        return idx, d[np.arange(len(d)), idx]

    def predict_names(self, emb: np.ndarray) -> List[str]:
        idx, _ = self.predict(emb)
        names = self.names
        return [names[i] for i in idx]

    # ---- persistence ----------------------------------------------------
    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"dim": self.dim, "k": self.k,
                       "species": [e.to_json() for e in self.species.values()]}, f)

    @classmethod
    def load(cls, path: str) -> "PrototypeStore":
        with open(path, encoding="utf-8") as f:
            d = json.load(f)
        s = cls(d["dim"], d["k"])
        for e in d["species"]:
            ent = SpeciesEntry.from_json(e)
            s.species[ent.name] = ent
        return s

    def nbytes(self) -> int:
        return sum(e.prototypes.nbytes for e in self.species.values())

    def snapshot(self) -> Dict[str, np.ndarray]:
        """Copy of mean prototype per species (for drift measurement)."""
        return {n: e.prototypes.mean(0).copy() for n, e in self.species.items()}
