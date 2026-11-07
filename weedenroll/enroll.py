"""Field enrollment workflow.

Operator captures k images of an unknown plant → node embeds them →
prototypes written to the store → open-set threshold extended → governance
event logged.  No gradient updates, no network.

``Enroller.enroll`` returns a before/after report so DR5 ("no harm to
existing classes") can be checked on the device if a validation cache is
available, and so the benchmark can record it.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np

from .embed.base import Backbone
from .govern import GovernanceLogger
from .openset import OpenSetRejector
from .store import PrototypeStore


@dataclass
class EnrollReport:
    name: str
    n_shots: int
    platform: str
    embed_ms: float
    write_ms: float
    total_ms: float
    store_bytes: int
    n_species_after: int
    network_calls: int = 0           # always 0 — asserted in tests
    notes: dict = field(default_factory=dict)


class Enroller:
    def __init__(self, backbone: Backbone, store: PrototypeStore,
                 rejector: Optional[OpenSetRejector] = None,
                 logger: Optional[GovernanceLogger] = None):
        self.backbone = backbone
        self.store = store
        self.rejector = rejector
        self.logger = logger

    def enroll(self, name: str, images: Sequence, platform: str = "ground",
               source: str = "field") -> EnrollReport:
        t0 = time.perf_counter()
        emb = self.backbone.embed_all(images)
        t1 = time.perf_counter()
        self.store.enroll(name, emb, source=source, platform=platform)
        if self.rejector is not None:
            self.rejector.add_class_threshold(name, emb)
        t2 = time.perf_counter()
        rep = EnrollReport(name=name, n_shots=len(emb), platform=platform,
                           embed_ms=(t1 - t0) * 1e3, write_ms=(t2 - t1) * 1e3,
                           total_ms=(t2 - t0) * 1e3, store_bytes=self.store.nbytes(),
                           n_species_after=len(self.store.names))
        if self.logger is not None:
            self.logger.log_enrollment(rep)
        return rep

    def enroll_embeddings(self, name: str, emb: np.ndarray, platform: str = "ground",
                          source: str = "field") -> EnrollReport:
        """Same as ``enroll`` but with precomputed embeddings (benchmarks)."""
        t1 = time.perf_counter()
        self.store.enroll(name, emb, source=source, platform=platform)
        if self.rejector is not None:
            self.rejector.add_class_threshold(name, emb)
        t2 = time.perf_counter()
        rep = EnrollReport(name=name, n_shots=len(emb), platform=platform,
                           embed_ms=0.0, write_ms=(t2 - t1) * 1e3, total_ms=(t2 - t1) * 1e3,
                           store_bytes=self.store.nbytes(), n_species_after=len(self.store.names))
        if self.logger is not None:
            self.logger.log_enrollment(rep)
        return rep
