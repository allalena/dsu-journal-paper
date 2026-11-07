from __future__ import annotations

import abc
from typing import Iterable, Sequence

import numpy as np


class L2Normalize:
    """Row-wise L2 normalisation with a numerical floor."""

    @staticmethod
    def __call__(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
        n = np.linalg.norm(x, axis=-1, keepdims=True)
        return x / np.maximum(n, eps)


class Backbone(abc.ABC):
    """Abstract embedding backbone.

    Contract:
      * ``embed(batch)`` returns an ``(N, D)`` float32 array of L2-normalised
        embeddings.  Normalisation is part of the contract so the store can
        use cosine distance as ``1 - dot``.
      * ``dim`` is the embedding dimension D.
      * ``preprocess(images)`` converts raw inputs (PIL images, file paths, or
        arrays) to whatever ``embed`` expects.  Default is identity.
    """

    name: str = "backbone"

    @property
    @abc.abstractmethod
    def dim(self) -> int: ...

    @abc.abstractmethod
    def embed(self, batch) -> np.ndarray: ...

    def preprocess(self, images: Sequence) -> object:
        return images

    def embed_all(self, items: Iterable, batch_size: int = 32) -> np.ndarray:
        """Embed an iterable in batches; returns ``(N, D)``."""
        out = []
        buf = []
        for it in items:
            buf.append(it)
            if len(buf) == batch_size:
                out.append(self.embed(self.preprocess(buf)))
                buf = []
        if buf:
            out.append(self.embed(self.preprocess(buf)))
        if not out:
            return np.zeros((0, self.dim), dtype=np.float32)
        return np.concatenate(out, axis=0).astype(np.float32)

    # Optional: size on disk, used for DR3 reporting.
    def footprint_bytes(self) -> int | None:
        return None
