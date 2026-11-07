from __future__ import annotations

import numpy as np

from .base import Backbone, L2Normalize


class SyntheticBackbone(Backbone):
    """Backbone for the synthetic benchmark.

    The synthetic dataset already produces latent vectors (see
    ``bench.datasets.SyntheticWeedDataset``).  This backbone applies the two
    things a real backbone would still do on top of its features:

    * L2 normalisation (contract of :class:`Backbone`), and
    * optional simulated INT8 quantisation (uniform 8-bit quantisation of the
      activation vector) so the FP32-vs-INT8 table can be exercised.

    ``latency_ms`` lets the profiler report a plausible per-frame time when
    no hardware is present; it is clearly labelled as simulated in outputs.
    """

    name = "synthetic"

    def __init__(self, dim: int = 128, precision: str = "fp32",
                 latency_ms: float = 0.0, model_bytes: int = 0):
        self._dim = int(dim)
        self.precision = precision
        self.latency_ms = float(latency_ms)
        self._model_bytes = int(model_bytes)
        self._norm = L2Normalize()

    @property
    def dim(self) -> int:
        return self._dim

    def embed(self, batch) -> np.ndarray:
        x = np.asarray(batch, dtype=np.float32)
        if x.ndim == 1:
            x = x[None, :]
        if self.precision == "int8":
            x = self._fake_quant(x)
        return self._norm(x).astype(np.float32)

    @staticmethod
    def _fake_quant(x: np.ndarray) -> np.ndarray:
        # Symmetric per-tensor 8-bit quantisation, then dequantise.
        scale = np.max(np.abs(x)) / 127.0 if np.max(np.abs(x)) > 0 else 1.0
        q = np.clip(np.round(x / scale), -127, 127)
        return q * scale

    def footprint_bytes(self) -> int | None:
        return self._model_bytes or None
