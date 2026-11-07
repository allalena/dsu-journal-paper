"""Embedding backbone interface.

Any backbone must implement :class:`Backbone`.  Three implementations ship:

* :class:`TorchBackbone`   — torchvision/timm model (FP32 or INT8), used on the
  training workstation and for ONNX export.
* :class:`OnnxBackbone`    — ONNX Runtime session; this is what the Jetson runs
  (TensorRT execution provider when available, CPU otherwise).
* :class:`SyntheticBackbone` — maps synthetic sample ids straight to embeddings,
  so the full pipeline can be exercised without images or weights.

To use the dissertation backbone, subclass :class:`Backbone` and implement
``embed`` (and optionally ``preprocess``) — nothing else in the package
depends on the architecture.
"""

from .base import Backbone, L2Normalize
from .synthetic import SyntheticBackbone

__all__ = ["Backbone", "L2Normalize", "SyntheticBackbone"]


def load_backbone(spec: dict) -> Backbone:
    """Factory used by the CLI.  ``spec`` comes from the YAML config."""
    kind = spec.get("kind", "synthetic")
    kw = {k: v for k, v in spec.items() if k not in ("kind", "compare_int8")}
    if kind == "synthetic":
        return SyntheticBackbone(**kw)
    if kind == "torch":
        from .torch_backbone import TorchBackbone
        return TorchBackbone(**kw)
    if kind == "onnx":
        from .onnx_backbone import OnnxBackbone
        return OnnxBackbone(**kw)
    raise ValueError(f"unknown backbone kind: {kind}")
