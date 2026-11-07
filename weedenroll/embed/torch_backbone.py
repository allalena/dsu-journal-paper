from __future__ import annotations

import os
from typing import Sequence

import numpy as np

from .base import Backbone, L2Normalize

try:
    import torch
    import torch.nn as nn
    from torchvision import models, transforms
    from PIL import Image
except ImportError as e:  # pragma: no cover
    raise ImportError("TorchBackbone needs torch, torchvision and pillow: "
                      "pip install 'weedenroll[torch]'") from e


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class _EmbedHead(nn.Module):
    """Backbone features -> projection -> L2-normalised embedding."""

    def __init__(self, features: nn.Module, in_dim: int, out_dim: int):
        super().__init__()
        self.features = features
        self.proj = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        f = self.features(x)
        if f.ndim > 2:
            f = torch.flatten(torch.nn.functional.adaptive_avg_pool2d(f, 1), 1)
        z = self.proj(f)
        return torch.nn.functional.normalize(z, dim=-1)


def build_model(arch: str = "mobilenet_v3_small", out_dim: int = 128,
                pretrained: bool = False) -> nn.Module:
    """Build a compact embedding model.

    Replace this with the dissertation backbone by returning any module whose
    forward yields an ``(N, out_dim)`` L2-normalised tensor.
    """
    weights = "DEFAULT" if pretrained else None
    if arch == "mobilenet_v3_small":
        m = models.mobilenet_v3_small(weights=weights)
        feats, in_dim = m.features, 576
    elif arch == "mobilenet_v3_large":
        m = models.mobilenet_v3_large(weights=weights)
        feats, in_dim = m.features, 960
    elif arch == "efficientnet_b0":
        m = models.efficientnet_b0(weights=weights)
        feats, in_dim = m.features, 1280
    elif arch in ("resnet18", "resnet34", "resnet50", "resnet101"):
        m = getattr(models, arch)(weights=weights)
        feats = nn.Sequential(*list(m.children())[:-1])     # drop the ImageNet fc
        in_dim = m.fc.in_features                          # 512 (18/34) or 2048 (50/101)
    else:
        raise ValueError(f"unknown arch {arch}")
    return _EmbedHead(feats, in_dim, out_dim)


class TorchBackbone(Backbone):
    name = "torch"

    def __init__(self, arch: str = "mobilenet_v3_small", out_dim: int = 128,
                 weights_path: str | None = None, pretrained: bool = False,
                 image_size: int = 224, device: str = "cpu", precision: str = "fp32"):
        self._dim = out_dim
        self.arch = arch
        self.image_size = image_size
        self.device = torch.device(device if torch.cuda.is_available() or device == "cpu" else "cpu")
        self.precision = precision
        sd = None
        if weights_path and os.path.exists(weights_path):
            sd = torch.load(weights_path, map_location="cpu")
            if isinstance(sd, dict) and "arch" in sd:          # checkpoint from weedenroll train
                arch, out_dim = sd["arch"], sd.get("out_dim", out_dim)
                self._dim = out_dim; self.arch = arch
        elif weights_path:
            print(f"[TorchBackbone] WARNING: weights_path {weights_path!r} not found; using an UNTRAINED "
                  f"projection head (run `weedenroll train` first for the paper's numbers)")
        if sd is None:
            # No checkpoint: the projection head is random.  Seed it so that two
            # processes (e.g. `enroll` then `predict`) build the same head and a
            # saved prototype store stays meaningful.
            torch.manual_seed(0)
        self.model = build_model(arch, out_dim, pretrained and sd is None)
        if sd is not None:
            missing, unexpected = self.model.load_state_dict(sd.get("state_dict", sd), strict=False)
            if missing:
                print(f"[TorchBackbone] warning: {len(missing)} missing keys when loading {weights_path}")
        if precision == "int8":
            # Dynamic quantisation of linear layers; quantized kernels exist only on
            # CPU, so the INT8 comparison always runs there (the Jetson INT8 numbers
            # come from the ONNX/TensorRT path, scripts/export_onnx.py + profiler).
            self.device = torch.device("cpu")
            self.model = torch.quantization.quantize_dynamic(
                self.model.cpu().eval(), {nn.Linear}, dtype=torch.qint8)
        self.model.eval().to(self.device)
        self.tf = transforms.Compose([
            transforms.Resize(int(image_size * 1.14)),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
        self._weights_path = weights_path

    @property
    def dim(self) -> int:
        return self._dim

    def preprocess(self, images: Sequence):
        tensors = []
        for im in images:
            if isinstance(im, (str, os.PathLike)):
                im = Image.open(im).convert("RGB")
            elif isinstance(im, np.ndarray):
                im = Image.fromarray(im).convert("RGB")
            tensors.append(self.tf(im))
        return torch.stack(tensors)

    @torch.no_grad()
    def embed(self, batch) -> np.ndarray:
        if not torch.is_tensor(batch):
            batch = self.preprocess(batch)
        z = self.model(batch.to(self.device))
        return z.cpu().numpy().astype(np.float32)

    def footprint_bytes(self) -> int | None:
        if self._weights_path and os.path.exists(self._weights_path):
            return os.path.getsize(self._weights_path)
        n = sum(p.numel() for p in self.model.parameters())
        return n * (1 if self.precision == "int8" else 4)

    def export_onnx(self, path: str, opset: int = 18):
        export_onnx(self.model, path, self.image_size, opset)
        return path


def export_onnx(model: nn.Module, path: str, image_size: int = 224, opset: int = 18) -> str:
    """ONNX export with a dynamic batch axis (works with the dynamo and legacy exporters).

    Weights are embedded in the single ``.onnx`` file (no ``.onnx.data`` sidecar)
    so one file can be copied to the Jetson.
    """
    model = model.cpu().eval()
    dummy = torch.randn(1, 3, image_size, image_size)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    try:
        batch = torch.export.Dim("batch", min=1, max=256)
        kw = dict(opset_version=opset, input_names=["image"], output_names=["embedding"],
                  dynamic_shapes={"x": {0: batch}}, dynamo=True)
        try:
            torch.onnx.export(model, (dummy,), path, external_data=False, **kw)
        except TypeError:                                   # older torch: no external_data keyword
            torch.onnx.export(model, (dummy,), path, **kw)
    except Exception:
        torch.onnx.export(model, dummy, path, opset_version=opset, input_names=["image"], output_names=["embedding"],
                          dynamic_axes={"image": {0: "batch"}, "embedding": {0: "batch"}}, dynamo=False)
    side = path + ".data"
    if os.path.exists(side):                                # fold any sidecar back into one file
        import onnx
        m = onnx.load(path, load_external_data=True)
        onnx.save_model(m, path, save_as_external_data=False)
        os.remove(side)
    return path
