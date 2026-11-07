from __future__ import annotations

import os
from typing import Sequence

import numpy as np

from .base import Backbone, L2Normalize

try:
    import onnxruntime as ort
    from PIL import Image
except ImportError as e:  # pragma: no cover
    raise ImportError("OnnxBackbone needs onnxruntime and pillow: "
                      "pip install 'weedenroll[onnx]'") from e

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class OnnxBackbone(Backbone):
    """ONNX Runtime backbone — the deployment path.

    On a Jetson with onnxruntime-gpu built for TensorRT, pass
    ``providers=["TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"]``
    and set ``trt_int8=True`` to request INT8 engine building (requires a
    calibration cache produced by ``scripts/calibrate_int8.py``).
    """

    name = "onnx"

    def __init__(self, model_path: str, image_size: int = 224,
                 providers: Sequence[str] | None = None, trt_int8: bool = False,
                 trt_cache_dir: str | None = None, precision: str = "fp32"):
        self.model_path = model_path
        self.image_size = image_size
        trt_int8 = trt_int8 or precision == "int8"
        self.precision = "int8" if trt_int8 else precision
        avail = ort.get_available_providers()
        if providers is None:
            providers = [p for p in ("TensorrtExecutionProvider", "CUDAExecutionProvider",
                                     "CPUExecutionProvider") if p in avail]
        provider_opts = []
        for p in providers:
            if p == "TensorrtExecutionProvider":
                o = {"trt_fp16_enable": "1", "trt_int8_enable": "1" if trt_int8 else "0",
                     "trt_engine_cache_enable": "1"}
                if trt_cache_dir:
                    o["trt_engine_cache_path"] = trt_cache_dir
                if trt_int8:
                    o["trt_int8_use_native_calibration_table"] = "0"
                provider_opts.append(o)
            else:
                provider_opts.append({})
        self.session = ort.InferenceSession(model_path, providers=list(providers),
                                            provider_options=provider_opts)
        self.input_name = self.session.get_inputs()[0].name
        self._dim = int(self.session.get_outputs()[0].shape[-1])
        self._norm = L2Normalize()

    @property
    def dim(self) -> int:
        return self._dim

    def preprocess(self, images: Sequence) -> np.ndarray:
        s = self.image_size
        out = np.empty((len(images), 3, s, s), dtype=np.float32)
        for i, im in enumerate(images):
            if isinstance(im, (str, os.PathLike)):
                im = Image.open(im).convert("RGB")
            elif isinstance(im, np.ndarray):
                im = Image.fromarray(im).convert("RGB")
            w, h = im.size
            r = int(s * 1.14)
            im = im.resize((r, int(r * h / w)) if w < h else (int(r * w / h), r), Image.BILINEAR)
            w, h = im.size
            l, t = (w - s) // 2, (h - s) // 2
            im = im.crop((l, t, l + s, t + s))
            a = (np.asarray(im, dtype=np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
            out[i] = a.transpose(2, 0, 1)
        return out

    def embed(self, batch) -> np.ndarray:
        x = np.asarray(batch, dtype=np.float32)
        z = self.session.run(None, {self.input_name: x})[0]
        return self._norm(z).astype(np.float32)

    def footprint_bytes(self) -> int | None:
        if not os.path.exists(self.model_path):
            return None
        n = os.path.getsize(self.model_path)
        if os.path.exists(self.model_path + ".data"):     # external-data sidecar, if any
            n += os.path.getsize(self.model_path + ".data")
        return n
