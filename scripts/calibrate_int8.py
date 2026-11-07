"""Static INT8 quantisation of the ONNX backbone with onnxruntime (CPU/any
device), as an alternative to TensorRT's own calibration.

    python scripts/calibrate_int8.py --onnx weights/student.onnx --images data/deepweeds \\
        --out weights/student_int8.onnx --n 300
"""
import argparse, os, random
import numpy as np
from onnxruntime.quantization import quantize_static, CalibrationDataReader, QuantType, QuantFormat
from weedenroll.embed.onnx_backbone import OnnxBackbone

p = argparse.ArgumentParser()
p.add_argument("--onnx", required=True); p.add_argument("--images", required=True)
p.add_argument("--out", required=True); p.add_argument("--n", type=int, default=300)
a = p.parse_args()

files = [os.path.join(r, f) for r, _, fs in os.walk(a.images) for f in fs if f.lower().endswith((".jpg", ".png", ".jpeg"))]
random.seed(0); random.shuffle(files); files = files[:a.n]
bb = OnnxBackbone(a.onnx, providers=["CPUExecutionProvider"])


class Reader(CalibrationDataReader):
    def __init__(self):
        self.it = iter(files)

    def get_next(self):
        f = next(self.it, None)
        return None if f is None else {bb.input_name: bb.preprocess([f])}


# Fold constants / infer shapes first (onnxruntime's recommended pre-processing);
# falls back to the raw model if the helper is unavailable.
src = a.onnx
try:
    from onnxruntime.quantization.shape_inference import quant_pre_process
    src = os.path.splitext(a.out)[0] + "_prep.onnx"
    quant_pre_process(a.onnx, src, skip_symbolic_shape=True)
except Exception as e:                      # pragma: no cover
    print("quant_pre_process skipped:", e); src = a.onnx
quantize_static(src, a.out, Reader(), quant_format=QuantFormat.QDQ,
                activation_type=QuantType.QUInt8, weight_type=QuantType.QInt8, per_channel=True)
if src != a.onnx and os.path.exists(src):
    os.remove(src)
print("wrote", a.out, os.path.getsize(a.out) / 1e6, "MB")
