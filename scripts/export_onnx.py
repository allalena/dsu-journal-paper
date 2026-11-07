"""Export the torch backbone to ONNX for the Jetson.

    python scripts/export_onnx.py --config configs/real.yaml --out weights/student.onnx

Then on the Jetson:
    trtexec --onnx=weights/student.onnx --int8 --calib=weights/calib.cache --saveEngine=weights/student_int8.plan
or let onnxruntime's TensorRT EP build the engine (configs/jetson.yaml).
"""
import argparse, yaml
from weedenroll.embed.torch_backbone import TorchBackbone

p = argparse.ArgumentParser()
p.add_argument("--config", required=True); p.add_argument("--out", default="weights/student.onnx")
a = p.parse_args()
with open(a.config, encoding="utf-8") as f:
    cfg = yaml.safe_load(f)["backbone"]
bb = TorchBackbone(**{k: v for k, v in cfg.items() if k not in ("kind", "compare_int8", "precision")}, precision="fp32")
print("exported", bb.export_onnx(a.out))
