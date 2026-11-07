"""Train the embedding backbone once: teacher → distilled compact student.

    weedenroll train --config configs/train.yaml

Protocol (Materials and Methods):

* **Base / held-out split.**  The backbone is trained only on the *base*
  species of DeepWeeds (plus the Negative class); ``holdout_species`` never
  appear in training, so the within-dataset enrollment experiment (DR4) is a
  true novel-class test.  Plant Seedlings and CoFly species are never seen
  during training either.
* **Teacher.**  ResNet-50 (ImageNet init) with a 128-d L2-normalised
  projection, trained with a cosine-softmax classifier (normalised-softmax /
  ArcFace-style margin) so that the embedding space is metric.
* **Student.**  MobileNetV3-Small (ImageNet init) with the same head,
  trained with the same cosine-softmax loss **plus** embedding distillation:
  ``1 - cos(z_student, z_teacher)`` on every image, and relational KD on the
  batch similarity matrix.  No weights from the teacher are copied; only its
  embedding geometry is transferred.
* **Output.**  ``weights/teacher.pt``, ``weights/student_distilled.pt`` (what
  configs/real.yaml loads) and ``weights/student.onnx`` for the Jetson.

Set ``epochs`` small (1–2) with ``max_images_per_class`` to smoke-test on a
laptop; the paper run uses the defaults in configs/train.yaml.
"""
from __future__ import annotations

import json
import math
import os
import random
import time
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, Dataset as TorchDataset
    from torchvision import transforms
    from PIL import Image
except ImportError as e:  # pragma: no cover
    raise ImportError("training needs torch, torchvision, pillow: pip install 'weedenroll[torch]'") from e

from .embed.torch_backbone import build_model, IMAGENET_MEAN, IMAGENET_STD

IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp")


# --------------------------------------------------------------------------
# data
# --------------------------------------------------------------------------
def list_imagefolder(root: str, exclude: Sequence[str] = (), max_per_class: Optional[int] = None,
                     seed: int = 0) -> Tuple[List[str], List[int], List[str]]:
    rng = random.Random(seed)
    classes = sorted(d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d)) and d not in exclude)
    files, labels = [], []
    for i, c in enumerate(classes):
        fs = sorted(f for f in os.listdir(os.path.join(root, c)) if f.lower().endswith(IMG_EXT))
        rng.shuffle(fs)
        if max_per_class:
            fs = fs[:max_per_class]
        files += [os.path.join(root, c, f) for f in fs]; labels += [i] * len(fs)
    return files, labels, classes


class ImageListDataset(TorchDataset):
    def __init__(self, files: List[str], labels: List[int], tf):
        self.files, self.labels, self.tf = files, labels, tf

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        return self.tf(Image.open(self.files[i]).convert("RGB")), self.labels[i]


def make_transforms(size: int = 224):
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(size, scale=(0.6, 1.0)),
        transforms.RandomHorizontalFlip(), transforms.RandomVerticalFlip(),
        transforms.RandomRotation(20),
        transforms.ColorJitter(0.3, 0.3, 0.2, 0.05),
        transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)])
    eval_tf = transforms.Compose([
        transforms.Resize(int(size * 1.14)), transforms.CenterCrop(size),
        transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)])
    return train_tf, eval_tf


def stratified_split(labels: List[int], val_frac: float, seed: int) -> Tuple[List[int], List[int]]:
    rng = np.random.default_rng(seed); labels = np.asarray(labels)
    tr, va = [], []
    for c in np.unique(labels):
        ix = np.where(labels == c)[0]; rng.shuffle(ix)
        n = max(1, int(len(ix) * val_frac)); va += ix[:n].tolist(); tr += ix[n:].tolist()
    return sorted(tr), sorted(va)


# --------------------------------------------------------------------------
# losses
# --------------------------------------------------------------------------
class CosineClassifier(nn.Module):
    """Normalised-softmax head with additive margin (ArcFace-lite).  Keeps the
    embedding metric so nearest-prototype works after training."""

    def __init__(self, dim: int, n_classes: int, scale: float = 30.0, margin: float = 0.2):
        super().__init__()
        self.w = nn.Parameter(torch.randn(n_classes, dim) * 0.01)
        self.scale, self.margin = scale, margin

    def forward(self, z, y=None):
        cos = F.linear(z, F.normalize(self.w, dim=-1))          # z already L2-normalised
        if y is None:
            return self.scale * cos
        onehot = F.one_hot(y, cos.shape[1]).float()
        return self.scale * (cos - self.margin * onehot)


def distill_loss(zs: torch.Tensor, zt: torch.Tensor, rkd_weight: float = 1.0) -> torch.Tensor:
    """Embedding KD: per-sample cosine + relational KD on the batch similarity matrix."""
    point = (1.0 - (zs * zt).sum(-1)).mean()
    ss, st = zs @ zs.t(), zt @ zt.t()
    rel = F.smooth_l1_loss(ss, st)
    return point + rkd_weight * rel


# --------------------------------------------------------------------------
# training loops
# --------------------------------------------------------------------------
def _device(pref: str = "auto") -> torch.device:
    if pref == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(pref)


@torch.no_grad()
def evaluate(model: nn.Module, head: CosineClassifier, loader: DataLoader, device) -> Dict[str, float]:
    model.eval(); head.eval()
    correct = n = 0; Z, Y = [], []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        z = model(x); logits = head(z)
        correct += (logits.argmax(1) == y).sum().item(); n += len(y)
        Z.append(z.cpu()); Y.append(y.cpu())
    Z, Y = torch.cat(Z), torch.cat(Y)
    # nearest-prototype accuracy (what the deployed system does)
    protos = torch.stack([F.normalize(Z[Y == c].mean(0), dim=-1) for c in range(int(Y.max()) + 1)])
    npa = ((Z @ protos.t()).argmax(1) == Y).float().mean().item()
    return {"acc_softmax": correct / max(n, 1), "acc_nearest_prototype": npa}


def _epoch(model, head, loader, opt, sched, device, teacher=None, kd_weight=1.0, rkd_weight=1.0,
           log_every=50) -> float:
    model.train(); head.train(); tot = 0.0; t0 = time.time()
    for i, (x, y) in enumerate(loader):
        x, y = x.to(device), y.to(device)
        z = model(x)
        loss = F.cross_entropy(head(z, y), y)
        if teacher is not None:
            with torch.no_grad():
                zt = teacher(x)
            loss = loss + kd_weight * distill_loss(z, zt, rkd_weight)
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        if sched is not None:
            sched.step()
        tot += loss.item()
        if (i + 1) % log_every == 0:
            print(f"    step {i + 1}/{len(loader)}  loss {tot / (i + 1):.4f}  {time.time() - t0:.0f}s")
    return tot / max(len(loader), 1)


def train_stage(name: str, arch: str, out_dim: int, train_loader, val_loader, n_classes: int, epochs: int,
                lr: float, device, weights_out: str, teacher: Optional[nn.Module] = None,
                kd_weight: float = 1.0, rkd_weight: float = 1.0, pretrained: bool = True,
                margin: float = 0.2, scale: float = 30.0, log_every: int = 50) -> nn.Module:
    model = build_model(arch, out_dim, pretrained).to(device)
    head = CosineClassifier(out_dim, n_classes, scale, margin).to(device)
    params = [{"params": model.parameters(), "lr": lr}, {"params": head.parameters(), "lr": lr * 10}]
    opt = torch.optim.AdamW(params, weight_decay=1e-4)
    steps = epochs * len(train_loader)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[lr, lr * 10], total_steps=max(steps, 1), pct_start=0.1)
    best = -1.0; hist = []
    for ep in range(epochs):
        print(f"[{name}] epoch {ep + 1}/{epochs}")
        tl = _epoch(model, head, train_loader, opt, sched, device, teacher, kd_weight, rkd_weight, log_every)
        ev = evaluate(model, head, val_loader, device) if val_loader is not None else {}
        hist.append({"epoch": ep + 1, "train_loss": tl, **ev}); print(f"    {hist[-1]}")
        score = ev.get("acc_nearest_prototype", -tl)
        if score > best:
            best = score
            os.makedirs(os.path.dirname(os.path.abspath(weights_out)), exist_ok=True)
            torch.save({"state_dict": model.state_dict(), "arch": arch, "out_dim": out_dim,
                        "head": head.state_dict(), "history": hist}, weights_out)
    sd = torch.load(weights_out, map_location="cpu")
    model.load_state_dict(sd["state_dict"]); model.eval()
    with open(weights_out + ".json", "w", encoding="utf-8") as f:
        json.dump({"arch": arch, "out_dim": out_dim, "best_val": best, "history": hist}, f, indent=2)
    return model


def run(cfg: dict) -> Dict[str, str]:
    seed = cfg.get("seed", 0); torch.manual_seed(seed); random.seed(seed); np.random.seed(seed)
    device = _device(cfg.get("device", "auto")); print(f"[train] device {device}")
    d = cfg["data"]
    files, labels, classes = list_imagefolder(d["root"], exclude=list(d.get("exclude", [])) + list(d.get("holdout_species", [])),
                                              max_per_class=d.get("max_images_per_class"), seed=seed)
    print(f"[train] base classes ({len(classes)}): {classes}")
    print(f"[train] held-out species (never trained on): {d.get('holdout_species', [])}")
    tr, va = stratified_split(labels, d.get("val_frac", 0.1), seed)
    train_tf, eval_tf = make_transforms(cfg.get("image_size", 224))
    bs, nw = cfg.get("batch_size", 64), cfg.get("num_workers", 4)
    train_loader = DataLoader(ImageListDataset([files[i] for i in tr], [labels[i] for i in tr], train_tf),
                              bs, shuffle=True, num_workers=nw, drop_last=True, pin_memory=device.type == "cuda")
    val_loader = DataLoader(ImageListDataset([files[i] for i in va], [labels[i] for i in va], eval_tf),
                            bs, shuffle=False, num_workers=nw)
    out = cfg.get("weights_dir", "weights"); os.makedirs(out, exist_ok=True)
    t = cfg.get("teacher", {}); s = cfg.get("student", {})
    teacher_path = os.path.join(out, "teacher.pt")
    if t.get("weights") and os.path.exists(t["weights"]):
        sd = torch.load(t["weights"], map_location="cpu")
        teacher = build_model(sd["arch"], sd["out_dim"], False); teacher.load_state_dict(sd["state_dict"])
        teacher = teacher.to(device).eval(); print(f"[train] loaded teacher {t['weights']}")
    else:
        teacher = train_stage("teacher", t.get("arch", "resnet50"), cfg.get("out_dim", 128), train_loader, val_loader,
                              len(classes), t.get("epochs", 15), t.get("lr", 3e-4), device, teacher_path,
                              pretrained=t.get("pretrained", True), margin=t.get("margin", 0.2),
                              scale=t.get("scale", 30.0), log_every=cfg.get("log_every", 50))
    for p in teacher.parameters():
        p.requires_grad_(False)
    student_path = os.path.join(out, "student_distilled.pt")
    student = train_stage("student", s.get("arch", "mobilenet_v3_small"), cfg.get("out_dim", 128), train_loader,
                          val_loader, len(classes), s.get("epochs", 30), s.get("lr", 1e-3), device, student_path,
                          teacher=teacher, kd_weight=s.get("kd_weight", 1.0), rkd_weight=s.get("rkd_weight", 1.0),
                          pretrained=s.get("pretrained", True), margin=s.get("margin", 0.2),
                          scale=s.get("scale", 30.0), log_every=cfg.get("log_every", 50))
    # ONNX for the Jetson
    onnx_path = os.path.join(out, "student.onnx")
    try:
        from .embed.torch_backbone import export_onnx
        export_onnx(student, onnx_path, cfg.get("image_size", 224))
        print(f"[train] exported {onnx_path}")
    except Exception as e:  # pragma: no cover
        print(f"[train] ONNX export skipped: {e}"); onnx_path = ""
    meta = {"classes": classes, "holdout_species": d.get("holdout_species", []), "teacher": teacher_path,
            "student": student_path, "onnx": onnx_path, "out_dim": cfg.get("out_dim", 128)}
    with open(os.path.join(out, "train_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    return meta
