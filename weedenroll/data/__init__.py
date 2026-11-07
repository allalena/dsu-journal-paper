"""Dataset acquisition: download → convert to ImageFolder → verify.

    pip install -e ".[data]"                    # pyarrow (+ kaggle, gdown optional)
    weedenroll data --dst data --all            # everything below
    weedenroll data --dst data --set deepweeds  # one set

Three public datasets, each with more than one source so one blocked host
does not stop the run.  ``--source auto`` (default) tries the sources in the
order listed; the first that succeeds is used.  The Hugging Face mirrors are
plain HTTPS downloads (no account, no token); Kaggle needs the Kaggle API
credentials (~/.kaggle/kaggle.json or KAGGLE_USERNAME / KAGGLE_KEY).

Dataset          Sources (in --source auto order)                                   Paper role
---------------  -----------------------------------------------------------------  ----------------------
deepweeds        hf      Project-AgML/DeepWeeds_classification (parquet, 493 MB)    ground, 8 species + Negative
                 kaggle  imsparsh/deepweeds
                 gdrive  images.zip from github.com/AlexOlsen/DeepWeeds (official)
plant_seedlings  hf      Project-AgML/plant_seedlings_aarhus (parquet, 1.7 GB)      ground, 12 species
                 kaggle  vbookshelf/v2-plant-seedlings-dataset
                 aarhus  vision.eng.au.dk NonsegmentedV2.zip (official; the page's
                         download links were broken when last checked, Oct 2026)
cofly            zenodo  record 6697343, CoFly-WeedDB.zip 1.2 GB (official)         UAV aerial, 3 species

All three are redistributed under their original licences (DeepWeeds CC BY 4.0,
Plant Seedlings CC BY-SA 4.0, CoFly-WeedDB CC BY 4.0).  The Hugging Face
CoFly mirror is *not* used: its masks are palette indices, not the official
``labels_1d`` class ids, so the species identity could not be verified.

After download, ``verify`` checks image counts against the published numbers
(DeepWeeds 17 509; Plant Seedlings V2 5 539; CoFly 201 annotated of 366 giving
377 per-plant crops: Field bindweed 283, Johnson grass 73, Purslane 21) so a
partial download is caught before the benchmark runs.
"""
from __future__ import annotations

import csv
import json
import os
import re
import shutil
import subprocess
import sys
import zipfile
from typing import Callable, Dict, Iterable, List, Optional, Sequence

HERE = os.path.dirname(__file__)
DEEPWEEDS_LABELS = os.path.join(HERE, "deepweeds_labels.csv")   # bundled from AlexOlsen/DeepWeeds (CC BY 4.0)

EXPECTED = {
    "deepweeds": {"images": 17509, "species": 9, "non_negative": 8403},
    "plant_seedlings": {"images": 5539, "species": 12},
    # per-plant crops from the 201 masks: Field bindweed 283, Johnson grass 73, Purslane 21
    "cofly": {"annotated": 201, "species": 3, "min_crops_per_species": 15},
}
COFLY_CLASSES = {1: "Johnson grass", 2: "Purslane", 3: "Field bindweed"}   # labels_1d pixel values (official README)

# Canonical species names (Giselsson et al., 2017).  Mirrors spell them
# differently (``black_grass``, ``Black-grass``, ``Scentless Mayweed`` ...);
# every source is mapped onto this list so the ImageFolder is identical
# whichever mirror was used.
SEEDLINGS_SPECIES = ["Black-grass", "Charlock", "Cleavers", "Common chickweed", "Common wheat",
                     "Fat hen", "Loose silky-bent", "Maize", "Scentless mayweed",
                     "Shepherd's purse", "Small-flowered cranesbill", "Sugar beet"]
# label ids of the Hugging Face mirror (alphabetical, see its README)
SEEDLINGS_HF_LABELS = ["black_grass", "charlock", "cleavers", "common_chickweed", "common_wheat", "fat_hen",
                       "loose_silkybent", "maize", "scentless_mayweed", "shepherds_purse",
                       "smallflowered_cranesbill", "sugar_beet"]

HF_RESOLVE = "https://huggingface.co/datasets/{repo}/resolve/main/{path}"
SOURCES = {
    "deepweeds": {
        "hf": {"repo": "Project-AgML/DeepWeeds_classification", "files": ["data/train-00000-of-00001.parquet"]},
        "kaggle": "imsparsh/deepweeds",
        "gdrive": "1xnK3B6K6KekDI55vwJ0vnc2IGoDga9cj",      # images.zip, github.com/AlexOlsen/DeepWeeds
    },
    "plant_seedlings": {
        "hf": {"repo": "Project-AgML/plant_seedlings_aarhus",
               "files": [f"data/train-0000{i}-of-00003.parquet" for i in range(3)]},
        "kaggle": "vbookshelf/v2-plant-seedlings-dataset",
        "aarhus": "https://vision.eng.au.dk/?download=/data/WeedData/NonsegmentedV2.zip",
    },
    "cofly": {"zenodo": "6697343"},
}
IMG_EXT = (".jpg", ".jpeg", ".png")


def _key(name: str) -> str:
    """Spelling-insensitive key: "Shepherd's purse" == "shepherds_purse" == "Shepherds Purse"."""
    return re.sub(r"[^a-z0-9]", "", name.lower())


_SEEDLINGS_CANON = {_key(s): s for s in SEEDLINGS_SPECIES}


def canonical_seedling(name: str) -> str:
    return _SEEDLINGS_CANON.get(_key(name), name)


# --------------------------------------------------------------------------
# download helpers
# --------------------------------------------------------------------------
def _run(cmd: List[str]) -> None:
    print("  $", " ".join(cmd)); subprocess.check_call(cmd)


def _have_kaggle() -> bool:
    if shutil.which("kaggle") is None:
        return False
    return bool(os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY")) or \
        os.path.exists(os.path.expanduser("~/.kaggle/kaggle.json"))


def kaggle_download(dataset: str, dst: str) -> str:
    """Download and unzip a Kaggle dataset into dst."""
    if not _have_kaggle():
        raise RuntimeError("Kaggle API not configured: pip install kaggle, then put kaggle.json in ~/.kaggle "
                           "(Kaggle > Settings > API > Create New Token) or set KAGGLE_USERNAME/KAGGLE_KEY.")
    os.makedirs(dst, exist_ok=True)
    _run(["kaggle", "datasets", "download", "-d", dataset, "-p", dst, "--unzip"])
    return dst


def http_download(url: str, path: str, expect_binary: bool = True) -> str:
    """Streamed download to ``path`` (skipped if it already exists).

    Refuses an HTML response when a binary file is expected: a mirror that
    answers with a login page or a broken redirect must fail loudly instead of
    leaving an HTML file named ``*.zip`` behind.
    """
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    if os.path.exists(path) and os.path.getsize(path) > 0:
        print(f"  exists: {path}"); return path
    print(f"  GET {url}")
    import urllib.request
    req = urllib.request.Request(url, headers={"User-Agent": "weedenroll/0.1 (dataset fetch)"})
    with urllib.request.urlopen(req, timeout=120) as r:
        ctype = r.headers.get("content-type", "")
        if expect_binary and ctype.startswith("text/html"):
            raise RuntimeError(f"{url} returned an HTML page, not a file (content-type {ctype})")
        total = int(r.headers.get("content-length") or 0); done = 0
        with open(path + ".part", "wb") as f:
            while True:
                chunk = r.read(1 << 20)
                if not chunk:
                    break
                f.write(chunk); done += len(chunk)
                if total:
                    print(f"\r  {done / 1e6:8.1f} / {total / 1e6:.1f} MB", end="", flush=True)
        print()
    os.replace(path + ".part", path)
    return path


def gdrive_download(file_id: str, path: str) -> str:
    """Google Drive file (large files need the virus-scan confirm token).

    Uses the direct ``drive.usercontent.google.com`` endpoint; falls back to
    ``gdown`` if Google answers with an HTML page instead of the file.
    """
    if os.path.exists(path) and os.path.getsize(path) > 0:
        return path
    url = f"https://drive.usercontent.google.com/download?id={file_id}&export=download&confirm=t"
    try:
        return http_download(url, path)
    except Exception as e:                     # quota page, changed endpoint ...
        print(f"  direct Drive download failed ({e}); trying gdown")
    try:
        import gdown
    except ImportError:
        raise RuntimeError("pip install gdown  (needed for the Google Drive source)")
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    out = gdown.download(id=file_id, output=path, quiet=False)
    if not out or not os.path.exists(path):
        raise RuntimeError(f"gdown could not fetch Google Drive file {file_id}")
    return path


def zenodo_files(record: str) -> Dict[str, str]:
    """Map filename -> download URL for a Zenodo record via its JSON API."""
    import urllib.request
    with urllib.request.urlopen(f"https://zenodo.org/api/records/{record}", timeout=60) as r:
        meta = json.load(r)
    return {f["key"]: f["links"]["self"] for f in meta["files"]}


def hf_download(repo: str, files: Sequence[str], dst: str) -> List[str]:
    """Fetch the given files of a Hugging Face dataset repo into dst (plain HTTPS, no token)."""
    out = []
    for f in files:
        out.append(http_download(HF_RESOLVE.format(repo=repo, path=f), os.path.join(dst, os.path.basename(f))))
    return out


def unzip(path: str, dst: str) -> str:
    print(f"  unzip {path} -> {dst}")
    with zipfile.ZipFile(path) as z:
        z.extractall(dst)
    return dst


def _find_dir(root: str, name: str) -> Optional[str]:
    if not os.path.isdir(root):
        return None
    for r, ds, _ in os.walk(root):
        if name in ds:
            return os.path.join(r, name)
    return None


def _count_images(root: str) -> Dict[str, int]:
    out = {}
    for cls in sorted(os.listdir(root)):
        p = os.path.join(root, cls)
        if os.path.isdir(p):
            out[cls] = sum(1 for f in os.listdir(p) if f.lower().endswith(IMG_EXT))
    return out


def _n_images(root: str) -> int:
    return sum(_count_images(root).values()) if os.path.isdir(root) else 0


def _place(src: str, dst: str, link: bool) -> bool:
    """Copy (or symlink) src -> dst unless dst already exists.  Returns True if written."""
    if os.path.exists(dst):
        return False
    (os.symlink if link else shutil.copy2)(os.path.abspath(src), dst)
    return True


def _write_unique(d: str, fname: str, data: bytes) -> bool:
    """Write bytes as d/fname; if a *different* file of that name exists, add a suffix."""
    base, ext = os.path.splitext(fname)
    t = os.path.join(d, fname); k = 1
    while os.path.exists(t):
        if os.path.getsize(t) == len(data):
            return False                       # same image already placed
        t = os.path.join(d, f"{base}_{k}{ext}"); k += 1
    with open(t, "wb") as f:
        f.write(data)
    return True


def parquet_to_imagefolder(parquets: Sequence[str], out: str, class_of_row: Callable[[dict], str],
                           default_ext: str = ".jpg") -> int:
    """Hugging Face ``image`` + label parquet shards -> ``out/<class>/<file>``.

    Rows are processed one row group at a time so memory stays bounded.  The
    original file name (``image.path``) is kept when the mirror preserved it.
    """
    try:
        import pyarrow.parquet as pq
    except ImportError as e:
        raise RuntimeError("pip install pyarrow  (needed to read the Hugging Face parquet mirrors)") from e
    n = 0
    for p in parquets:
        pf = pq.ParquetFile(p)
        stem = os.path.splitext(os.path.basename(p))[0]
        for g in range(pf.metadata.num_row_groups):
            for j, row in enumerate(pf.read_row_group(g).to_pylist()):
                cls = class_of_row(row)
                img = row["image"]
                fname = os.path.basename(img.get("path") or "") or f"{stem}_{g}_{j}{default_ext}"
                d = os.path.join(out, cls); os.makedirs(d, exist_ok=True)
                n += _write_unique(d, fname, img["bytes"])
    return n


def _order(source: str, available: Iterable[str]) -> List[str]:
    available = list(available)
    if source == "auto":
        return available
    if source not in available:
        raise ValueError(f"--source {source} not valid here; choose auto or one of {available}")
    return [source]


def _try_sources(name: str, order: Sequence[str], fetch: Callable[[str], int]) -> str:
    last: Optional[Exception] = None
    for src in order:
        print(f"  [{name}] source: {src}")
        try:
            n = fetch(src)
            print(f"  [{name}] {src}: placed {n} new images"); return src
        except Exception as e:                 # fall through to the next mirror
            print(f"  [{name}] {src} failed: {e}"); last = e
    raise RuntimeError(f"[{name}] no source succeeded (last error: {last})")


# --------------------------------------------------------------------------
# DeepWeeds
# --------------------------------------------------------------------------
def _deepweeds_labels() -> Dict[str, str]:
    with open(DEEPWEEDS_LABELS, encoding="utf-8") as f:
        return {row["Filename"]: row["Species"].strip() for row in csv.DictReader(f)}


def _deepweeds_from_images(images: str, out: str, link: bool) -> int:
    """Flat folder of the 17 509 jpgs + bundled label CSV -> ImageFolder."""
    n = 0
    for fn, species in _deepweeds_labels().items():
        s = os.path.join(images, fn)
        if os.path.exists(s):
            d = os.path.join(out, species); os.makedirs(d, exist_ok=True)
            n += _place(s, os.path.join(d, fn), link)
    return n


def _deepweeds_image_dir(raw: str) -> str:
    """Locate the folder that actually holds the jpgs (mirrors nest it differently)."""
    cand = _find_dir(raw, "images")
    if cand and any(f.endswith(".jpg") for f in os.listdir(cand)):
        return cand
    for r, _, fs in os.walk(raw):
        if sum(f.endswith(".jpg") for f in fs) > 100:
            return r
    raise RuntimeError(f"no jpg folder found under {raw}")


def get_deepweeds(dst: str, source: str = "auto", link: bool = False) -> str:
    raw = os.path.join(dst, "raw", "deepweeds"); out = os.path.join(dst, "deepweeds")
    if _n_images(out) >= EXPECTED["deepweeds"]["images"]:
        print(f"  deepweeds: complete in {out}"); return out
    labels = _deepweeds_labels()

    def fetch(src: str) -> int:
        if src == "hf":
            spec = SOURCES["deepweeds"]["hf"]
            files = hf_download(spec["repo"], spec["files"], os.path.join(raw, "hf"))
            mism = []

            def cls(row):
                sp = row["species"].strip()
                fn = os.path.basename(row["image"].get("path") or "")
                if fn in labels and labels[fn] != sp:
                    mism.append(fn)
                return sp
            n = parquet_to_imagefolder(files, out, cls)
            if mism:
                raise RuntimeError(f"{len(mism)} images disagree with the official label file, e.g. {mism[:3]}")
            return n
        if src == "kaggle":
            kaggle_download(SOURCES["deepweeds"]["kaggle"], os.path.join(raw, "kaggle"))
            return _deepweeds_from_images(_deepweeds_image_dir(os.path.join(raw, "kaggle")), out, link)
        if src == "gdrive":
            z = gdrive_download(SOURCES["deepweeds"]["gdrive"], os.path.join(raw, "gdrive", "images.zip"))
            unzip(z, os.path.join(raw, "gdrive", "images"))
            return _deepweeds_from_images(_deepweeds_image_dir(os.path.join(raw, "gdrive")), out, link)
        raise ValueError(src)

    _try_sources("deepweeds", _order(source, SOURCES["deepweeds"]), fetch)
    print(f"  deepweeds: {_n_images(out)} images in {out}")
    return out


# --------------------------------------------------------------------------
# Plant Seedlings (V2, non-segmented, 5 539 images)
# --------------------------------------------------------------------------
def _seedlings_from_folders(root: str, out: str, link: bool) -> int:
    """Find the directory whose children are the 12 class folders and copy them."""
    cls_root = None
    for r, ds, _ in os.walk(root):
        if sum(1 for d in ds if _key(d) in _SEEDLINGS_CANON) >= 6:
            cls_root = r; break
    if cls_root is None:
        raise RuntimeError(f"could not find the 12 class folders under {root}")
    n = 0
    for cls in sorted(os.listdir(cls_root)):
        p = os.path.join(cls_root, cls)
        if not os.path.isdir(p) or _key(cls) not in _SEEDLINGS_CANON:
            continue
        d = os.path.join(out, canonical_seedling(cls)); os.makedirs(d, exist_ok=True)
        for f in os.listdir(p):
            if f.lower().endswith(IMG_EXT):
                n += _place(os.path.join(p, f), os.path.join(d, f), link)
    return n


def get_plant_seedlings(dst: str, source: str = "auto", link: bool = False) -> str:
    raw = os.path.join(dst, "raw", "plant_seedlings"); out = os.path.join(dst, "plant_seedlings")
    if _n_images(out) >= EXPECTED["plant_seedlings"]["images"]:
        print(f"  plant_seedlings: complete in {out}"); return out

    def fetch(src: str) -> int:
        if src == "hf":
            spec = SOURCES["plant_seedlings"]["hf"]
            files = hf_download(spec["repo"], spec["files"], os.path.join(raw, "hf"))
            return parquet_to_imagefolder(files, out, lambda row: canonical_seedling(SEEDLINGS_HF_LABELS[row["label"]]),
                                          default_ext=".png")
        if src == "kaggle":
            kaggle_download(SOURCES["plant_seedlings"]["kaggle"], os.path.join(raw, "kaggle"))
            return _seedlings_from_folders(os.path.join(raw, "kaggle"), out, link)
        if src == "aarhus":
            z = http_download(SOURCES["plant_seedlings"]["aarhus"], os.path.join(raw, "aarhus", "NonsegmentedV2.zip"))
            unzip(z, os.path.join(raw, "aarhus"))
            return _seedlings_from_folders(os.path.join(raw, "aarhus"), out, link)
        raise ValueError(src)

    _try_sources("plant_seedlings", _order(source, SOURCES["plant_seedlings"]), fetch)
    print(f"  plant_seedlings: {_n_images(out)} images in {out}")
    return out


# --------------------------------------------------------------------------
# CoFly-WeedDB (Zenodo 6697343)
# --------------------------------------------------------------------------
def get_cofly(dst: str, source: str = "auto", min_size: int = 48, pad: float = 0.15, link: bool = False) -> str:
    raw = os.path.join(dst, "raw", "cofly"); out = os.path.join(dst, "cofly_crops")
    _order(source, SOURCES["cofly"])             # only zenodo is valid
    if _find_dir(raw, "labels_1d") is None:
        os.makedirs(raw, exist_ok=True)
        files = zenodo_files(SOURCES["cofly"]["zenodo"])
        zips = [k for k in files if k.lower().endswith(".zip")]
        if not zips:
            raise RuntimeError(f"no zip in Zenodo record: {list(files)}")
        for k in zips:
            extract_cofly(http_download(files[k], os.path.join(raw, k)), raw)
    images, masks = _find_dir(raw, "images"), _find_dir(raw, "labels_1d")
    if images is None or masks is None:
        raise RuntimeError(f"CoFly layout not found under {raw} (expected images/ and labels_1d/)")
    n = crop_cofly(images, masks, out, min_size, pad)
    print(f"  cofly: wrote {n} new per-plant crops to {out}")
    return out


def extract_cofly(zip_path: str, raw: str) -> int:
    """Extract only ``images/`` and ``labels_1d/`` from CoFly-WeedDB.zip, renaming each
    file to its ``ID_xxxxx`` stem.

    The archive's file names embed GPS and date (~110 characters) and the
    ``overlay``/``labels`` folders are not needed; the short names keep the
    paths under Windows' 260-character limit and save ~800 MB of disk.
    """
    n = 0
    print(f"  extract images/ and labels_1d/ from {zip_path} -> {raw}")
    with zipfile.ZipFile(zip_path) as z:
        for info in z.infolist():
            parts = info.filename.replace("\\", "/").split("/")
            if info.is_dir() or len(parts) < 2 or parts[-2] not in ("images", "labels_1d"):
                continue
            stem, ext = os.path.splitext(parts[-1])
            m = re.match(r"(ID_\d+)", stem)
            short = (m.group(1) if m else stem[:40]) + ext.lower()
            d = os.path.join(raw, parts[-2]); os.makedirs(d, exist_ok=True)
            t = os.path.join(d, short)
            if not os.path.exists(t):
                with z.open(info) as src, open(t, "wb") as dst:
                    shutil.copyfileobj(src, dst)
                n += 1
    print(f"  extracted {n} files")
    return n


def crop_cofly(images: str, masks: str, out: str, min_size: int = 48, pad: float = 0.15) -> int:
    """Each connected component of a class mask -> one square crop (``<class>/<stem>_<id>_<i>.jpg``)."""
    import numpy as np
    from PIL import Image
    from scipy import ndimage
    n = 0
    for fn in sorted(os.listdir(masks)):
        if not fn.lower().endswith(".png"):
            continue
        stem = os.path.splitext(fn)[0]
        ip = next((os.path.join(images, stem + e) for e in (".png", ".jpg", ".JPG", ".jpeg")
                   if os.path.exists(os.path.join(images, stem + e))), None)
        if ip is None:
            continue
        m = np.asarray(Image.open(os.path.join(masks, fn)))
        if m.ndim == 3:
            m = m[..., 0]
        im = None
        for val, cls in COFLY_CLASSES.items():
            lab, k = ndimage.label(m == val)
            for i in range(1, k + 1):
                ys, xs = np.where(lab == i)
                if len(ys) < 64:                          # drop specks
                    continue
                if im is None:
                    im = Image.open(ip).convert("RGB")
                W, H = im.size
                y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
                s = max(y1 - y0, x1 - x0, min_size) * (1 + 2 * pad)
                cy, cx = (y0 + y1) / 2, (x0 + x1) / 2
                box = (int(max(0, cx - s / 2)), int(max(0, cy - s / 2)), int(min(W, cx + s / 2)), int(min(H, cy + s / 2)))
                if box[2] - box[0] < min_size or box[3] - box[1] < min_size:
                    continue
                d = os.path.join(out, cls); os.makedirs(d, exist_ok=True)
                t = os.path.join(d, f"{stem[:40]}_{val}_{i}.jpg")   # short stem: CoFly names embed GPS + date
                if not os.path.exists(t):
                    im.crop(box).save(t, quality=92); n += 1
    return n


# --------------------------------------------------------------------------
# verify
# --------------------------------------------------------------------------
def verify(dst: str) -> Dict[str, dict]:
    report = {}
    p = os.path.join(dst, "deepweeds")
    if os.path.isdir(p):
        c = _count_images(p); tot = sum(c.values()); nn = tot - c.get("Negative", 0)
        report["deepweeds"] = {"images": tot, "non_negative": nn, "species": len(c),
                               "ok": tot == EXPECTED["deepweeds"]["images"] and len(c) == EXPECTED["deepweeds"]["species"],
                               "per_class": c}
    p = os.path.join(dst, "plant_seedlings")
    if os.path.isdir(p):
        c = _count_images(p); tot = sum(c.values())
        report["plant_seedlings"] = {"images": tot, "species": len(c),
                                     "ok": abs(tot - EXPECTED["plant_seedlings"]["images"]) <= 5
                                     and len(c) == EXPECTED["plant_seedlings"]["species"], "per_class": c}
    p = os.path.join(dst, "cofly_crops")
    if os.path.isdir(p):
        c = _count_images(p)
        report["cofly"] = {"crops": sum(c.values()), "species": len(c),
                           "ok": len(c) == EXPECTED["cofly"]["species"]
                           and min(c.values(), default=0) >= EXPECTED["cofly"]["min_crops_per_species"],
                           "per_class": c}
    for k, v in report.items():
        print(f"  {k:18s} {'OK ' if v['ok'] else 'CHECK'}  " + ", ".join(f"{a}={b}" for a, b in v.items() if a != "per_class"))
    return report


GETTERS: Dict[str, Callable] = {"deepweeds": get_deepweeds, "plant_seedlings": get_plant_seedlings, "cofly": get_cofly}


def main(argv=None):
    import argparse
    p = argparse.ArgumentParser(prog="weedenroll data")
    p.add_argument("--dst", default="data")
    p.add_argument("--set", choices=list(GETTERS), action="append")
    p.add_argument("--all", action="store_true")
    p.add_argument("--source", default="auto", help="auto | hf | kaggle | gdrive | aarhus | zenodo")
    p.add_argument("--link", action="store_true", help="symlink instead of copy (saves disk; folder sources only)")
    p.add_argument("--verify-only", action="store_true")
    a = p.parse_args(argv)
    if not a.verify_only:
        sets = list(GETTERS) if a.all or not a.set else a.set
        for s in sets:
            print(f"[weedenroll data] {s}")
            GETTERS[s](a.dst, a.source, link=a.link)
    print("[weedenroll data] verify")
    rep = verify(a.dst)
    with open(os.path.join(a.dst, "verify.json"), "w", encoding="utf-8") as f:
        json.dump(rep, f, indent=2)
    return 0 if rep and all(v["ok"] for v in rep.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
