import io
import os
import tempfile

import numpy as np
import pytest
from PIL import Image

from weedenroll.data import (crop_cofly, verify, parquet_to_imagefolder, canonical_seedling,
                             main as data_main, COFLY_CLASSES, DEEPWEEDS_LABELS, SEEDLINGS_SPECIES,
                             SEEDLINGS_HF_LABELS)


def test_bundled_deepweeds_labels():
    import csv
    with open(DEEPWEEDS_LABELS, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 17509
    assert len({r["Species"] for r in rows}) == 9


def test_seedling_names_are_canonical_across_mirrors():
    # Hugging Face ids, Kaggle/Aarhus folder names and the canonical list all map to the same 12 names
    assert [canonical_seedling(x) for x in SEEDLINGS_HF_LABELS] == SEEDLINGS_SPECIES
    kaggle = ["Black-grass", "Charlock", "Cleavers", "Common Chickweed", "Common wheat", "Fat Hen", "Loose Silky-bent",
              "Maize", "Scentless Mayweed", "Shepherds Purse", "Small-flowered Cranesbill", "Sugar beet"]
    assert [canonical_seedling(x) for x in kaggle] == SEEDLINGS_SPECIES
    assert canonical_seedling("not a seedling") == "not a seedling"


def _jpeg_bytes(seed: int, size=(32, 32)) -> bytes:
    rng = np.random.default_rng(seed)
    buf = io.BytesIO()
    Image.fromarray(rng.integers(0, 255, (*size, 3), dtype=np.uint8)).save(buf, "JPEG")
    return buf.getvalue()


def test_parquet_to_imagefolder_keeps_names_and_classes():
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    rows = [{"image": {"bytes": _jpeg_bytes(i), "path": f"img{i % 3}.jpg"}, "label": i % 2, "species": ["A", "B"][i % 2]}
            for i in range(6)]
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "shard.parquet")
        pq.write_table(pa.Table.from_pylist(rows), p, row_group_size=4)      # two row groups
        out = os.path.join(d, "out")
        n = parquet_to_imagefolder([p], out, lambda r: r["species"])
        assert n == 6
        assert sorted(os.listdir(out)) == ["A", "B"]
        assert len(os.listdir(os.path.join(out, "A"))) == 3 and len(os.listdir(os.path.join(out, "B"))) == 3
        assert "img0.jpg" in os.listdir(os.path.join(out, "A"))               # original file name kept
        # a second run places nothing new (idempotent)
        assert parquet_to_imagefolder([p], out, lambda r: r["species"]) == 0


def test_cofly_crops_from_masks():
    with tempfile.TemporaryDirectory() as d:
        im, mk = os.path.join(d, "images"), os.path.join(d, "labels_1d"); os.makedirs(im); os.makedirs(mk)
        rgb = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)
        mask = np.zeros((720, 1280), np.uint8)
        mask[100:200, 100:220] = 1          # Johnson grass
        mask[400:460, 600:700] = 2          # Purslane
        mask[50:120, 900:1000] = 3          # Field bindweed
        mask[600:605, 10:15] = 3            # speck (dropped)
        Image.fromarray(rgb).save(os.path.join(im, "ID_00001.png"))
        Image.fromarray(mask).save(os.path.join(mk, "ID_00001.png"))
        out = os.path.join(d, "crops")
        n = crop_cofly(im, mk, out)
        assert n == 3
        assert sorted(os.listdir(out)) == sorted(COFLY_CLASSES.values())
        assert crop_cofly(im, mk, out) == 0                                   # idempotent


def test_verify_flags_partial_set():
    with tempfile.TemporaryDirectory() as d:
        src = os.path.join(d, "deepweeds", "Lantana"); os.makedirs(src)
        for i in range(3):
            Image.fromarray(np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)).save(os.path.join(src, f"{i}.jpg"))
        rep = verify(d)
        assert rep["deepweeds"]["images"] == 3 and rep["deepweeds"]["ok"] is False   # partial set is flagged


def test_data_main_verify_only_exit_code():
    with tempfile.TemporaryDirectory() as d:
        src = os.path.join(d, "deepweeds", "Lantana"); os.makedirs(src)
        Image.fromarray(np.zeros((8, 8, 3), np.uint8)).save(os.path.join(src, "a.jpg"))
        assert data_main(["--dst", d, "--verify-only"]) == 1          # incomplete -> non-zero
        assert os.path.exists(os.path.join(d, "verify.json"))
