# weedenroll

Offline few-shot weed recognition with field enrollment on edge devices.
Companion code for the CIGR Journal paper and the IGI Global chapter
*Train Once, Enroll in the Field: Offline Few-Shot Weed Recognition with
Prototype Enrollment on UAV-Class Edge Hardware*.

The system: a compact embedding backbone is trained once (distillation →
ONNX, FP16/INT8 TensorRT engine on the target), and species live as **prototypes** in an on-device store. A new weed
is enrolled in the field from 5–10 images with **no retraining, no network**.
Unknown plants are flagged by distance-based **open-set rejection**, and a
local **governance log** records confidence, rejection rate and prototype
drift per session.

```
weedenroll/
  embed/        Backbone interface: torch (train/export), onnx (Jetson), synthetic (CI)
  store.py      Prototype store — nearest-prototype classification, JSON persistence
  openset.py    Distance threshold calibration + rejection
  enroll.py     Field enrollment workflow (write-only)
  govern.py     Session governance log (JSONL)
  data/         Dataset download (Hugging Face / Kaggle / official), ImageFolder conversion, verification
  bench/        Datasets, episodic benchmark (DR1,4,5,6,7), hardware profiler (DR2,3)
  cost.py       Season cost model (DR8) with sensitivity analysis
  report.py     CIGR-style figures (6) and tables (5) + DOCX bundle
  cli.py        weedenroll run | profile | cost | enroll | predict
configs/        synthetic.yaml (pipeline self-test), real.yaml (the paper), train.yaml, jetson.yaml
colab/          notebook that runs the paper pipeline on a Colab GPU
scripts/        ONNX export, INT8 calibration (dataset preparation is `weedenroll data`)
tests/          pytest suite
```

## Quick start (no data, no GPU)

```bash
pip install -e ".[dev,docx,torch,onnx]"
pytest -q
weedenroll run --config configs/synthetic.yaml
```

`outputs/synthetic/` then contains every figure (`figures/*.png|pdf` +
`captions.json`), every table (`tables/*.csv|md`), raw per-episode results
(`results/*.csv`), the governance logs, and `paper_assets.docx` with all
figures and tables in one file. **Everything from this config is labelled
synthetic** — it exercises the pipeline; the numbers are not the paper's.

## The paper run (real data) — four commands

```bash
# 1. Data: download (no account needed), convert to ImageFolder, verify counts against the published numbers
pip install -e ".[data]"               # pyarrow for the Hugging Face mirrors; kaggle/gdown only for the alternates
weedenroll data --dst data --all

# 2. Backbone: ResNet-50 teacher → distilled MobileNetV3-Small student (GPU: ~1–2 h on DeepWeeds)
weedenroll train --config configs/train.yaml

# 3. Optional: hardware profile on a device (copy weights/student.onnx there first).
#    Without it, DR2 is reported as *projected* (see below).
weedenroll profile --config configs/jetson.yaml --out outputs/hardware_results.json

# 4. Everything else: benchmarks, cost model, figures, tables, DOCX
weedenroll run --config configs/real.yaml

# 5. Re-build figures, tables and the DOCX from saved results (no benchmarks re-run)
weedenroll report --config configs/real.yaml
```

### Datasets and sources

`weedenroll data --source auto` tries the sources below in order and uses the
first that succeeds (`--source hf|kaggle|gdrive|aarhus|zenodo` forces one).
All links were checked on 17 Oct 2026.

| Set | Images | Platform | Sources, in `auto` order | Licence |
|---|---|---|---|---|
| DeepWeeds (Olsen et al., 2019) | 17 509, 8 species + Negative | ground | 1. Hugging Face [`Project-AgML/DeepWeeds_classification`](https://huggingface.co/datasets/Project-AgML/DeepWeeds_classification) (one 493 MB parquet, no account; keeps the original file names and species strings) 2. Kaggle [`imsparsh/deepweeds`](https://www.kaggle.com/datasets/imsparsh/deepweeds) (token) 3. official `images.zip` on Google Drive from [AlexOlsen/DeepWeeds](https://github.com/AlexOlsen/DeepWeeds) (491 MB) | CC BY 4.0 |
| Plant Seedlings V2 (Giselsson et al., 2017) | 5 539, 12 species | ground | 1. Hugging Face [`Project-AgML/plant_seedlings_aarhus`](https://huggingface.co/datasets/Project-AgML/plant_seedlings_aarhus) (three parquets, 1.7 GB, no account) 2. Kaggle [`vbookshelf/v2-plant-seedlings-dataset`](https://www.kaggle.com/datasets/vbookshelf/v2-plant-seedlings-dataset) (token) 3. official [Aarhus University page](https://vision.eng.au.dk/plant-seedlings-dataset/) `NonsegmentedV2.zip` (its download links were broken when checked) | CC BY-SA 4.0 |
| CoFly-WeedDB (Krestenitis et al., 2022) | 201 annotated UAV images (of 366) → 377 per-plant crops (Field bindweed 283, Johnson grass 73, Purslane 21) | UAV aerial | official [Zenodo record 6697343](https://zenodo.org/records/6697343), `CoFly-WeedDB.zip` (1.2 GB; `images/`, `labels_1d/` with 0 background, 1 Johnson grass, 2 purslane, 3 field bindweed) | CC BY 4.0 |

Direct file URLs (what the code fetches):

```
https://huggingface.co/datasets/Project-AgML/DeepWeeds_classification/resolve/main/data/train-00000-of-00001.parquet
https://huggingface.co/datasets/Project-AgML/plant_seedlings_aarhus/resolve/main/data/train-00000-of-00003.parquet   (…00001…, …00002…)
https://drive.usercontent.google.com/download?id=1xnK3B6K6KekDI55vwJ0vnc2IGoDga9cj&export=download&confirm=t         (DeepWeeds images.zip)
https://zenodo.org/api/records/6697343/files/CoFly-WeedDB.zip/content
```

`weedenroll data` ends with a verification step that fails (exit 1) if any
set is incomplete, and writes `data/verify.json` with per-class counts for
Table 1. The DeepWeeds label file is bundled (`weedenroll/data/deepweeds_labels.csv`);
the Hugging Face copy is cross-checked against it file by file. Plant
Seedlings class folders are renamed to one canonical spelling whichever
mirror was used. The Hugging Face CoFly mirror is deliberately not used: its
masks are palette indices, not the official `labels_1d` class ids.

Other public weed sets that fit the same ImageFolder protocol if you want a
fourth source: [Weed25](https://github.com/Pm-Wang/Weed25) (25 species, ground),
[CottonWeedID15](https://www.kaggle.com/datasets/yuzhenlu/cottonweedid15) (15 species, ground),
[WeedCrop](https://universe.roboflow.com/) UAV sets on Roboflow Universe. Point a
`data.sets` entry at any `<root>/<species>/<image>` folder.

### Held-out protocol (why the numbers are a real novel-class test)

`configs/train.yaml` excludes three DeepWeeds species (`holdout_species`)
from backbone training. `configs/real.yaml` restricts the within-dataset
enrollment episodes to those species, so DR4/DR5/DR6 are measured on species
the backbone has never seen. Plant Seedlings and CoFly are never used in
training. Keep the two lists identical; `weedenroll run` also reads them from
`weights/train_meta.json`.

### DR7 uses real UAV imagery only

No public dataset images the DeepWeeds species from a UAV, so `configs/real.yaml`
has no paired ground/aerial set and DR7 is reported from CoFly-WeedDB alone:
species seen only from the air, enrolled from k UAV crops with the
ground-trained backbone (Figure 4, single panel). If you collect paired UAV
images of any DeepWeeds species, add an `aerial_shared` entry to `data.sets`
and the degradation-and-recovery experiment (two-panel Figure 4) runs as well.

## Run on Google Colab (A100)

`colab/weedenroll_colab.ipynb` runs the four commands above on a Colab GPU and
saves data, weights and `outputs/real` to Google Drive. Zip the project folder
(or use `weedenroll-src.zip`), open the notebook in Colab, choose an A100
runtime and run all cells. About 1 to 1.5 h end to end. DR3 (footprint) is
measured on any host; DR2 (latency, power) is *projected* unless
`weedenroll profile` is run on a device.

## DR2 without a device: projected, not measured

The paper run has no Jetson. Table 2 and the DR2 row of the scorecard are
therefore built as follows, and say so in every caption:

* the FP32 latency measured on the benchmark host CPU (8.0 ms per frame on
  the Colab x86 CPU) is reported as an **upper bound** for the target: a
  single server core without TensorRT;
* one **projected** row for the NVIDIA Jetson Orin Nano 8 GB (15 W mode) is
  taken from published batch-1 measurements on that board:
  EfficientNet-B0 (4.0 M parameters) at 3.19 ms per image with TensorRT
  FP16 and a MobileNetV4 student at 2.24 ms (Nishikawa, 2026,
  arXiv:2607.26238), and a MobileNetV3 classifier at 21.7 ms per crop in
  PyTorch FP32 without TensorRT (Mishra and Lone, 2026, arXiv:2606.17241).
  MobileNetV3-Small (2.5 M parameters) is lighter than both, so 2–4 ms with
  TensorRT FP16 and at most 22 ms unoptimised are the projected bounds;
* power is given as the 15 W power-mode cap of the board, not a measurement;
* the host `int8` rows (dynamic quantisation of the Linear layers only) are
  dropped from Tables 2 and 3 because they are not a deployment path.

The DR2 status is `projected (not measured on device)`. Running the profiler
on any Jetson replaces the projection with a measurement:

## Measuring DR2 on a Jetson Orin Nano (optional)

```bash
# on the workstation
weedenroll train ... produces weights/student.onnx automatically
# on the Jetson: onnxruntime-gpu (JetPack wheel with TensorRT EP), tegrastats on PATH
sudo nvpmodel -m 0 && sudo jetson_clocks          # or the 15 W mode you report
weedenroll profile --config configs/jetson.yaml --out outputs/hardware_results.json
```

`weedenroll profile` samples `tegrastats` (VDD_IN) while running 300 frames
at batch 1 and reports latency (mean, SD, p95), throughput, power, energy per
frame, model size, store size, peak RSS and 10-shot enrollment time, for
FP32 and INT8 (TensorRT engines). `configs/real.yaml` reads the JSON; Table 2,
the DR2/DR3 scorecard rows and the cost model's energy term pick it up.

## On-device use

```bash
weedenroll enroll  --config configs/jetson.yaml --store store.json --name "Parthenium" --images shots/ --platform aerial
weedenroll predict --config configs/jetson.yaml --store store.json --images frames/ --threshold 0.45
```

Both run with sockets disabled (`bench.episodes.no_network`), which is how
DR1 is evidenced.

## Mapping to the paper

| Output | Paper element | DR |
|---|---|---|
| `fig1_architecture` | Fig. 1 system | — |
| `fig2_enrollment` | Fig. 2 enrollment F1 vs k; known-class drop | DR4, DR5 |
| `fig3_openset` | Fig. 3 ROC and distance distributions | DR6 |
| `fig4_transfer` | Fig. 4 ground→aerial degradation and recovery | DR7 |
| `fig5_cost` | Fig. 5 cost breakdown, workload and species sweeps | DR8 |
| `fig6_sensitivity` | Fig. 6 tornado vs cloud and vs retraining | DR8 |
| `table1_datasets` | Table 1 | — |
| `table2_hardware` | Table 2 | DR2, DR3 |
| `table3_results` | Table 3 (mean ± SE, n) | DR4–DR7 |
| `table4_cost`, `table4b_cost_assumptions` | Table 4 | DR8 |
| `table5_scorecard` | Table 5 DR target vs achieved | all |

## What was executed where

* Synthetic self-test suite, all benchmarks, cost model, figures, tables, DOCX: run end to end (Windows, CPU).
* Downloads and conversion: executed for the Hugging Face DeepWeeds and Plant Seedlings mirrors and the
  Zenodo CoFly archive, on Windows and on Colab (counts verified: 17 509 / 5 539 / 377 crops).
  The Kaggle and Google Drive alternates were checked to exist and serve the file, not downloaded.
* Training (teacher → distilled student → ONNX), checkpoint loading, ONNX Runtime inference, static INT8
  quantisation, the on-device `enroll`/`predict` commands and the full `configs/real.yaml` pipeline:
  executed on real images (subset smoke runs on the CPU workstation; full run on a Colab A100).
* Not executed here: the Jetson TensorRT path and `weedenroll profile` (need the device);
  DR2 is projected from the host CPU upper bound and published Orin Nano benchmarks.
* Runs on Windows, Linux and macOS (text files are UTF-8, console output falls back safely, CoFly file
  names are shortened on extraction to stay under Windows' path limit).

## Reproducibility

All episodes are seeded (`benchmark.seed`); per-episode rows are written to
`results/*.csv` so every mean ± SE in the tables can be recomputed.
