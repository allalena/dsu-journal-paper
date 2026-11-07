"""Command line: ``weedenroll run --config configs/synthetic.yaml``.

Sub-commands
  run       full pipeline → outputs/<run>/{results/*.csv, figures/*, tables/*, paper_assets.docx}
  profile   hardware profile only (run this on the Jetson)
  report    re-build cost model, figures, tables and DOCX from a finished run (no benchmarks)
  cost      cost model only
  enroll    enroll a species on device from a folder of images
  predict   classify a folder of images with the on-device store
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import asdict

import numpy as np
import pandas as pd
import yaml

from .bench.datasets import make_synthetic_suite, load_dataset
from .bench.episodes import EpisodicBenchmark, no_network
from .bench.profile import HardwareProfiler
from .cost import CostAssumptions, SeasonCostModel
from .embed import load_backbone
from .enroll import Enroller
from .govern import GovernanceLogger
from .openset import OpenSetRejector
from .store import PrototypeStore
from . import report as R


def _load_cfg(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _datasets(cfg: dict):
    if cfg["data"].get("kind") == "synthetic":
        s = cfg["data"]
        return make_synthetic_suite(s.get("dim", 128), s.get("seed", 0), s.get("n_per_class", 120),
                                    s.get("spread", 0.35)), True
    ds = {k: load_dataset(v) for k, v in cfg["data"]["sets"].items()}
    return ds, False


def cmd_run(args):
    cfg = _load_cfg(args.config)
    out = os.path.join(cfg.get("output_dir", "outputs"), cfg.get("run_name", time.strftime("%Y%m%d-%H%M%S")))
    res_dir, fig_dir, tab_dir, log_dir = [os.path.join(out, d) for d in ("results", "figures", "tables", "governance")]
    for d in (res_dir, fig_dir, tab_dir, log_dir):
        os.makedirs(d, exist_ok=True)
    datasets, synthetic = _datasets(cfg)
    bcfg = dict(cfg["backbone"])
    backbone = load_backbone(bcfg)
    b = cfg.get("benchmark", {})
    bench = EpisodicBenchmark(backbone, b.get("shots", [1, 5, 10]), b.get("episodes", 20), b.get("n_novel", 2),
                              b.get("test_frac", 0.3), b.get("openset_rule", "global"), b.get("openset_quantile", 0.95),
                              b.get("prototypes_per_class", 1), b.get("seed", 0), log_dir)
    print(f"[weedenroll] run -> {out}  (synthetic={synthetic}, backbone={backbone.name})")

    with no_network():
        # ---- closed-set FP32 vs INT8 -----------------------------------------
        known = bench.known_class_table(datasets)
        if bcfg.get("compare_int8", True):
            b8 = load_backbone({**bcfg, "precision": "int8"})
            known = pd.concat([known, EpisodicBenchmark(b8, seed=b.get("seed", 0)).known_class_table(datasets)])
        known.to_csv(os.path.join(res_dir, "closed_set.csv"), index=False)
        print("[weedenroll] closed-set done")

        # ---- enrollment / open-set ------------------------------------------
        base = datasets[b.get("base", "deepweeds")]
        holdout = b.get("holdout_species")            # species the backbone never saw (real runs)
        if holdout is None and os.path.exists(os.path.join(cfg.get("weights_dir", "weights"), "train_meta.json")):
            with open(os.path.join(cfg.get("weights_dir", "weights"), "train_meta.json"), encoding="utf-8") as f:
                holdout = json.load(f).get("holdout_species")
        if holdout:
            print(f"[weedenroll] novel species restricted to held-out set: {holdout}")
        enr = bench.enrollment_episodes(base, tag="within", novel_pool=holdout)
        cross = b.get("cross_source", "seedlings")
        if cross in datasets:
            enr = pd.concat([enr, bench.enrollment_episodes(base, datasets[cross], tag="cross")])
        enr.to_csv(os.path.join(res_dir, "enrollment.csv"), index=False)
        roc = bench.roc_curve_example(base)
        np.savez(os.path.join(res_dir, "roc_example.npz"), **{k: v for k, v in roc.items() if isinstance(v, np.ndarray)},
                 global_thr=roc["global_thr"], auroc=roc["auroc"])
        print("[weedenroll] enrollment/open-set done")

        # ---- transfer ------------------------------------------------------------
        ground = datasets[b.get("ground", "deepweeds")]
        shared = datasets.get(b.get("aerial_shared", "aerial_shared"))
        disjoint = datasets.get(b.get("aerial_disjoint", "cofly"))
        if shared is None:
            print("[weedenroll] no paired ground/aerial set: DR7 is reported from aerial-only enrollment (CoFly)")
        tr = bench.transfer_episodes(ground, shared, disjoint)
        tr.to_csv(os.path.join(res_dir, "transfer.csv"), index=False)
        print("[weedenroll] transfer done")

    # ---- hardware ----------------------------------------------------------------
    hw_path = cfg.get("hardware_results")
    hw = []
    if hw_path and os.path.exists(hw_path):
        with open(hw_path, encoding="utf-8") as f:
            hw = json.load(f)
    else:
        for prec in (["fp32", "int8"] if bcfg.get("compare_int8", True) else ["fp32"]):
            bb = load_backbone({**bcfg, "precision": prec})
            hw.append(asdict(HardwareProfiler(bb, n_frames=b.get("profile_frames", 200)).run()))
    with open(os.path.join(res_dir, "hardware.json"), "w", encoding="utf-8") as f:
        json.dump(hw, f, indent=2)
    print("[weedenroll] hardware profile done")

    _build_report(cfg, out, datasets, synthetic, known, enr, tr, roc, hw)


def _table1_from_disk(tab_dir: str) -> pd.DataFrame:
    p = os.path.join(tab_dir, "table1_datasets.csv")
    return pd.read_csv(p) if os.path.exists(p) else pd.DataFrame()


def _build_report(cfg: dict, out: str, datasets, synthetic: bool, known: pd.DataFrame, enr: pd.DataFrame,
                  tr: pd.DataFrame, roc: dict, hw: list):
    """Cost model, figures, tables, DOCX and governance summary for a run directory.
    Shared by ``run`` (fresh results) and ``report`` (results re-read from disk)."""
    res_dir, fig_dir, tab_dir, log_dir = [os.path.join(out, d) for d in ("results", "figures", "tables", "governance")]
    for d in (res_dir, fig_dir, tab_dir, log_dir):
        os.makedirs(d, exist_ok=True)
    # ---- cost -------------------------------------------------------------------
    ca = CostAssumptions(**cfg.get("cost", {}))
    jet = [h for h in hw if not np.isnan(h.get("energy_mj_per_frame", np.nan)) and not h.get("simulated")]
    if jet:
        ca = ca.override(energy_mj_per_frame=jet[0]["energy_mj_per_frame"])
    model = SeasonCostModel(ca)
    cost_tbl = model.table(); be = model.breakeven_frames(); sp = model.species_sweep()
    sens_c, sens_r = model.sensitivity(versus="cloud"), model.sensitivity(versus="retrain")
    for name, df in [("cost_table", cost_tbl), ("cost_breakeven", be), ("cost_species", sp),
                     ("cost_sensitivity_vs_cloud", sens_c), ("cost_sensitivity_vs_retrain", sens_r),
                     ("cost_nodes", model.nodes_sweep())]:
        df.to_csv(os.path.join(res_dir, f"{name}.csv"), index=False)
    print("[weedenroll] cost model done")

    # ---- figures ----------------------------------------------------------------
    caps = {}
    R.fig_architecture(fig_dir, caps)
    R.fig_enrollment(enr, fig_dir, caps)
    R.fig_openset(enr, roc, fig_dir, caps)
    R.fig_transfer(tr, fig_dir, caps)
    R.fig_cost(cost_tbl, be, sp, ca, fig_dir, caps)
    R.fig_sensitivity(sens_c, sens_r, fig_dir, caps)
    with open(os.path.join(fig_dir, "captions.json"), "w", encoding="utf-8") as f:
        json.dump(caps, f, indent=2, ensure_ascii=False)

    # ---- tables -----------------------------------------------------------------
    tables = {
        "table1_datasets": (R.table_datasets(datasets, synthetic) if datasets is not None
                            else _table1_from_disk(tab_dir)),
        "table2_hardware": R.table_hardware(hw),
        "table3_results": R.table_results(known, enr, tr),
        "table4_cost": R.table_cost(cost_tbl, ca),
        "table4b_cost_assumptions": R.table_assumptions(ca),
        "table5_scorecard": R.table_scorecard(known, enr, tr, hw, cost_tbl),
    }
    tcaps = {
        "table1_datasets": "Table 1. Datasets and their role in the evaluation protocol.",
        "table2_hardware": "Table 2. Inference latency, power, energy and footprint (DR2, DR3). Latency over n frames at batch size 1. Rows measured on the benchmark host CPU are an upper bound for the target device; the Orin Nano row is projected from published batch-1 measurements on that board (Nishikawa, 2026, arXiv:2607.26238; Mishra and Lone, 2026, arXiv:2606.17241) and is not a measurement of this system. Power is board input (VDD_IN) from tegrastats where measured.",
        "table3_results": "Table 3. Recognition, enrollment, open-set and transfer results (mean ± SE over n episodes).",
        "table4_cost": "Table 4. Season-level cost breakdown of the three deployment options (base case).",
        "table4b_cost_assumptions": "Table 4 (continued). Cost model assumptions.",
        "table5_scorecard": "Table 5. Design requirements (DR1–DR8): target versus achieved."
                            + ("" if any(R._is_jetson(h) for h in hw) else
                               " DR2 is projected from the host-CPU upper bound and published Orin Nano benchmarks, not measured on the device."),
    }
    for k, df in tables.items():
        df.to_csv(os.path.join(tab_dir, f"{k}.csv"), index=False)
        with open(os.path.join(tab_dir, f"{k}.md"), "w", encoding="utf-8") as f:
            f.write(tcaps[k] + "\n\n" + df.to_markdown(index=False, disable_numparse=True) + "\n")
    docx = R.write_docx(tables, caps, tcaps, os.path.join(out, "paper_assets.docx"), fig_dir)
    # governance summary
    sessions = GovernanceLogger.read_sessions(log_dir)
    pd.DataFrame(sessions).drop(columns=["drift"], errors="ignore").to_csv(os.path.join(res_dir, "governance_sessions.csv"), index=False)
    print(f"[weedenroll] figures -> {fig_dir}\n[weedenroll] tables -> {tab_dir}\n[weedenroll] docx -> {docx}")
    print("\n" + tables["table5_scorecard"].to_string(index=False))


def cmd_report(args):
    """Re-build everything downstream of the benchmarks from ``results/*.csv``."""
    cfg = _load_cfg(args.config)
    out = args.out or os.path.join(cfg.get("output_dir", "outputs"), cfg["run_name"])
    res_dir, tab_dir = os.path.join(out, "results"), os.path.join(out, "tables")
    known = pd.read_csv(os.path.join(res_dir, "closed_set.csv"))
    enr = pd.read_csv(os.path.join(res_dir, "enrollment.csv"))
    tr = pd.read_csv(os.path.join(res_dir, "transfer.csv"))
    if "k" in tr and tr.k.notna().all():
        tr["k"] = tr.k.astype(int)
    with np.load(os.path.join(res_dir, "roc_example.npz")) as d:
        roc = {k: d[k] for k in d.files}
    roc["global_thr"] = float(roc["global_thr"]); roc["auroc"] = float(roc["auroc"])
    hw_path = cfg.get("hardware_results")
    if not (hw_path and os.path.exists(hw_path)):
        hw_path = os.path.join(res_dir, "hardware.json")
    with open(hw_path, encoding="utf-8") as f:
        hw = json.load(f)
    try:
        datasets, synthetic = _datasets(cfg)
    except Exception as e:                      # images not on this machine: keep Table 1 from the run
        print(f"[weedenroll] datasets not loadable here ({e}); Table 1 re-used from {tab_dir}")
        datasets, synthetic = None, False
    print(f"[weedenroll] report -> {out}  (results re-read from {res_dir}, hardware from {hw_path})")
    _build_report(cfg, out, datasets, synthetic, known, enr, tr, roc, hw)


def cmd_data(args):
    from .data import main as data_main
    sys.exit(data_main(args.rest))


def cmd_train(args):
    from .train import run
    meta = run(_load_cfg(args.config))
    print(json.dumps(meta, indent=2))


def cmd_profile(args):
    cfg = _load_cfg(args.config)
    res = []
    for prec in cfg.get("precisions", ["fp32", "int8"]):
        bb = load_backbone({**cfg["backbone"], "precision": prec})
        r = HardwareProfiler(bb, cfg.get("image_size", 224), 1, n_frames=cfg.get("frames", 300),
                             device_name=cfg.get("device_name")).run()
        print(json.dumps(asdict(r), indent=2)); res.append(r)
    HardwareProfiler.save(res, args.out)
    print(f"[weedenroll] saved {args.out}")


def cmd_cost(args):
    cfg = _load_cfg(args.config) if args.config else {}
    m = SeasonCostModel(CostAssumptions(**cfg.get("cost", {})))
    print(m.table().to_string(index=False))
    print("\n" + m.sensitivity().to_string(index=False))


def cmd_enroll(args):
    cfg = _load_cfg(args.config)
    bb = load_backbone(cfg["backbone"])
    store = PrototypeStore.load(args.store) if os.path.exists(args.store) else PrototypeStore(bb.dim)
    logger = GovernanceLogger(args.log_dir)
    rej = OpenSetRejector(store)
    imgs = [os.path.join(args.images, f) for f in sorted(os.listdir(args.images))]
    with no_network():
        rep = Enroller(bb, store, rej, logger).enroll(args.name, imgs, platform=args.platform)
    store.save(args.store); logger.close(store.snapshot())
    print(json.dumps(asdict(rep), indent=2))


def cmd_predict(args):
    cfg = _load_cfg(args.config)
    bb = load_backbone(cfg["backbone"])
    store = PrototypeStore.load(args.store)
    rej = OpenSetRejector(store); rej.global_thr = args.threshold
    imgs = [os.path.join(args.images, f) for f in sorted(os.listdir(args.images))]
    with no_network():
        z = bb.embed_all(imgs); r = rej.predict(z)
    for f, i, d, u in zip(imgs, r.pred_idx, r.distance, r.is_unknown):
        print(f"{os.path.basename(f)}\t{'UNKNOWN' if u else store.names[i]}\t{d:.3f}")


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "data":
        # `weedenroll data --dst data --all`: argparse.REMAINDER does not forward
        # options that start with "--", so hand the raw arguments to the data CLI.
        from .data import main as data_main
        sys.exit(data_main(argv[1:]))
    # Windows consoles default to a legacy code page; tables and captions contain "±" and "→".
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8", errors="replace")
            except (ValueError, OSError):
                pass
    p = argparse.ArgumentParser(prog="weedenroll")
    sp = p.add_subparsers(dest="cmd", required=True)
    a = sp.add_parser("run"); a.add_argument("--config", required=True); a.set_defaults(fn=cmd_run)
    a = sp.add_parser("data", help="download + convert + verify datasets"); a.add_argument("rest", nargs=argparse.REMAINDER); a.set_defaults(fn=cmd_data)
    a = sp.add_parser("train"); a.add_argument("--config", required=True); a.set_defaults(fn=cmd_train)
    a = sp.add_parser("profile"); a.add_argument("--config", required=True); a.add_argument("--out", default="outputs/hardware_results.json"); a.set_defaults(fn=cmd_profile)
    a = sp.add_parser("cost"); a.add_argument("--config"); a.set_defaults(fn=cmd_cost)
    a = sp.add_parser("report", help="re-build figures/tables/DOCX from a finished run")
    a.add_argument("--config", required=True); a.add_argument("--out", help="run directory (default: output_dir/run_name)")
    a.set_defaults(fn=cmd_report)
    a = sp.add_parser("enroll"); a.add_argument("--config", required=True); a.add_argument("--store", required=True)
    a.add_argument("--name", required=True); a.add_argument("--images", required=True); a.add_argument("--platform", default="ground")
    a.add_argument("--log-dir", default="governance"); a.set_defaults(fn=cmd_enroll)
    a = sp.add_parser("predict"); a.add_argument("--config", required=True); a.add_argument("--store", required=True)
    a.add_argument("--images", required=True); a.add_argument("--threshold", type=float, default=0.5); a.set_defaults(fn=cmd_predict)
    args = p.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()
