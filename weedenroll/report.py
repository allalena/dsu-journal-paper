"""CIGR-style figures and tables.

Conventions (CIGR Journal author guidelines):
  * Times New Roman; 10 pt text in figures ≤ 10 cm wide, 12 pt wider.
  * Error bars with SE or SD stated, and n given, in every caption.
  * Captions, not titles, on figures; units in table headings; horizontal
    rules only in tables.
  * Figures must survive grayscale printing: every series has a distinct
    marker and line style as well as a colour.

Colours are the validated categorical palette from the dataviz reference
(blue, orange, aqua, yellow, magenta, green) in fixed slot order.
"""
from __future__ import annotations

import json
import os
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np
import pandas as pd

from .bench.metrics import summarize

PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
MARKERS = ["o", "s", "^", "D", "v", "P"]
LINES = ["-", "--", "-.", ":", (0, (3, 1, 1, 1)), (0, (5, 2))]
INK, INK2, GRID = "#1a1a19", "#5a5a57", "#d9d9d5"

CM = 1 / 2.54
W1, W2 = 8.0 * CM, 15.2 * CM           # 6.0 in text width of the CIGR page (and half of it)


def fmt_num(v) -> str:
    """CIGR number style: 4-digit numbers closed up (2084), 5+ digits space-grouped (17 509)."""
    a = abs(float(v))
    body = f"{a:,.0f}".replace(",", " ") if a >= 10000 else f"{a:.0f}"
    return ("\u2212" if v < 0 else "") + body


def _style():
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Liberation Serif", "DejaVu Serif"],
        "font.size": 10, "axes.labelsize": 10, "axes.titlesize": 10,
        "legend.fontsize": 10, "xtick.labelsize": 10, "ytick.labelsize": 10,   # CIGR: figure text 10-12 pt
        "axes.edgecolor": INK2, "axes.linewidth": 0.6, "axes.spines.top": False,
        "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
        "grid.linewidth": 0.5, "axes.axisbelow": True, "xtick.color": INK2,
        "ytick.color": INK2, "axes.labelcolor": INK, "text.color": INK,
        "legend.frameon": False, "lines.linewidth": 1.4, "lines.markersize": 5,
        "savefig.dpi": 300, "figure.dpi": 120, "errorbar.capsize": 2,
    })


def _save(fig, out_dir: str, name: str, caption: str, captions: Dict[str, str]):
    os.makedirs(out_dir, exist_ok=True)
    if not getattr(fig, "_layout_done", False):
        fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"{name}.png"), bbox_inches="tight")
    fig.savefig(os.path.join(out_dir, f"{name}.pdf"), bbox_inches="tight")
    plt.close(fig)
    captions[name] = caption


def _mse(df: pd.DataFrame, by: List[str], col: str) -> pd.DataFrame:
    g = df.groupby(by)[col]
    out = g.agg(mean="mean", sd=lambda v: v.std(ddof=1) if len(v) > 1 else 0.0, n="count").reset_index()
    out["se"] = out["sd"] / np.sqrt(out["n"])
    return out


# ==========================================================================
# Figure 1 — system architecture
# ==========================================================================
def fig_architecture(out_dir: str, captions: Dict[str, str]):
    _style()
    fig, ax = plt.subplots(figsize=(W2, 10.5 * CM))
    ax.set_xlim(0, 100); ax.set_ylim(0, 63); ax.axis("off"); ax.grid(False)
    TL, BL = 2.7, 2.5          # title / body line heights in data units (10 pt text at this scale)

    def box(x, y, w, h, title, body="", fc="#f3f3f0", ec=INK2, lw=0.8):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.3,rounding_size=1.0",
                                    fc=fc, ec=ec, lw=lw))
        nt = title.count("\n") + 1
        nb = body.count("\n") + 1 if body else 0
        total = TL * nt + (BL * nb + 0.8 if body else 0)
        top = y + h / 2 + total / 2
        ax.text(x + w / 2, top, title, ha="center", va="top", fontsize=10, weight="bold", linespacing=1.05)
        if body:
            ax.text(x + w / 2, top - TL * nt - 0.8, body, ha="center", va="top", fontsize=10,
                    color=INK2, linespacing=1.1)

    def arrow(x0, y0, x1, y1, text="", ls="-", tdx=0, tdy=1.0, ha="center", head=True):
        ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>" if head else "-",
                                     mutation_scale=9, lw=0.9, color=INK, linestyle=ls,
                                     shrinkA=0.5, shrinkB=0.5))
        if text:
            ax.text((x0 + x1) / 2 + tdx, (y0 + y1) / 2 + tdy, text, ha=ha, va="bottom", fontsize=10, color=INK2)

    # ---- off-line region ------------------------------------------------
    ax.add_patch(FancyBboxPatch((1, 46), 98, 16, boxstyle="round,pad=0.3", fc="none", ec=GRID, lw=0.8,
                                linestyle=(0, (2, 2))))
    ax.text(2.5, 61.3, "Train once (workstation, on-line)", fontsize=10, color=INK2, va="top", style="italic")
    box(4, 47.5, 27, 10.5, "Teacher model", "large CNN,\nground-level images")
    box(36.5, 47.5, 27, 10.5, "Distillation", "compact student\nembedding network")
    box(69, 47.5, 27, 10.5, "Export", "ONNX graph; FP16/INT8\nTensorRT engine on target")
    arrow(31, 52.75, 36.5, 52.75); arrow(63.5, 52.75, 69, 52.75)
    # deploy path: down, left, down into the backbone
    arrow(82.5, 47.5, 82.5, 43.5, ls="--", head=False); arrow(82.5, 43.5, 28.5, 43.5, ls="--", head=False)
    arrow(28.5, 43.5, 28.5, 35, "deploy once", ls="--", tdx=1.2, tdy=-1.2, ha="left")

    # ---- edge-node region ------------------------------------------------------
    ax.add_patch(FancyBboxPatch((1, 1), 98, 41, boxstyle="round,pad=0.3", fc="none", ec=INK2, lw=0.9))
    ax.text(97.5, 41.3, "Edge node (UAV / UGV, off-line)", fontsize=10, color=INK2, va="top", ha="right", style="italic")
    box(2, 23, 12, 12, "Camera", "frames")
    box(17, 23, 22, 12, "Embedding\nbackbone", "FP16/INT8 engine,\n224 px")
    box(50, 23, 22, 12, "Prototype\nstore", "nearest prototype,\ncosine distance")
    box(80, 27.5, 18, 9.5, "Decision", "species,\nactuation")
    box(80, 8, 18, 14.5, "Field\nenrollment", "operator captures\nk shots, confirms\nlabel")
    box(47.5, 2, 27, 10, "Governance log", "confidence, rejection\nrate, prototype drift", fc="#e9eef7")
    arrow(14, 29, 17, 29); arrow(39, 29, 50, 29, "embedding")
    arrow(72, 31.5, 80, 33, "known", tdx=0, tdy=1.2)
    arrow(72, 26.5, 80, 20, "unknown", tdx=-2.5, tdy=-5.0)
    arrow(80, 11, 66, 23, "enroll:\nwrite prototypes", tdx=-10.5, tdy=-3.0)
    arrow(56, 23, 56, 12, "log", tdx=-1.0, ha="right")
    _save(fig, out_dir, "fig1_architecture",
          "Figure 1. System architecture. The embedding backbone is trained once off-line and deployed "
          "as an ONNX graph (built into an FP16 or INT8 TensorRT engine on the target); inference, open-set rejection, field enrollment and governance logging all "
          "run on the edge node with no network connection. Enrollment writes prototypes to the store; "
          "no model weights change after deployment.", captions)


# ==========================================================================
# Figure 2 — enrollment (DR4) and harm to known classes (DR5)
# ==========================================================================
def fig_enrollment(enr: pd.DataFrame, out_dir: str, captions: Dict[str, str]):
    _style()
    fig, axes = plt.subplots(1, 2, figsize=(W2, 7.6 * CM))
    tags = list(enr["tag"].unique())
    labels = {"within": "Same dataset (DeepWeeds)",
              "cross": "Second dataset (Plant Seedlings)"}
    ax = axes[0]
    for i, t in enumerate(tags):
        s = _mse(enr[enr.tag == t], ["k"], "novel_f1")
        ax.errorbar(s.k, s["mean"], yerr=s["se"], color=PALETTE[i], marker=MARKERS[i], ls=LINES[i],
                    label=labels.get(t, t), lw=1.4)
    ax.axhline(0.80, color=INK2, lw=0.8, ls=(0, (1, 2))); ax.text(enr.k.max(), 0.805, "DR4 target", ha="right", fontsize=10, color=INK2)
    ax.set_xlabel("Shots per enrolled species, k"); ax.set_ylabel("Macro-F1 on novel species")
    ax.set_xticks(sorted(enr.k.unique())); ax.set_ylim(0, 1.02)
    ax.text(-0.12, 1.14, "(a)", transform=ax.transAxes, weight="bold")

    ax = axes[1]
    for i, t in enumerate(tags):
        s = _mse(enr[enr.tag == t], ["k"], "known_f1_drop_pts")
        ax.errorbar(s.k, s["mean"], yerr=s["se"], color=PALETTE[i], marker=MARKERS[i], ls=LINES[i],
                    label=labels.get(t, t), lw=1.4)
    ax.axhline(2.0, color=INK2, lw=0.8, ls=(0, (1, 2))); ax.text(enr.k.max(), 2.05, "DR5 limit", ha="right", fontsize=10, color=INK2)
    ax.axhline(0, color=INK2, lw=0.5)
    ax.set_xlabel("Shots per enrolled species, k"); ax.set_ylabel("Known-class F1 drop (points)")
    ax.set_xticks(sorted(enr.k.unique()))
    # keep the DR5 limit in view but never clip the data (small runs can show large drops)
    sd = _mse(enr, ["tag", "k"], "known_f1_drop_pts")
    ax.set_ylim(min(-0.3, float((sd["mean"] - sd["se"]).min()) - 0.2), max(2.4, float((sd["mean"] + sd["se"]).max()) + 0.2))
    # legend of (a) applies; a second legend here overlapped the k = 1 points and the DR5 label
    ax.text(-0.12, 1.14, "(b)", transform=ax.transAxes, weight="bold")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=2, title="Novel species drawn from", bbox_to_anchor=(0.5, 0.0),
               frameon=False)
    fig.tight_layout(rect=[0, 0.2, 1, 1]); fig._layout_done = True
    n = int(enr.groupby(["tag", "k"]).size().min())
    _save(fig, out_dir, "fig2_enrollment",
          f"Figure 2. Field enrollment without retraining. (a) Macro-F1 on newly enrolled species versus "
          f"number of enrollment shots k; (b) change in macro-F1 on the species already in the store after "
          f"enrollment (DR5). Points are means over n = {n} episodes; error bars are ± 1 SE. The legend applies to both panels.", captions)


# ==========================================================================
# Figure 3 — open-set (DR6)
# ==========================================================================
def fig_openset(enr: pd.DataFrame, roc: dict, out_dir: str, captions: Dict[str, str]):
    _style()
    fig, axes = plt.subplots(1, 2, figsize=(W2, 6.5 * CM))
    ax = axes[0]
    ax.plot(roc["fpr"], roc["tpr"], color=PALETTE[0], lw=1.6, label=f"ROC (AUROC = {roc['auroc']:.3f})")
    ax.plot([0, 1], [0, 1], color=GRID, lw=0.8, ls="--")
    # operating point at the calibrated threshold
    thr = roc["global_thr"]
    fpr_op = float((roc["score_known"] > thr).mean()); tpr_op = float((roc["score_unknown"] > thr).mean())
    ax.plot([fpr_op], [tpr_op], marker="o", color=PALETTE[1], ms=7, mec="white", mew=1.2, ls="none",
            label=f"95th-pct threshold\n(FPR {fpr_op:.2f}, TPR {tpr_op:.2f})")
    ax.set_xlabel("False rejection rate on known species"); ax.set_ylabel("Detection rate on unenrolled species")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.02); ax.legend(loc="lower right")
    ax.text(-0.12, 1.14, "(a)", transform=ax.transAxes, weight="bold")

    ax = axes[1]
    bins = np.linspace(0, max(roc["score_known"].max(), roc["score_unknown"].max()) * 1.05, 40)
    ax.hist(roc["score_known"], bins, color=PALETTE[0], alpha=0.75, label="Known species", density=True)
    ax.hist(roc["score_unknown"], bins, color=PALETTE[1], alpha=0.75, label="Unenrolled species", density=True,
            hatch="////", edgecolor="white", lw=0.3)
    ax.axvline(thr, color=INK, lw=0.9, ls="--")
    ax.text(thr + 0.01, ax.get_ylim()[1] * 0.62, f"threshold\n= {thr:.2f}", fontsize=10, va="top", ha="left")
    ax.set_xlim(left=max(0.0, bins[0] - 0.05))
    ax.set_xlabel("Cosine distance to nearest prototype"); ax.set_ylabel("Density"); ax.legend(loc="upper left")
    ax.text(-0.12, 1.14, "(b)", transform=ax.transAxes, weight="bold")
    a = summarize(enr[enr.k == enr.k.min()]["auroc"])
    _save(fig, out_dir, "fig3_openset",
          f"Figure 3. Open-set rejection of unenrolled species (DR6). (a) ROC of the nearest-prototype distance "
          f"as an unknown-ness score for one representative episode, with the operating point at the calibrated "
          f"95th-percentile threshold; (b) distance distributions behind it. Across n = {a['n']} episodes the "
          f"mean AUROC was {a['mean']:.3f} ± {a['se']:.3f} (SE).", captions)


# ==========================================================================
# Figure 4 — ground → aerial transfer (DR7)   [headline]
# ==========================================================================
def fig_transfer(tr: pd.DataFrame, out_dir: str, captions: Dict[str, str], aerial_label: str = "aerial"):
    _style()
    if "ground→ground" not in set(tr.condition):
        return _fig_transfer_disjoint_only(tr, out_dir, captions)
    fig, axes = plt.subplots(1, 2, figsize=(W2, 7.6 * CM), gridspec_kw={"width_ratios": [1, 1.7]})
    ks = sorted(tr[tr.k > 0].k.unique())
    # (a) the degradation bar pair
    ax = axes[0]
    base = tr[tr.k == 0]
    s = _mse(base, ["condition"], "macro_f1").set_index("condition")
    order = ["ground→ground", "ground→aerial (no enrollment)"]
    vals = [s.loc[o, "mean"] for o in order]; errs = [s.loc[o, "se"] for o in order]
    bars = ax.bar([0, 1], vals, yerr=errs, color=[PALETTE[0], PALETTE[1]], width=0.6, edgecolor="white", lw=0.5)
    bars[1].set_hatch("////")
    for x, v in zip([0, 1], vals):
        ax.text(x, v + 0.03, f"{v:.2f}", ha="center", fontsize=10)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["Ground\nqueries", "Aerial\nqueries"])
    ax.set_ylabel("Macro-F1, ground-image prototypes"); ax.set_ylim(0, 1.08)
    ax.text(-0.3, 1.10, "(a)", transform=ax.transAxes, weight="bold")

    # (b) recovery vs k
    ax = axes[1]
    conds = [("aerial enrollment (replace)", 2, "Aerial queries: aerial prototypes replace ground"),
             ("aerial enrollment (merge)", 3, "Aerial queries: aerial prototypes added to ground"),
             ("ground after merge", 4, "Ground queries after adding aerial prototypes"),
             ("aerial-only enrollment (disjoint species)", 5, "Aerial-only species, enrolled from the air (CoFly)")]
    for cond, ci, lab in conds:
        d = tr[tr.condition == cond]
        if d.empty:
            continue
        m = _mse(d, ["k"], "macro_f1")
        ax.errorbar(m.k, m["mean"], yerr=m["se"], color=PALETTE[ci % len(PALETTE)], marker=MARKERS[ci % 6],
                    ls=LINES[ci % 6], label=lab, lw=1.4)
    ax.axhline(vals[0], color=PALETTE[0], lw=0.9, ls=(0, (1, 2)))
    ax.text(ks[0] + 0.2, vals[0] - 0.02, "ground→ground reference", ha="left", va="top", fontsize=10, color=PALETTE[0])
    ax.axhline(vals[1], color=PALETTE[1], lw=0.9, ls=(0, (1, 2)))
    ax.text(ks[-1], vals[1] - 0.02, "ground→aerial, no enrollment", ha="right", va="top", fontsize=10, color=PALETTE[1])
    ax.set_xlabel("Aerial enrollment shots per species, k"); ax.set_ylabel("Macro-F1")
    ax.set_xticks(ks); ax.set_ylim(0, 1.05)
    ax.text(-0.1, 1.04, "(b)", transform=ax.transAxes, weight="bold")
    h, l = ax.get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=2, fontsize=10, bbox_to_anchor=(0.5, 0.0), frameon=False)
    fig.tight_layout(rect=[0, 0.16, 1, 1]); fig._layout_done = True
    n = int(tr.groupby(["condition", "k"]).size().min())
    _save(fig, out_dir, "fig4_transfer",
          f"Figure 4. Ground-to-aerial transfer and recovery by field enrollment (DR7). (a) A store built from "
          f"ground-level images applied to ground versus aerial queries of the same species; (b) the aerial "
          f"score recovered by enrolling k aerial shots per species, either replacing or merging with the ground "
          f"prototypes, with the merged store re-checked on ground queries, and species seen only from the air. "
          f"Means over n = {n} episodes; error bars ± 1 SE.", captions)


def _fig_transfer_disjoint_only(tr: pd.DataFrame, out_dir: str, captions: Dict[str, str]):
    fig, ax = plt.subplots(figsize=(W1, 6.5 * CM))
    d = tr[tr.condition == "aerial-only enrollment (disjoint species)"]
    m = _mse(d, ["k"], "macro_f1")
    ax.errorbar(m.k, m["mean"], yerr=m["se"], color=PALETTE[5], marker=MARKERS[5], ls=LINES[0], lw=1.4)
    ax.set_xlabel("Aerial enrollment shots per species, k"); ax.set_ylabel("Macro-F1, aerial species")
    ax.set_xticks(sorted(d.k.unique())); ax.set_ylim(0, 1.05)
    n = int(d.groupby("k").size().min())
    _save(fig, out_dir, "fig4_transfer",
          f"Figure 4. Aerial enrollment with a ground-trained backbone (DR7). Species seen only from the air "
          f"(CoFly-WeedDB) enrolled from k UAV image crops per species and evaluated on held-out crops. Means over "
          f"n = {n} episodes; error bars ± 1 SE.", captions)


# ==========================================================================
# Figure 5 — season cost (DR8)
# ==========================================================================
def fig_cost(cost_tbl: pd.DataFrame, breakeven: pd.DataFrame, species: pd.DataFrame, a, out_dir: str,
             captions: Dict[str, str]):
    _style()
    fig, axes = plt.subplots(1, 3, figsize=(W2, 9.0 * CM), gridspec_kw={"width_ratios": [1.15, 1, 1]})
    short = {"A": "A: edge + enrollment", "B": "B: cloud inference", "C": "C: edge + retraining"}
    ax = axes[0]
    comps = [("hardware", "Hardware (amortised)"), ("energy", "Energy"), ("connectivity", "Connectivity"),
             ("cloud_inference", "Cloud inference"), ("data_transfer", "Data transfer"),
             ("enrollment_labour", "Enrollment labour"), ("retraining", "Retraining"), ("redeployment", "Redeployment")]
    hatches = ["", "", "//", "", "\\\\", "", "xx", ".."]
    cols = [PALETTE[0], PALETTE[2], PALETTE[3], PALETTE[1], PALETTE[3], PALETTE[2], PALETTE[1], PALETTE[4]]
    x = np.arange(len(cost_tbl)); bottom = np.zeros(len(cost_tbl))
    for (c, lab), h, col in zip(comps, hatches, cols):
        v = cost_tbl[c].values
        if v.sum() <= 0:
            continue
        ax.bar(x, v, bottom=bottom, color=col, hatch=h, edgecolor="white", lw=0.6, width=0.62, label=lab)
        bottom += v
    for xi, t in zip(x, cost_tbl["total"]):
        ax.text(xi, t * 1.02, fmt_num(t), ha="center", fontsize=10)
    ax.set_xticks(x); ax.set_xticklabels(["A", "B", "C"])
    ax.set_ylabel("Season cost (USD)"); ax.set_ylim(0, cost_tbl["total"].max() * 1.15)
    ax.text(-0.45, 1.10, "(a)", transform=ax.transAxes, weight="bold")
    h, l = ax.get_legend_handles_labels()

    ax = axes[1]
    for i, (opt, d) in enumerate(breakeven.groupby("option", sort=False)):
        ax.plot(d.frames_per_node_season, d.total, color=PALETTE[i], ls=LINES[i], label=short[opt[0]], lw=1.5)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.axvline(a.frames_per_node_season, color=INK2, lw=0.8, ls=(0, (1, 2)))
    ax.text(a.frames_per_node_season * 1.15, ax.get_ylim()[0] * 1.15, "base", fontsize=10, va="bottom", ha="left", color=INK2)
    ax.set_xlabel("Frames per node per season"); ax.set_ylabel("Season cost (USD, log scale)")
    ax.text(-0.3, 1.10, "(b)", transform=ax.transAxes, weight="bold")
    h2, l2 = ax.get_legend_handles_labels()

    ax = axes[2]
    for i, (opt, d) in enumerate(species.groupby("option", sort=False)):
        ax.plot(d.new_species_per_season, d.total, color=PALETTE[i], marker=MARKERS[i], ls=LINES[i], lw=1.4, ms=3.5)
    ax.axvline(a.new_species_per_season, color=INK2, lw=0.8, ls=(0, (1, 2)))
    ax.set_xlabel("New species per season"); ax.set_ylabel("Season cost (USD)")
    ax.text(-0.3, 1.10, "(c)", transform=ax.transAxes, weight="bold")

    fig.legend(h2, l2, loc="lower center", ncol=3, fontsize=10, bbox_to_anchor=(0.5, 0.19), frameon=False)
    fig.legend(h, l, loc="lower center", ncol=3, fontsize=10, bbox_to_anchor=(0.5, 0.0), frameon=False)
    fig.tight_layout(rect=[0, 0.30, 1, 1]); fig._layout_done = True
    _save(fig, out_dir, "fig5_cost",
          f"Figure 5. Season-level cost of three deployment options for a fleet of {a.n_nodes} nodes (DR8). "
          f"(a) Cost breakdown at the base case of {fmt_num(a.frames_per_node_season)} frames per node and "
          f"{a.new_species_per_season} newly encountered species per season; (b) total cost as the inference workload "
          f"varies; (c) total cost as the number of new species varies, the driver of retraining cost. Dotted "
          f"vertical lines mark the base case. Assumptions in Table 4.", captions)


# ==========================================================================
# Figure 6 — sensitivity tornadoes
# ==========================================================================
NICE = {"new_species_per_season": "New species per season", "ml_engineer_usd_hour": "ML engineer rate (USD/h)",
        "edge_hw_life_years": "Edge hardware life (years)", "label_images_per_species": "Labelled images per species",
        "enrollment_minutes_per_species": "Enrollment time per species (min)", "operator_usd_hour": "Operator rate (USD/h)",
        "edge_hw_usd": "Edge hardware cost (USD)", "cloud_inference_usd_per_1000": "Cloud inference (USD/1000)",
        "cellular_usd_node_month": "Cellular plan (USD/node/month)", "frames_per_node_season": "Frames per node per season",
        "egress_usd_gb": "Data egress (USD/GB)", "electricity_usd_kwh": "Electricity (USD/kWh)"}


def fig_sensitivity(sens_cloud: pd.DataFrame, sens_retrain: pd.DataFrame, out_dir: str, captions: Dict[str, str]):
    _style()
    fig, axes = plt.subplots(2, 1, figsize=(W2, 12.5 * CM),
                             gridspec_kw={"height_ratios": [max(1, int((sens_cloud.swing > 1e-6).sum())),
                                                            max(1, int((sens_retrain.swing > 1e-6).sum()))]})
    bases = []
    for ax, sens, tag, title in [(axes[0], sens_cloud, "(a)", "versus B: cloud inference"),
                                 (axes[1], sens_retrain, "(b)", "versus C: edge + retraining")]:
        s = sens[sens.swing > 1e-6].head(8).iloc[::-1]
        y = np.arange(len(s)); base = float(sens.saving_base.iloc[0]); bases.append(base)
        ax.barh(y, s.saving_low - base, color=PALETTE[1], hatch="////", edgecolor="white", lw=0.4, height=0.62, label="Parameter −50 %")
        ax.barh(y, s.saving_high - base, color=PALETTE[0], edgecolor="white", lw=0.4, height=0.62, label="Parameter +50 %")
        ax.axvline(0, color=INK, lw=0.8)
        ax.set_yticks(y); ax.set_yticklabels([NICE.get(p, p.replace("_", " ")) for p in s.parameter], fontsize=10)
        ax.set_xlabel(f"Change in saving of A (USD) {title}\nbase saving USD {fmt_num(base)}")
        ax.text(-0.02, 1.0, tag, transform=ax.transAxes, ha="right", va="bottom", weight="bold")
    axes[0].legend(loc="lower right", fontsize=10)
    _save(fig, out_dir, "fig6_sensitivity",
          f"Figure 6. One-at-a-time sensitivity of the season saving of option A (offline edge with field enrollment) "
          f"to ± 50 % changes in each cost assumption, (a) relative to cloud inference (base saving USD {fmt_num(bases[0])}) and "
          f"(b) relative to a closed-world edge model with periodic retraining (base saving USD {fmt_num(bases[1])}). "
          f"Parameters shared by both options cancel and are omitted.", captions)


# ==========================================================================
# Tables
# ==========================================================================
def table_datasets(datasets: dict, synthetic: bool) -> pd.DataFrame:
    rows = []
    roles = {"deepweeds": "Backbone domain; known-class baseline; held-out species for enrollment",
             "seedlings": "Second enrollment source (cross-dataset)",
             "aerial_shared": "Aerial views of DeepWeeds species: ground→aerial transfer",
             "cofly": "Aerial-only species: aerial enrollment"}
    for name, ds in datasets.items():
        rows.append({"Dataset": name + (" (synthetic)" if synthetic else ""), "Platform": ds.platform,
                     "Species": len(ds.species), "Images": len(ds),
                     "Images per species (min–max)": f"{min(ds.class_counts().values())}–{max(ds.class_counts().values())}",
                     "Role": roles.get(name, "")})
    return pd.DataFrame(rows)


# Published NVIDIA Jetson Orin Nano measurements used to *project* DR2 when the
# device itself is not available.  Both are arXiv papers reporting batch-1
# latencies on the 8 GB Orin Nano; neither is a measurement of this code.
ORIN_NANO_PUBLISHED = [
    {"ref": "Nishikawa (2026), arXiv:2607.26238", "model": "EfficientNet-B0, 4.0 M parameters",
     "runtime": "TensorRT FP16, JetPack 5.1.2", "mode": "15 W MAXN", "latency_ms": 3.19, "fps": 313,
     "note": "batch 1, incl. host-device transfer; a MobileNetV4 student measured 2.24 ms; "
             "PTQ-INT8 2.37 ms GPU compute"},
    {"ref": "Mishra and Lone (2026), arXiv:2606.17241", "model": "MobileNetV3 classifier",
     "runtime": "PyTorch FP32, no TensorRT", "mode": "MAXN SUPER", "latency_ms": 21.69, "fps": None,
     "note": "per-crop wall-clock time on the GPU inside a streaming pipeline"},
]
PROJECTED_ORIN_NANO = {
    "device": "NVIDIA Jetson Orin Nano 8 GB, 15 W mode (projected)",
    "latency_text": "2–4 (TensorRT FP16); ≤ 22 (PyTorch FP32)",
    "fps_text": "250–450 (TensorRT FP16)",
    "power_text": "≤ 15 (power-mode cap)",
    "dr2_text": "2–4 ms (TensorRT FP16) and at most 22 ms (PyTorch FP32) at the 15 W power-mode cap",
    "basis": "projected from published Orin Nano benchmarks (Nishikawa, 2026; Mishra and Lone, 2026)",
}


def _is_jetson(r: dict) -> bool:
    d = str(r.get("device", "")).lower()
    return "jetson" in d or "orin" in d


def table_hardware(hw: List[dict]) -> pd.DataFrame:
    """Measured rows first.  Host-CPU rows are an upper bound for the target
    device.  Host-side ``int8`` rows are dynamic quantisation of the Linear
    layers only (not a deployment path) and are dropped.  If no device row is
    present, one projected Orin Nano row is appended from the literature."""
    rows = []
    for r in hw:
        jet = _is_jetson(r)
        if r["precision"] == "int8" and not jet:
            continue
        basis = ("simulated" if r.get("simulated") else
                 ("measured on device" if jet else "measured on benchmark host (upper bound for the target)"))
        rows.append({"Device": r["device"], "Backbone": r["backbone"], "Precision": r["precision"].upper(),
                     "Latency, mean ± SD (ms)": f"{r['latency_ms_mean']:.1f} ± {r['latency_ms_sd']:.1f}",
                     "Latency, p95 (ms)": f"{r['latency_ms_p95']:.1f}", "Throughput (frames/s)": f"{r['fps']:.0f}",
                     "Power (W)": "n/a" if np.isnan(r["power_w_mean"]) else f"{r['power_w_mean']:.1f}",
                     "Energy (mJ/frame)": "n/a" if np.isnan(r["energy_mj_per_frame"]) else f"{r['energy_mj_per_frame']:.0f}",
                     "Model (MB)": "n/a" if np.isnan(r["model_mb"]) else f"{r['model_mb']:.1f}",
                     "Store (kB)": f"{r['store_kb']:.1f}", "Peak RSS (MB)": f"{r['rss_mb_peak']:.0f}",
                     "10-shot enrollment (ms)": f"{r['enroll_ms_10shot']:.1f}",
                     "n frames": r["n_frames"], "Basis": basis})
    ref = next((r for r in hw if r["precision"] == "fp32" and not r.get("simulated")), None)
    if ref is not None and not any(_is_jetson(r) for r in hw):     # real host run, no device: project
        P = PROJECTED_ORIN_NANO
        rows.append({"Device": P["device"], "Backbone": ref["backbone"], "Precision": "FP16 (TensorRT)",
                     "Latency, mean ± SD (ms)": P["latency_text"], "Latency, p95 (ms)": "n/a",
                     "Throughput (frames/s)": P["fps_text"], "Power (W)": P["power_text"], "Energy (mJ/frame)": "n/a",
                     "Model (MB)": "n/a" if np.isnan(ref["model_mb"]) else f"{ref['model_mb'] / 2:.1f} (FP16)",
                     "Store (kB)": f"{ref['store_kb']:.1f}", "Peak RSS (MB)": "n/a", "10-shot enrollment (ms)": "n/a",
                     "n frames": "—", "Basis": P["basis"]})
    return pd.DataFrame(rows)


def table_results(known: pd.DataFrame, enr: pd.DataFrame, tr: pd.DataFrame) -> pd.DataFrame:
    """Closed-set + enrollment + open-set + transfer, mean ± SE, n."""
    rows = []
    known = known[known.precision == "fp32"]   # host dynamic-INT8 rows are not a deployment path
    for _, r in known.iterrows():
        rows.append({"Experiment": f"Closed-set, {r.dataset} ({r.platform}, {r.precision.upper()})", "k": "all",
                     "Metric": "macro-F1", "Mean ± SE": f"{r.macro_f1:.3f}", "n": r.n_test})
    for (tag, k), d in enr.groupby(["tag", "k"]):
        for col, lab in [("novel_f1", "novel macro-F1 (DR4)"), ("known_f1_drop_pts", "known macro-F1 drop, pts (DR5)")]:
            s = summarize(d[col])
            rows.append({"Experiment": f"Enrollment, {tag}", "k": k, "Metric": lab,
                         "Mean ± SE": f"{s['mean']:.3f} ± {s['se']:.3f}", "n": s["n"]})
    d = enr[enr.k == enr.k.min()]
    for tag, dd in d.groupby("tag"):
        for col, lab in [("auroc", "AUROC (DR6)"), ("tpr_at_5fpr", "TPR at 5 % FPR"), ("known_frr", "known false-rejection rate")]:
            s = summarize(dd[col])
            rows.append({"Experiment": f"Open-set, {tag}", "k": "—", "Metric": lab,
                         "Mean ± SE": f"{s['mean']:.3f} ± {s['se']:.3f}", "n": s["n"]})
    for (cond, k), dd in tr.groupby(["condition", "k"]):
        s = summarize(dd["macro_f1"])
        rows.append({"Experiment": f"Transfer: {cond}", "k": k if k else "—", "Metric": "macro-F1 (DR7)",
                     "Mean ± SE": f"{s['mean']:.3f} ± {s['se']:.3f}", "n": s["n"]})
    return pd.DataFrame(rows)


def table_cost(cost_tbl: pd.DataFrame, a) -> pd.DataFrame:
    t = cost_tbl.copy()
    t = t.rename(columns={"option": "Option", "hardware": "Hardware", "energy": "Energy", "connectivity": "Connectivity",
                          "cloud_inference": "Cloud inference", "data_transfer": "Data transfer",
                          "enrollment_labour": "Enrollment labour", "retraining": "Retraining",
                          "redeployment": "Redeployment", "total": "Total", "usd_per_node_season": "Per node",
                          "usd_per_1000_decisions": "Per 1000 decisions"})
    for c in t.columns[1:]:
        t[c] = t[c].map(lambda v: f"{v:,.2f}")
    t.columns = [c if c == "Option" else f"{c} (USD)" for c in t.columns]
    return t


def table_assumptions(a) -> pd.DataFrame:
    from dataclasses import asdict
    d = asdict(a); notes = d.pop("source_notes")
    rows = [{"Parameter": k.replace("_", " "), "Value": v, "Source / note": notes.get(k, "")} for k, v in d.items()
            if v is not None]
    return pd.DataFrame(rows)


def table_scorecard(known, enr, tr, hw: List[dict], cost_tbl) -> pd.DataFrame:
    """DR1–DR8: target vs achieved."""
    kmax = enr.k.max()
    e10 = enr[(enr.k == kmax)]
    s_nov = summarize(e10.novel_f1); s_drop = summarize(e10.known_f1_drop_pts); s_auc = summarize(enr[enr.k == enr.k.min()].auroc)
    jet = [h for h in hw if _is_jetson(h)]
    fp32 = [h for h in hw if h["precision"] == "fp32"]
    h = jet[0] if jet else (fp32[0] if fp32 else (hw[0] if hw else None))
    if "ground→ground" in set(tr.condition):
        g2g = tr[tr.condition == "ground→ground"].macro_f1.mean(); g2a = tr[tr.condition == "ground→aerial (no enrollment)"].macro_f1.mean()
        rec = tr[(tr.condition == "aerial enrollment (merge)") & (tr.k == kmax)].macro_f1.mean()
        dr7 = f"ground→ground {g2g:.3f}; ground→aerial {g2a:.3f}; after {kmax}-shot aerial enrollment {rec:.3f}"
    else:                                   # CoFly only: species enrolled from the air with a ground-trained backbone
        d = tr[tr.condition == "aerial-only enrollment (disjoint species)"]
        parts = [f"k = {k}: {summarize(g.macro_f1)['mean']:.3f}" for k, g in d.groupby("k")]
        dr7 = "aerial-only enrollment (CoFly), macro-F1 " + "; ".join(parts) if parts else "no aerial set"
    A, B, C = cost_tbl.total.values
    rows = [
        ("DR1", "Fully offline inference and enrollment", "No network calls at runtime",
         "0 socket calls (benchmark run with sockets disabled)", "met"),
        ("DR2", "Real-time inference on UAV-class hardware", "< 50 ms per frame at 224 px, < 15 W",
         (f"{h['latency_ms_mean']:.1f} ms, {h['power_w_mean']:.1f} W on {h['device']}" if h and jet and not np.isnan(h["power_w_mean"])
          else (f"{h['latency_ms_mean']:.1f} ms on {h['device']} (simulated backbone)" if h and h.get("simulated")
                else (f"{h['latency_ms_mean']:.1f} ms measured on {h['device']} (upper bound); projected on Orin Nano: "
                      f"{PROJECTED_ORIN_NANO['dr2_text']}, from published benchmarks" if h else "not measured"))),
         ("met" if h["latency_ms_mean"] < 50 and h["power_w_mean"] < 15 else "not met") if jet
         else ("pending device run" if not h or h.get("simulated")
               else ("projected (not measured on device)" if h["latency_ms_mean"] < 50 else "not measured"))),
        ("DR3", "Compact model footprint", "< 25 MB model + prototype store",
         f"{h['model_mb']:.1f} MB + {h['store_kb']:.1f} kB" if h and not np.isnan(h["model_mb"]) else "n/a",
         "pending" if not h or np.isnan(h["model_mb"]) else ("met" if h["model_mb"] < 25 else "not met")),
        ("DR4", "Field enrollment without retraining", "macro-F1 on new species > 0.80 at 5–10 shots",
         f"{s_nov['mean']:.3f} ± {s_nov['se']:.3f} at k = {kmax} (n = {s_nov['n']})", "met" if s_nov["mean"] > 0.80 else "not met"),
        ("DR5", "No harm to existing classes", "< 2-point drop in known-class macro-F1",
         f"{s_drop['mean']:.2f} ± {s_drop['se']:.2f} points at k = {kmax}", "met" if s_drop["mean"] < 2 else "not met"),
        ("DR6", "Open-set detection of unenrolled plants", "AUROC > 0.85",
         f"{s_auc['mean']:.3f} ± {s_auc['se']:.3f} (n = {s_auc['n']})", "met" if s_auc["mean"] > 0.85 else "not met"),
        ("DR7", "Transfer across imaging platforms", "Documented degradation and recovery", dr7, "documented"),
        ("DR8", "Cost-effectiveness", "Per-node cost and break-even vs cloud and retraining",
         f"season cost A ${A:,.0f} vs B ${B:,.0f} vs C ${C:,.0f}", "met" if A < min(B, C) else "not met"),
    ]
    return pd.DataFrame(rows, columns=["ID", "Design requirement", "Target", "Result", "Status"])


# ==========================================================================
# DOCX export of tables + captions
# ==========================================================================
def write_docx(tables: Dict[str, pd.DataFrame], captions: Dict[str, str], table_captions: Dict[str, str],
               path: str, fig_dir: str):
    try:
        from docx import Document
        from docx.shared import Pt, Cm
        from docx.enum.text import WD_ALIGN_PARAGRAPH
    except ImportError:
        return None
    doc = Document()
    st = doc.styles["Normal"]; st.font.name = "Times New Roman"; st.font.size = Pt(12)
    doc.add_heading("Figures and tables — generated by weedenroll", level=1)
    for name, cap in captions.items():
        png = os.path.join(fig_dir, f"{name}.png")
        if os.path.exists(png):
            doc.add_picture(png, width=Cm(16))
        p = doc.add_paragraph(cap); p.paragraph_format.space_after = Pt(18)
    for key, df in tables.items():
        p = doc.add_paragraph(); r = p.add_run(table_captions.get(key, key)); r.bold = True
        t = doc.add_table(rows=1, cols=len(df.columns)); t.style = "Light Shading"
        for i, c in enumerate(df.columns):
            t.rows[0].cells[i].text = str(c)
        for _, row in df.iterrows():
            cells = t.add_row().cells
            for i, v in enumerate(row):
                cells[i].text = str(v)
        for row in t.rows:
            for c in row.cells:
                for para in c.paragraphs:
                    for run in para.runs:
                        run.font.size = Pt(9); run.font.name = "Times New Roman"
        doc.add_paragraph()
    doc.save(path)
    return path
