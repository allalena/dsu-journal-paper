"""Episodic benchmark for DR1, DR4, DR5, DR6, DR7.

Protocol (mirrors the paper's Materials and Methods):

* **Known/novel split.** In each episode, ``n_novel`` species of the base
  dataset are withheld; the remaining species form the *known* store built
  from the train split.  Novel species are then enrolled from ``k`` shots
  drawn from their train split and evaluated on their test split.
* **DR4** macro-F1 on novel species after k-shot enrollment.
* **DR5** macro-F1 on known species before vs after enrollment (harm).
* **DR6** AUROC of the unknown-ness score, known-test vs novel-test, *before*
  enrollment; plus TPR at 5 % FPR and the ROC curve.
* **DR7** ground→aerial transfer: known store built from ground images,
  evaluated on aerial images of the same species (degradation); then
  aerial prototypes enrolled from k aerial shots (recovery).  A second
  aerial set with disjoint species (CoFly) measures aerial-only enrollment.
* **DR1** the whole benchmark runs inside :func:`no_network`, which makes any
  socket creation raise; ``network_calls`` is therefore 0 by construction.

Every metric is reported per episode so the report can state mean ± SE and n.
"""
from __future__ import annotations

import contextlib
import socket
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from ..embed.base import Backbone
from ..enroll import Enroller
from ..govern import GovernanceLogger
from ..openset import OpenSetRejector
from ..store import PrototypeStore
from .datasets import Dataset
from .metrics import macro_f1, accuracy, auroc_unknown, roc_points, tpr_at_fpr


@contextlib.contextmanager
def no_network():
    """Block socket creation for the duration (DR1 evidence)."""
    real = socket.socket

    class _Blocked(socket.socket):
        def __init__(self, *a, **k):
            raise RuntimeError("network access attempted during offline benchmark")

    socket.socket = _Blocked
    try:
        yield
    finally:
        socket.socket = real


class EpisodicBenchmark:
    def __init__(self, backbone: Backbone, shots: Sequence[int] = (1, 5, 10),
                 episodes: int = 20, n_novel: int = 2, test_frac: float = 0.3,
                 openset_rule: str = "global", openset_quantile: float = 0.95,
                 prototypes_per_class: int = 1, seed: int = 0,
                 log_dir: Optional[str] = None):
        self.backbone = backbone
        self.shots = list(shots)
        self.episodes = episodes
        self.n_novel = n_novel
        self.test_frac = test_frac
        self.rule = openset_rule
        self.quantile = openset_quantile
        self.k_proto = prototypes_per_class
        self.seed = seed
        self.log_dir = log_dir

    # ------------------------------------------------------------------
    def _store_from(self, ds: Dataset, species: Sequence[str], platform: str) -> PrototypeStore:
        emb = ds.embeddings(self.backbone)
        st = PrototypeStore(self.backbone.dim, self.k_proto)
        for s in species:
            st.enroll(s, emb[ds.indices_of(s)], source="train", platform=platform)
        return st

    def _eval(self, store: PrototypeStore, ds: Dataset, species: Sequence[str]) -> Dict[str, float]:
        emb = ds.embeddings(self.backbone)
        ix = np.concatenate([ds.indices_of(s) for s in species])
        y = ds.labels[ix]
        yhat = np.asarray(store.predict_names(emb[ix]))
        return {"macro_f1": macro_f1(y, yhat, labels=list(species)), "acc": accuracy(y, yhat)}

    # ------------------------------------------------------------------
    def known_class_table(self, datasets: Dict[str, Dataset]) -> pd.DataFrame:
        """Closed-set baseline: all species known, train→test (per dataset)."""
        rows = []
        for name, ds in datasets.items():
            tr, te = ds.split(self.test_frac, self.seed)
            st = self._store_from(tr, ds.species, ds.platform)
            m = self._eval(st, te, ds.species)
            rows.append({"dataset": name, "platform": ds.platform, "n_species": len(ds.species),
                         "n_test": len(te), "precision": getattr(self.backbone, "precision", "fp32"),
                         **m})
        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    def enrollment_episodes(self, base: Dataset, novel_source: Optional[Dataset] = None,
                            tag: str = "within", novel_pool: Optional[Sequence[str]] = None) -> pd.DataFrame:
        """DR4/DR5/DR6.  If ``novel_source`` is given, novel species are drawn
        from it (cross-dataset enrollment) and all of ``base`` is known.
        ``novel_pool`` restricts which species may be novel — with a trained
        backbone this must be the species held out from training."""
        rng = np.random.default_rng(self.seed)
        base_tr, base_te = base.split(self.test_frac, self.seed)
        if novel_source is not None:
            nov_tr, nov_te = novel_source.split(self.test_frac, self.seed)
        pool = list(novel_pool) if novel_pool else None
        if pool is not None:
            missing = [s for s in pool if s not in base.species]
            if len(missing) == len(pool):
                raise ValueError(f"none of the held-out species {pool} exist in {base.name}; "
                                 f"its species are {list(base.species)}")
            if missing:
                print(f"[benchmark] warning: held-out species not in {base.name}: {missing}")
        rows = []
        for ep in range(self.episodes):
            if novel_source is None:
                cand = [s for s in base.species if s in pool] if pool else list(base.species)
                novel = list(rng.choice(cand, min(self.n_novel, len(cand)), replace=False))
                known = [s for s in base.species if s not in novel]
                nov_tr, nov_te = base_tr, base_te
            else:
                novel = list(rng.choice(novel_source.species, self.n_novel, replace=False))
                known = list(base.species)
            store = self._store_from(base_tr, known, base.platform)
            rej = OpenSetRejector(store, self.rule, self.quantile)
            rej.calibrate(base_te.embeddings(self.backbone)[np.isin(base_te.labels, known)],
                          base_te.labels[np.isin(base_te.labels, known)])
            logger = GovernanceLogger(self.log_dir, f"ep{ep}-{tag}") if self.log_dir else None
            if logger:
                logger.set_baseline(store.snapshot())

            # --- before enrollment: known F1 and open-set ----------------
            before = self._eval(store, base_te, known)
            emb_te = base_te.embeddings(self.backbone)
            kn_ix = np.isin(base_te.labels, known)
            nov_emb_te = nov_te.embeddings(self.backbone)
            nov_ix = np.isin(nov_te.labels, novel)
            s_known = rej.score(emb_te[kn_ix])
            s_unknown = rej.score(nov_emb_te[nov_ix])
            auroc = auroc_unknown(s_known, s_unknown)
            tpr5 = tpr_at_fpr(s_known, s_unknown, 0.05)
            r = rej.predict(emb_te[kn_ix])
            frr = float(r.is_unknown.mean())                   # known wrongly rejected
            r2 = rej.predict(nov_emb_te[nov_ix])
            tdr = float(r2.is_unknown.mean())                  # novel correctly flagged
            if logger:
                logger.log_inference(np.r_[r.distance, r2.distance], np.r_[r.is_unknown, r2.is_unknown])

            # --- enrollment at each k (fresh copy of the store) -----------
            for k in self.shots:
                st_k = _copy_store(store)
                rej_k = OpenSetRejector(st_k, self.rule, self.quantile)
                rej_k.global_thr, rej_k.class_thr = rej.global_thr, dict(rej.class_thr)
                enr = Enroller(self.backbone, st_k, rej_k, logger)
                nov_emb_tr = nov_tr.embeddings(self.backbone)
                for s in novel:
                    ix = nov_tr.indices_of(s)
                    shots_ix = rng.choice(ix, min(k, len(ix)), replace=False)
                    enr.enroll_embeddings(s, nov_emb_tr[shots_ix], platform=nov_tr.platform)
                after_known = self._eval(st_k, base_te, known)
                after_novel = self._eval(st_k, nov_te, novel)
                # overall on union of known + novel test
                rows.append({
                    "tag": tag, "episode": ep, "k": k, "novel": "|".join(novel),
                    "known_f1_before": before["macro_f1"], "known_f1_after": after_known["macro_f1"],
                    "known_f1_drop_pts": 100 * (before["macro_f1"] - after_known["macro_f1"]),
                    "novel_f1": after_novel["macro_f1"], "novel_acc": after_novel["acc"],
                    "auroc": auroc, "tpr_at_5fpr": tpr5, "known_frr": frr, "novel_tdr": tdr,
                    "store_bytes": st_k.nbytes(), "network_calls": 0,
                })
            if logger:
                logger.close(store.snapshot())
        return pd.DataFrame(rows)

    def roc_curve_example(self, base: Dataset, seed: int = 0):
        """One representative ROC (first episode's split) for the figure."""
        rng = np.random.default_rng(seed)
        tr, te = base.split(self.test_frac, self.seed)
        novel = list(rng.choice(base.species, self.n_novel, replace=False))
        known = [s for s in base.species if s not in novel]
        store = self._store_from(tr, known, base.platform)
        rej = OpenSetRejector(store, self.rule, self.quantile)
        emb = te.embeddings(self.backbone)
        kn, nv = np.isin(te.labels, known), np.isin(te.labels, novel)
        rej.calibrate(emb[kn], te.labels[kn])
        s_k, s_u = rej.score(emb[kn]), rej.score(emb[nv])
        fpr, tpr, thr = roc_points(s_k, s_u)
        return {"fpr": fpr, "tpr": tpr, "thr": thr, "score_known": s_k, "score_unknown": s_u,
                "global_thr": rej.global_thr, "auroc": auroc_unknown(s_k, s_u)}

    # ------------------------------------------------------------------
    def transfer_episodes(self, ground: Dataset, aerial_shared: Optional[Dataset],
                          aerial_disjoint: Optional[Dataset] = None) -> pd.DataFrame:
        """DR7.  ``aerial_shared`` must contain the same species as ``ground``;
        pass None to run only the aerial-only (disjoint species) part."""
        rng = np.random.default_rng(self.seed + 7)
        g_tr, g_te = ground.split(self.test_frac, self.seed)
        if aerial_shared is not None:
            species = [s for s in ground.species if s in aerial_shared.species]
            a_tr, a_te = aerial_shared.split(self.test_frac, self.seed)
        rows = []
        for ep in range(self.episodes):
            if aerial_shared is None:
                if aerial_disjoint is None:
                    break
                d_tr, d_te = aerial_disjoint.split(self.test_frac, self.seed)
                d_emb = d_tr.embeddings(self.backbone)
                for k in self.shots:
                    st_d = PrototypeStore(self.backbone.dim, self.k_proto)
                    for s in aerial_disjoint.species:
                        ix = rng.choice(d_tr.indices_of(s), min(k, len(d_tr.indices_of(s))), replace=False)
                        st_d.enroll(s, d_emb[ix], source="field", platform="aerial")
                    rows.append({"episode": ep, "condition": "aerial-only enrollment (disjoint species)",
                                 "k": k, **self._eval(st_d, d_te, aerial_disjoint.species)})
                continue
            store = self._store_from(g_tr, species, "ground")
            gg = self._eval(store, g_te, species)                  # ground proto → ground query
            ga = self._eval(store, a_te, species)                  # ground proto → aerial query
            rows.append({"episode": ep, "condition": "ground→ground", "k": 0, **gg})
            rows.append({"episode": ep, "condition": "ground→aerial (no enrollment)", "k": 0, **ga})
            a_emb_tr = a_tr.embeddings(self.backbone)
            for k in self.shots:
                # (a) replace: aerial prototypes only
                st_r = _copy_store(store)
                for s in species:
                    ix = rng.choice(a_tr.indices_of(s), min(k, len(a_tr.indices_of(s))), replace=False)
                    st_r.enroll(s, a_emb_tr[ix], source="field", platform="aerial", replace=True)
                rows.append({"episode": ep, "condition": "aerial enrollment (replace)", "k": k,
                             **self._eval(st_r, a_te, species)})
                # (b) merge: ground + aerial prototypes averaged
                st_m = _copy_store(store)
                for s in species:
                    ix = rng.choice(a_tr.indices_of(s), min(k, len(a_tr.indices_of(s))), replace=False)
                    st_m.enroll(s, a_emb_tr[ix], source="field", platform="aerial", replace=False)
                rows.append({"episode": ep, "condition": "aerial enrollment (merge)", "k": k,
                             **self._eval(st_m, a_te, species)})
                # and the merged store must not have harmed ground queries
                rows.append({"episode": ep, "condition": "ground after merge", "k": k,
                             **self._eval(st_m, g_te, species)})
            if aerial_disjoint is not None:
                d_tr, d_te = aerial_disjoint.split(self.test_frac, self.seed)
                d_emb = d_tr.embeddings(self.backbone)
                for k in self.shots:
                    st_d = PrototypeStore(self.backbone.dim, self.k_proto)
                    for s in aerial_disjoint.species:
                        ix = rng.choice(d_tr.indices_of(s), min(k, len(d_tr.indices_of(s))), replace=False)
                        st_d.enroll(s, d_emb[ix], source="field", platform="aerial")
                    rows.append({"episode": ep, "condition": "aerial-only enrollment (disjoint species)",
                                 "k": k, **self._eval(st_d, d_te, aerial_disjoint.species)})
        return pd.DataFrame(rows)


def _copy_store(store: PrototypeStore) -> PrototypeStore:
    import copy
    return copy.deepcopy(store)
