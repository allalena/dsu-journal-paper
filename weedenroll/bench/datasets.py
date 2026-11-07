"""Datasets.

Real datasets are expected in ImageFolder layout::

    <root>/<species_name>/<image>.jpg

``weedenroll data`` (module :mod:`weedenroll.data`) downloads DeepWeeds, Plant
Seedlings V2 and CoFly-WeedDB and converts them (CSV labels, class folders,
segmentation masks → per-plant crops) into that layout.

The synthetic suite reproduces the *structure* of the three public datasets
(species counts, a ground and an aerial platform, a domain shift between
them) in latent space so that every benchmark, figure and table can be run
and checked before real images or a Jetson are available.  Results from the
synthetic suite are labelled "synthetic" everywhere they appear.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..embed.base import Backbone

IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")


@dataclass
class Dataset:
    """A flat list of items with string labels and a platform tag."""
    name: str
    items: list                      # image paths or latent vectors
    labels: np.ndarray               # (N,) str
    platform: str                    # "ground" | "aerial"
    species: List[str] = field(default_factory=list)
    _emb_cache: Dict[str, np.ndarray] = field(default_factory=dict, repr=False)

    def __post_init__(self):
        self.labels = np.asarray(self.labels)
        if not self.species:
            self.species = sorted(set(self.labels.tolist()))

    def __len__(self):
        return len(self.items)

    def indices_of(self, name: str) -> np.ndarray:
        return np.where(self.labels == name)[0]

    def embeddings(self, backbone: Backbone, batch_size: int = 64) -> np.ndarray:
        key = f"{backbone.name}:{getattr(backbone, 'precision', 'fp32')}:{id(backbone)}"
        if key not in self._emb_cache:
            self._emb_cache[key] = backbone.embed_all(self.items, batch_size)
        return self._emb_cache[key]

    def subset(self, idx: Sequence[int], name: Optional[str] = None) -> "Dataset":
        idx = np.asarray(idx)
        d = Dataset(name or self.name, [self.items[i] for i in idx], self.labels[idx],
                    self.platform, species=self.species)
        for k, v in self._emb_cache.items():
            d._emb_cache[k] = v[idx]
        return d

    def split(self, test_frac: float = 0.3, seed: int = 0) -> Tuple["Dataset", "Dataset"]:
        """Stratified train/test split."""
        rng = np.random.default_rng(seed)
        tr, te = [], []
        for s in self.species:
            ix = self.indices_of(s)
            rng.shuffle(ix)
            n_te = max(1, int(round(len(ix) * test_frac)))
            te.extend(ix[:n_te].tolist())
            tr.extend(ix[n_te:].tolist())
        return self.subset(sorted(tr), self.name + "/train"), self.subset(sorted(te), self.name + "/test")

    def class_counts(self) -> Dict[str, int]:
        return {s: int((self.labels == s).sum()) for s in self.species}


# --------------------------------------------------------------------------
# Real images
# --------------------------------------------------------------------------
class ImageFolderDataset(Dataset):
    def __init__(self, root: str, name: Optional[str] = None, platform: str = "ground",
                 max_per_class: Optional[int] = None, exclude: Sequence[str] = ()):
        items, labels = [], []
        for cls in sorted(os.listdir(root)):
            p = os.path.join(root, cls)
            if not os.path.isdir(p) or cls in exclude:
                continue
            files = sorted(f for f in os.listdir(p) if f.lower().endswith(IMG_EXT))
            if max_per_class:
                files = files[:max_per_class]
            for f in files:
                items.append(os.path.join(p, f))
                labels.append(cls)
        super().__init__(name or os.path.basename(root.rstrip("/")), items, np.asarray(labels), platform)


# --------------------------------------------------------------------------
# Synthetic latent-space suite
# --------------------------------------------------------------------------
class SpeciesGeometry:
    """Latent geometry shared by all platforms of one synthetic world.

    Latent space = ``n_species`` class axes + ``n_nuisance`` nuisance axes
    (lighting, soil, growth stage).  Each species mean is a unit vector with
    cosine ``correlation`` to a common "plant" axis plus its own class axis.
    A fixed orthonormal map ``P`` embeds the latent into R^dim; a little
    isotropic noise is added there so the embedding is not exactly low-rank.
    """

    def __init__(self, species_names: Sequence[str], dim: int = 128, n_nuisance: int = 8,
                 correlation: float = 0.6, seed: int = 0):
        self.names = list(species_names)
        self.n = len(self.names)
        self.dim = dim
        self.n_nuisance = n_nuisance
        self.L = self.n + n_nuisance + 1                     # +1 common axis
        rng = np.random.default_rng(seed)
        P, _ = np.linalg.qr(rng.standard_normal((dim, self.L)))
        self.P = P.astype(np.float32)                         # (dim, L) orthonormal columns
        m = np.zeros((self.n, self.L), np.float32)
        m[:, 0] = correlation
        for i in range(self.n):
            m[i, 1 + i] = np.sqrt(1 - correlation ** 2)
        self.means = m                                         # latent unit means


class SyntheticWeedDataset(Dataset):
    """Samples from a :class:`SpeciesGeometry` for one imaging platform.

    * ``spread``      within-species noise std along the class axes (the part
                      that causes confusion); nuisance axes get ``spread``
                      too; the 128-D ambient gets ``iso_noise`` per dim.
    * ``platform="aerial"`` applies a domain shift in the latent:
        ``separation``  < 1 shrinks the class-axis signal (small plants),
        ``shift_mix``   mixes neighbouring class axes (species look alike),
        ``shift_vec``   translates along the nuisance axes (background),
        ``shift_noise`` adds extra class-axis noise.
    Sharing one geometry between a ground and an aerial dataset represents the
    same species imaged from two platforms, which is what DR7 measures.
    """

    def __init__(self, name: str, geom: SpeciesGeometry, names: Sequence[str] | None = None,
                 n_per_class: int = 120, spread: float = 0.35, platform: str = "ground",
                 separation: float = 1.0, shift_mix: float = 0.0, shift_vec: float = 0.0,
                 shift_noise: float = 0.0, iso_noise: float = 0.02, seed: int = 0,
                 shift_seed: int = 1234):
        rng = np.random.default_rng(seed)
        names = list(names) if names is not None else list(geom.names)
        n, L = geom.n, geom.L
        cls = slice(1, 1 + n); nui = slice(1 + n, L)
        means = geom.means.copy()
        means[:, cls] *= separation
        # Domain shift is a property of the *platform*, so it is drawn from
        # ``shift_seed`` (shared by every aerial dataset of one world), not from
        # the sample seed.  Class-subspace rotation: identity blended with a
        # random orthogonal matrix, re-orthonormalised, so classes stay as
        # separable as before but no longer line up with ground prototypes.
        srng = np.random.default_rng(shift_seed)
        M = np.eye(n)
        if shift_mix > 0:
            Q, _ = np.linalg.qr(srng.standard_normal((n, n)))
            M, Rm = np.linalg.qr((1 - shift_mix) * np.eye(n) + shift_mix * Q)
            M = M * np.sign(np.diag(Rm))[None, :]          # fix QR sign ambiguity
        t = np.zeros(L, np.float32); t[nui] = srng.standard_normal(L - 1 - n) * shift_vec
        items, labels = [], []
        for s in names:
            i = geom.names.index(s)
            z = np.repeat(means[i][None, :], n_per_class, 0)
            z[:, cls] += rng.standard_normal((n_per_class, n)) * (spread + shift_noise)
            z[:, nui] += rng.standard_normal((n_per_class, L - 1 - n)) * spread
            z[:, cls] = z[:, cls] @ M.T
            z += t[None, :]
            x = z @ geom.P.T + rng.standard_normal((n_per_class, geom.dim)) * iso_noise
            items.extend(list(x.astype(np.float32)))
            labels.extend([s] * n_per_class)
        super().__init__(name, items, np.asarray(labels), platform, species=names)
        self.meta = dict(spread=spread, separation=separation, shift_mix=shift_mix, shift_vec=shift_vec,
                         shift_noise=shift_noise, seed=seed)


DEEPWEEDS_SPECIES = ["Chinee apple", "Lantana", "Parkinsonia", "Parthenium",
                     "Prickly acacia", "Rubber vine", "Siam weed", "Snake weed"]
SEEDLINGS_SPECIES = ["Black-grass", "Charlock", "Cleavers", "Common chickweed", "Common wheat",
                     "Fat hen", "Loose silky-bent", "Maize", "Scentless mayweed",
                     "Shepherd's purse", "Small-flowered cranesbill", "Sugar beet"]
COFLY_SPECIES = ["Johnson grass", "Field bindweed", "Purslane"]


def make_synthetic_suite(dim: int = 128, seed: int = 0, n_per_class: int = 120,
                         spread: float = 0.22) -> Dict[str, Dataset]:
    """Mirror of the public-dataset protocol in latent space.

    Returns:
      deepweeds   ground, 8 species  (backbone 'training' domain + held-out species)
      seedlings   ground, 12 species (second enrollment source)
      cofly       aerial, 3 species shared with... nothing (real CoFly species)
      aerial_shared  aerial views of the 8 DeepWeeds species — the paired
                     ground/aerial set used for the transfer experiment.
    """
    # One geometry for every species so that cross-dataset species are
    # neighbours in the same embedding space, as they would be for a real
    # backbone.
    geom = SpeciesGeometry(DEEPWEEDS_SPECIES + SEEDLINGS_SPECIES + COFLY_SPECIES, dim, seed=seed + 10)
    aerial = dict(separation=0.8, shift_mix=0.6, shift_vec=0.2, shift_noise=0.04, shift_seed=seed + 99)
    suite = {
        "deepweeds": SyntheticWeedDataset("deepweeds", geom, DEEPWEEDS_SPECIES, n_per_class, spread,
                                          "ground", seed=seed + 1),
        "seedlings": SyntheticWeedDataset("seedlings", geom, SEEDLINGS_SPECIES, n_per_class, spread * 1.1,
                                          "ground", seed=seed + 2),
        "cofly": SyntheticWeedDataset("cofly", geom, COFLY_SPECIES, n_per_class, spread, "aerial",
                                      seed=seed + 3, **aerial),
        "aerial_shared": SyntheticWeedDataset("aerial_shared", geom, DEEPWEEDS_SPECIES, n_per_class, spread,
                                              "aerial", seed=seed + 4, **aerial),
    }
    return suite


def load_dataset(spec: dict) -> Dataset:
    """From YAML: {kind: imagefolder, root: ..., platform: ...}."""
    if spec.get("kind", "imagefolder") == "imagefolder":
        if not os.path.isdir(spec["root"]):
            raise FileNotFoundError(f"dataset root {spec['root']!r} not found; run `weedenroll data` first")
        return ImageFolderDataset(spec["root"], spec.get("name"), spec.get("platform", "ground"),
                                  spec.get("max_per_class"), spec.get("exclude", ()))
    raise ValueError(f"unknown dataset kind {spec}")
