import os
import socket
import tempfile

import numpy as np
import pytest

from weedenroll import PrototypeStore, OpenSetRejector, Enroller, GovernanceLogger
from weedenroll.embed import SyntheticBackbone
from weedenroll.bench.datasets import make_synthetic_suite
from weedenroll.bench.episodes import EpisodicBenchmark, no_network
from weedenroll.cost import CostAssumptions, SeasonCostModel


def _clusters(n_cls=4, n=30, dim=32, seed=0):
    rng = np.random.default_rng(seed)
    mu = rng.standard_normal((n_cls, dim)); mu /= np.linalg.norm(mu, axis=1, keepdims=True)
    X = np.concatenate([mu[i] + 0.2 * rng.standard_normal((n, dim)) for i in range(n_cls)])
    y = np.repeat([f"s{i}" for i in range(n_cls)], n)
    return X.astype(np.float32), y


def test_store_roundtrip_and_predict():
    X, y = _clusters()
    st = PrototypeStore(32)
    for s in np.unique(y):
        st.enroll(s, X[y == s])
    pred = np.asarray(st.predict_names(X))
    assert (pred == y).mean() > 0.95
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "store.json"); st.save(p)
        st2 = PrototypeStore.load(p)
        assert st2.names == st.names
        np.testing.assert_allclose(st2.distances(X[:5]), st.distances(X[:5]), atol=1e-6)


def test_enrollment_is_write_only_and_offline():
    X, y = _clusters(n_cls=5)
    bb = SyntheticBackbone(dim=32)
    st = PrototypeStore(32)
    for s in ["s0", "s1", "s2", "s3"]:
        st.enroll(s, X[y == s])
    rej = OpenSetRejector(st); rej.calibrate(X[y != "s4"], y[y != "s4"])
    # s4 is unknown before enrollment
    assert rej.predict(X[y == "s4"]).is_unknown.mean() > 0.8
    with tempfile.TemporaryDirectory() as d:
        lg = GovernanceLogger(d, "t")
        with no_network():
            with pytest.raises(RuntimeError):
                socket.socket()          # the guard really blocks sockets
            rep = Enroller(bb, st, rej, lg).enroll("s4", list(X[y == "s4"][:5]))
        assert rep.network_calls == 0 and rep.n_shots == 5
        assert "s4" in st.names
        assert rej.predict(X[y == "s4"]).is_unknown.mean() < 0.3
        s = lg.close(st.snapshot())
        assert s.n_enrollments == 1


def test_openset_auroc_on_synthetic():
    suite = make_synthetic_suite(dim=64, n_per_class=60)
    bb = SyntheticBackbone(dim=64)
    b = EpisodicBenchmark(bb, shots=[5], episodes=3, n_novel=2)
    df = b.enrollment_episodes(suite["deepweeds"], tag="within")
    assert df.auroc.mean() > 0.85
    assert df.novel_f1.mean() > 0.8
    assert df.known_f1_drop_pts.mean() < 2


def test_transfer_degrades_then_recovers():
    suite = make_synthetic_suite(dim=64, n_per_class=60)
    bb = SyntheticBackbone(dim=64)
    b = EpisodicBenchmark(bb, shots=[10], episodes=3)
    df = b.transfer_episodes(suite["deepweeds"], suite["aerial_shared"], suite["cofly"])
    gg = df[df.condition == "ground→ground"].macro_f1.mean()
    ga = df[df.condition == "ground→aerial (no enrollment)"].macro_f1.mean()
    rec = df[df.condition == "aerial enrollment (replace)"].macro_f1.mean()
    assert ga < gg and rec > ga


def test_int8_simulation_close_to_fp32():
    X, _ = _clusters()
    z32 = SyntheticBackbone(32, "fp32").embed(X); z8 = SyntheticBackbone(32, "int8").embed(X)
    assert np.abs((z32 * z8).sum(1) - 1).max() < 0.01


def test_cost_model_consistency():
    m = SeasonCostModel(CostAssumptions())
    t = m.table()
    assert len(t) == 3 and (t.total > 0).all()
    s = m.sensitivity()
    assert set(s.columns) >= {"parameter", "swing"}
    # more new species must raise retraining options but not option A's retraining line
    sp = m.species_sweep(5)
    c = sp[sp.option.str.startswith("C")].total.values
    assert np.all(np.diff(c) > 0)
