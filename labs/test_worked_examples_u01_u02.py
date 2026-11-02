"""Recompute every number quoted in the worked examples of Units 01 and 02.

If a test fails, the text is wrong (or the code changed): fix whichever is in error.
"""
import math
import pathlib
import sys

import numpy as np
import pytest

SPINE = pathlib.Path(__file__).resolve().parent.parent / "spine"
sys.path.insert(0, str(SPINE))
from common import encode, load_corpus, load_lines, nll, split_lines  # noqa: E402
from stage01_bigram import fit_bigram  # noqa: E402

TR, VA, VOCAB = load_corpus()
V = len(VOCAB)


def H2(p):
    return -(p * math.log(p) + (1 - p) * math.log(1 - p))


def KL2(p, q):
    return p * math.log(p / q) + (1 - p) * math.log((1 - p) / (1 - q))


# ----------------------------------------------------------------------------- Unit 01
def test_we_1_1_coin():
    assert H2(0.7) == pytest.approx(0.6109, abs=5e-5)
    assert KL2(0.5, 0.7) == pytest.approx(0.0872, abs=5e-5)
    assert math.log(2) + KL2(0.5, 0.7) == pytest.approx(0.7803, abs=5e-5)
    assert H2(0.6) == pytest.approx(0.6730, abs=5e-5)
    assert KL2(0.6, 0.7) == pytest.approx(0.0226, abs=5e-5)
    assert H2(0.6) + KL2(0.6, 0.7) == pytest.approx(0.6956, abs=5e-5)


def _toy(alpha):
    voc = ["a", "b", "_"]
    tr = [voc.index(c) for c in "ab_ab_ba"]
    ho = [voc.index(c) for c in "ba_aa"]
    C = np.zeros((3, 3))
    for a, b in zip(tr[:-1], tr[1:]):
        C[a, b] += 1
    P = (C + alpha) / (C + alpha).sum(1, keepdims=True)
    with np.errstate(divide="ignore"):
        lt = -np.log(P[tr[:-1], tr[1:]])
        lh = -np.log(P[ho[:-1], ho[1:]])
    return lt, lh


def test_we_1_2_bigram_by_hand():
    lt, lh = _toy(0.0)
    assert lt.sum() == pytest.approx(3.2958, abs=5e-5)
    assert lt.mean() == pytest.approx(0.4708, abs=5e-5)
    assert np.isinf(lh.mean())
    lt, lh = _toy(1.0)
    assert lt.sum() == pytest.approx(5.3391, abs=5e-5) and lt.mean() == pytest.approx(0.7627, abs=5e-5)
    assert lh.sum() == pytest.approx(5.2338, abs=5e-5) and lh.mean() == pytest.approx(1.3084, abs=5e-5)
    assert lh.mean() > math.log(3)


def test_we_1_3_spine_baselines():
    P = fit_bigram(TR, V, 0.1)
    assert nll(P, VA) / math.log(2) == pytest.approx(3.065, abs=5e-4) and math.exp(nll(P, VA)) == pytest.approx(8.37, abs=5e-3)
    assert V == 50 and len(TR) == 385_274 and len(VA) == 42_881
    assert math.log(V) == pytest.approx(3.912, abs=5e-4)
    c = np.bincount(TR, minlength=V) + 0.1
    u = c / c.sum()
    assert -np.mean(np.log(u[VA])) == pytest.approx(3.060, abs=5e-4)
    assert -np.mean(np.log(u[TR])) == pytest.approx(3.060, abs=5e-4)
    P = fit_bigram(TR, V, 0.1)
    assert nll(P, TR) == pytest.approx(2.123, abs=5e-4)
    assert nll(P, VA) == pytest.approx(2.124, abs=5e-4)
    assert nll(fit_bigram(TR, V, 1e12), VA) == pytest.approx(math.log(V), abs=1e-6)


@pytest.mark.parametrize("n,train,held", [(500, 1.925, 2.763), (2000, 2.057, 2.371), (8000, 2.098, 2.198),
                                          (32000, 2.126, 2.148), (128000, 2.120, 2.127), (385274, 2.123, 2.124)])
def test_we_1_4_learning_curve(n, train, held):
    P = fit_bigram(TR[:n], V, 0.1)
    assert nll(P, TR[:n]) == pytest.approx(train, abs=5e-4)
    assert nll(P, VA) == pytest.approx(held, abs=5e-4)


@pytest.mark.parametrize("alpha,train,held", [(0.001, 1.977, 2.523), (0.01, 1.986, 2.415), (0.03, 2.004, 2.378), (0.1, 2.057, 2.371),
                                              (0.3, 2.170, 2.428), (1, 2.413, 2.607), (3, 2.759, 2.892)])
def test_we_1_5_choosing_alpha(alpha, train, held):
    small = TR[:2000]
    P = fit_bigram(small, V, alpha)
    assert nll(P, small) == pytest.approx(train, abs=5e-4)
    assert nll(P, VA) == pytest.approx(held, abs=5e-4)


def test_we_1_5_unseen_pairs():
    small = TR[:2000]
    C = np.zeros((V, V))
    np.add.at(C, (small[:-1], small[1:]), 1)
    assert (C > 0).sum() == 269
    assert (C[VA[:-1], VA[1:]] == 0).mean() == pytest.approx(0.054, abs=5e-4)
    assert 1 - 269 / 2500 == pytest.approx(0.892, abs=5e-4)


def test_we_1_6_hoeffding():
    assert math.sqrt(math.log(40) / 20000) == pytest.approx(0.0136, abs=5e-5)
    assert math.sqrt((math.log(1000) + math.log(40)) / 20000) == pytest.approx(0.0230, abs=5e-5)


def test_we_1_7_standard_errors():
    P = fit_bigram(TR, V, 0.1)
    _, held = split_lines(load_lines())
    per, chars = [], []
    for s in held:
        ids = encode("\n" + s, VOCAB)
        loss = -np.log(P[ids[:-1], ids[1:]])
        per.append(loss.mean())
        chars.append(loss)
    per, chars = np.array(per), np.concatenate(chars)
    assert len(per) == 140 and per.mean() == pytest.approx(2.1204, abs=5e-5)
    assert per.std(ddof=1) == pytest.approx(0.0617, abs=5e-5)
    assert per.std(ddof=1) / math.sqrt(140) == pytest.approx(0.0052, abs=5e-5)
    assert len(chars) == 42_741 and chars.std(ddof=1) == pytest.approx(1.244, abs=5e-4)
    assert chars.std(ddof=1) / math.sqrt(len(chars)) == pytest.approx(0.0060, abs=5e-5)
    x = chars - chars.mean()
    assert (x[:-1] * x[1:]).mean() / x.var() == pytest.approx(-0.32, abs=5e-3)
    assert ((1.96 + 0.84) * 0.5 / 0.02) ** 2 == pytest.approx(4900, abs=1)


def test_we_1_8_memorization_bits():
    assert 50_000 * math.log2(10) == pytest.approx(166_096, abs=1)


def test_unit01_problem_answers():
    f08 = (3 * -math.log(0.8) + 2 * -math.log(0.2)) / 5
    f06 = (3 * -math.log(0.6) + 2 * -math.log(0.4)) / 5
    assert f08 == pytest.approx(0.7777, abs=5e-5) and f06 == pytest.approx(0.6730, abs=5e-5)
    assert math.ceil(math.log(200) / (2 * 0.005 ** 2)) == 105_967
    assert math.ceil((math.log(1000) + math.log(200)) / (2 * 0.005 ** 2)) == 244_122


# ----------------------------------------------------------------------------- Unit 02
from math import comb  # noqa: E402

import stage02_embeddings as S2  # noqa: E402


def test_we_2_1_xor():
    pts = [(0, 0, 0), (0, 1, 1), (1, 0, 1), (1, 1, 0)]
    w, b = (1, 1, -2), -0.5
    scores = [w[0] * x1 + w[1] * x2 + w[2] * x1 * x2 + b for x1, x2, _ in pts]
    assert scores == [-0.5, 0.5, 0.5, -0.5]
    assert all((s > 0) == bool(y) for s, (_, _, y) in zip(scores, pts))
    # no affine separator on a fine grid of (w1, w2, b)
    g = np.linspace(-3, 3, 61)
    W1, W2, B = np.meshgrid(g, g, g, indexing="ij")
    ok = (B <= 0) & (W2 + B > 0) & (W1 + B > 0) & (W1 + W2 + B <= 0)
    assert not ok.any()


def test_we_2_2_one_hot_readout_is_a_table():
    W = np.log(np.array([[1 / 5, 3 / 5, 1 / 5], [2 / 6, 1 / 6, 3 / 6], [2 / 5, 2 / 5, 1 / 5]]))
    z = W.T @ np.array([0, 1, 0])
    assert np.allclose(z, [-1.0986, -1.7918, -0.6931], atol=5e-5)
    p = np.exp(z) / np.exp(z).sum()
    assert np.allclose(p, [2 / 6, 1 / 6, 3 / 6])


@pytest.mark.parametrize("k,params", [(1, 9426), (2, 11474), (3, 13522), (4, 15570), (6, 19666)])
def test_we_2_3_neural_ngram_parameters(k, params):
    p = S2.ngram_init(50, k, d=16, H=128)
    assert sum(v.size for v in p.values()) == params == 7378 + 2048 * k


@pytest.mark.parametrize("k,seen", [(1, 50), (2, 401), (3, 2078), (4, 4502), (6, 12903)])
def test_we_2_3_contexts_seen(k, seen):
    contexts = {tuple(TR[i - k : i]) for i in range(k, len(TR))}
    assert len(contexts) == seen


def test_we_2_4_and_2_5_factorized_bigram_width_and_geometry():
    p, _ = S2.train(TR, V, d=16, steps=1500)
    assert S2.heldout_nll(p, VA) == pytest.approx(2.15, abs=0.01)
    assert sum(v.size for v in p.values()) == 1106
    near3 = [c for c, _ in S2.nearest(p, VOCAB, "3", k=4)]
    assert all(c.isdigit() for c in near3)
    nearT = [c for c, _ in S2.nearest(p, VOCAB, "T", k=3)]
    assert all(c.isupper() for c in nearT)


def test_we_2_7_lookup_gradient():
    G = np.array([[1.0, 2], [3, 4], [5, 6]])
    dE = np.zeros((3, 2))
    np.add.at(dE, [0, 2, 0], G)
    U = np.eye(3)[[0, 2, 0]]
    assert np.array_equal(dE, [[6, 8], [0, 0], [3, 4]]) and np.array_equal(dE, U.T @ G)
    bug = np.zeros((3, 2))
    bug[[0, 2, 0]] = G
    assert np.array_equal(bug[0], [5, 6])


def cover(N, p):
    return 2 * sum(comb(N - 1, k) for k in range(p))


def test_we_2_8_cover_count():
    assert cover(4, 3) == 14 and cover(4, 4) == 16
    assert cover(10, 3) == 92


def test_unit02_problem_answers():
    assert 256 * 32 + 8 * 32 * 512 + 512 + 512 * 256 + 256 == 271_104
    assert math.ceil(8 * math.log(1e4) / 0.1 ** 2) == 7369
    assert math.ceil(8 * math.log(1e4) / 0.01 ** 2) == 736_828
