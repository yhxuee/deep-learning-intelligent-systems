"""Verification targets for spine stages 01 and 03."""
import pathlib
import sys

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from common import load_corpus, nll  # noqa: E402
from stage01_bigram import fit_bigram  # noqa: E402
from stage03_neural_bigram import loss_and_grad, train  # noqa: E402

TRAIN, VAL, VOCAB = load_corpus()
V = len(VOCAB)


def test_stage01_beats_uniform_on_held_out():
    P = fit_bigram(TRAIN, V, alpha=0.1)
    assert nll(P, VAL) < np.log(V) - 1.0


def test_stage01_large_alpha_recovers_uniform_exactly():
    P = fit_bigram(TRAIN, V, alpha=1e12)
    assert nll(P, VAL) == pytest.approx(np.log(V), abs=1e-6)


def test_stage01_rows_are_distributions():
    P = fit_bigram(TRAIN, V, alpha=0.5)
    assert np.allclose(P.sum(axis=1), 1.0)


def test_stage03_init_loss_is_ln_V():
    loss, _ = loss_and_grad(np.zeros((V, V)), TRAIN[:-1], TRAIN[1:])
    assert loss == pytest.approx(np.log(V), abs=1e-12)


def test_stage03_gradient_matches_finite_differences():
    rng = np.random.default_rng(0)
    W = rng.normal(0, 0.1, (V, V))
    x, y = TRAIN[:200], TRAIN[1:201]
    _, g = loss_and_grad(W, x, y)
    for _ in range(20):
        i, j = int(x[rng.integers(len(x))]), int(rng.integers(V))
        Wp, Wm = W.copy(), W.copy()
        Wp[i, j] += 1e-6
        Wm[i, j] -= 1e-6
        num = (loss_and_grad(Wp, x, y)[0] - loss_and_grad(Wm, x, y)[0]) / 2e-6
        assert num == pytest.approx(g[i, j], rel=1e-5, abs=1e-10)


def test_stage03_converges_to_count_mle_and_never_below():
    mle = nll(fit_bigram(TRAIN, V, alpha=0.0), TRAIN)
    _, hist = train(TRAIN, V, steps=1000)
    assert hist.min() >= mle - 1e-9, "a neural bigram cannot beat the count MLE on its own training data"
    assert hist[-1] - mle < 0.01
