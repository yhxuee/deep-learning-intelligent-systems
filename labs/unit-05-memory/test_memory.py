"""Invariant tests for Atlas nodes mech-linear-recurrence and the linear-attention memory (Unit 05)."""
import pathlib
import sys

import numpy as np
import pytest

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from _loader import load  # noqa: E402

mem = load(HERE, "memory")
rng = np.random.default_rng(0)


def _system(N=4, m=2, rho=0.8, seed=0):
    r = np.random.default_rng(seed)
    S = r.normal(size=(N, N))
    S = (S + S.T) / 2                                # symmetric -> normal, real spectrum
    A = rho * S / np.max(np.abs(np.linalg.eigvalsh(S)))
    return A, r.normal(size=(N, m))


def test_closed_form_matches_loop():
    A, B = _system()
    xs, h0 = rng.normal(size=(25, 2)), rng.normal(size=4)
    assert np.allclose(mem.recurrence_loop(A, B, xs, h0), mem.recurrence_closed_form(A, B, xs, h0), atol=1e-10)


def test_convolution_equivalence_with_zero_initial_state():
    A, B = _system()
    xs = rng.normal(size=(20, 2))
    hs = mem.recurrence_loop(A, B, xs, np.zeros(4))
    K = mem.convolution_kernel(A, B, 20)
    conv = np.array([sum(K[t - k] @ xs[k] for k in range(t + 1)) for t in range(20)])
    assert np.allclose(hs, conv, atol=1e-10)


def test_spectral_radius_below_one_decays_geometrically():
    A, _ = _system(rho=0.8)
    n = mem.influence_norms(A, 60)
    assert np.all(np.diff(n[1:]) <= 1e-12)           # normal matrix: monotone decay
    assert n[60] == pytest.approx(0.8 ** 60, rel=1e-6)


def test_spectral_radius_above_one_explodes():
    A, _ = _system(rho=1.1)
    assert mem.influence_norms(A, 60)[60] > 100


def test_non_normal_transient_growth():
    A = np.array([[0.9, 5.0], [0.0, 0.9]])            # spectral radius 0.9, yet ...
    n = mem.influence_norms(A, 200)
    assert 19.0 < n.max() < 20.0 and int(np.argmax(n)) in (9, 10)
    assert n[200] < 1e-3                               # ... it does decay eventually


def test_scalar_half_life_known_value():
    assert mem.half_life(0.9) == pytest.approx(6.5788, abs=1e-4)


def test_linear_attention_exact_recall_up_to_dk_orthonormal_keys():
    d_k, d_v = 16, 8
    Q, _ = np.linalg.qr(rng.normal(size=(d_k, d_k)))
    keys, values = Q.T, rng.normal(size=(d_k, d_v))
    M = mem.LinearAttentionMemory(d_k, d_v)
    for k, v in zip(keys, values):
        M.write(k, v)
    for k, v in zip(keys, values):
        assert np.allclose(M.read(k), v, atol=1e-10)


def test_linear_attention_interference_beyond_dk():
    d_k, d_v, n_pairs = 16, 8, 64
    keys = rng.normal(size=(n_pairs, d_k))
    keys /= np.linalg.norm(keys, axis=1, keepdims=True)
    values = rng.normal(size=(n_pairs, d_v))
    M = mem.LinearAttentionMemory(d_k, d_v)
    for k, v in zip(keys, values):
        M.write(k, v)
    rel_err = np.mean([np.linalg.norm(M.read(k) - v) / np.linalg.norm(v) for k, v in zip(keys, values)])
    assert rel_err > 0.5, "with 4x more pairs than key dimensions, exact recall is impossible"
