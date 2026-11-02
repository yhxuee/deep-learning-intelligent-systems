"""Recompute every exact number quoted in the worked examples of Units 06-08."""
import math
import pathlib
import sys

import numpy as np
import pytest

SPINE = pathlib.Path(__file__).resolve().parent.parent / "spine"
sys.path.insert(0, str(SPINE))
import model as M  # noqa: E402
import stage06_scaling as S6  # noqa: E402

A100_FLOPS, A100_BW, A100_W = 312e12, 2.039e12, 400.0


# ----------------------------------------------------------------------------- Unit 06
def test_we_6_1_spine_flops():
    cfg = M.Config(vocab=50, d=64, n_layer=3, n_head=4, max_len=128)
    p = M.init_params(cfg)
    assert M.matmul_params(cfg) == 150_656
    assert M.n_params(p, cfg) == 160_704 and M.n_params(p, cfg, embedding=False) == 149_312
    per_tok = 6 * M.matmul_params(cfg) + 12 * cfg.n_layer * 128 * cfg.d
    assert per_tok == 1_198_848
    assert 12 * cfg.n_layer * 128 * cfg.d / per_tok == pytest.approx(0.246, abs=5e-4)
    M.FLOPS["count"] = 0
    x = np.zeros((16, 128), dtype=int)
    M.loss_and_grads(p, cfg, x, x)
    assert M.FLOPS["count"] == 2_455_240_704 == 2048 * per_tok
    assert M.FLOPS["count"] * 4000 == pytest.approx(9.82e12, rel=1e-3)
    assert 16 * M.n_params(p, cfg) / 1e6 == pytest.approx(2.57, abs=0.01)


def test_we_6_2_to_6_5_rules_of_thumb():
    assert 6 * 175e9 * 300e9 == pytest.approx(3.15e23)
    assert 16 * 7e9 / 1e9 == 112
    n = 4096
    assert 2 * n ** 3 / (3 * n * n * 2) == pytest.approx(1365.3, abs=0.1)
    assert A100_FLOPS / A100_BW == pytest.approx(153.0, abs=0.1)
    assert 14e9 / A100_BW * 1e3 == pytest.approx(6.87, abs=0.01)
    assert A100_BW / 14e9 == pytest.approx(145.6, abs=0.1)
    assert 1.4e10 / A100_FLOPS * 1e3 == pytest.approx(0.045, abs=5e-4)


def test_we_6_7_chinchilla_flat_optimum():
    E, A, B, a, b = 1.69, 406.4, 410.7, 0.34, 0.28
    N, D = S6.compute_optimal(5.76e23, A, B, a, b)
    assert N == pytest.approx(3.22e10, rel=5e-3) and D == pytest.approx(2.98e12, rel=5e-3)
    assert D / N == pytest.approx(92.6, abs=0.1)
    L = lambda n, d: E + A / n ** a + B / d ** b  # noqa: E731
    assert L(N, D) == pytest.approx(1.9307, abs=5e-5)
    assert L(70e9, 1.4e12) == pytest.approx(1.9366, abs=5e-5)
    assert b / (a + b) == pytest.approx(0.4516, abs=5e-5)


def test_we_6_8_energy():
    e = A100_W / (0.4 * A100_FLOPS)
    assert e == pytest.approx(3.205e-12, rel=1e-3)
    assert 3.14e23 * e / 3.6e9 == pytest.approx(279.6, abs=0.1)
    assert 1.380649e-23 * 300 * math.log(2) == pytest.approx(2.871e-21, rel=1e-3)


def test_unit06_problem_answers():
    assert 6e9 * 2e10 / (0.4 * A100_FLOPS) / 86400 == pytest.approx(11.1, abs=0.1)
    for B, I in [(1, 1.0), (16, 15.9), (256, 227.6)]:
        assert B * 4096 / (2 * B + 4096) == pytest.approx(I, abs=0.1)
    assert 0.34 * 0.28 / 0.62 == pytest.approx(0.154, abs=5e-4)


# ----------------------------------------------------------------------------- Unit 07
def test_we_7_1_chain_rule():
    losses = [-math.log(0.5), -math.log(0.4), -math.log(0.9)]
    assert sum(losses) == pytest.approx(1.7148, abs=5e-5) == -math.log(0.18)
    assert sum(losses) / 3 == pytest.approx(0.5716, abs=5e-5)


def test_we_7_2_free_targets():
    from common import load_corpus, load_lines
    import stage07_pretrain as S7

    final, _ = S7.clean(load_lines())
    tr, _, _ = load_corpus(train_lines=final)
    assert len(final) == 1187 and len(tr) == 363_418
    assert 4000 * 16 * 128 / len(tr) == pytest.approx(22.5, abs=0.05)


@pytest.mark.parametrize("J,P", [(0.3, 0.122), (0.5, 0.644), (0.7, 0.988), (0.8, 0.9998)])
def test_we_7_3_s_curve(J, P):
    assert 1 - (1 - J ** 4) ** 16 == pytest.approx(P, abs=5e-4)


def test_we_7_7_infonce_ceiling():
    assert math.log(8) == pytest.approx(2.08, abs=5e-3)
    assert math.log(1024) == pytest.approx(6.93, abs=5e-3)
    assert math.log(65536) == pytest.approx(11.09, abs=5e-3)


def test_unit07_problem_answers():
    assert sum(-math.log(p) for p in [0.2, 0.5, 0.5, 0.9]) == pytest.approx(3.101, abs=5e-4)
    for J, P in [(0.5, 0.470), (0.7, 0.975), (0.9, 1.000)]:
        assert 1 - (1 - J ** 5) ** 20 == pytest.approx(P, abs=5e-4)
    assert (1 / 20) ** (1 / 5) == pytest.approx(0.549, abs=5e-4)
    assert math.log(4096) - 2.0 == pytest.approx(6.32, abs=5e-3)
