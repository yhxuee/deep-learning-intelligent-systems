"""Every number quoted in the worked examples of Units 03–05, recomputed.

If a test here fails after an edit to the text or the code, the text and the
computation disagree — fix whichever is wrong. Sections are cited per test.
"""
import math

import numpy as np
import pytest

approx = pytest.approx


# ----------------------------------------------------------------------------- Unit 03
def test_u03_line_fit_gradient_descent():  # §3.2.1, Worked Example 3.1
    x, y = np.array([1.0, 2.0, 3.0]), np.array([2.0, 3.9, 6.1])
    L = lambda w: np.mean((w * x - y) ** 2)
    dL = lambda w: np.mean(2 * (w * x - y) * x)
    assert (x @ y) / (x @ x) == approx(2.007142857)
    assert 2 * np.mean(x ** 2) == approx(28 / 3)
    w, iterates = 0.0, []
    for _ in range(5):
        w -= 0.1 * dL(w)
        iterates.append(w)
    assert iterates == approx([1.873333, 1.998222, 2.006548, 2.007103, 2.007140], abs=1e-6)
    assert L(2.007140) == approx(0.006429, abs=1e-6)


def test_u03_parabola_step_sizes():  # §3.2.3
    for eta, factor in [(0.1, 0.6), (0.25, 0.0), (0.4, -0.6), (0.6, -1.4)]:
        assert 1 - eta * 4 == approx(factor)
    assert (-1.4) ** 4 == approx(3.8416)


def test_u03_neuron_and_fanout():  # §3.3.3, Worked Example 3.2
    a = math.tanh(0.1)
    assert a == approx(0.099668, abs=1e-6)
    zbar = a * (1 - a * a)
    assert (zbar, 2 * zbar) == approx((0.098678, 0.197356), abs=1e-6)
    a_, b_ = 0.5, -1.0
    assert 2 * a_ + b_ + (1 - math.tanh(a_) ** 2) == approx(0.786448, abs=1e-6)


def test_u03_two_layer_network():  # Worked Example 3.3
    x = np.array([1.0, 2.0])
    W1, b1 = np.array([[0.5, -0.25], [0.3, 0.1]]), np.array([0.1, -0.2])
    w2, b2 = np.array([1.0, -2.0]), 0.05
    h = np.tanh(W1 @ x + b1)
    yhat = w2 @ h + b2
    assert yhat == approx(-0.432957, abs=1e-6)
    assert 0.5 * (yhat - 1) ** 2 == approx(1.026683, abs=1e-6)
    zbar = (yhat - 1) * w2 * (1 - h ** 2)
    assert zbar == approx([-1.418723, 2.622704], abs=1e-6)
    assert np.outer(zbar, x)[0, 1] == approx(-2.837445, abs=1e-6)


def test_u03_softmax_cross_entropy():  # Worked Example 3.4
    s = np.array([2.0, 1.0, 0.1])
    p = np.exp(s) / np.exp(s).sum()
    assert p == approx([0.659001, 0.242433, 0.098566], abs=1e-6)
    assert -math.log(p[0]) == approx(0.417030, abs=1e-6)


def test_u03_conditioning_and_momentum():  # Worked Example 3.5, §3.5.4, §3.5.7
    k = 10.0
    assert (k - 1) / (k + 1) == approx(0.818182, abs=1e-6)
    assert math.log(1e-3) / math.log(9 / 11) == approx(34.42, abs=0.01)
    r = (math.sqrt(k) - 1) / (math.sqrt(k) + 1)
    assert (r, r ** 2, 4 / (math.sqrt(k) + 1) ** 2) == approx((0.519494, 0.269874, 0.230886), abs=1e-6)


def test_u03_finite_difference_optimum():  # Worked Example 3.6
    u = 2.0 ** -53
    h = (3 * u * math.sin(1) / math.cos(1)) ** (1 / 3)
    assert h == approx(8.03e-6, rel=1e-2)
    assert h * h * math.cos(1) / 6 + u * math.sin(1) / h == approx(1.74e-11, rel=1e-2)


def test_u03_adam_first_step():  # Worked Example 3.7
    g = np.array([0.001, -50.0, 3.0])
    assert g / (np.abs(g) + 1e-8) == approx([0.99999, -1.0, 1.0], abs=1e-5)


# ----------------------------------------------------------------------------- Unit 04
def test_u04_soft_lookup():  # §4.2.7, Worked Example 4.1
    for beta, out in [(1, 4.03), (2, 5.36), (4, 6.70), (8, 6.99)]:
        w = math.exp(beta) / (math.exp(beta) + 3)
        assert w * 7 + (1 - w) / 3 * 4 == approx(out, abs=0.006)


def test_u04_best_affine_outputs_mean():  # Worked Example 4.2
    # minimize ||F - y||^2 subject to F_A + F_D - F_B - F_C = 0
    y, c = np.array([3.0, 7.0, 7.0, 3.0]), np.array([1.0, -1.0, -1.0, 1.0])
    F = y - c * (c @ y) / (c @ c)
    assert F == approx([5, 5, 5, 5])


def test_u04_attention_forward_backward():  # Worked Examples 4.3–4.4
    Q = np.array([[1.0, 0], [0, 1], [1, 1]])
    K = np.array([[1.0, 0], [1, 1], [0, 1]])
    V = np.array([[1.0, 2], [0, 1], [3, 0]])
    S = Q @ K.T / math.sqrt(2)
    S = np.where(np.tril(np.ones((3, 3))) > 0, S, -np.inf)
    A = np.exp(S - S.max(1, keepdims=True))
    A /= A.sum(1, keepdims=True)
    assert A[2] == approx([0.2483, 0.5035, 0.2483], abs=1e-4)
    assert (A @ V)[2] == approx([0.9930, 1.0000], abs=1e-4)
    dY = np.array([[1.0, 0], [0, 1], [1, -1]])
    dA = dY @ V.T
    dS = A * (dA - (dA * A).sum(1, keepdims=True))
    assert dS[2] == approx([-0.2465, -0.5000, 0.7465], abs=1e-4)
    assert (dS @ K / math.sqrt(2))[2] == approx([-0.5279, 0.1743], abs=1e-4)
    assert (dS.T @ Q / math.sqrt(2))[0] == approx([-0.1743, -0.0179], abs=1e-4)


def test_u04_rope_relative():  # Worked Example 4.5
    R = lambda a: np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])
    q, k, th = np.array([1.0, 0]), np.array([0.6, 0.8]), 0.5
    for (m, n), want in [((3, 1), 0.9974), ((5, 3), 0.9974), ((10, 8), 0.9974), ((1, 3), -0.3490)]:
        assert (R(m * th) @ q) @ (R(n * th) @ k) == approx(want, abs=1e-4)


def test_u04_gpt2_counts_flops_cache():  # Worked Examples 4.6–4.8
    d, L, V, T = 768, 12, 50257, 1024
    assert 4 * d * d == 2_359_296
    assert 12 * d * d + 13 * d == 7_087_872
    assert L * (12 * d * d + 13 * d) + V * d + T * d + 2 * d == 124_439_808
    assert (24 * T * d * d, 4 * T * T * d) == approx((1.45e10, 3.22e9), rel=5e-3)
    assert 12 * T * T * 2 == 25_165_824
    assert 2 * L * T * d * 2 == 37_748_736


def test_u04_online_softmax():  # Worked Example 4.9, §4.8.1
    s, v = np.array([1.0, 3.0, 2.0]), np.array([10.0, 20.0, 30.0])
    direct = (np.exp(s) / np.exp(s).sum()) @ v
    m, l, o = 3.0, math.exp(-2) + 1, 10 * math.exp(-2) + 20
    m2 = max(m, 2.0)
    l, o = math.exp(m - m2) * l + math.exp(2 - m2), math.exp(m - m2) * o + 30 * math.exp(2 - m2)
    assert direct == approx(21.547, abs=1e-3) and o / l == approx(direct, rel=1e-12)


def test_u04_gqa_cache_sizes():  # §4.8.2
    per_token = lambda heads: 2 * 80 * heads * 128 * 2
    assert [per_token(h) for h in (64, 8, 1)] == [2_621_440, 327_680, 40_960]


# ----------------------------------------------------------------------------- Unit 05
def test_u05_ema_table_and_tradeoffs():  # §5.2.3–5.2.6
    h, out = 20.0, []
    for x in [20.0, 20.4, 19.8, 25.0, 20.1, 20.0]:
        h = 0.8 * h + 0.2 * x
        out.append(h)
    assert out == approx([20.0, 20.08, 20.024, 21.0192, 20.83536, 20.668288])
    assert 0.9 ** 20 == approx(0.1216, abs=1e-4)
    for a, hl, t95, vr in [(0.5, 1.0, 4.32, 0.3333), (0.9, 6.58, 28.43, 0.0526), (0.99, 68.97, 298.07, 0.00503)]:
        assert math.log(0.5) / math.log(a) == approx(hl, abs=0.01)
        assert math.log(0.05) / math.log(a) == approx(t95, abs=0.01)
        assert (1 - a) / (1 + a) == approx(vr, abs=1e-4)


def test_u05_discretizations():  # §5.2.7–5.2.8, §5.3.9
    assert [math.exp(-r) for r in (0.1, 1.0, 2.5)] == approx([0.905, 0.368, 0.082], abs=1e-3)
    tau, dt = 2.0, 0.5
    A, B = -1 / tau, 1 / tau
    assert (math.exp(dt * A), (math.exp(dt * A) - 1) / A * B) == approx((math.exp(-dt / tau), 1 - math.exp(-dt / tau)))


def test_u05_filter_design_and_detector():  # Worked Examples 5.1–5.2
    assert 15 / 17 == approx(0.8824, abs=1e-4)
    assert math.exp(math.log(0.05) / 60) == approx(0.9513, abs=1e-4)
    assert -1 / math.log(0.9) == approx(9.49, abs=0.01)
    y = [0.95 ** t - 0.5 ** t for t in range(1, 6)]
    assert y == approx([0.45, 0.6525, 0.732375, 0.752006, 0.742531], abs=1e-6)
    assert int(np.argmax(y)) + 1 == 4


def test_u05_kernel_by_diagonalization():  # Worked Example 5.3
    A, B, C = np.array([[0.5, 0.5], [0, 0.8]]), np.array([0, 1.0]), np.array([1.0, 0])
    for j in range(8):
        assert C @ np.linalg.matrix_power(A, j) @ B == approx(5 / 3 * (0.8 ** j - 0.5 ** j))


def test_u05_rotation_and_nonnormal():  # Worked Examples 5.4–5.5
    th = math.pi / 6
    assert [0.9 ** t * math.cos(t * th) for t in (1, 2, 5, 6, 12)] == approx([0.779, 0.405, -0.511, -0.531, 0.282], abs=1e-3)
    A10 = np.linalg.matrix_power(np.array([[0.9, 5.0], [0, 0.9]]), 10)
    assert A10[0, 1] == approx(19.371, abs=1e-3) and np.linalg.norm(A10, 2) == approx(19.38, abs=0.01)


def test_u05_bptt_by_hand():  # Worked Example 5.6
    w, xs, hs = 0.5, [1.0, -0.5, 0.25], [0.0]
    for x in xs:
        hs.append(math.tanh(w * hs[-1] + x))
    assert hs[1:] == approx([0.761594, -0.118642, 0.188401], abs=1e-6)
    hbar, wbar, abars = hs[3] - 0.5, 0.0, []
    for t in (3, 2, 1):
        abar = hbar * (1 - hs[t] ** 2)
        abars.append(abar)
        wbar += abar * hs[t - 1]
        hbar = abar * w
    assert wbar == approx(-0.077177, abs=1e-6)
    assert abars == approx([-0.300538, -0.148154, -0.031110], abs=1e-6)


def test_u05_cross_talk_and_scan():  # Worked Examples 5.7–5.8
    S = np.array([[2.0, 0], [0, 3.0]]) + np.outer([1.0, 1.0], np.array([1.0, 1.0]) / math.sqrt(2))
    assert S @ np.array([1.0, 0]) == approx([2.7071, 0.7071], abs=1e-4)
    r3 = S @ (np.array([1.0, 1.0]) / math.sqrt(2))
    assert np.linalg.norm(r3 - 1) / math.sqrt(2) == approx(1.80, abs=0.01)
    comb = lambda p, q: (q[0] * p[0], q[0] * p[1] + q[1])
    pairs = [(0.5, 1.0), (0.8, 2.0), (0.9, -1.0), (0.5, 4.0)]
    assert comb(comb(pairs[0], pairs[1]), comb(pairs[2], pairs[3])) == approx((0.18, 4.76))


def test_u05_filter_gain_and_kalman():  # §5.5.1, §5.6.2
    a = 0.9
    gain = lambda w: (1 - a) / abs(1 - a * np.exp(-1j * w))
    assert [gain(w) for w in (0, math.pi / 8, math.pi / 2, math.pi)] == approx([1.0, 0.261, 0.0743, 0.0526], abs=1e-3)
    q, r = 0.01, 1.0
    P = (q + math.sqrt(q * q + 4 * q * r)) / 2
    assert 1 - P / (P + r) == approx(0.905, abs=1e-3)


def test_u05_counting_bound_example():  # §5.5.2
    assert 1000 * math.log2(50257) == approx(15617, abs=1)
