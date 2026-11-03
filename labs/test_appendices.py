"""Recompute every number quoted in the worked examples and selected answers of Appendices A-H."""
import cmath
import math

import numpy as np
import pytest


# ----------------------------------------------------------------------------- A · Linear algebra
def test_a_projection_and_least_squares():
    y, a = np.array([1.0, 2, 2]), np.array([1.0, 1, 0])
    proj = a * (a @ y) / (a @ a)
    assert proj == pytest.approx([1.5, 1.5, 0]) and (y - proj) @ a == pytest.approx(0)
    assert np.linalg.norm(y - proj) == pytest.approx(math.sqrt(4.5)) == pytest.approx(2.121, abs=5e-4)
    X, yy = np.array([[1.0, 0], [1, 1], [1, 2]]), np.array([1.0, 2, 2])
    assert X.T @ X == pytest.approx(np.array([[3, 3], [3, 5]])) and X.T @ yy == pytest.approx([5, 6])
    beta = np.linalg.solve(X.T @ X, X.T @ yy)
    assert beta == pytest.approx([7 / 6, 0.5])
    resid = yy - X @ beta
    assert resid == pytest.approx([-1 / 6, 1 / 3, -1 / 6]) and resid.sum() == pytest.approx(0)


def test_a_eigen_svd_condition():
    A = np.array([[2.0, 1], [1, 2]])
    assert np.linalg.eigvalsh(A) == pytest.approx([1, 3]) and np.linalg.cond(A) == pytest.approx(3)
    M = np.array([[3.0, 0], [4, 5]])
    s = np.linalg.svd(M, compute_uv=False)
    assert s == pytest.approx([math.sqrt(45), math.sqrt(5)]) and s == pytest.approx([6.708, 2.236], abs=5e-4)
    assert sorted(np.linalg.eigvals(M).real) == pytest.approx([3, 5]) and s[0] / s[1] == pytest.approx(3)
    U, S, Vt = np.linalg.svd(M)
    M1 = S[0] * np.outer(U[:, 0], Vt[0])
    assert np.linalg.norm(M - M1, 2) == pytest.approx(s[1]) and np.linalg.norm(M - M1) == pytest.approx(s[1])


def test_a_transient_growth_and_answers():
    A = np.array([[0.5, 2], [0, 0.5]])
    norms = [np.linalg.norm(np.linalg.matrix_power(A, t), 2) for t in range(7)]
    assert norms == pytest.approx([1, 2.118, 2.031, 1.510, 1.004, 0.627, 0.376], abs=5e-4)
    for t in range(1, 6):
        assert np.linalg.matrix_power(A, t)[0, 1] == pytest.approx(4 * t * 0.5 ** t)
    assert np.linalg.cond(np.diag([4.0, 1])) == 4
    D = np.diag([0.9, 0.5])
    assert [np.linalg.norm(np.linalg.matrix_power(D, t), 2) for t in range(5)] == pytest.approx([0.9 ** t for t in range(5)])


# ----------------------------------------------------------------------------- B · Calculus
def test_b_gradient_softmax_and_lse():
    f = lambda x, y: x * x * y + math.sin(y)  # noqa: E731
    h = 1e-6
    gx = (f(1 + h, 0) - f(1 - h, 0)) / (2 * h)
    gy = (f(1, h) - f(1, -h)) / (2 * h)
    assert (gx, gy) == pytest.approx((0, 2), abs=1e-8)
    z = np.array([1.0, 0, 0])
    p = np.exp(z) / np.exp(z).sum()
    assert p == pytest.approx([0.5761, 0.2119, 0.2119], abs=1e-4)
    J = np.diag(p) - np.outer(p, p)
    assert J == pytest.approx(np.array([[0.2442, -0.1221, -0.1221], [-0.1221, 0.1670, -0.0449],
                                        [-0.1221, -0.0449, 0.1670]]), abs=1e-4)
    assert J.sum(1) == pytest.approx(0, abs=1e-12) and np.all(np.linalg.eigvalsh(J) > -1e-12)
    assert p - np.eye(3)[1] == pytest.approx([0.5761, -0.7881, 0.2119], abs=1e-4)
    zz = np.array([1000.0, 999.0])
    m = zz.max()
    assert m + math.log(np.exp(zz - m).sum()) == pytest.approx(1000.3133, abs=1e-4)


def test_b_finite_difference_errors():
    errs = {h: (math.exp(h) - math.exp(-h)) / (2 * h) - 1 for h in (1e-1, 1e-2, 1e-3)}
    for h, e in errs.items():
        assert e == pytest.approx(h * h / 6, rel=1e-3)
    assert errs[1e-1] == pytest.approx(1.67e-3, abs=1e-5)
    assert abs((math.exp(1e-8) - math.exp(-1e-8)) / 2e-8 - 1) > 1e-10      # roundoff dominates
    assert (1.1 ** 3 - 1) / 0.1 - 3 == pytest.approx(0.31) and (1.1 ** 3 - 0.9 ** 3) / 0.2 - 3 == pytest.approx(0.01)


# ----------------------------------------------------------------------------- C · Probability
def test_c_bayes_gaussian_chebyshev():
    post = 0.01 * 0.9 / (0.01 * 0.9 + 0.99 * 0.05)
    assert post == pytest.approx(0.154, abs=5e-4) and 0.01 * 0.9 + 0.99 * 0.05 == pytest.approx(0.0585)
    assert 0.1 * 0.9 / (0.1 * 0.9 + 0.9 * 0.05) == pytest.approx(0.667, abs=5e-4)
    assert 1 - 0.8 ** 2 == pytest.approx(0.36)
    assert math.erfc(2 / math.sqrt(2)) == pytest.approx(0.0455, abs=5e-5)
    # Gumbel-max reproduces softmax
    rng = np.random.default_rng(0)
    z = np.array([1.0, 0.0, -0.5])
    draws = np.argmax(z + rng.gumbel(size=(200_000, 3)), axis=1)
    assert np.bincount(draws, minlength=3) / 200_000 == pytest.approx(np.exp(z) / np.exp(z).sum(), abs=5e-3)


# ----------------------------------------------------------------------------- D · Information theory
def kl(p, q):
    p, q = np.asarray(p, float), np.asarray(q, float)
    return float(np.sum(p * np.log(p / q)))


def test_d_entropy_kl_mi():
    q = np.array([0.5, 0.3, 0.15, 0.05])
    H = -np.sum(q * np.log(q))
    assert H == pytest.approx(1.142, abs=5e-4) and H / math.log(2) == pytest.approx(1.648, abs=5e-4)
    assert kl([0.5, 0.5], [0.9, 0.1]) == pytest.approx(0.511, abs=5e-4)
    assert kl([0.9, 0.1], [0.5, 0.5]) == pytest.approx(0.368, abs=5e-4)
    pxy = np.array([[0.4, 0.1], [0.1, 0.4]])
    mi = kl(pxy.ravel(), np.outer(pxy.sum(1), pxy.sum(0)).ravel())
    h02 = -0.2 * math.log(0.2) - 0.8 * math.log(0.8)
    assert h02 == pytest.approx(0.500, abs=5e-4) and mi == pytest.approx(math.log(2) - h02) == pytest.approx(0.193, abs=5e-4)
    assert 0.5 * math.log(2 * math.pi * math.e) == pytest.approx(1.419, abs=5e-4)
    p3 = np.array([0.7, 0.2, 0.1])
    assert math.log(3) - (-np.sum(p3 * np.log(p3))) == pytest.approx(0.297, abs=5e-4)


def test_d_gaussian_kl_and_coding():
    klg = lambda m1, s1, m2, s2: math.log(s2 / s1) + (s1 ** 2 + (m1 - m2) ** 2) / (2 * s2 ** 2) - 0.5  # noqa: E731
    assert klg(0, 1, 1, 2) == pytest.approx(0.443, abs=5e-4) and klg(1, 2, 0, 1) == pytest.approx(1.307, abs=5e-4)
    x = np.linspace(-15, 15, 300_001)
    dx = x[1] - x[0]
    N = lambda m, s: np.exp(-(x - m) ** 2 / (2 * s * s)) / (math.sqrt(2 * math.pi) * s)  # noqa: E731
    assert np.sum(N(0, 1) * np.log(N(0, 1) / N(1, 2))) * dx == pytest.approx(klg(0, 1, 1, 2), abs=1e-6)
    p = np.array([0.5, 0.25, 0.125, 0.125])
    lengths = np.array([1, 2, 3, 3])
    assert np.sum(2.0 ** -lengths) == 1 and p @ lengths == 1.75 == pytest.approx(-np.sum(p * np.log2(p)))


def test_d_forward_versus_reverse_kl():
    x = np.linspace(-12, 12, 240_001)
    dx = x[1] - x[0]
    N = lambda m, s: np.exp(-(x - m) ** 2 / (2 * s * s)) / (math.sqrt(2 * math.pi) * s)  # noqa: E731
    p = 0.5 * N(-2, 0.5) + 0.5 * N(2, 0.5)
    s_mm = math.sqrt(0.25 + 4)
    assert s_mm == pytest.approx(2.06, abs=5e-3)
    fwd = lambda q: float(np.sum(p * np.log(p / q)) * dx)  # noqa: E731
    rev = lambda q: float(np.sum(q * np.log(q / p)) * dx)  # noqa: E731
    assert fwd(N(0, s_mm)) == pytest.approx(0.72, abs=5e-3) and fwd(N(2, 0.5)) == pytest.approx(15.3, abs=0.05)
    assert rev(N(0, s_mm)) == pytest.approx(2.10, abs=5e-3) and rev(N(2, 0.5)) == pytest.approx(math.log(2), abs=1e-3)


# ----------------------------------------------------------------------------- E · Statistics
def test_e_intervals_and_power():
    z, n, p = 1.96, 10, 0.7
    wald = (p - z * math.sqrt(p * (1 - p) / n), p + z * math.sqrt(p * (1 - p) / n))
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    assert wald == pytest.approx((0.416, 0.984), abs=5e-4) and (c - h, c + h) == pytest.approx((0.397, 0.892), abs=5e-4)
    assert 1.96 * math.sqrt(0.81 * 0.19 / 500) == pytest.approx(0.034, abs=5e-4)
    assert math.ceil(1.96 ** 2 * 0.81 * 0.19 / 0.01 ** 2) == 5913
    assert (2.8 * 0.5 / 0.1) ** 2 == pytest.approx(196)
    assert (2.8 * 0.09 / 0.05) ** 2 == pytest.approx(25.4, abs=0.05)
    assert math.ceil(math.log(2 / 0.05) / (2 * 0.01 ** 2)) == 18445
    assert (1 + 7) / (2 + 10) == pytest.approx(0.667, abs=5e-4)


def test_e_multiple_comparisons():
    ps = [0.001, 0.008, 0.039, 0.041, 0.042, 0.060, 0.074, 0.205, 0.212, 0.216]

    def bh(alpha):
        ks = [i + 1 for i, q in enumerate(sorted(ps)) if q <= (i + 1) * alpha / len(ps)]
        return max(ks) if ks else 0

    assert sum(q <= 0.05 / 10 for q in ps) == 1 and bh(0.05) == 2 and bh(0.10) == 6
    assert sum(q <= 0.05 for q in ps) == 5


def test_e_bernstein_and_maximum():
    val = 2 * math.exp(-1000 * 1e-4 / (2 * 0.0196 + 2 * 0.98 * 0.01 / 3))
    assert val == pytest.approx(0.22, abs=5e-3)
    x = np.linspace(-10, 10, 200_001)
    dx = x[1] - x[0]
    phi = np.exp(-x * x / 2) / math.sqrt(2 * math.pi)
    Phi = np.cumsum(phi) * dx
    emax = np.sum(x * 100 * phi * Phi ** 99) * dx
    assert emax == pytest.approx(2.51, abs=5e-3) and math.sqrt(2 * math.log(100)) == pytest.approx(3.03, abs=5e-3)


# ----------------------------------------------------------------------------- F · Optimization
def test_f_rates_newton_maxent():
    L, mu = 10.0, 1.0
    rates = [1 - mu / L, (L - mu) / (L + mu), (math.sqrt(L / mu) - 1) / (math.sqrt(L / mu) + 1)]
    assert rates == pytest.approx([0.9, 0.818, 0.519], abs=5e-4)
    steps = [math.log(1e-6) / math.log(r) for r in rates]
    assert steps == pytest.approx([131, 69, 21], abs=1)
    k = 1e4
    big = [math.log(1e-6) / math.log(r) for r in (1 - 1 / k, (k - 1) / (k + 1), (math.sqrt(k) - 1) / (math.sqrt(k) + 1))]
    assert big[0] == pytest.approx(138_000, rel=0.01) and big[1] == pytest.approx(69_000, rel=0.01) and big[2] == pytest.approx(690, rel=0.01)
    x, errs = 0.5, []
    for _ in range(5):
        x = 2 * x - x * x
        errs.append(1 - x)
    assert errs == pytest.approx([0.25, 0.0625, 0.00390625, 1.52587890625e-05, 2.3283064365386963e-10])
    lo, hi = -5.0, 5.0
    for _ in range(200):
        lam = (lo + hi) / 2
        w = np.exp(lam * np.arange(1, 7))
        lo, hi = (lam, hi) if (np.arange(1, 7) @ w) / w.sum() < 4.5 else (lo, lam)
    w = np.exp(lam * np.arange(1, 7))
    p = w / w.sum()
    assert lam == pytest.approx(0.371, abs=5e-4)
    assert p == pytest.approx([0.054, 0.079, 0.114, 0.165, 0.240, 0.348], abs=1e-3)
    assert -np.sum(p * np.log(p)) == pytest.approx(1.614, abs=5e-4)


def test_f_gibbs_variational_principle():
    rng = np.random.default_rng(3)
    r, tau = rng.normal(size=5), 0.7
    pstar = np.exp(r / tau) / np.exp(r / tau).sum()
    value = lambda p: p @ r - tau * np.sum(p * np.log(p))  # noqa: E731
    assert value(pstar) == pytest.approx(tau * math.log(np.exp(r / tau).sum()))
    for _ in range(100):
        assert value(rng.dirichlet(np.ones(5))) <= value(pstar) + 1e-12


# ----------------------------------------------------------------------------- G · Linear systems
def test_g_ema_and_realizations():
    a = 0.9
    assert math.log(2) / -math.log(a) == pytest.approx(6.58, abs=5e-3)
    assert math.log(2) / -math.log(0.99) == pytest.approx(69.0, abs=0.05)
    G = lambda w: abs((1 - a) / (1 - a * cmath.exp(-1j * w)))  # noqa: E731
    assert G(0) == pytest.approx(1) and G(math.pi / 2) == pytest.approx(0.074, abs=5e-4)
    assert G(math.pi) == pytest.approx(0.053, abs=5e-4)
    A, B, C = np.diag([0.5, 1.2]), np.array([1.0, 0]), np.array([1.0, 0])
    ctrb = np.column_stack([B, A @ B])
    obsv = np.vstack([C, C @ A])
    assert np.linalg.matrix_rank(ctrb) == 1 and np.linalg.matrix_rank(obsv) == 1
    assert [C @ np.linalg.matrix_power(A, k) @ B for k in range(6)] == pytest.approx([0.5 ** k for k in range(6)])
    A2, B2, C2 = np.array([[0.9, 1], [0, 0.9]]), np.array([0.0, 1]), np.array([1.0, 0])
    assert np.linalg.matrix_rank(np.column_stack([B2, A2 @ B2])) == 2
    assert np.linalg.matrix_rank(np.vstack([C2, C2 @ A2])) == 2
    g = [C2 @ np.linalg.matrix_power(A2, k) @ B2 for k in range(1, 15)]
    assert g == pytest.approx([k * 0.9 ** (k - 1) for k in range(1, 15)])
    assert int(np.argmax(g)) + 1 in (9, 10)


def test_g_discretization_and_fft():
    lam = -10.0
    assert 2 / abs(lam) == pytest.approx(0.2) and 1 + 0.3 * lam == pytest.approx(-2)
    assert math.exp(0.1 * lam) == pytest.approx(0.368, abs=5e-4) and 1 + 0.1 * lam == pytest.approx(0)
    rng = np.random.default_rng(0)
    T = 16
    x, k = rng.normal(size=T), rng.normal(size=T)
    linear = np.convolve(x, k)[:T]
    n = 2 * T - 1
    circ = np.real(np.fft.ifft(np.fft.fft(x, n) * np.fft.fft(k, n)))[:T]
    assert circ == pytest.approx(linear)
    wrapped = np.real(np.fft.ifft(np.fft.fft(x) * np.fft.fft(k)))
    assert not np.allclose(wrapped, linear)
    M1, M2 = np.array([[0.0, 2], [0, 0]]), np.array([[0.0, 0], [2, 0]])
    assert max(abs(np.linalg.eigvals(M1))) < 1 and max(abs(np.linalg.eigvals(M2))) < 1
    assert max(abs(np.linalg.eigvals(M1 @ M2))) == pytest.approx(4)


# ----------------------------------------------------------------------------- H · Markov decision processes
def test_h_chains_and_bellman():
    P = np.array([[0.9, 0.1], [0.3, 0.7]])
    w, v = np.linalg.eig(P.T)
    pi = np.real(v[:, np.argmax(np.real(w))])
    assert pi / pi.sum() == pytest.approx([0.75, 0.25]) and sorted(np.real(w)) == pytest.approx([0.6, 1])
    mu, traj = np.array([1.0, 0]), []
    for _ in range(4):
        traj.append(mu[0])
        mu = mu @ P
    assert traj == pytest.approx([1, 0.9, 0.84, 0.804])
    P3 = np.array([[0.5, 0.5, 0], [0.25, 0.5, 0.25], [0, 0.5, 0.5]])
    assert np.array([0.25, 0.5, 0.25]) @ P3 == pytest.approx([0.25, 0.5, 0.25])
    Pp, r = np.array([[0.9, 0.1], [0.5, 0.5]]), np.array([0.0, 1])
    V = np.linalg.solve(np.eye(2) - 0.9 * Pp, r)
    assert V == pytest.approx([45 / 32, 95 / 32])
    Vk, k = np.zeros(2), 0
    while np.abs(Vk - V).max() >= 1e-3:
        Vk, k = r + 0.9 * Pp @ Vk, k + 1
    assert k == 71 and math.ceil(math.log(1e-3 / V.max()) / math.log(0.9)) == 76
    assert np.linalg.solve(np.eye(2) - 0.5 * Pp, r) == pytest.approx([0.125, 1.375])
    assert np.linalg.solve(np.eye(2) - 0.99 * Pp, r) == pytest.approx([16.39, 18.05], abs=5e-3)


def test_h_occupancy_form_matches_episodic_form():
    """Example H.4 in the occupancy form: only the t = 1 state carries a decision."""
    gamma, theta = 0.5, 0.0
    p1 = 1 / (1 + math.exp(-theta))
    d1 = (1 - gamma) * gamma          # discounted occupancy of the decision state (visited at t = 1)
    Q = {1: 1.0, 0: 0.0}
    grad = d1 / (1 - gamma) * sum(prob * Q[a] * (a - p1) for a, prob in ((1, p1), (0, 1 - p1)))
    assert grad == pytest.approx(0.125)
    b_star = (0.8 * 1 * 0.2 ** 2) / (0.8 * 0.2 ** 2 + 0.2 * 0.8 ** 2)
    assert b_star == pytest.approx(0.2)
