"""Regression tests for the corrections made after the scientific audit of v2.0 (findings K04-K14).

The Unit 09 findings (K01-K03) are tested next to the code they concern, in spine/test_spine.py.
Each test here pins a counterexample or a number that distinguishes the corrected statement from the
one it replaced, so that the error cannot silently return.
"""
import math

import numpy as np
import pytest


# ----------------------------------------------------------------------------- K04 (Unit 05 §5.5.1)
def test_k04_transfer_poles_can_miss_eigenvalues_of_a_nonminimal_realization():
    A = np.diag([0.5, 1.2])
    B = np.array([[1.0], [0.0]])
    C = np.array([[1.0, 0.0]])
    for z in (2.0, 3.0 + 1.0j, -1.5, 0.9j + 1.1):
        G = (C @ np.linalg.inv(np.eye(2) - A / z) @ B)[0, 0]
        assert G == pytest.approx(1 / (1 - 0.5 / z))           # only the pole 0.5 is visible
    # impulse response from h0 = 0 decays like 0.5^t: input-output stable
    h, ys = np.zeros(2), []
    for t in range(60):
        h = A @ h + B[:, 0] * (t == 0)
        ys.append((C @ h)[0])
    assert abs(ys[-1]) < 1e-15 and max(abs(y) for y in ys) == 1.0
    # yet a nonzero initial state in the hidden mode grows like 1.2^t: internally unstable
    h = np.array([0.0, 1e-9])
    for _ in range(200):
        h = A @ h
    assert h[1] > 1e6 and max(abs(np.linalg.eigvals(A))) == pytest.approx(1.2)


# ----------------------------------------------------------------------------- K05 (Unit 05 §5.5.4)
def _multiplies_dense(N, T):
    return T * N * N            # h <- A h, one dense N x N product per step


def _multiplies_diagonal(N, T):
    return T * N                # h <- a * h, elementwise


@pytest.mark.parametrize("N", [4, 16, 64])
def test_k05_dense_recurrence_is_quadratic_in_state_size(N):
    T = 50
    rng = np.random.default_rng(N)
    A = rng.normal(size=(N, N)) / (2 * math.sqrt(N))
    a = np.diag(A).copy()
    count = {"dense": 0, "diag": 0}
    h, g = rng.normal(size=N), rng.normal(size=N)
    for _ in range(T):
        h = A @ h
        count["dense"] += A.size
        g = a * g
        count["diag"] += a.size
    assert count["dense"] == _multiplies_dense(N, T) and count["diag"] == _multiplies_diagonal(N, T)
    assert count["dense"] / count["diag"] == N


# ----------------------------------------------------------------------------- K07 (Unit 10 §10.3.4)
def test_k07_top_label_and_multiclass_brier_differ():
    def multiclass(p, y):
        return float(np.sum((np.asarray(p) - np.eye(len(p))[y]) ** 2))

    def top_label(p, y):
        return float((max(p) - (int(np.argmax(p)) == y)) ** 2)

    p, q = [0.6, 0.2, 0.2], [0.6, 0.4, 0.0]
    assert top_label(p, 0) == pytest.approx(0.16) and top_label(q, 0) == pytest.approx(0.16)
    assert multiclass(p, 0) == pytest.approx(0.24) and multiclass(q, 0) == pytest.approx(0.32)
    # the multiclass score is proper for the whole vector: reporting the true distribution minimizes it
    true = np.array([0.5, 0.3, 0.2])
    expected = lambda f: sum(true[y] * multiclass(f, y) for y in range(3))  # noqa: E731
    rng = np.random.default_rng(0)
    for _ in range(200):
        other = rng.dirichlet(np.ones(3))
        assert expected(true) <= expected(other) + 1e-12


# ----------------------------------------------------------------------------- K09 (Unit 01 §1.5.1)
def test_k09_two_sided_uniform_bound_radius():
    K, delta, n = 1000, 0.05, 10_000
    eps = math.sqrt((math.log(K) + math.log(2 / delta)) / (2 * n))
    assert eps == pytest.approx(math.sqrt(math.log(2 * K / delta) / (2 * n)))
    # per-member two-sided Hoeffding tail at this radius is exactly delta / K
    assert 2 * math.exp(-2 * n * eps ** 2) == pytest.approx(delta / K)


# ----------------------------------------------------------------------------- K12 (Unit 10 Example 10.8)
def test_k12_precision_and_power_sample_sizes_differ():
    v = 0.82 * 0.18 + 0.78 * 0.22
    precision = v * (1.96 / 0.04) ** 2
    power = (1.96 + 0.84) ** 2 * v / 0.04 ** 2
    assert precision == pytest.approx(766.4, abs=0.1)
    assert power == pytest.approx(1564.1, abs=0.1)
    # with the precision-sized sample, a true difference of 0.04 is detected only about half the time
    se = math.sqrt(v / 767)
    phi = lambda x: 0.5 * (1 + math.erf(x / math.sqrt(2)))  # noqa: E731
    assert 1 - phi(1.96 - 0.04 / se) == pytest.approx(0.5, abs=0.01)
    se = math.sqrt(v / 1565)
    assert 1 - phi(1.96 - 0.04 / se) == pytest.approx(0.80, abs=0.01)


# ----------------------------------------------------------------------------- K13 (Unit 01 §1.3.3)
def test_k13_bigram_loss_is_normalized_by_transitions():
    text = "ab_ab_ba"
    pairs = list(zip(text, text[1:]))
    assert len(pairs) == len(text) - 1 == 7
    counts = {}
    for a, b in pairs:
        counts.setdefault(a, {}).setdefault(b, 0)
        counts[a][b] += 1
    total = -sum(math.log(counts[a][b] / sum(counts[a].values())) for a, b in pairs)
    assert total / (len(text) - 1) == pytest.approx(0.4708, abs=1e-4)


# ----------------------------------------------------------------------------- K14 (Unit 10 §10.5.3)
def conformal_threshold(scores, alpha):
    n = len(scores)
    k = math.ceil((n + 1) * (1 - alpha))
    return math.inf if k > n else float(np.sort(scores)[k - 1])


def test_k14_conformal_threshold_is_infinite_when_the_index_exceeds_n():
    scores = np.arange(1, 10) / 10.0          # n = 9
    assert conformal_threshold(scores, 0.05) == math.inf    # ceil(10 * 0.95) = 10 > 9
    assert conformal_threshold(scores, 0.2) == pytest.approx(0.8)   # ceil(10 * 0.8) = 8


def test_k14_split_conformal_marginal_coverage():
    rng = np.random.default_rng(0)
    n, alpha, trials = 19, 0.1, 20_000
    hits = 0
    for _ in range(trials):
        s = rng.exponential(size=n + 1)       # exchangeable scores, last one is the test point
        hits += s[-1] <= conformal_threshold(s[:-1], alpha)
    # exact coverage for continuous scores: ceil((n+1)(1-alpha)) / (n+1) = 18/20
    assert hits / trials == pytest.approx(18 / 20, abs=0.01)


# ----------------------------------------------------------------------------- Unit 10 §10.5.1 and §10.5.4
def test_unit10_brier_honesty_identity_and_bh_example():
    p = 0.3
    for q in np.linspace(0, 1, 101):
        expected_brier = p * (q - 1) ** 2 + (1 - p) * q**2
        assert expected_brier == pytest.approx((q - p) ** 2 + p * (1 - p))
        assert expected_brier >= p * (1 - p) - 1e-12

    p_values = np.array([0.003, 0.012, 0.021, 0.080, 0.200])
    thresholds = np.arange(1, 6) * 0.05 / 5
    passing = np.flatnonzero(p_values <= thresholds)
    assert passing[-1] + 1 == 3
    assert np.sum(p_values <= 0.05 / 5) == 1
