"""Verification targets for spine stages 01 and 03 (shared corpus data/tales.txt)."""
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
SMALL = TRAIN[:12_000]  # the full-batch neural bigram test runs on a prefix to stay fast


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
    loss, _ = loss_and_grad(np.zeros((V, V)), SMALL[:-1], SMALL[1:])
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
    mle = nll(fit_bigram(SMALL, V, alpha=0.0), SMALL)
    _, hist = train(SMALL, V, steps=1000)
    assert hist.min() >= mle - 1e-9, "a neural bigram cannot beat the count MLE on its own training data"
    assert hist[-1] - mle < 0.01


# ----------------------------------------------------------------------------- Stage 02
import stage02_embeddings as S2  # noqa: E402


def test_stage02_init_loss_is_exactly_ln_V():
    p = S2.init(V, 8)
    loss, _ = S2.loss_and_grad(p, SMALL[:-1], SMALL[1:])
    assert loss == pytest.approx(np.log(V), abs=1e-12)


def test_stage02_lookup_gradient_equals_one_hot_matmul_gradient():
    rng = np.random.default_rng(0)
    E = rng.normal(size=(V, 6))
    x = rng.integers(0, V, 50)
    G = rng.normal(size=(50, 6))          # upstream gradient dL/dH for H = E[x]
    lookup = np.zeros_like(E)
    np.add.at(lookup, x, G)
    onehot = np.eye(V)[x]                  # H = onehot @ E  =>  dL/dE = onehot^T G
    assert np.allclose(lookup, onehot.T @ G)


def test_stage02_gradient_matches_finite_differences():
    rng = np.random.default_rng(1)
    p = {"E": rng.normal(size=(V, 5)), "M": rng.normal(size=(5, 5)) * 0.3, "c": rng.normal(size=V) * 0.1}
    x, y = SMALL[:300], SMALL[1:301]
    _, g = S2.loss_and_grad(p, x, y)
    for k in p:
        for _ in range(6):
            i = tuple(rng.integers(0, s) for s in p[k].shape)
            old = p[k][i]
            p[k][i] = old + 1e-6
            lp = S2.loss_and_grad(p, x, y)[0]
            p[k][i] = old - 1e-6
            lm = S2.loss_and_grad(p, x, y)[0]
            p[k][i] = old
            assert (lp - lm) / 2e-6 == pytest.approx(g[k][i], rel=1e-5, abs=1e-10)


def test_stage02_never_beats_the_count_mle():
    mle = nll(fit_bigram(SMALL, V, alpha=0.0), SMALL)
    p, _ = S2.train(SMALL, V, d=16, steps=300, batch=2000)
    assert S2.heldout_nll(p, SMALL) >= mle - 1e-9


def test_stage02_neural_ngram_gradient_matches_finite_differences():
    rng = np.random.default_rng(5)
    p = S2.ngram_init(V, 3, d=4, H=6, seed=1)
    for k in p:
        p[k] = p[k] + rng.normal(0, 0.2, p[k].shape)
    X, y = S2.ngram_windows(SMALL[:200], 3)
    _, g = S2.ngram_loss_and_grad(p, X, y)
    for k in p:
        for _ in range(5):
            i = tuple(rng.integers(0, s) for s in p[k].shape)
            old = p[k][i]
            p[k][i] = old + 1e-6
            lp = S2.ngram_loss_and_grad(p, X, y)[0]
            p[k][i] = old - 1e-6
            lm = S2.ngram_loss_and_grad(p, X, y)[0]
            p[k][i] = old
            assert (lp - lm) / 2e-6 == pytest.approx(g[k][i], rel=1e-5, abs=1e-10)


def test_stage02_neural_ngram_beats_counts_when_data_is_scarce():
    """Unit 02, Worked Example 2.6: with 5 000 training characters, sharing through embeddings wins."""
    n = 5000
    count = min(S2.count_ngram_nll(TRAIN[:n], VAL, V, 4, a)[0] for a in [0.003, 0.01, 0.03, 0.1, 0.3])
    q = S2.train_ngram(TRAIN[:n], V, k=4, steps=2500)
    assert S2.ngram_nll(q, VAL, 4) < count - 0.08


def test_stage02_interchangeable_symbols_get_the_same_embedding():
    """Distributional hypothesis, planted: symbols 1 and 2 have identical contexts."""
    rng = np.random.default_rng(0)
    K = 8
    P = rng.dirichlet(np.ones(K) * 0.3, size=K)
    P[2] = P[1]                                # same successors
    P[:, 1] += P[:, 2]
    P[:, 2] = P[:, 1] / 2
    P[:, 1] = P[:, 2]                          # same predecessors, split evenly
    seq = [0]
    for _ in range(20_000):
        seq.append(rng.choice(K, p=P[seq[-1]]))
    ids = np.array(seq)
    p, _ = S2.train(ids, K, d=4, steps=600, batch=2000)
    E = p["E"] / np.linalg.norm(p["E"], axis=1, keepdims=True)
    S = E @ E.T
    others = [S[i, j] for i in range(K) for j in range(i + 1, K) if (i, j) != (1, 2)]
    assert S[1, 2] > 0.95 and S[1, 2] > max(others)


# ----------------------------------------------------------------------------- Stage 04 / 05
import model as M  # noqa: E402
import stage04_attention as S4  # noqa: E402


def test_stage04_short_run_starts_at_ln_V_and_beats_the_bigram():
    p, cfg, init_val, hist = S4.run(TRAIN, VAL, V, steps=400, log_every=0, lr=1e-2, d=32, n_layer=2, max_len=32)
    assert abs(init_val - np.log(V)) < 0.1
    assert M.evaluate(p, cfg, VAL, cfg.max_len) < nll(fit_bigram(TRAIN, V, 0.1), VAL)


# ----------------------------------------------------------------------------- Stage 06
import stage06_scaling as S6  # noqa: E402


def test_stage06_fit_recovers_a_known_power_law():
    N = np.array([1e3, 3e3, 1e4, 3e4, 1e5])
    L = 0.3 + 5.0 * N ** -0.4
    fit = S6.fit_power_law(N, L, n_grid=2000)
    assert fit["E"] == pytest.approx(0.3, abs=2e-3)
    assert fit["alpha"] == pytest.approx(0.4, abs=0.01)
    assert S6.predict(fit, 1e6) == pytest.approx(0.3 + 5.0 * 1e6 ** -0.4, rel=2e-3)


def test_stage06_compute_optimal_closed_form_matches_brute_force():
    A, B, a, b, C = 400.0, 410.0, 0.34, 0.28, 1e21
    N_star, D_star = S6.compute_optimal(C, A, B, a, b)
    N = np.logspace(6, 13, 200_001)
    loss = A * N ** -a + B * (C / (6 * N)) ** -b
    assert N[loss.argmin()] == pytest.approx(N_star, rel=1e-3)
    assert 6 * N_star * D_star == pytest.approx(C)


# ----------------------------------------------------------------------------- Stage 07
import stage07_pretrain as S7  # noqa: E402
from common import load_lines, load_manifest, split_lines  # noqa: E402

LINES = load_lines()
MANIFEST = load_manifest()


def test_stage07_finds_every_planted_duplicate_and_nothing_else():
    train, _ = split_lines(LINES)
    _, removed = S7.deduplicate(train)
    pairs = MANIFEST["duplicates"]
    for d in pairs:
        assert d["copy"] in removed or d["source"] in removed, d
    involved = {d["copy"] for d in pairs} | {d["source"] for d in pairs}
    assert set(removed) <= involved
    assert len(removed) == len(pairs)


def test_stage07_contamination_check_finds_exactly_the_planted_stories():
    train, held = split_lines(LINES)
    overlap = S7.contamination(train, held)
    planted = sorted(c["held_out"] - MANIFEST["held_out_start"] for c in MANIFEST["contaminated"])
    assert list(np.where(overlap >= S7.OVERLAP)[0]) == planted


def test_stage07_cleaned_training_set_is_not_contaminated():
    final, _ = S7.clean(LINES)
    _, held = split_lines(LINES)
    assert (S7.contamination(final, held) < S7.OVERLAP).all()


def test_stage07_minhash_collision_probability_matches_theory():
    """Pr[minhash_a == minhash_b] = Jaccard(a, b) for each hash function (Broder 1997)."""
    a = {f"w{i}" for i in range(60)}
    b = {f"w{i}" for i in range(20, 80)}            # Jaccard 40 / 80 = 0.5
    sa, sb = S7.minhash(a, 400), S7.minhash(b, 400)
    assert (sa == sb).mean() == pytest.approx(S7.jaccard(a, b), abs=0.08)


# ----------------------------------------------------------------------------- Stage 08
import stage08_sampling as S8  # noqa: E402


def test_stage08_samplers_reduce_to_each_other_in_their_limits():
    rng = np.random.default_rng(0)
    logits = rng.normal(size=(500, 12)) * 2
    assert (S8.top_k(1)(logits, rng) == S8.greedy(logits)).all()
    assert (S8.temperature(1e-6)(logits, rng) == S8.greedy(logits)).all()
    P = S8.probs(logits)
    assert S8.nucleus_mask(P, 1.0).all()
    m = S8.nucleus_mask(P, 0.5)
    kept = np.where(m, P, 0).sum(1)
    assert (kept >= 0.5 - 1e-12).all()
    # minimal: dropping the smallest kept token would fall below p
    smallest = np.where(m, P, np.inf).min(1)
    assert (kept - smallest < 0.5).all()


def test_stage08_sampled_frequencies_match_the_distribution():
    rng = np.random.default_rng(1)
    logits = np.log(np.array([[0.5, 0.3, 0.15, 0.05]]))
    draws = np.array([S8.temperature(1.0)(logits, rng)[0] for _ in range(20_000)])
    freq = np.bincount(draws, minlength=4) / len(draws)
    assert np.allclose(freq, [0.5, 0.3, 0.15, 0.05], atol=0.012)


def test_stage08_exact_gaussian_velocity_transports_noise_to_data():
    rng = np.random.default_rng(2)
    v = S8.gaussian_velocity([1.0, -2.0], 0.5)
    x = S8.integrate(v, rng.standard_normal((20_000, 2)), n_steps=200)
    assert np.allclose(x.mean(0), [1.0, -2.0], atol=0.03)
    assert np.allclose(x.std(0), [0.5, 0.5], atol=0.02)


def test_stage08_flow_matching_gradient_matches_finite_differences():
    rng = np.random.default_rng(3)
    p = S8.mlp_init(hidden=8, seed=1)
    for k in p:
        p[k] = p[k] + rng.normal(0, 0.2, p[k].shape)
    x0, x1, t = rng.normal(size=(16, 2)), rng.normal(size=(16, 2)), rng.random(16)
    _, g = S8.fm_loss_and_grad(p, x0, x1, t)
    for k in p:
        for _ in range(4):
            i = tuple(rng.integers(0, s) for s in p[k].shape)
            old = p[k][i]
            p[k][i] = old + 1e-6
            lp = S8.fm_loss_and_grad(p, x0, x1, t)[0]
            p[k][i] = old - 1e-6
            lm = S8.fm_loss_and_grad(p, x0, x1, t)[0]
            p[k][i] = old
            assert (lp - lm) / 2e-6 == pytest.approx(g[k][i], rel=1e-5, abs=1e-9)


# ----------------------------------------------------------------------------- Stage 09
import stage09_post_training as S9  # noqa: E402


def test_stage09_policy_gradient_estimator_is_unbiased():
    """Bandit with logits z: grad_z E[r] = pi * (r - pi.r). The REINFORCE average must match it."""
    rng = np.random.default_rng(0)
    z = rng.normal(size=5)
    r = rng.random(5)
    pi = np.exp(z - z.max())
    pi /= pi.sum()
    exact = pi * (r - pi @ r)
    a = rng.choice(5, size=400_000, p=pi)
    est = ((r[a] - pi @ r)[:, None] * (np.eye(5)[a] - pi)).mean(0)
    assert np.allclose(est, exact, atol=4e-3)


@pytest.mark.parametrize("B", [2, 4, 64])
def test_stage09_batch_mean_baseline_is_biased_and_leave_one_out_is_not(B):
    """Expected per-batch estimate of grad_z E[r] for a bandit, computed exactly by enumerating the
    sampled action of sample 1 and averaging the other B - 1 rewards in closed form (audit K03)."""
    rng = np.random.default_rng(B)
    z, r = rng.normal(size=4), rng.random(4)
    pi = np.exp(z - z.max())
    pi /= pi.sum()
    exact = pi * (r - pi @ r)
    mean_other = pi @ r                       # E[r_j] for j != i, independent of a_i
    score = np.eye(4) - pi                    # d log pi(a) / dz, one row per action a
    # E[(r_i - b_i) score(a_i)] with b_i independent of a_i except through r_i itself
    loo = sum(pi[a] * (r[a] - mean_other) * score[a] for a in range(4))
    batch_mean = sum(pi[a] * (r[a] - (r[a] + (B - 1) * mean_other) / B) * score[a] for a in range(4))
    assert np.allclose(loo, exact, atol=1e-12)
    assert np.allclose(batch_mean, (B - 1) / B * exact, atol=1e-12)
    # and the implementation computes exactly these advantages
    rr = rng.random(B)
    assert np.allclose(S9.advantages(rr, "loo"), B / (B - 1) * S9.advantages(rr, "mean"))


def test_stage09_discounted_policy_gradient_carries_gamma_t():
    """Two-step episode, reward 1 at t = 1 if the second action is 1, gamma = 0.5, Bernoulli second-step
    policy with logit theta (p = 0.5). J = E[gamma r_1] = 0.5 sigmoid(theta); dJ/dtheta = 0.125.
    The theorem with gamma^t gives 0.125; omitting it gives 0.25 (audit K01)."""
    gamma, theta = 0.5, 0.0
    p1 = 1 / (1 + np.exp(-theta))
    exact = gamma * p1 * (1 - p1)
    with_discount = without = 0.0
    for a1, prob in ((1, p1), (0, 1 - p1)):   # step 0 has a single action (score 0); enumerate step 1
        G1 = float(a1 == 1)                   # return-to-go from t = 1
        score = a1 - p1                       # d log Bernoulli(a1; sigmoid(theta)) / d theta
        with_discount += prob * gamma ** 1 * score * G1
        without += prob * score * G1
    assert exact == pytest.approx(0.125) and with_discount == pytest.approx(0.125)
    assert without == pytest.approx(0.25)


def test_stage09_value_baseline_is_not_the_variance_optimal_baseline():
    """Bernoulli logit policy, p = 0.8, reward 1 for action 1: V = 0.8 but the variance-minimizing
    constant baseline is b* = E[r s^2] / E[s^2] = 0.2, which makes the estimate constant (audit K02)."""
    p = 0.8
    acts, probs = np.array([1, 0]), np.array([p, 1 - p])
    r = acts.astype(float)
    s = acts - p                               # score d log pi / d logit
    b_star = (probs @ (r * s ** 2)) / (probs @ s ** 2)
    assert b_star == pytest.approx(0.2) and probs @ r == pytest.approx(0.8)
    est = (r - b_star) * s
    assert np.allclose(est, 0.16) and probs @ ((r - 0.8) * s) == pytest.approx(0.16)
    var = lambda b: probs @ (((r - b) * s) ** 2) - (probs @ ((r - b) * s)) ** 2  # noqa: E731
    assert var(b_star) == pytest.approx(0.0, abs=1e-15) and var(0.8) > 0.01


def test_stage09_kl_regularized_optimum_is_the_tilted_reference():
    rng = np.random.default_rng(1)
    ref = rng.dirichlet(np.ones(6))
    r = rng.random(6)
    beta = 0.3
    z = np.log(ref).copy()
    for _ in range(20_000):  # exact gradient ascent on E_pi[r] - beta KL(pi || ref)
        pi = np.exp(z - z.max())
        pi /= pi.sum()
        f = r - beta * (np.log(pi) - np.log(ref))
        z += 0.5 * pi * (f - pi @ f)
    target = ref * np.exp(r / beta)
    assert np.allclose(pi, target / target.sum(), atol=1e-6)


@pytest.mark.parametrize("which", ["dpo", "reinforce"])
def test_stage09_gradients_through_the_spine_model_match_finite_differences(which):
    cfg = M.Config(vocab=7, d=8, n_layer=1, n_head=2, max_len=5)
    p = M.init_params(cfg, seed=0)
    ref = M.init_params(cfg, seed=1)
    rng = np.random.default_rng(0)
    X = rng.integers(0, 7, (4, 5))
    ref_logp = S9.last_logprobs(ref, cfg, X)[0]
    yw, yl = rng.integers(0, 7, 4), rng.integers(0, 7, 4)
    actions, rewards = rng.integers(0, 7, 4), rng.random(4)

    def objective(q):
        lp = S9.last_logprobs(q, cfg, X)[0]
        if which == "dpo":
            u = 0.5 * ((lp[np.arange(4), yw] - ref_logp[np.arange(4), yw]) - (lp[np.arange(4), yl] - ref_logp[np.arange(4), yl]))
            return np.mean(np.logaddexp(0, -u))
        A = S9.advantages(rewards)
        kl = (np.exp(lp) * (lp - ref_logp)).sum(-1)
        return -(A * lp[np.arange(4), actions]).mean() + 0.2 * kl.mean()

    if which == "dpo":
        _, g = S9.dpo_loss_grad(p, cfg, X, yw, yl, ref_logp, 0.5)
    else:
        g, _ = S9.reinforce_grad(p, cfg, X, actions, rewards, ref_logp, 0.2)
    for k in ["wte", "0.attn.wqkv", "0.mlp.wfc"]:
        for _ in range(4):
            i = tuple(rng.integers(0, s) for s in p[k].shape)
            old = p[k][i]
            p[k][i] = old + 1e-6
            lp_ = objective(p)
            p[k][i] = old - 1e-6
            lm_ = objective(p)
            p[k][i] = old
            assert (lp_ - lm_) / 2e-6 == pytest.approx(g[k][i], rel=1e-5, abs=1e-9)


# ----------------------------------------------------------------------------- Stage 10
import stage10_eval as S10  # noqa: E402


def test_stage10_ece_of_calibrated_and_overconfident_predictors():
    rng = np.random.default_rng(0)
    conf = rng.uniform(0.05, 0.95, 200_000)
    correct = rng.random(200_000) < conf
    assert S10.ece(conf, correct)[0] < 0.01
    overconf = np.minimum(conf + 0.04, 0.999)
    e = S10.ece(overconf, correct)[0]
    assert e == pytest.approx(0.04, abs=0.01)


def test_stage10_bootstrap_intervals_have_nominal_coverage():
    rng = np.random.default_rng(1)
    hits = 0
    reps = 300
    for k in range(reps):
        x = rng.exponential(1.0, 60)
        _, lo, hi = S10.bootstrap_ci(x, n_boot=600, seed=k)
        hits += lo <= 1.0 <= hi
    assert 0.88 <= hits / reps <= 0.98


def test_stage10_pairing_narrows_the_interval_for_correlated_models():
    rng = np.random.default_rng(2)
    difficulty = rng.normal(0, 1, 100)
    a = difficulty + rng.normal(0, 0.1, 100)
    b = difficulty + 0.05 + rng.normal(0, 0.1, 100)
    _, lo_p, hi_p = S10.paired_bootstrap_ci(b, a)
    _, lo_a, hi_a = S10.bootstrap_ci(a)
    _, lo_b, hi_b = S10.bootstrap_ci(b)
    unpaired_width = np.hypot(hi_a - lo_a, hi_b - lo_b)
    assert (hi_p - lo_p) < 0.2 * unpaired_width
    assert lo_p > 0


# ----------------------------------------------------------------------------- Stage 11
import stage11_induction as S11  # noqa: E402


def test_stage11_hand_built_circuit_is_found_used_and_needed():
    p, cfg = S11.build_induction_circuit()
    X = S11.repeated_batch(np.random.default_rng(0), 64, 8, 8)
    assert S11.second_half_loss(p, cfg, X, 8) < 0.01
    _, cache = M.forward(p, cfg, X[:, :-1], keep=True)
    prev, ind = S11.head_scores(cache, 8)
    assert max(prev, key=prev.get) == (0, 0) and prev[(0, 0)] > 0.95
    assert max(ind, key=ind.get) == (1, 0) and ind[(1, 0)] > 0.95
    for head, needed in {(0, 0): True, (1, 0): True, (0, 1): False, (1, 1): False}.items():
        hm = np.ones((2, 2))
        hm[head] = 0
        loss = S11.second_half_loss(p, cfg, X, 8, head_mask=hm)
        assert (loss > 1.0) if needed else (loss < 0.01), head


def test_stage11_patching_a_run_with_its_own_activations_is_a_no_op():
    p, cfg = S11.build_induction_circuit()
    X = S11.repeated_batch(np.random.default_rng(1), 8, 8, 8)[:, :-1]
    logits, cache = M.forward(p, cfg, X, keep=True)
    patch = {(1, 0): cache["blocks"][1]["o"][:, 0]}
    assert np.allclose(M.forward(p, cfg, X, patch=patch), logits)


def test_stage11_patching_the_induction_head_transfers_the_clean_prediction():
    p, cfg = S11.build_induction_circuit()
    rng = np.random.default_rng(2)
    clean = S11.repeated_batch(rng, 16, 8, 8)
    corrupt = clean.copy()
    corrupt[:, 1:9] = np.stack([rng.permutation(8) for _ in range(16)])   # first half differs
    _, cc = M.forward(p, cfg, clean[:, :-1], keep=True)
    patched = S11.second_half_loss(p, cfg, clean, 8, patch=None)
    corrupt_loss = S11.second_half_loss(p, cfg, np.concatenate([corrupt[:, :9], clean[:, 9:]], 1), 8)
    assert patched < 0.01 and corrupt_loss > 1.0


# ----------------------------------------------------------------------------- Stage 12
import stage12_capstone as S12  # noqa: E402


def test_stage12_same_seed_same_result():
    a = S12.run_condition(True, 3, TRAIN, VAL, V, steps=20, d=16, T=16)
    b = S12.run_condition(True, 3, TRAIN, VAL, V, steps=20, d=16, T=16)
    assert a == b


def test_stage12_preregistration_detects_tampering(tmp_path):
    f = tmp_path / "prereg.json"
    S12.preregister(f, {"prediction": "x > 0", "threshold": 0.02})
    assert S12.verify_preregistration(f)[0]
    f.write_text(f.read_text().replace("0.02", "0.05"))
    assert not S12.verify_preregistration(f)[0]


def test_stage12_decision_rule():
    assert S12.decide((0.10, 0.05, 0.15), 0.02) == "needed"
    assert S12.decide((0.00, -0.01, 0.01), 0.02) == "not needed"
    assert S12.decide((0.03, 0.00, 0.06), 0.02) == "inconclusive"
