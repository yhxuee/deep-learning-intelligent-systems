"""Recompute every exact number quoted in the worked examples and selected answers of Units 09-12."""
import hashlib
import json
import math
import pathlib
import sys

import numpy as np
import pytest

SPINE = pathlib.Path(__file__).resolve().parent.parent / "spine"
sys.path.insert(0, str(SPINE))
import stage10_eval as S10  # noqa: E402
import stage12_capstone as S12  # noqa: E402

sigmoid = lambda x: 1.0 / (1.0 + math.exp(-x))  # noqa: E731
softplus_neg = lambda u: math.log1p(math.exp(-u))  # DPO / Bradley-Terry loss  # noqa: E731


# ----------------------------------------------------------------------------- Unit 09
def test_we_9_1_9_2_bandit_gradient_and_baseline():
    pi, r = np.full(3, 1 / 3), np.array([1.0, 0.0, 0.0])
    grad = pi * (r - pi @ r)
    assert grad == pytest.approx([0.2222, -0.1111, -0.1111], abs=1e-4)
    for b, var in ((0.0, 0.0988), (1 / 3, 0.0247)):
        g1 = np.array([(r[a] - b) * ((a == 0) - pi[0]) for a in range(3)])
        assert pi @ g1 == pytest.approx(0.2222, abs=1e-4)
        assert pi @ g1 ** 2 - (pi @ g1) ** 2 == pytest.approx(var, abs=1e-4)


def test_we_9_3_tilted_optimum():
    ref, r = np.array([0.2, 0.5, 0.3]), np.array([1.0, 0.0, 0.0])
    expected = {0.1: ((0.9998, 0.0001, 0.0001), 0.839), 0.5: ((0.6488, 0.2195, 0.1317), 0.412),
                2.0: ((0.2919, 0.4426, 0.2655), 0.244)}
    for beta, (pstar, value) in expected.items():
        w = ref * np.exp(r / beta)
        assert w / w.sum() == pytest.approx(pstar, abs=1e-4)
        assert beta * math.log(w.sum()) == pytest.approx(value, abs=1e-3)
        # the optimum attains its value: E[r] - beta KL(pi* || ref) = beta ln Z
        p = w / w.sum()
        assert p @ r - beta * np.sum(p * np.log(p / ref)) == pytest.approx(beta * math.log(w.sum()))
    assert math.exp(2) == pytest.approx(7.39, abs=5e-3)


def test_we_9_4_bellman():
    assert 0.9 * 1.0 == 0.9
    q = 0.5
    assert 0.9 * (1 - q) / (1 - 0.9 * q) == pytest.approx(0.8182, abs=1e-4)


def test_we_9_5_preferences_and_dpo():
    assert sigmoid(1) == pytest.approx(0.7311, abs=1e-4)
    assert -math.log(sigmoid(1)) == pytest.approx(0.3133, abs=1e-4)
    assert softplus_neg(0) == pytest.approx(0.6931, abs=1e-4)
    u = 0.1 * (math.log(2) + math.log(2))
    assert u == pytest.approx(0.1386, abs=1e-4) and softplus_neg(u) == pytest.approx(0.6262, abs=1e-4)


def test_we_9_8_kl_reference_point():
    assert 0.8 * math.log(0.8 / 0.2) + 0.2 * math.log(0.2 / 0.8) == pytest.approx(0.832, abs=1e-3)
    assert 1.34 / math.log(2) == pytest.approx(1.93, abs=5e-3)


def test_unit_09_selected_answers():
    z, r = np.array([1.0, 0, 0]), np.array([0.0, 1, 0])
    pi = np.exp(z) / np.exp(z).sum()
    assert pi == pytest.approx([0.576, 0.212, 0.212], abs=1e-3)
    J = pi @ r
    assert J == pytest.approx(0.212, abs=1e-3)
    assert pi * (r - J) == pytest.approx([-0.122, 0.167, -0.045], abs=1e-3)
    w = np.array([0.5 * math.e, 0.5])
    assert w / w.sum() == pytest.approx([0.731, 0.269], abs=1e-3)
    assert sigmoid(0.5) == pytest.approx(0.6225, abs=1e-4) and -math.log(sigmoid(0.5)) == pytest.approx(0.474, abs=1e-3)
    assert 1 / (1 - 0.9) == pytest.approx(10) and 1 / (1 - 0.99) == pytest.approx(100)
    assert softplus_neg(0.1) == pytest.approx(0.644, abs=1e-3) and softplus_neg(1.0) == pytest.approx(0.313, abs=1e-3)
    assert 1 - (1 - 1e-6) ** 64 == pytest.approx(6.4e-5, rel=1e-4)
    # Problems 9.2 and 9.6-9.8.
    assert math.exp(4) / (math.exp(4) + 1) == pytest.approx(0.982, abs=1e-3)
    group = np.array([1.0, 0.0, 0.0, 0.0])
    mean_adv = group - group.mean()
    loo_adv = group - (group.sum() - group) / 3
    assert mean_adv == pytest.approx(0.75 * loo_adv)
    q = 1 - (1 - 1e-4) ** 64
    assert q == pytest.approx(0.00638, abs=1e-5)
    assert math.ceil(math.log(0.5) / (64 * math.log(1 - 1e-4))) == 109
    assert 2 / 3 == pytest.approx((4 / 27) / (2 / 9))


# ----------------------------------------------------------------------------- Unit 10
def test_we_10_1_ece_by_hand():
    conf = np.array([0.9] * 5 + [0.6] * 5)
    correct = np.array([1, 1, 1, 0, 0] + [1, 1, 1, 0, 0])
    assert S10.ece(conf, correct, n_bins=10)[0] == pytest.approx(0.15)
    assert correct.mean() == pytest.approx(0.6) and conf.mean() == pytest.approx(0.75)


def test_we_10_3_paired_vs_unpaired_widths():
    assert 0.327 - 0.240 == pytest.approx(0.087)
    marginal_half_04 = (0.345 - 0.327) / 2
    marginal_half_07 = (0.240 - 0.222) / 2
    rough_unpaired_half = math.hypot(marginal_half_04, marginal_half_07)
    rough_unpaired_full = 2 * rough_unpaired_half
    paired_full = 0.110 - 0.101
    assert rough_unpaired_half == pytest.approx(0.0127, abs=5e-5)
    assert rough_unpaired_full == pytest.approx(0.0255, abs=5e-5)
    assert paired_full / rough_unpaired_full == pytest.approx(0.354, abs=5e-4)


def test_we_10_8_prompts_needed():
    v = 0.82 * 0.18 + 0.78 * 0.22
    assert v == pytest.approx(0.3192)
    assert v * (1.96 / 0.04) ** 2 == pytest.approx(766.4, abs=0.1)
    assert (1.96 + 0.84) ** 2 * v / 0.04**2 == pytest.approx(1564.08)
    se_400 = math.sqrt(v / 400)
    assert se_400 == pytest.approx(0.02825, abs=5e-6)
    assert (0.04 - 1.96 * se_400, 0.04 + 1.96 * se_400) == pytest.approx((-0.015, 0.095), abs=5e-4)


def test_unit_10_selected_answers():
    conf = np.array([0.8] * 10 + [0.4] * 10)
    correct = np.array([1] * 7 + [0] * 3 + [1] * 6 + [0] * 4)
    assert correct.mean() == 0.65 and conf.mean() == pytest.approx(0.6)
    assert S10.ece(conf, correct, n_bins=10)[0] == pytest.approx(0.15)
    assert 1.96 * math.sqrt(0.81 * 0.19 / 500) == pytest.approx(0.034, abs=5e-4)
    assert math.ceil((1.96 / 0.01) ** 2 * 0.81 * 0.19) == 5913
    assert math.ceil((1.96 / 0.01) ** 2 * 0.25) == 9604
    assert 1 - 0.95 ** 20 == pytest.approx(0.642, abs=5e-4)
    assert math.ceil(100 * 0.9) == 90
    assert 0.2 * 0.30 + 0.8 * 0.05 == pytest.approx(0.10) and 0.1 * 0.30 + 0.9 * 0.05 == pytest.approx(0.075)
    assert 0.8 / 0.9 == pytest.approx(0.889, abs=5e-4)
    counts = np.array([118, 140, 149, 70, 90, 57, 54, 53, 4389])
    mean_conf = np.array([0.165, 0.256, 0.344, 0.444, 0.547, 0.651, 0.753, 0.854, 0.998])
    bin_acc = np.array([0.110, 0.207, 0.282, 0.357, 0.511, 0.561, 0.574, 0.792, 0.998])
    rounded_table_ece = np.sum(counts * np.abs(mean_conf - bin_acc)) / counts.sum()
    assert counts.sum() == 5120
    assert rounded_table_ece == pytest.approx(0.00977, abs=5e-6)
    assert counts[-1] / counts.sum() == pytest.approx(0.8572, abs=5e-5)
    brier = 0.5 * (0.7 * 0.2**2 + 0.3 * 0.8**2) + 0.5 * (0.6 * 0.6**2 + 0.4 * 0.4**2)
    calibration = 0.5 * (0.8 - 0.7) ** 2 + 0.5 * (0.4 - 0.6) ** 2
    resolution = 0.5 * (0.7 - 0.65) ** 2 + 0.5 * (0.6 - 0.65) ** 2
    uncertainty = 0.65 * 0.35
    assert brier == pytest.approx(0.25)
    assert (calibration, resolution, uncertainty) == pytest.approx((0.025, 0.0025, 0.2275))
    assert calibration - resolution + uncertainty == pytest.approx(brier)


# ----------------------------------------------------------------------------- Unit 11
def test_unit_11_attention_and_logits():
    s = 60 / math.sqrt(18)
    assert s == pytest.approx(14.14, abs=5e-3)
    weight = lambda n: math.exp(s) / (math.exp(s) + n - 1)  # noqa: E731
    assert 1 - weight(4) == pytest.approx(2.2e-6, rel=0.05)
    assert 1 - weight(16) == pytest.approx(1.1e-5, rel=0.05)
    assert 1 - weight(1000) == pytest.approx(7.2e-4, rel=0.05)
    assert 20 / math.sqrt(18) == pytest.approx(4.7, abs=0.05)
    s20 = 20 / math.sqrt(18)
    assert math.exp(s20) / (math.exp(s20) + 15) == pytest.approx(0.881, abs=1e-3)
    assert math.exp(10) / (math.exp(10) + math.e + 7) == pytest.approx(0.9996, abs=1e-4)
    assert math.exp(3) / (math.exp(3) + math.e + 7) == pytest.approx(0.674, abs=1e-3)
    assert (8.77 - 4.4) / (8.77 - 0.0004) == pytest.approx(0.50, abs=5e-3)


# ----------------------------------------------------------------------------- Unit 12
WITH = [1.71, 1.81, 1.8241, 1.8778, 1.8224]
WITHOUT = [1.8735, 1.7713, 1.748, 1.823, 1.7782]
WITH_R = [1.8544, 1.7983, 1.8633, 1.7433, 1.7428]
WITHOUT_R = [1.8032, 1.917, 1.8133, 1.8172, 1.8301]


def test_we_12_1_registration_digest():
    rec = json.loads((SPINE / "checkpoints" / "stage12_preregistration.json").read_text())
    digest = hashlib.sha256(json.dumps(rec["record"], sort_keys=True).encode()).hexdigest()
    assert digest == rec["sha256"] == "e66b6c1a6bd4f8f6f0eb25b4c01b17eb0880752a6cd9e7e8ad0d99517972491f"
    edited = dict(rec["record"], threshold=0.03)
    assert hashlib.sha256(json.dumps(edited, sort_keys=True).encode()).hexdigest() != digest


def test_we_12_2_to_12_4_seed_variance_pairing_power():
    w, wo = np.array(WITH), np.array(WITHOUT)
    d = wo - w
    assert (w.mean(), wo.mean(), d.mean()) == pytest.approx((1.8089, 1.7988, -0.0101), abs=1e-4)
    assert (w.std(ddof=1), wo.std(ddof=1)) == pytest.approx((0.061, 0.050), abs=1e-3)
    assert d.std(ddof=1) == pytest.approx(0.098, abs=1e-3)
    assert np.delete(d, 0).mean() == pytest.approx(-0.053, abs=1e-3)
    # pairing did not pair: negative correlation, paired sd above the independent sd
    assert math.sqrt(w.var(ddof=1) + wo.var(ddof=1)) == pytest.approx(0.079, abs=1e-3)
    assert np.corrcoef(w, wo)[0, 1] == pytest.approx(-0.56, abs=0.01)
    wr, wor = np.array(WITH_R), np.array(WITHOUT_R)
    assert np.corrcoef(wr, wor)[0, 1] == pytest.approx(-0.18, abs=0.01)
    # power
    assert (2.8 * d.std(ddof=1) / 0.02) ** 2 == pytest.approx(189, abs=1)
    pooled = np.r_[d, wor - wr]
    assert pooled.std(ddof=1) == pytest.approx(0.088, abs=1e-3)
    assert (2.8 * pooled.std(ddof=1) / 0.02) ** 2 == pytest.approx(152, abs=1)
    assert d.std(ddof=1) / math.sqrt(5) == pytest.approx(0.044, abs=1e-3)
    assert 0.02 * math.sqrt(5) / 2.8 == pytest.approx(0.016, abs=5e-4)
    assert (2.8 * 0.09 / 0.05) ** 2 == pytest.approx(25, abs=0.5)


def test_we_12_5_forking_paths():
    assert [round(1 - 0.95 ** k, 3) for k in (1, 5, 10, 20)] == [0.05, 0.226, 0.401, 0.642]
    d, dr = np.array(WITHOUT) - np.array(WITH), np.array(WITHOUT_R) - np.array(WITH_R)
    assert dr.mean() == pytest.approx(0.036, abs=1e-3)
    assert np.r_[d, dr].mean() == pytest.approx(0.013, abs=1e-3)


def test_we_12_7_capstone_decisions():
    assert S12.decide((-0.0101, -0.0634, 0.0794), 0.02) == "inconclusive"
    assert S12.decide((0.0358, -0.0257, 0.0972), 0.02) == "inconclusive"
    assert -0.0634 <= -0.0101 <= 0.0794 and -0.0257 <= -0.0101 <= 0.0972
    assert 0.0358 - (-0.0101) == pytest.approx(0.046, abs=1e-3)
    assert math.hypot(0.044, 0.036) == pytest.approx(0.057, abs=1e-3)


def test_unit_12_ppv_and_selected_answers():
    ppv = lambda prior, power, alpha=0.05: power * prior / (power * prior + alpha * (1 - prior))  # noqa: E731
    assert ppv(0.1, 0.8) == pytest.approx(0.64)
    assert ppv(0.1, 0.2) == pytest.approx(0.31, abs=5e-3)
    assert ppv(0.5, 0.8) == pytest.approx(0.94, abs=5e-3)
    assert (2.8 * 0.05 / 0.02) ** 2 == pytest.approx(49) and math.ceil((2.8 * 0.05 / 0.05) ** 2) == 8
    assert 1 - 0.95 ** 10 == pytest.approx(0.401, abs=5e-4)
    assert [S12.decide((0, lo, hi), 0.02) for lo, hi in ((0.03, 0.08), (-0.01, 0.015), (0.01, 0.05))] == [
        "needed", "not needed", "inconclusive"]
    assert 0.08 / 0.03 == pytest.approx(2.7, abs=0.05)
    assert 0.07 / math.hypot(0.03, 0.02) == pytest.approx(1.94, abs=5e-3)
    assert [round(0.06 * math.sqrt(2 * (1 - rho)), 3) for rho in (0.8, 0.0, -0.5)] == [0.038, 0.085, 0.104]
