"""Properties the "beyond language" experiments rely on (Experiments 2.7, 5.9, 8.7, 10.7)."""
import pathlib
import sys

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import beyond_language as B  # noqa: E402


def nll(w, X, y):
    p = np.clip(B.predict(w, X), 1e-9, 1 - 1e-9)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def accuracy(w, X, y):
    return float(((B.predict(w, X) > 0.5) == y).mean())


def test_point_clouds_need_an_invariant_representation():
    X, y = B.point_clouds(2000, np.random.default_rng(0))
    Xt, yt = B.point_clouds(5000, np.random.default_rng(100))
    # permuting the points of a cloud does not change its invariant features
    perm = X[:, ::-1]
    assert np.allclose(B.radial_features(X), B.radial_features(perm))
    F, Ft = B.radial_features(X), B.radial_features(Xt)
    f_mean, f_std = F.mean(0), F.std(0)
    F, Ft = (F - f_mean) / f_std, (Ft - f_mean) / f_std
    C, Ct = X.reshape(len(X), -1), Xt.reshape(len(Xt), -1)
    c_mean, c_std = C.mean(0), C.std(0)
    C, Ct = (C - c_mean) / c_std, (Ct - c_mean) / c_std
    assert accuracy(B.logistic_regression(F, y), Ft, yt) > 0.8    # invariant summary generalizes
    assert accuracy(B.logistic_regression(C, y), Ct, yt) < 0.6    # raw coordinates: near chance


def test_sensor_ema_with_kalman_alpha_is_best():
    x, r = B.sensor_stream(5000, np.random.default_rng(1))
    a = B.steady_state_kalman_alpha(0.01, 0.25)
    assert a == pytest.approx(0.819, abs=1e-3)
    mse = {al: np.mean((B.ema(r, al)[100:] - x[100:]) ** 2) for al in (0.5, a, 0.95, 0.99)}
    assert min(mse, key=mse.get) == a
    assert mse[a] < 0.25 / 4                                      # far below the raw sensor noise


def test_double_well_is_symmetric_with_two_modes():
    s = B.double_well_samples(20_000, np.random.default_rng(2))
    assert s.shape == (20_000, 2)
    assert (s[:, 0] > 0).mean() == pytest.approx(0.5, abs=0.02)
    assert abs(s[:, 0]).mean() == pytest.approx(0.93, abs=0.05)  # mass concentrates near x = ±1
    assert (abs(s[:, 0]) < 0.2).mean() < 0.05                     # little mass at the barrier


def test_multisite_artifact_helps_in_training_and_hurts_at_a_new_site():
    Xtr, ytr, site, Xt, yt = B.multisite(3000, np.random.default_rng(3))
    assert np.corrcoef(Xtr[:, 2], ytr)[0, 1] > 0.15 and abs(np.corrcoef(Xt[:, 2], yt)[0, 1]) < 0.05
    w_all, w_sig = B.logistic_regression(Xtr, ytr), B.logistic_regression(Xtr[:, :2], ytr)
    assert nll(w_all, Xtr, ytr) < nll(w_sig, Xtr[:, :2], ytr)     # the shortcut looks better in training
    assert nll(w_all, Xt, yt) > nll(w_sig, Xt[:, :2], yt) + 0.02  # and is worse at the held-out site
