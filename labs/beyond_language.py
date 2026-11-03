"""Synthetic data for the "beyond language" experiments (Experiments 2.7, 5.9, 8.7 and 10.7).

The spine project is a language model; these generators let the same principles be tested on other kinds
of data, to see which conclusions depend on language and which do not. Each generator is small,
deterministic given its seed, and documented with the property its experiment relies on — properties that
labs/test_beyond_language.py checks.
"""
import numpy as np


# ----------------------------------------------------------------------------- Unit 02: sets of points
def point_clouds(n, rng, k=6, noise=0.05):
    """n clouds of k 2-D points: label 0 = points on a circle, label 1 = points on a square's boundary,
    each randomly rotated, scaled and listed in random order. The label depends on the *set*, not on the
    order of its points: a model that reads the concatenated coordinates must learn that invariance from
    data; a model that sums per-point features has it by construction (Unit 02, §2.5.4)."""
    X = np.empty((n, k, 2))
    y = rng.integers(0, 2, n)
    for i in range(n):
        t = rng.uniform(0, 2 * np.pi, k)
        if y[i] == 0:
            pts = np.stack([np.cos(t), np.sin(t)], 1)
        else:
            s = rng.uniform(-1, 1, k)
            side = rng.integers(0, 4, k)
            pts = np.stack([np.where(side < 2, np.where(side == 0, 1.0, -1.0), s),
                            np.where(side >= 2, np.where(side == 2, 1.0, -1.0), s)], 1)
        a = rng.uniform(0, 2 * np.pi)
        R = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
        X[i] = rng.uniform(0.5, 1.5) * pts @ R.T + noise * rng.normal(size=(k, 2))
        X[i] = X[i][rng.permutation(k)]
    return X, y


def radial_features(X):
    """Per-point features (radius, radius^2, radius^4) of rotation-invariant geometry, mean-pooled: a
    permutation- and rotation-invariant summary of each cloud."""
    r = np.linalg.norm(X, axis=-1)
    r = r / r.mean(axis=1, keepdims=True)          # scale-invariant
    return np.stack([r.std(1), (r ** 2).std(1), (r ** 4).mean(1)], 1)


# ----------------------------------------------------------------------------- Unit 05: a noisy sensor
def sensor_stream(T, rng, q=0.01, r=0.25, x0=20.0):
    """A temperature that drifts as a random walk (variance q per step) read by a sensor with noise
    variance r. Returns (true state, readings). The steady-state Kalman filter for this model is an
    exponential moving average whose alpha follows from q and r (Unit 05, §5.6.2)."""
    x = x0 + np.cumsum(np.sqrt(q) * rng.normal(size=T))
    return x, x + np.sqrt(r) * rng.normal(size=T)


def steady_state_kalman_alpha(q, r):
    """EMA weight on the previous estimate equivalent to the steady-state Kalman filter of a random walk:
    the prior variance P solves P = (P r / (P + r)) + q, the gain is K = P / (P + r), and alpha = 1 - K."""
    P = (q + np.sqrt(q * q + 4 * q * r)) / 2
    return 1 - P / (P + r)


def ema(readings, alpha, h0=None):
    h = readings[0] if h0 is None else h0
    out = np.empty_like(readings)
    for t, x in enumerate(readings):
        h = alpha * h + (1 - alpha) * x
        out[t] = h
    return out


# ----------------------------------------------------------------------------- Unit 08: a physical density
def double_well_samples(n, rng, beta=3.0):
    """Exact samples from the Boltzmann density p(x, y) ∝ exp(-beta U) with U = (x^2 - 1)^2 + y^2 / 2, a
    two-mode distribution from statistical physics, by rejection sampling. Used as data for flow matching
    (Unit 08) in place of the synthetic ring: are both modes covered, and with the right weights?"""
    out = []
    while sum(len(o) for o in out) < n:
        x = rng.uniform(-2.2, 2.2, 4 * n)
        y = rng.normal(0, 1 / np.sqrt(beta), 4 * n)            # exact in y
        accept = rng.uniform(size=4 * n) < np.exp(-beta * (x * x - 1) ** 2)
        out.append(np.stack([x[accept], y[accept]], 1))
    return np.concatenate(out)[:n]


# ----------------------------------------------------------------------------- Unit 10: several sites
def multisite(n_per_site, rng, prevalence=(0.1, 0.3, 0.5), test_prevalence=0.2, artifact_strength=2.0):
    """A binary condition observed at several sites. Two features carry real signal; a third, a site
    artefact (think: the scanner model), is correlated with the label across *training* sites only because
    prevalence differs between sites. The held-out site has its own prevalence and an artefact value
    unrelated to the label. Returns (X, y, site) for training sites and (X, y) for the held-out site.

    WARNING: synthetic data for a methods exercise. Real clinical models need prospective, multi-site
    external validation and clinical oversight before any deployment (Unit 10, §10.6.1)."""
    Xs, ys, ss = [], [], []
    for s, prev in enumerate(prevalence):
        y = (rng.uniform(size=n_per_site) < prev).astype(int)
        signal = rng.normal(size=(n_per_site, 2)) + 0.8 * y[:, None]
        artifact = artifact_strength * prev + 0.3 * rng.normal(size=n_per_site)   # site-level, tracks prevalence
        Xs.append(np.column_stack([signal, artifact])), ys.append(y), ss.append(np.full(n_per_site, s))
    y_t = (rng.uniform(size=n_per_site) < test_prevalence).astype(int)
    X_t = np.column_stack([rng.normal(size=(n_per_site, 2)) + 0.8 * y_t[:, None],
                           artifact_strength * 0.5 + 0.3 * rng.normal(size=n_per_site)])
    return np.concatenate(Xs), np.concatenate(ys), np.concatenate(ss), X_t, y_t


def logistic_regression(X, y, steps=3000, lr=0.5, l2=1e-3):
    """Plain gradient descent on the mean cross-entropy, with a bias column added."""
    Xb = np.column_stack([X, np.ones(len(X))])
    w = np.zeros(Xb.shape[1])
    for _ in range(steps):
        p = 1 / (1 + np.exp(-Xb @ w))
        w -= lr * (Xb.T @ (p - y) / len(y) + l2 * w)
    return w


def predict(w, X):
    return 1 / (1 + np.exp(-np.column_stack([X, np.ones(len(X))]) @ w))
