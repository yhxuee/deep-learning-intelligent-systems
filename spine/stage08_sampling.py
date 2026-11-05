"""Spine Stage 08 — sampling from the spine model, and a 2-D flow-matching model (Unit 08).

Part A — decoding the spine model: greedy, temperature, top-k and nucleus (top-p) sampling.
Part B — flow matching on a toy 2-D distribution (8 Gaussians on a ring): a small MLP v_theta(x, t)
is trained to predict x1 - x0 along straight paths x_t = (1 - t) x0 + t x1 between noise x0 and data
x1 (Lipman et al. 2023; Liu et al. 2023), then sampled by integrating dx/dt = v_theta(x, t) with
Euler steps — the ODE discretization of Unit 05.

Verification targets (test_spine.py):
  * top-p with p = 1 and temperature 1 is plain sampling; top-k with k = 1 is greedy;
    sampled frequencies match the distribution
  * for Gaussian data the optimal velocity field is known in closed form; integrating it maps
    N(0, I) exactly onto the data distribution
  * the MLP's hand-derived gradient matches finite differences; training reduces the loss
"""
import argparse

import numpy as np

# ----------------------------------------------------------------------------- Part A: samplers


def probs(logits, tau=1.0):
    z = logits / max(tau, 1e-8)
    z = z - z.max(-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(-1, keepdims=True)


def draw(P, rng):
    """One sample per row by inverting the cumulative distribution."""
    u = rng.random((P.shape[0], 1))
    return np.minimum((P.cumsum(-1) < u).sum(-1), P.shape[-1] - 1)


def greedy(logits, rng=None):
    return logits.argmax(-1)


def temperature(tau):
    return lambda logits, rng: draw(probs(logits, tau), rng)


def top_k(k, tau=1.0):
    def f(logits, rng):
        P = probs(logits, tau)
        kth = np.sort(P, -1)[:, -k][:, None]
        P = np.where(P >= kth, P, 0.0)
        return draw(P / P.sum(-1, keepdims=True), rng)

    return f


def nucleus_mask(P, top_p):
    """Smallest set of tokens, taken in order of decreasing probability, whose total mass >= top_p."""
    order = np.argsort(-P, -1)
    sorted_P = np.take_along_axis(P, order, -1)
    keep_sorted = (sorted_P.cumsum(-1) - sorted_P) < top_p      # keep a token if the mass before it is < p
    mask = np.zeros_like(P, dtype=bool)
    np.put_along_axis(mask, order, keep_sorted, -1)
    return mask


def top_p(p, tau=1.0):
    def f(logits, rng):
        P = probs(logits, tau)
        P = np.where(nucleus_mask(P, p), P, 0.0)
        return draw(P / P.sum(-1, keepdims=True), rng)

    return f


# ----------------------------------------------------------------------------- Part B: flow matching


def ring(n, rng, k=8, radius=2.0, std=0.15):
    angles = 2 * np.pi * rng.integers(0, k, n) / k
    return np.stack([radius * np.cos(angles), radius * np.sin(angles)], 1) + std * rng.standard_normal((n, 2))


def mlp_init(hidden=64, seed=0):
    rng = np.random.default_rng(seed)
    return {
        "W1": rng.normal(0, 1 / np.sqrt(3), (3, hidden)), "b1": np.zeros(hidden),
        "W2": rng.normal(0, 1 / np.sqrt(hidden), (hidden, hidden)), "b2": np.zeros(hidden),
        "W3": rng.normal(0, 1 / np.sqrt(hidden), (hidden, 2)) * 0.1, "b3": np.zeros(2),
    }


def mlp(p, x, t):
    z0 = np.concatenate([x, t[:, None]], 1)
    a1 = np.tanh(z0 @ p["W1"] + p["b1"])
    a2 = np.tanh(a1 @ p["W2"] + p["b2"])
    return a2 @ p["W3"] + p["b3"], (z0, a1, a2)


def fm_loss_and_grad(p, x0, x1, t):
    """Conditional flow-matching loss  mean || v(x_t, t) - (x1 - x0) ||^2  and its gradient."""
    xt = (1 - t)[:, None] * x0 + t[:, None] * x1
    v, (z0, a1, a2) = mlp(p, xt, t)
    r = v - (x1 - x0)
    n = len(x0)
    loss = float((r ** 2).sum(1).mean())
    dv = 2 * r / n
    g = {"W3": a2.T @ dv, "b3": dv.sum(0)}
    da2 = dv @ p["W3"].T * (1 - a2 ** 2)
    g["W2"], g["b2"] = a1.T @ da2, da2.sum(0)
    da1 = da2 @ p["W2"].T * (1 - a1 ** 2)
    g["W1"], g["b1"] = z0.T @ da1, da1.sum(0)
    return loss, g


def fm_train(steps=3000, batch=512, lr=3e-3, seed=0, data=ring):
    rng = np.random.default_rng(seed)
    p = mlp_init(seed=seed)
    m = {k: np.zeros_like(v) for k, v in p.items()}
    s = {k: np.zeros_like(v) for k, v in p.items()}
    hist = []
    for i in range(1, steps + 1):
        x1 = data(batch, rng)
        x0 = rng.standard_normal((batch, 2))
        t = rng.random(batch)
        loss, g = fm_loss_and_grad(p, x0, x1, t)
        hist.append(loss)
        for k in p:
            m[k] = 0.9 * m[k] + 0.1 * g[k]
            s[k] = 0.999 * s[k] + 0.001 * g[k] ** 2
            p[k] -= lr * (m[k] / (1 - 0.9 ** i)) / (np.sqrt(s[k] / (1 - 0.999 ** i)) + 1e-8)
    return p, np.array(hist)


def integrate(velocity, x0, n_steps=100):
    """Euler integration of dx/dt = velocity(x, t) from t = 0 to t = 1."""
    x = x0.copy()
    h = 1.0 / n_steps
    for k in range(n_steps):
        t = np.full(len(x), k * h)
        x = x + h * velocity(x, t)
    return x


def integrate_midpoint(velocity, x0, n_steps=100):
    """Explicit midpoint (second-order Runge--Kutta) integration on t in [0, 1]."""
    x = x0.copy()
    h = 1.0 / n_steps
    for k in range(n_steps):
        t = np.full(len(x), k * h)
        k1 = velocity(x, t)
        t_mid = t + 0.5 * h
        k2 = velocity(x + 0.5 * h * k1, t_mid)
        x = x + h * k2
    return x


def gaussian_velocity(mu, sigma):
    """Exact marginal velocity for x0 ~ N(0, I), x1 ~ N(mu, sigma^2 I) and straight paths:
    v*(x, t) = E[x1 - x0 | x_t = x] = mu + (t sigma^2 - (1 - t)) / s_t^2 (x - t mu),
    s_t^2 = (1 - t)^2 + t^2 sigma^2."""
    mu = np.asarray(mu, float)

    def v(x, t):
        s2 = (1 - t) ** 2 + (t * sigma) ** 2
        return mu + ((t * sigma ** 2 - (1 - t)) / s2)[:, None] * (x - t[:, None] * mu)

    return v


def mode_coverage(samples, k=8, radius=2.0, tol=0.5):
    """Fraction of samples within tol of some mode, and how many of the k modes received a sample."""
    angles = 2 * np.pi * np.arange(k) / k
    modes = np.stack([radius * np.cos(angles), radius * np.sin(angles)], 1)
    dist = np.linalg.norm(samples[:, None, :] - modes[None], axis=2)
    near = dist.min(1) < tol
    return float(near.mean()), int(len(set(dist.argmin(1)[near])))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fm-steps", type=int, default=3000)
    args = ap.parse_args()
    from common import decode, encode, load_checkpoint
    import model as M

    rng = np.random.default_rng(0)
    ck = load_checkpoint("stage07")
    if ck is not None:
        p, cfg, _ = ck
        from common import load_corpus

        vocab = load_corpus()[2]
        prompt = encode("Lily had a ", vocab)[None, :].repeat(3, 0)
        for name, s in [("greedy", greedy), ("tau=0.7", temperature(0.7)), ("tau=1.5", temperature(1.5)), ("top-p 0.9", top_p(0.9))]:
            out = M.generate(p, cfg, prompt, 90, s, rng)
            print(f"--- {name}")
            for row in out:
                print("   ", decode(row, vocab).split("\n")[0])
    fm, hist = fm_train(steps=args.fm_steps)
    x = integrate(lambda x, t: mlp(fm, x, t)[0], rng.standard_normal((2000, 2)))
    frac, modes = mode_coverage(x)
    print(f"flow matching: loss {hist[:100].mean():.3f} -> {hist[-100:].mean():.3f};  {frac:.0%} of samples near a mode; {modes}/8 modes covered")


if __name__ == "__main__":
    main()
