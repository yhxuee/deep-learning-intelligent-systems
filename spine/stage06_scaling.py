"""Spine Stage 06 — a scaling-law fit with a pre-registered prediction (Unit 06).

Train five widths of the spine model on the same token budget, fit
    L(N) = E + A * N^(-alpha)          (N = non-embedding parameters, Kaplan et al. 2020)
write the prediction for a sixth, larger width to disk *before* training it, then train it and
compare.  Also: the compute-optimal split between parameters and data for a law of the form
    L(N, D) = E + A N^(-alpha) + B D^(-beta)     subject to  C = 6 N D     (Hoffmann et al. 2022).

Verification targets (test_spine.py):
  * the fitting routine recovers known (E, A, alpha) from synthetic data
  * the closed-form compute-optimal N agrees with brute-force minimization
  * measured training FLOPs equal 6 N + 12 L T d per token (test_model.py)
"""
import argparse
import json
import time

import numpy as np

import model as M
from common import CKPT_DIR, load_corpus


def fit_power_law(N, L, n_grid=400):
    """Least-squares fit of L = E + A N^-alpha.  For each candidate E on a grid, log(L - E) is linear
    in log N, so (log A, alpha) follow from ordinary least squares; keep the E with the smallest
    squared error in L itself."""
    N, L = np.asarray(N, float), np.asarray(L, float)
    best = None
    for E in np.linspace(0.0, L.min() * 0.999, n_grid):
        y = np.log(L - E)
        X = np.stack([np.ones_like(N), -np.log(N)], 1)
        (logA, alpha), *_ = np.linalg.lstsq(X, y, rcond=None)
        pred = E + np.exp(logA) * N ** (-alpha)
        sse = float(((pred - L) ** 2).sum())
        if best is None or sse < best[0]:
            best = (sse, E, float(np.exp(logA)), float(alpha))
    return {"E": best[1], "A": best[2], "alpha": best[3], "sse": best[0]}


def predict(fit, N):
    return fit["E"] + fit["A"] * np.asarray(N, float) ** (-fit["alpha"])


def compute_optimal(C, A, B, alpha, beta):
    """argmin_{N,D} A N^-alpha + B D^-beta  s.t. 6 N D = C.
    Setting the derivative of A N^-a + B (C/6N)^-b to zero gives
        N* = G (C/6)^(b/(a+b)),  G = (a A / (b B))^(1/(a+b)),  D* = (C/6) / N*."""
    G = (alpha * A / (beta * B)) ** (1.0 / (alpha + beta))
    N = G * (C / 6.0) ** (beta / (alpha + beta))
    return N, (C / 6.0) / N


def train_size(d, tr, va, V, steps, seed=0, n_layer=2, T=64, B=16):
    cfg = M.Config(vocab=V, d=d, n_layer=n_layer, n_head=4, max_len=T)
    p = M.init_params(cfg, seed=seed)
    M.train(p, cfg, tr, steps=steps, T=T, B=B, lr=3e-3, seed=seed)
    return M.n_params(p, cfg, embedding=False), M.evaluate(p, cfg, va, T)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--widths", default="8,12,16,24,32")
    ap.add_argument("--predict", type=int, default=48)
    args = ap.parse_args()
    tr, va, vocab = load_corpus()
    V = len(vocab)
    widths = [int(w) for w in args.widths.split(",")]
    rows = []
    for d in widths:
        t = time.time()
        N, L = train_size(d, tr, va, V, args.steps)
        rows.append((d, N, L))
        print(f"d={d:3d}  N={N:7d}  held-out NLL {L:.4f}  ({time.time() - t:.0f}s)", flush=True)
    fit = fit_power_law([r[1] for r in rows], [r[2] for r in rows])
    cfg6 = M.Config(vocab=V, d=args.predict, n_layer=2, n_head=4, max_len=64)
    N6 = M.n_params(M.init_params(cfg6), cfg6, embedding=False)
    pred = float(predict(fit, N6))
    CKPT_DIR.mkdir(exist_ok=True)
    (CKPT_DIR / "stage06_prediction.json").write_text(json.dumps({"fit": fit, "d": args.predict, "N": int(N6), "predicted": pred}, indent=1))
    print(f"fit: E={fit['E']:.3f}  A={fit['A']:.2f}  alpha={fit['alpha']:.3f};  pre-registered prediction for d={args.predict} (N={N6}): {pred:.4f}")
    _, L6 = train_size(args.predict, tr, va, V, args.steps)
    print(f"observed: {L6:.4f}   error {L6 - pred:+.4f} nats")
    tokens = args.steps * 16 * 64
    print(f"token budget per run: {tokens:,}")


if __name__ == "__main__":
    main()
