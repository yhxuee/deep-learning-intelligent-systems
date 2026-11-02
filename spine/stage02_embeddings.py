"""Spine Stage 02 — learned embeddings with a tied read-out (Unit 02).

Model: each character a has a vector e_a in R^d (rows of E, V x d).  The score for "b follows a" is
    z_ab = e_a^T M e_b + c_b,
so the same table E is used to read the input and to score the output (tied embeddings), and the
V x V bigram table of Stage 01 is replaced by a rank-d factorization with V d + d^2 + V parameters.

Verification targets:
  * at initialization (M = 0, c = 0) the loss is exactly ln V
  * the gradient of a table lookup equals the gradient of a one-hot matrix product (Unit 02 §2.3)
  * training loss never goes below the count-based maximum-likelihood bigram (the best any bigram can do)
  * characters that occur in identical contexts get nearly identical embeddings (distributional hypothesis)
"""
import argparse

import numpy as np

from common import DEFAULT_DATA, load_corpus, nll
from stage01_bigram import fit_bigram


def init(V, d, seed=0):
    rng = np.random.default_rng(seed)
    return {"E": rng.normal(0, 1 / np.sqrt(d), (V, d)), "M": np.zeros((d, d)), "c": np.zeros(V)}


def logits(p, x):
    return p["E"][x] @ p["M"] @ p["E"].T + p["c"]


def loss_and_grad(p, x, y):
    """Mean cross-entropy and its gradient.  z = H M E^T + c with H = E[x]."""
    E, M = p["E"], p["M"]
    H = E[x]
    z = H @ M @ E.T + p["c"]
    z = z - z.max(1, keepdims=True)
    logp = z - np.log(np.exp(z).sum(1, keepdims=True))
    N = len(x)
    loss = -logp[np.arange(N), y].mean()
    dz = np.exp(logp)
    dz[np.arange(N), y] -= 1.0
    dz /= N
    g = {"c": dz.sum(0), "M": H.T @ dz @ E}
    dE = dz.T @ (H @ M)                 # E as the read-out (output side)
    dH = dz @ E @ M.T                   # E as the lookup (input side) ...
    np.add.at(dE, x, dH)                # ... scatter-added into the rows that were read
    g["E"] = dE
    return loss, g


def train(ids, V, d=16, steps=1500, lr=0.03, seed=0, batch=4096):
    rng = np.random.default_rng(seed)
    p = init(V, d, seed)
    m = {k: np.zeros_like(v) for k, v in p.items()}
    s = {k: np.zeros_like(v) for k, v in p.items()}
    hist = []
    for t in range(1, steps + 1):
        i = rng.integers(0, len(ids) - 1, batch)
        loss, g = loss_and_grad(p, ids[i], ids[i + 1])
        hist.append(loss)
        for k in p:
            m[k] = 0.9 * m[k] + 0.1 * g[k]
            s[k] = 0.999 * s[k] + 0.001 * g[k] ** 2
            p[k] -= lr * (m[k] / (1 - 0.9 ** t)) / (np.sqrt(s[k] / (1 - 0.999 ** t)) + 1e-8)
    return p, np.array(hist)


def heldout_nll(p, ids):
    z = logits(p, ids[:-1])
    z = z - z.max(1, keepdims=True)
    logp = z - np.log(np.exp(z).sum(1, keepdims=True))
    return float(-logp[np.arange(len(ids) - 1), ids[1:]].mean())


def nearest(p, vocab, ch, k=4):
    E = p["E"] / np.linalg.norm(p["E"], axis=1, keepdims=True)
    i = vocab.index(ch)
    sims = E @ E[i]
    order = [j for j in np.argsort(-sims) if j != i][:k]
    return [(vocab[j], float(sims[j])) for j in order]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=DEFAULT_DATA)
    ap.add_argument("--d", type=int, default=16)
    ap.add_argument("--steps", type=int, default=1500)
    args = ap.parse_args()
    tr, va, vocab = load_corpus(args.data)
    V = len(vocab)
    p, hist = train(tr, V, d=args.d, steps=args.steps)
    print(f"V = {V}, d = {args.d}: {V * args.d + args.d ** 2 + V} parameters vs {V * V} for the count table")
    print(f"init loss {hist[0]:.4f} (ln V = {np.log(V):.4f});  held-out NLL {heldout_nll(p, va):.4f}")
    print(f"count bigram (alpha=0.1) held-out NLL {nll(fit_bigram(tr, V, 0.1), va):.4f}")
    for ch in ["a", "3", "T", "."]:
        if ch in vocab:
            print(f"nearest to {ch!r}: " + ", ".join(f"{c!r} {s:.2f}" for c, s in nearest(p, vocab, ch)))


if __name__ == "__main__":
    main()
