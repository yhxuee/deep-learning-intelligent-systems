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

Second model (Bengio et al. 2003): a *neural n-gram*.  The embeddings of the previous k characters are
concatenated, passed through one tanh layer and read out:  z = W2 tanh(W1 [e_{t-k}; ...; e_{t-1}] + b1) + b2.
Its parameter count grows linearly in k, while a count table needs up to V^k rows; Unit 02 §2.4 compares them.
"""
from collections import defaultdict
import math
import argparse

import numpy as np

from common import DEFAULT_DATA, load_corpus, load_corpus_three_way, nll
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


# ----------------------------------------------------------------------------- neural n-gram (Bengio et al. 2003)
def ngram_windows(ids, k):
    X = np.stack([ids[i : len(ids) - k + i] for i in range(k)], 1)
    return X, ids[k:]


def ngram_init(V, k, d=16, H=128, seed=0):
    rng = np.random.default_rng(seed)
    return {"E": rng.normal(0, 0.3, (V, d)), "W1": rng.normal(0, 1 / np.sqrt(k * d), (k * d, H)), "b1": np.zeros(H),
            "W2": rng.normal(0, 0.1 / np.sqrt(H), (H, V)), "b2": np.zeros(V)}


def ngram_loss_and_grad(p, X, y):
    N, k = X.shape
    d = p["E"].shape[1]
    x = p["E"][X].reshape(N, k * d)
    h = np.tanh(x @ p["W1"] + p["b1"])
    z = h @ p["W2"] + p["b2"]
    z = z - z.max(1, keepdims=True)
    logp = z - np.log(np.exp(z).sum(1, keepdims=True))
    loss = -logp[np.arange(N), y].mean()
    dz = np.exp(logp)
    dz[np.arange(N), y] -= 1.0
    dz /= N
    g = {"W2": h.T @ dz, "b2": dz.sum(0)}
    da = (dz @ p["W2"].T) * (1 - h * h)
    g["W1"], g["b1"] = x.T @ da, da.sum(0)
    dE = np.zeros_like(p["E"])
    np.add.at(dE, X, (da @ p["W1"].T).reshape(N, k, d))   # each position's slice goes back to its token's row
    g["E"] = dE
    return loss, g


def train_ngram(ids, V, k=4, d=16, H=128, steps=2500, lr=3e-3, batch=512, seed=0, weight_decay=1e-4):
    X, y = ngram_windows(ids, k)
    p = ngram_init(V, k, d, H, seed)
    rng = np.random.default_rng(seed)
    m = {a: np.zeros_like(b) for a, b in p.items()}
    s = {a: np.zeros_like(b) for a, b in p.items()}
    for t in range(1, steps + 1):
        i = rng.integers(0, len(y), batch)
        _, g = ngram_loss_and_grad(p, X[i], y[i])
        for a in p:
            ga = g[a] + weight_decay * p[a]
            m[a] = 0.9 * m[a] + 0.1 * ga
            s[a] = 0.999 * s[a] + 0.001 * ga * ga
            p[a] -= lr * (m[a] / (1 - 0.9 ** t)) / (np.sqrt(s[a] / (1 - 0.999 ** t)) + 1e-8)
    return p


def ngram_nll(p, ids, k):
    X, y = ngram_windows(ids, k)
    return float(ngram_loss_and_grad(p, X, y)[0])


def count_ngram_nll(train_ids, test_ids, V, k, alpha):
    """Add-alpha count model on the previous k characters; unseen contexts get the uniform distribution."""
    C = defaultdict(lambda: np.zeros(V))
    for i in range(k, len(train_ids)):
        C[tuple(train_ids[i - k : i])][train_ids[i]] += 1
    total = 0.0
    for i in range(k, len(test_ids)):
        c = C.get(tuple(test_ids[i - k : i]))
        total += math.log(V) if c is None else -math.log((c[test_ids[i]] + alpha) / (c.sum() + alpha * V))
    return total / (len(test_ids) - k), len(C)


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
    print("count vs neural 4-gram (untouched test NLL) as the training text grows:")
    eval_tr, tune, test, eval_vocab = load_corpus_three_way(args.data)
    assert vocab == eval_vocab and np.array_equal(tr, eval_tr)
    for n in [5000, 20000, 80000, len(tr)]:
        alphas = [0.003, 0.01, 0.03, 0.1, 0.3]
        alpha = min(alphas, key=lambda a: count_ngram_nll(tr[:n], tune, V, 4, a)[0])
        cnt = count_ngram_nll(tr[:n], test, V, 4, alpha)[0]
        q = train_ngram(tr[:n], V, k=4)
        print(f"   n={n:7d}  alpha={alpha:.3f}  count-test {cnt:.3f}  neural-test {ngram_nll(q, test, 4):.3f}")


if __name__ == "__main__":
    main()
