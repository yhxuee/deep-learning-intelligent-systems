"""Spine Stage 03 — neural bigram trained with a hand-derived gradient (Unit 03).

Model: logits = W[x_t]  (W is V x V, initialized to 0), loss = mean cross-entropy.
Gradient (derived in Unit 03): dL/dlogits = (softmax(logits) - onehot(y)) / N,
scattered back into the rows of W indexed by x_t.  Optimizer: Adam (Atlas: opt-adamw, lambda = 0).

Verification targets:
  * loss at initialization is exactly ln V
  * training loss never goes below the count-based maximum-likelihood bigram's training loss
    (the optimum of this model class) and converges to within 0.01 nats of it
"""
import argparse

import numpy as np

from common import DEFAULT_DATA, load_corpus, nll
from stage01_bigram import fit_bigram


def log_softmax(z):
    m = z.max(axis=1, keepdims=True)
    return z - m - np.log(np.exp(z - m).sum(axis=1, keepdims=True))


def loss_and_grad(W, x, y):
    logp = log_softmax(W[x])
    N = len(x)
    loss = -logp[np.arange(N), y].mean()
    dlogits = np.exp(logp)
    dlogits[np.arange(N), y] -= 1.0
    dlogits /= N
    dW = np.zeros_like(W)
    np.add.at(dW, x, dlogits)          # rows reused by many positions: accumulate (+=)
    return loss, dW


def train(ids, V, steps=3000, lr=0.1, betas=(0.9, 0.999), eps=1e-8, log_every=0):
    x, y = ids[:-1], ids[1:]
    W = np.zeros((V, V))
    m, v = np.zeros_like(W), np.zeros_like(W)
    history = []
    for t in range(1, steps + 1):
        loss, g = loss_and_grad(W, x, y)
        history.append(loss)
        m = betas[0] * m + (1 - betas[0]) * g
        v = betas[1] * v + (1 - betas[1]) * g * g
        W -= lr * (m / (1 - betas[0] ** t)) / (np.sqrt(v / (1 - betas[1] ** t)) + eps)
        if log_every and t % log_every == 0:
            print(f"step {t:5d}  train loss {loss:.4f}")
    final, _ = loss_and_grad(W, x, y)
    history.append(final)
    return W, np.array(history)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=DEFAULT_DATA)
    ap.add_argument("--steps", type=int, default=3000)
    args = ap.parse_args()
    train_ids, val_ids, vocab = load_corpus(args.data)
    V = len(vocab)
    W, hist = train(train_ids, V, steps=args.steps, log_every=500)
    mle = nll(fit_bigram(train_ids, V, alpha=0.0), train_ids)
    print(f"init loss {hist[0]:.4f}  (ln V = {np.log(V):.4f})")
    print(f"final train loss {hist[-1]:.4f}  vs count-MLE optimum {mle:.4f}  (gap {hist[-1] - mle:+.4f} nats)")


if __name__ == "__main__":
    main()
