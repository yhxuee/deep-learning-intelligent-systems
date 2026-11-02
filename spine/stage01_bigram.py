"""Spine Stage 01 — count-based bigram language model (Unit 01).

Verification targets:
  * held-out NLL is below the uniform baseline ln V
  * as alpha -> infinity, the model becomes exactly uniform (NLL = ln V)
"""
import argparse

import numpy as np

from common import DEFAULT_DATA, load_corpus, nll


def fit_bigram(ids, V, alpha=1.0):
    """Add-alpha smoothed bigram: P[a, b] = (count(a->b) + alpha) / (count(a) + alpha V)."""
    counts = np.zeros((V, V))
    np.add.at(counts, (ids[:-1], ids[1:]), 1.0)
    counts += alpha
    return counts / counts.sum(axis=1, keepdims=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=DEFAULT_DATA)
    ap.add_argument("--alpha", type=float, default=0.1)
    args = ap.parse_args()
    train, val, vocab = load_corpus(args.data)
    V = len(vocab)
    P = fit_bigram(train, V, args.alpha)
    print(f"vocabulary V = {V};  uniform baseline ln V = {np.log(V):.4f} nats/char")
    print(f"train NLL = {nll(P, train):.4f}   held-out NLL = {nll(P, val):.4f}   (alpha = {args.alpha})")


if __name__ == "__main__":
    main()
