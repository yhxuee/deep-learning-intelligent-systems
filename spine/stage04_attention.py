"""Spine Stage 04 — add causal self-attention: the spine model becomes a Transformer (Unit 04).

Uses the shared model in model.py (pre-LN decoder, tied embeddings, learned positions) whose every
gradient is checked against finite differences in test_model.py.  Trains on the raw training text
(duplicates and all — Stage 07 cleans it) and saves checkpoints/stage04.npz.

Verification targets (test_spine.py):
  * every attention invariant: causality, KV-cache equivalence, gradient check (test_model.py)
  * loss at initialization is within 0.05 nats of ln V
  * after training, held-out NLL is below the best bigram (Stage 01) — the model now uses more
    than one previous character
"""
import argparse

import numpy as np

import model as M
from common import DEFAULT_DATA, load_corpus, nll, save_checkpoint
from stage01_bigram import fit_bigram

DEFAULT = dict(d=64, n_layer=2, n_head=4, max_len=64)


def run(train_ids, val_ids, V, steps=1500, seed=0, log_every=250, lr=3e-3, **cfg_overrides):
    cfg = M.Config(vocab=V, **{**DEFAULT, **cfg_overrides})
    p = M.init_params(cfg, seed=seed)
    init_val = M.evaluate(p, cfg, val_ids, cfg.max_len)
    hist = M.train(p, cfg, train_ids, steps=steps, T=cfg.max_len, B=16, lr=lr, seed=seed,
                   val_ids=val_ids, log_every=log_every)
    return p, cfg, init_val, hist


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=DEFAULT_DATA)
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--no-save", action="store_true")
    args = ap.parse_args()
    tr, va, vocab = load_corpus(args.data)
    V = len(vocab)
    p, cfg, init_val, hist = run(tr, va, V, steps=args.steps)
    val = M.evaluate(p, cfg, va, cfg.max_len)
    bigram = nll(fit_bigram(tr, V, 0.1), va)
    print(f"parameters {M.n_params(p, cfg):,};  init held-out NLL {init_val:.4f} (ln V = {np.log(V):.4f})")
    print(f"held-out NLL {val:.4f}  vs  best bigram {bigram:.4f}")
    if not args.no_save:
        save_checkpoint("stage04", p, cfg, steps=args.steps, val_nll=val, data="tales.txt (raw)")


if __name__ == "__main__":
    main()
