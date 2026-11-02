"""Shared utilities for the spine project: data loading, character vocabulary, splits."""
import pathlib

import numpy as np

DEFAULT_DATA = pathlib.Path(__file__).parent / "data" / "sample.txt"


def load_corpus(path=DEFAULT_DATA, val_fraction=0.1):
    """Return (train_ids, val_ids, vocab). Contiguous split: the last val_fraction of the text is held out."""
    text = pathlib.Path(path).read_text(encoding="utf-8")
    vocab = sorted(set(text))
    stoi = {c: i for i, c in enumerate(vocab)}
    ids = np.array([stoi[c] for c in text], dtype=np.int64)
    n_val = int(len(ids) * val_fraction)
    return ids[:-n_val], ids[-n_val:], vocab


def nll(prob_table, ids):
    """Mean negative log-likelihood (nats/token) of a bigram table P[a, b] = p(b | a) on a sequence."""
    p = prob_table[ids[:-1], ids[1:]]
    return float(-np.mean(np.log(p)))
