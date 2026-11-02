"""Spine Stage 05 — KV-cache decoding and a recall stress test (Unit 05).

Verification targets:
  * logits from cached, token-by-token decoding equal a full recompute (test_model.py)
  * measured cache size equals the formula 2 * L * heads * d_head * t * B * bytes (Unit 04 §4.3.10)
Measurements: time per generated token with and without the cache, and whether the model can still
use a name introduced at the start of a story when the question comes late (recall vs. distance).
"""
import time

import numpy as np

import model as M
from common import decode, encode, load_checkpoint, load_corpus


def greedy(logits, rng):
    return logits.argmax(-1)


def timing(p, cfg, prompt_len=8, n_new=48):
    prompt = np.zeros((1, prompt_len), dtype=int)
    out = {}
    for use_cache in (False, True):
        t = time.perf_counter()
        M.generate(p, cfg, prompt, n_new, greedy, None, use_cache=use_cache)
        out[use_cache] = (time.perf_counter() - t) / n_new
    return out


def recall_by_distance(p, cfg, vocab, rng, n=40):
    """Probability that the model, after '<Name> had ... <filler> ... with ', continues with the
    friend's name introduced at the start, as a function of how much filler text separates them."""
    names = ["Mia", "Tom", "Lily", "Ben", "Sam", "Anna", "Max", "Zoe"]
    fillers = ["", "Then the cup fell. ", "Then the cup fell. It was sad. They sat down. "]
    res = {}
    for f in fillers:
        hits = 0.0
        for _ in range(n):
            a, b = rng.choice(names, 2, replace=False)
            text = f"{a} had a red kite. One sunny day {a} went to the park with {b}. {f}{a} and "
            ids = encode(text[-cfg.max_len :], vocab)[None, :]
            logits = M.forward(p, cfg, ids)[0, -1]
            pr = np.exp(logits - logits.max())
            pr /= pr.sum()
            hits += pr[vocab.index(b[0])]
        res[len(f)] = hits / n
    return res


def main():
    ck = load_checkpoint("stage07") or load_checkpoint("stage04")
    if ck is None:
        raise SystemExit("run stage04_attention.py (or stage07_pretrain.py) first")
    p, cfg, meta = ck
    _, va, vocab = load_corpus()
    cache = M.KVCache(cfg, B=1)
    for t in va[:32]:
        M.decode_step(p, cfg, np.array([t]), cache)
    print(f"cache after 32 tokens: {cache.nbytes()} bytes (formula: {M.kv_cache_bytes(cfg, 1, 32)})")
    tm = timing(p, cfg)
    print(f"time per token: recompute {tm[False] * 1e3:.2f} ms, cached {tm[True] * 1e3:.2f} ms")
    rng = np.random.default_rng(0)
    for dist, pr in recall_by_distance(p, cfg, vocab, rng).items():
        print(f"filler {dist:3d} chars: p(first letter of the friend's name) = {pr:.2f}")
    sample = M.generate(p, cfg, encode("Mia had a ", vocab)[None, :], 50, greedy, rng)[0]
    print("greedy:", decode(sample, vocab))


if __name__ == "__main__":
    main()
