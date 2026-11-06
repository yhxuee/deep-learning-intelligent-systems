"""Spine Stage 07 — clean the data, then pretrain (Unit 07).

1. Exact deduplication: hash of the normalized story.
2. Near-duplicate detection: word 5-gram shingles, MinHash signatures and locality-sensitive hashing
   (Broder 1997), candidates confirmed by exact Jaccard similarity >= JACCARD.
3. Contamination check: for every held-out story, the fraction of its word 8-grams that also occur
   in the training text; stories above OVERLAP are removed from *training* (the held-out set is
   never edited, so numbers stay comparable with Stages 01-06).
4. Pretrain the spine model on the cleaned text with a 128-character context; save
   checkpoints/stage07.npz, the model used by Stages 08-12.

Verification targets (test_spine.py):
  * every planted duplicate and every planted contamination in data/tales_manifest.json is found
  * after cleaning, no held-out story has overlap >= OVERLAP with the training text
"""
import argparse
import hashlib
import re

import numpy as np

import model as M
from common import load_corpus, load_lines, save_checkpoint, split_lines

JACCARD = 0.7
OVERLAP = 0.8
N_HASH, BANDS = 64, 16          # 16 bands x 4 rows: a pair with Jaccard s collides with prob 1 - (1 - s^4)^16


def normalize(text):
    return re.sub(r"\s+", " ", text.lower()).strip()


def words(text):
    return re.findall(r"[a-z0-9]+", text.lower())


def shingles(text, n=5):
    w = words(text)
    return {" ".join(w[i : i + n]) for i in range(max(1, len(w) - n + 1))}


def jaccard(a, b):
    return len(a & b) / max(1, len(a | b))


def _h(s, seed):
    return int.from_bytes(hashlib.blake2b(s.encode(), digest_size=8, person=seed.to_bytes(16, "little")).digest(), "little")


def minhash(sh, n_hash=N_HASH):
    """Signature: for each of n_hash hash functions, the minimum hash over the shingle set."""
    return np.array([min(_h(s, k) for s in sh) for k in range(n_hash)], dtype=np.uint64)


def near_duplicate_pairs(lines, threshold=JACCARD):
    sh = [shingles(t) for t in lines]
    sig = np.stack([minhash(s) for s in sh])
    rows = N_HASH // BANDS
    buckets = {}
    for i in range(len(lines)):
        for b in range(BANDS):
            key = (b, sig[i, b * rows : (b + 1) * rows].tobytes())
            buckets.setdefault(key, []).append(i)
    cand = {(i, j) for ids in buckets.values() for a, i in enumerate(ids) for j in ids[a + 1 :]}
    return sorted((i, j, jaccard(sh[i], sh[j])) for i, j in cand if jaccard(sh[i], sh[j]) >= threshold)


def deduplicate(lines):
    """Return (kept indices, removed indices with reasons). The first occurrence is kept."""
    removed = {}
    seen = {}
    for i, t in enumerate(lines):
        h = hashlib.sha256(normalize(t).encode()).hexdigest()
        if h in seen:
            removed[i] = ("exact", seen[h])
        else:
            seen[h] = i
    for i, j, s in near_duplicate_pairs(lines):
        if j not in removed and i not in removed:
            removed[j] = ("near", i)
    kept = [i for i in range(len(lines)) if i not in removed]
    return kept, removed


def ngrams(text, n=8):
    w = words(text)
    return {" ".join(w[i : i + n]) for i in range(len(w) - n + 1)}


def contamination(train_lines, held_lines, n=8):
    """For each held-out story: fraction of its n-grams that occur anywhere in the training text."""
    index = set()
    for t in train_lines:
        index |= ngrams(t, n)
    out = []
    for t in held_lines:
        g = ngrams(t, n)
        out.append(len(g & index) / max(1, len(g)))
    return np.array(out)


def clean(lines, val_fraction=0.1):
    train, held = split_lines(lines, val_fraction)
    kept, removed = deduplicate(train)
    train_d = [train[i] for i in kept]
    held_sh = [ngrams(t) for t in held]
    # remove training stories that reproduce a held-out story
    drop = set()
    for k, t in zip(kept, train_d):
        g = ngrams(t)
        for hs in held_sh:
            if hs and len(g & hs) / len(hs) >= OVERLAP:
                drop.add(k)
                break
    final = [train[i] for i in kept if i not in drop]
    return final, {"n_train": len(train), "removed_dups": removed, "removed_contaminated": sorted(drop)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--no-save", action="store_true")
    args = ap.parse_args()
    lines = load_lines()
    train, held = split_lines(lines)
    before = contamination(train, held)
    final, report = clean(lines)
    after = contamination(final, held)
    print(f"training stories {report['n_train']} -> {len(final)}: removed {len(report['removed_dups'])} duplicates "
          f"and {len(report['removed_contaminated'])} contaminating copies")
    print(f"held-out stories with 8-gram overlap >= {OVERLAP}: before {int((before >= OVERLAP).sum())}, after {int((after >= OVERLAP).sum())}")
    print(f"median overlap of a clean held-out story: {np.median(after):.2f}  (templated text shares many 8-grams)")
    tr, va, vocab = load_corpus(train_lines=final)
    cfg = M.Config(vocab=len(vocab), d=64, n_layer=3, n_head=4, max_len=128)
    p = M.init_params(cfg, seed=0)
    M.train(p, cfg, tr, steps=args.steps, T=128, B=16, lr=3e-3, seed=0, val_ids=va, log_every=500)
    val = M.evaluate(p, cfg, va, 128)
    print(f"held-out NLL {val:.4f} nats/char")
    if not args.no_save:
        save_checkpoint("stage07", p, cfg, steps=args.steps, val_nll=val, data="tales.txt (deduplicated, decontaminated)")


if __name__ == "__main__":
    main()
