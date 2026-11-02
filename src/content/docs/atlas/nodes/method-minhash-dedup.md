---
title: MinHash near-duplicate detection
description: Jaccard similarity of shingle sets, MinHash signatures whose collision probability equals the Jaccard similarity, and banding to find candidate pairs without comparing all pairs.
atlas:
  id: method-minhash-dedup
  type: method
  mastery: understand-and-verify
  core_units: [7, 10]
  prerequisites: []
  related: [method-contamination-check]
  references: [broder-1997, lee-2022]
  last_reviewed: 2026-11-02
---

## In one paragraph

Represent each document by its set of shingles (word $n$-grams). Resemblance is the Jaccard similarity $J = |S_1\cap S_2|/|S_1\cup S_2|$. A MinHash keeps, for a random ordering $h$ of shingles, the minimum $\min_{s\in S}h(s)$; two documents' minima agree with probability exactly $J$. A signature of $n$ minima split into $b$ bands of $r$ rows makes two documents a candidate when any band matches, with probability $1 - (1 - J^r)^b$ — an S-curve around $(1/b)^{1/r}$. Candidates are confirmed by exact Jaccard similarity.

## Invariants

- $\Pr[\text{minhash agree}] = J$; with $n$ hashes the agreement fraction has standard error $\sqrt{J(1-J)/n}$.
- Identical documents always collide; documents with no shared shingle never do.
- Candidate probability is monotone in $J$ and equals $1 - (1-J^r)^b$ (spine `test_stage07_minhash_collision_probability_matches_theory`).
- On the spine corpus, every planted duplicate pair is found and nothing else is removed (spine `test_stage07_finds_every_planted_duplicate_and_nothing_else`).

## Known values

| Quantity | Value |
|---|---|
| Candidate probability, $b = 16$, $r = 4$, $J = 0.3 / 0.5 / 0.7 / 0.8$ | 0.122 / 0.644 / 0.988 / 0.9998 |
| Steepness threshold $(1/b)^{1/r}$ for $b = 16$, $r = 4$ | 0.5 |
| Spine corpus: planted pairs found (exact / near) | 67 of 67 (37 / 30), no false positives |
| Spine corpus: typical near-duplicate $J$; most similar distinct pair (first 400) | 0.775; 0.245 |

## Common bugs

- Using Python's built-in `hash` for MinHash: it is salted per process, so signatures differ between runs.
- Shingling characters instead of words (or vice versa) without re-tuning the threshold.
- Removing *both* members of a duplicate pair instead of keeping one.
- Treating a banding candidate as a duplicate without confirming the exact Jaccard similarity.
