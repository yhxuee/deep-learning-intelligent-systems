---
title: "6.9 Experiments"
description: Six experiments — the FLOP counter, a roofline on your own machine, decoding with and without a cache, a pre-registered scaling-law fit, the compute-optimal trade-off at toy scale, and recomputation.
sidebar: { label: "6.9 Experiments", order: 9 }
---

Write the prediction before running. Code: [`spine/model.py`](https://github.com/yhxuee/deep-learning-intelligent-systems/tree/main/spine) (FLOP counter `mm`), `spine/stage05_kv_cache.py`, `spine/stage06_scaling.py`.

## Experiment 6.1 — Counting every FLOP

**Tests:** Proposition 6.1, Worked Example 6.1. **Predict:** the counted FLOPs of one `loss_and_grads` call for $V = 13$, $d = 16$, $L = 3$, $T = 10$, $B = 4$.

**Procedure.** Reset `model.FLOPS["count"]`, run one call, compare with $BT(6N_{\text{mm}} + 12LTd)$. Then remove the first layer's input-gradient product (it is not needed if the embedding is frozen) and predict the new count before measuring.

**Look for:** exact equality (the spine test asserts it); the change when the embedding is frozen equals $2BTd\cdot d$ per removed product — derive which products disappear.

## Experiment 6.2 — A roofline on your machine

**Tests:** §6.2.5, §6.3.3. **Predict:** the size $n$ at which square matrix products on your CPU stop getting faster per FLOP.

**Procedure.** Time `A @ B` for $n\times n$ float32 matrices, $n$ from 16 to 4 096, and plot achieved GFLOP/s against arithmetic intensity $n/6$ (float32: 4 bytes). Measure memory bandwidth by timing a large array copy. Then time one spine training step and compute its utilization from Proposition 6.1.

**Look for:** a rising region (memory-bound) and a plateau (compute-bound) with the knee near peak FLOP/s divided by bandwidth; the spine step far below the plateau, because its products are small and its many elementwise operations are memory-bound.

## Experiment 6.3 — Decoding: recompute versus cache

**Tests:** §6.3.4, Unit 04 §4.3.10. **Predict:** the ratio of per-token time with and without the KV cache after 8, 32 and 120 tokens.

**Procedure.** `python stage05_kv_cache.py` reports times for a short generation; extend it to measure per-token time as a function of position, and the cache size against `kv_cache_bytes`.

**Look for:** recomputation cost growing with position, cached cost nearly flat; cache bytes equal to the formula exactly.

## Experiment 6.4 — A pre-registered scaling fit

**Tests:** §6.5.2, Worked Example 6.6. **Predict:** before running, write the held-out NLL you expect for width 48 *after* seeing the five smaller widths' results but *before* training it — or let the script do it: it saves the fit and the prediction to `checkpoints/stage06_prediction.json` before training the sixth model.

**Procedure.** `python stage06_scaling.py` (about ten minutes on a laptop). Repeat with three seeds; then with half the token budget.

**Look for:** whether the prediction falls inside the seed-to-seed spread; how the fitted exponent and floor change with the budget — and what that says about which resource limited the small models.

## Experiment 6.5 — The compute-optimal trade-off at toy scale

**Tests:** Proposition 6.2. **Predict:** for a fixed budget of $2\times10^{12}$ training FLOPs, which of widths 16, 24, 32, 48 reaches the lowest held-out NLL when each is trained for as many tokens as the budget allows.

**Procedure.** For each width, compute tokens $= C/(6N_{\text{mm}} + 12LTd)$ from Proposition 6.1, convert to steps, train, evaluate. Plot loss against width at fixed compute.

**Look for:** a U-shape — too small a model saturates, too large a model sees too few tokens; locate its minimum and compare with Proposition 6.2 using constants fitted in Experiment 6.4.

## Experiment 6.6 — Trading FLOPs for memory

**Tests:** §6.3.2. **Predict:** peak activation memory and step time for a 6-layer spine model with and without recomputing each block's activations in the backward pass.

**Procedure.** Implement block-level recomputation in a copy of `model.py` (keep only each block's input in the forward pass; recompute the block before its backward). Measure memory with `tracemalloc` and time with `time.perf_counter`; verify gradients are unchanged to $10^{-12}$.

**Look for:** activation memory falling by about the number of cached tensors per block; time rising by roughly one extra forward pass.
