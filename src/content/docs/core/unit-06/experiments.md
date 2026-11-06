---
title: "6.9 Experiments"
description: Six reproducible investigations of FLOP accounting, the roofline, KV caching, scaling-law prediction, compute-optimal allocation, and activation recomputation—with controls and failure interpretations.
sidebar: { label: "6.9 Experiments", order: 9 }
---

These are investigations, not demonstrations with guaranteed endings. Before each run, copy the small table below into your notes. A result that contradicts your prediction is useful if the measurement is valid.

| Field | Write before running |
|---|---|
| hypothesis | one sentence that could be wrong |
| independent variable | the quantity you deliberately change |
| controls | what must remain identical |
| metric | units, aggregation and uncertainty |
| failure criterion | what result would reject the hypothesis |

Run commands from the repository root. The relevant code is `spine/model.py`, `spine/stage05_kv_cache.py` and `spine/stage06_scaling.py`. Save the command, machine information, random seed and raw rows; a plot without provenance is not a result.

## Experiment 6.1 — Make the FLOP formula executable

**Question.** Does the implementation perform exactly the matrix products counted in Proposition 6.1?

Use $V=13$, $d=16$, $L=3$, $T=10$ and $B=4$. First calculate on paper:

$$
N_{\text{mm}}=12Ld^2+Vd=9{,}424,
$$

$$
F_{\text{step}}=BT(6N_{\text{mm}}+12LTd)=2{,}492{,}160.
$$

Then reset `model.FLOPS["count"]`, call `loss_and_grads` once, and compare the integer counter with your prediction. Change only $T$ to 5, 10, 20 and record the count divided by $BT$.

**Control.** Keep $V,d,L,B$ and all architecture switches fixed. The counter includes only calls through `mm`; it is not a hardware performance counter.

**Interpretation.** Exact equality verifies an implementation invariant: the declared matrix products match the derivation. It does not say that softmax, normalization, indexing or Python cost zero machine instructions. Plotting FLOPs/token against $T$ should give a line with intercept $6N_{\text{mm}}$ and slope $12Ld$. A mismatch means either the code path or the algebra changed; inspect the individual `mm` shapes before changing the formula.

## Experiment 6.2 — Draw a roofline from your own machine

**Question.** At what arithmetic intensity does matrix multiplication change from bandwidth-sensitive to compute-sensitive on this machine?

1. Create float32 matrices and time `A @ B` for $n=16,32,\ldots,4096$. Warm up each size, run at least five timed repeats, and keep the median.
2. Compute achieved rate as $2n^3/t$ FLOP/s. Under the idealized “read $A$ and $B$ once, write $C$ once” traffic, intensity is
   $$I_{\min}=\frac{2n^3}{3(4n^2)}=\frac{n}{6}\ \text{FLOPs/byte}.$$
3. Estimate large-array copy bandwidth using arrays bigger than the last-level cache. Count both the bytes read and written, and report that convention.
4. Plot achieved GFLOP/s against $I_{\min}$. Mark the measured bandwidth line $P=\beta I$ and the observed large-matrix plateau.
5. Time a spine training step. Divide Proposition 6.1's count by wall time, then by the **measured plateau**, not a marketing peak, to obtain a local utilization ratio.

**Controls and traps.** Record dtype, array layout, BLAS library and thread count. Close other heavy work. Small matrices are dominated by launch and interpreter overhead; copies may be served by cache; thermal throttling makes later trials slower. $I_{\min}$ is an optimistic traffic model, not measured bytes.

**Interpretation.** You may see a rising region and a plateau, but a laptop CPU is not obliged to draw a clean textbook roofline. The experiment succeeds if your axes and units let you explain deviations rather than erase them.

## Experiment 6.3 — What the KV cache saves—and what it does not

**Question.** How do recomputing the whole prefix and caching keys/values scale with context position?

First run `python spine/stage05_kv_cache.py`. Before trusting timing, verify the correctness control already encoded in `spine/test_model.py`: at every position, cached logits agree with a full recomputation within tolerance.

Next construct the same random model and fixed batch for context lengths that fit its `max_len`, for example 8, 16, 32, 64 and 120. For each length, warm up both paths and report the median of at least ten runs. Record:

- milliseconds for the next token with full-prefix recomputation;
- milliseconds for one cached decoding step;
- cache bytes from `cache.nbytes()` and from $2BLtd\,b$, where $b=8$ for the spine model's default float64 arrays;
- the speed-up ratio, with confidence intervals or at least min–max spread.

**Prediction.** Full recomputation repeats all earlier weight products and attention work. Cached decoding reuses earlier $K,V$, but its new query still attends to all $t$ cached positions; therefore cached time may grow with $t$. It should grow more slowly—not remain flat by mathematical necessity.

**Failure interpretation.** At this tiny NumPy scale, concatenation, allocation and interpreter overhead can make caching slower. That does not refute the arithmetic saving; it shows that saved FLOPs were not the measured bottleneck. Report both the exact-logit control and the timing result.

## Experiment 6.4 — Make a scaling law predict before it explains

**Question.** Can a fit to five smaller widths predict an unseen sixth width?

Run:

```bash
python spine/stage06_scaling.py --seed 0
```

The script trains widths 8, 12, 16, 24 and 32 at the same token budget, fits $L(N)=E+AN^{-\alpha}$, and writes the width-48 prediction to `checkpoints/stage06_prediction.json` **before** training width 48. Preserve that file before another run overwrites it.

Repeat with `--seed 1` and `--seed 2`; then repeat the three seeds with `--steps 1000`. Do not refit after seeing width 48 and call the result a prediction. For each condition, record the predicted NLL, observed NLL, signed error, seed spread and fitted $(E,A,\alpha)$.

**Controls.** Keep corpus split, tokenizer, optimizer schedule, context length and evaluation windows fixed. “Same steps” here means the same number of training tokens, not the same FLOPs: wider models consume more compute.

**Interpretation.** Ask whether errors are comparable to seed variation and whether halving the token budget changes the fitted exponent or floor. With only five noisy points, parameter estimates may be unstable even when interpolation looks smooth. A miss is evidence about this fit and range, not evidence that no scaling relation exists.

## Experiment 6.5 — Allocate a fixed compute budget

**Question.** At fixed arithmetic, is it better to train a smaller model on more tokens or a larger model on fewer?

Choose widths such as 8, 12, 16, 24, 32 and 48, with fixed $L,T,B$. For each width, compute its per-token matrix-product count

$$
c(d)=6N_{\text{mm}}(d)+12LTd,
$$

then assign $D(d)=\lfloor C/c(d)\rfloor$ training tokens and convert to whole steps $\lfloor D/(BT)\rfloor$. Pick $C$ so every model receives enough steps to learn and the full sweep is affordable; write $C$ before inspecting losses. Report the unused remainder caused by rounding.

Train at least three seeds, plot held-out NLL against width with error bars, and compare the empirical minimum with Proposition 6.2 using constants fitted independently in Experiment 6.4.

**Do not promise a U-shape.** The toy sweep may be monotone or noisy because the candidate widths miss the optimum, the fitted separable law is poor, or small models are optimization-limited. The conclusion is conditional: “among these widths, under this corpus, schedule and budget…”. If one width receives only a handful of steps, enlarge $C$ rather than interpreting an untrained model as a scaling law.

## Experiment 6.6 — Buy memory with recomputation

**Question.** How many activation bytes can be saved by discarding block internals, and how much extra time buys that saving?

Make a separate experimental copy of the training path. In the baseline, retain every array required by `backward`. In the recomputation version, retain each block's input and any irreducible boundary state, discard its internal attention/MLP arrays, and rerun that block's forward computation immediately before its backward computation.

Use the same 6-layer parameters and batch for both paths. Check the loss and every parameter gradient against the baseline before timing; report maximum absolute and relative error. Then:

1. sum `nbytes` of the retained activation arrays using one documented no-double-counting rule;
2. optionally measure process peak resident memory, noting that a Python allocator may retain freed buffers;
3. warm up and report median step time over repeated alternating trials;
4. count the extra `mm` FLOPs rather than assuming the time ratio.

**Prediction.** Retained activation bytes should fall, while FLOPs and usually wall time rise. The exact ratios depend on which boundaries are kept and whether allocation or memory traffic dominates. If gradients change materially, you have changed the computation—not implemented checkpointing. If explicit saved bytes fall but process RSS does not, distinguish live tensors from allocator-reserved memory.

## What to hand in

For each experiment, submit the preregistered prediction, exact command and seed, raw table, one plot where useful, a correctness control, and a five-sentence interpretation that includes at least one alternative explanation. The goal is not six pretty curves; it is six auditable arguments from measurements.
