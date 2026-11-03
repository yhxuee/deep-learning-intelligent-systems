---
title: "6.4 Worked examples"
description: Eight examples — every FLOP of the spine model, the 6N rule for GPT-3, a 7B training budget, arithmetic intensity, decoding latency, a pre-registered scaling fit that missed, the flat compute-optimal optimum, and energy per FLOP — with what the Landauer limit does and does not say.
sidebar: { label: "6.4 Worked examples", order: 4 }
---

Exact numbers are recomputed by `labs/test_worked_examples_u06_u08.py`; the FLOP identity is asserted by `spine/test_model.py`; the scaling fit comes from `spine/stage06_scaling.py`.

## Worked Example 6.1 — Every FLOP of the spine model

**Setting.** The Stage 07 model: $V = 50$, $d = 64$, $L = 3$, $h = 4$, $T = 128$, batch $B = 16$.

**Counting.** $N_{\text{mm}} = 12Ld^2 + Vd = 147\,456 + 3\,200 = 150\,656$. Per token (Proposition 6.1): $6N_{\text{mm}} + 12LTd = 903\,936 + 294\,912 = 1\,198\,848$ FLOPs. Per step ($BT = 2\,048$ tokens): $2\,455\,240\,704$ — exactly what the instrumented counter reports. The 4 000-step run: $9.82\times10^{12}$ FLOPs.

**Reading.** Attention is 24.6% of the training FLOPs here because the context (128) is large relative to the width (64): $T/(T + 6d) = 0.25$ for the blocks. The model has 160 704 parameters (149 312 excluding embeddings); its training state at 16 bytes per parameter is 2.6 MB. This answers the prediction prompt of §6.1: about 2.5 billion FLOPs per step.

## Worked Example 6.2 — The 6N rule for GPT-3

**Setting.** 175 billion parameters, 300 billion training tokens.

**Computation.** $6\times1.75\times10^{11}\times3\times10^{11} = 3.15\times10^{23}$ FLOPs. The paper reports $3.14\times10^{23}$ ([Brown et al. 2020](/references/#brown-2020)).

**Reading.** Two numbers and a constant reproduce the published training compute to within 0.3%. At 40% utilization of A100s it is about 700 000 GPU-hours ($3.15\times10^{23}/(0.4\times3.12\times10^{14})$ s).

## Worked Example 6.3 — What a 7B model needs to train

**Setting.** $N = 7\times10^9$, mixed precision, Adam.

**Computation.** 16 bytes per parameter → 112 GB for weights, gradients and optimizer state, before activations.

**Reading.** More than one 80 GB device. Sharding the state across 8 devices brings it to 14 GB each, leaving room for activations — the motivation for the sharded optimizers of §6.5.5.

## Worked Example 6.4 — Arithmetic intensity

**Setting.** A100: 312 TFLOP/s, 2.039 TB/s; ratio 153 FLOP/byte.

**Square product, $n = 4\,096$, 16-bit.** $I = n/3 = 1\,365$ FLOP/byte: compute-bound with a factor of 9 to spare.

**Matrix–vector product.** $I\approx1$: memory-bound, at most $1/153$ of peak.

**Reading.** The same hardware is either 99% or 0.7% "fast" depending on the shape of the product. Batching turns matrix–vector products into matrix–matrix products.

## Worked Example 6.5 — Why decoding is slow

**Setting.** A 7-billion-parameter model in 16-bit precision, batch 1, short context, one A100.

**Computation.** Compute time per token: $1.4\times10^{10}/3.12\times10^{14} = 0.045$ ms. Memory time per token (read all weights): $1.4\times10^{10}/2.039\times10^{12} = 6.87$ ms. The step takes at least 6.87 ms: at most 146 tokens per second.

**Reading.** The arithmetic is 150 times faster than the memory. This answers Question 2 of §6.1: the 22 000 tokens per second suggested by FLOPs was the wrong limit. Batch 64 would raise throughput nearly 64-fold at almost the same latency per step.

## Worked Example 6.6 — A pre-registered scaling fit that missed

**Setting.** Spine model, 2 layers, widths 8–32, each trained for 2 000 steps of 16 × 64 characters (2.05 million tokens). Fit $L = E + AN^{-\alpha}$ to the five results; write the prediction for width 48 to disk; then train width 48.

| width $d$ | 8 | 12 | 16 | 24 | 32 | **48** |
|---|---|---|---|---|---|---|
| non-embedding parameters $N$ | 1 696 | 3 696 | 6 464 | 14 304 | 25 216 | 56 256 |
| held-out NLL | 2.04 | 1.74 | 1.04 | 0.50 | 0.39 | **0.35** |

**Fit on all five:** $E = 0.00$, $\alpha = 0.67$; prediction for width 48: **0.23**. Observed: **0.35** — off by 0.12 nats, far outside any reasonable noise.

**Diagnosis.** The losses do not follow one power law: between widths 12 and 16 the model crosses a capacity threshold below which it cannot represent the story templates at all. Fitted through the transition, the curve plunges and the floor is driven to zero. Fitting only the three widths past the transition gives $E = 0.33$ and predicts 0.34 for width 48 — within 0.01 of the observation.

**Reading.** The law was fine; the choice of points was not. And that choice is exactly the kind of decision that must be made *before* seeing the result — choosing it afterwards to rescue the prediction would be the analytic flexibility Unit 12 warns against. The honest report is: the registered prediction failed; the failure revealed a regime change; a new prediction, with the point-selection rule registered, is needed for width 64. The floor of 0.33 is not the entropy of the stories — it is the best a 2-layer model reaches with 2 million tokens.

## Worked Example 6.7 — The flat compute-optimal optimum

**Setting.** Hoffmann et al.'s approach-3 law: $E = 1.69$, $A = 406.4$, $B = 410.7$, $\alpha = 0.34$, $\beta = 0.28$; budget $C = 5.76\times10^{23}$ FLOPs.

**Computation.** Proposition 6.2: $N^\star = 3.22\times10^{10}$, $D^\star = 2.98\times10^{12}$ (93 tokens per parameter), predicted loss 1.9307. The model actually trained, 70 billion parameters on 1.4 trillion tokens: predicted loss 1.9366.

**Reading.** A factor of 2.2 in model size costs 0.006 nats under this fit. The location of the optimum is sensitive to the fitted constants; its depth is not. This is why different estimation methods (and a later replication, [Besiroglu et al. 2024](/references/#besiroglu-2024)) could disagree about the optimal size by a factor of two without anyone's model being badly trained.

## Worked Example 6.8 — Energy per FLOP, and what Landauer's limit does not say

**Setting.** A100 at 400 W and 40% utilization of 312 TFLOP/s.

**Computation.** $e = 400/(0.4\times3.12\times10^{14}) = 3.2\times10^{-12}$ J per FLOP. GPT-3's $3.14\times10^{23}$ FLOPs → $1.0\times10^{12}$ J ≈ 280 MWh of accelerator energy. [Patterson et al. (2021)](/references/#patterson-2021) estimate 1 287 MWh for the actual run (older V100 hardware, whole data centre). The Landauer limit at 300 K is $kT\ln2 = 2.87\times10^{-21}$ J per bit erased.

**Reading.** Hardware generation and data-centre overheads change the answer by a factor of four. The Landauer figure is *per bit erased*, not per FLOP: turning it into a floor per operation would need a model of how many bits a floating-point multiply–add irreversibly erases in a given circuit, so the ratio of the two numbers is not a measure of how much improvement physics permits. Energy per useful FLOP is an engineering variable, and arithmetic intensity is one of its main levers.
