---
title: "6.10 Problems and further study"
description: Twenty graded problems on FLOP and memory accounting, arithmetic intensity, scaling laws, compute-optimal allocation and parallel training, with selected answers and an annotated reading list.
sidebar: { label: "6.10 Problems", order: 10 }
---

Difficulty: ★ routine · ★★ combines ideas · ★★★ proof or open-ended. **[AI-off]** problems are for independent practice and assessment.

## Computation

**6.1 ★ [AI-off]** How many FLOPs does it take to train a 1-billion-parameter model on 20 billion tokens (ignore attention)? At 40% utilization of an A100, how many GPU-days is that?

**6.2 ★ [AI-off]** A model has $d = 4\,096$, $L = 32$, $V = 32\,000$, tied embeddings. Compute $N_{\text{mm}}$ and the attention fraction of training FLOPs at $T = 2\,048$ and at $T = 32\,768$.

**6.3 ★** Compute the training-state memory (16 bytes per parameter) for 1B, 7B and 70B parameters. How many 80 GB devices does each need just for that state, if it is partitioned evenly?

**6.4 ★** Compute the arithmetic intensity of a $B\times4\,096$ by $4\,096\times4\,096$ product in 16-bit numbers for $B = 1, 16, 256$. Which are compute-bound on an A100?

**6.5 ★★** A 13-billion-parameter model in 16-bit precision is served on one A100 with a context of 2 048 tokens, 40 layers, $d = 5\,120$, no grouped-query attention. Compute the KV-cache bytes per sequence and the largest batch that fits next to the weights in 80 GB.

**6.6 ★★** Using the approach-3 constants of §6.5.1, compute $N^\star$, $D^\star$ and the predicted loss for $C = 10^{21}$ and $C = 10^{24}$. How do tokens per parameter change?

## Derivation

**6.7 ★ [AI-off]** Derive the FLOPs of the backward pass of $Y = XW$ and show it is twice the forward pass.

**6.8 ★★ [AI-off]** Prove Proposition 6.1, including the backward cost of the two attention products.

**6.9 ★★ [AI-off]** Prove Proposition 6.2. Then show that the minimum loss at the optimum scales as $L^\star - E\propto C^{-\alpha\beta/(\alpha+\beta)}$.

**6.10 ★★** Derive the bytes sent per device by a ring all-reduce of $G$ bytes on $p$ devices.

**6.11 ★★** For a Transformer block, derive the condition on $T$ and $d$ under which attention FLOPs exceed MLP FLOPs.

## Proof and analysis

**6.12 ★★** Show that with $\sqrt L$ checkpoints evenly spaced, activation memory is $O(\sqrt L)$ blocks and recomputation costs at most one extra forward pass.

**6.13 ★★** Show that $f(N) = AN^{-\alpha} + B(C/6)^{-\beta}N^{\beta}$ is convex in $u = \ln N$, so the stationary point of Proposition 6.2 is the global minimum.

**6.14 ★★** Expand $f$ to second order around $N^\star$ in $u = \ln N$ and compute the loss increase for $N = 2N^\star$ with the approach-3 constants at $C = 5.76\times10^{23}$. Compare with Worked Example 6.7.

**6.15 ★★★** Using the manifold heuristic of §6.5.4, derive the exponent for a piecewise-constant approximation and for a piecewise-polynomial approximation of degree $k$. What do measured exponents near 0.1 imply about $d_{\text{int}}$ under each?

**6.16 ★★★** Amdahl's law assumes a fixed problem. Derive Gustafson's law for a problem that grows with $p$, and explain which law describes data-parallel training with a fixed per-device batch.

## Implementation and design

**6.17 ★ [AI-off]** Write a function that returns training FLOPs per token, training-state bytes and KV-cache bytes per token for any decoder configuration. Check it against `model.FLOPS` and `kv_cache_bytes`.

**6.18 ★★** Implement the fit of §6.5.2 with uncertainty: bootstrap the five points (Unit 10) and report an interval for the prediction at width 48.

**6.19 ★★** Measure how the spine model's held-out NLL at fixed width depends on training tokens, and fit $L(D)$. Combine with Experiment 6.4 to fit $L(N, D)$ on the spine corpus.

**6.20 ★★★** You must serve a model to 10 000 users with at most 100 ms per token. Design the choice of model size, precision, batch size and cache strategy, stating the arithmetic for each decision.

## Selected answers

- **6.1** $6\times10^9\times2\times10^{10} = 1.2\times10^{20}$ FLOPs; at $0.4\times312\times10^{12}$ FLOP/s that is $9.6\times10^5$ s $\approx11$ GPU-days.
- **6.3** 16, 112 and 1 120 GB: one, two and fourteen 80 GB devices.
- **6.4** $I = 2\cdot B\cdot4096^2/(2(4096B + 4096^2 + 4096B))\approx B/(1 + 2B/4096)$: about 1, 16 and 228 FLOP/byte; only $B = 256$ exceeds 153.
- **6.9** Substituting $N^\star$ and $D^\star$ makes both terms proportional to $C^{-\alpha\beta/(\alpha+\beta)}$; with $\alpha = 0.34$, $\beta = 0.28$ the exponent is 0.154.
- **6.10** Reduce-scatter and all-gather each send $(p-1)/p\cdot G$: total $2G(p-1)/p$.

## Further reading

**Accounting.** [Kaplan et al. 2020](/references/#kaplan-2020), §2 — the FLOP and parameter counts used everywhere. [Korthikanti et al. 2023](/references/#korthikanti-2023) — exact activation memory of production Transformers.

**Scaling laws.** [Hoffmann et al. 2022](/references/#hoffmann-2022) and the replication by [Besiroglu et al. 2024](/references/#besiroglu-2024) — read together, as a lesson in fitting and reporting. [Muennighoff et al. 2023](/references/#muennighoff-2023) — what happens when data run out.

**Systems.** [Williams et al. 2009](/references/#williams-2009) — the roofline. [Pope et al. 2023](/references/#pope-2023) — inference costs analysed with exactly the tools of §6.3. [Rajbhandari et al. 2020](/references/#rajbhandari-2020) and [Narayanan et al. 2021](/references/#narayanan-2021) — how training is split across thousands of devices.

**Costs beyond FLOPs.** [Patterson et al. 2021](/references/#patterson-2021) — energy and emissions of large training runs, and how much depends on where and how they run.
