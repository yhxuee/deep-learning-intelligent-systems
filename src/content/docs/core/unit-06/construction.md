---
title: "6.3 Mathematical construction"
description: Exact FLOP accounting for a Transformer block with a proof that the instrumented count equals 6N + 12LTd per token, activation memory and recomputation, arithmetic intensity of matrix products and decoding, the roofline, and the KV cache as a memory-traffic problem.
sidebar: { label: "6.3 Construction", order: 3 }
---

## 6.3.1 Every product in a Transformer block

Take the pre-LN decoder of Unit 04 (§4.3.9) with width $d$, $L$ layers, $h$ heads of width $d_h = d/h$, a $4d$-wide MLP, sequence length $T$, vocabulary $V$, tied embeddings. For one sequence, the forward pass of one block performs these products:

| Product | Shape | FLOPs |
|---|---|---|
| $Q, K, V$ projections | $(T\times d)(d\times 3d)$ | $6Td^2$ |
| scores $QK^{\top}$ (all heads) | $h\times(T\times d_h)(d_h\times T)$ | $2T^2d$ |
| mixing $AV$ (all heads) | $h\times(T\times T)(T\times d_h)$ | $2T^2d$ |
| output projection | $(T\times d)(d\times d)$ | $2Td^2$ |
| MLP up | $(T\times d)(d\times 4d)$ | $8Td^2$ |
| MLP down | $(T\times 4d)(4d\times d)$ | $8Td^2$ |

Per block: $24Td^2 + 4T^2d$. The read-out $XE^{\top}$ adds $2TVd$. Embedding lookups are free (Unit 02, Proposition 2.1).

**Proposition 6.1.** With every product's input gradient computed, one training step on $B$ sequences of length $T$ costs exactly
$$
F_{\text{step}} = BT\,\big(6N_{\text{mm}} + 12LTd\big),\qquad N_{\text{mm}} = 12Ld^2 + Vd,
$$
FLOPs in matrix products, where $N_{\text{mm}}$ counts the parameters that enter a product once per token.

*Proof.* Forward: $L(24Td^2 + 4T^2d) + 2TVd = 2T(12Ld^2 + Vd) + 4LT^2d = 2TN_{\text{mm}} + 4LT^2d$ per sequence. Backward: each weight product $Y = XW$ needs $\bar W$ and $\bar X$, each as costly as the forward product, giving $2\times2TN_{\text{mm}}$. Each attention product has two activation inputs; for $S = QK^{\top}$ the backward needs $\bar Q = \bar SK$ and $\bar K = \bar S^{\top}Q$, and for $Y = AV$ it needs $\bar A = \bar YV^{\top}$ and $\bar V = A^{\top}\bar Y$ — again twice the forward cost, $8LT^2d$. Total per sequence: $6TN_{\text{mm}} + 12LT^2d$. Multiply by $B$ and divide by the $BT$ tokens. $\blacksquare$

The spine model counts every product with a wrapper `mm(a, b)` that adds $2mnk$ to a global counter, and `test_model.py` asserts that one `loss_and_grads` call adds *exactly* $BT(6N_{\text{mm}} + 12LTd)$ — a derivation checked to the last FLOP. The **attention fraction** of the training cost is $12LTd/(6N_{\text{mm}} + 12LTd)$; ignoring the read-out it is $T/(T + 6d)$, so attention dominates only when the context is longer than about six times the width. For GPT-style models with $d$ in the thousands and $T$ of a few thousand, attention is a minority of training FLOPs; for long-context models it is not.

## 6.3.2 Activation memory and recomputation

The backward pass needs, for each block, the inputs of every product and nonlinearity: in the spine implementation the cached values are $x_{\text{in}}$, the normalized input, $Q$, $K$, $V$, the attention weights $A$ ($hT^2$ numbers), the head outputs, the MLP input, pre-activation and activation ($4d$ each). Per sequence and layer that is roughly

$$
M_{\text{act}} \approx T\,(c\,d) + hT^2
$$

numbers with $c\approx 16$ for this layout ([Korthikanti et al. 2023](/references/#korthikanti-2023) give the exact count for production layouts). The $hT^2$ term — the attention matrix — is what IO-aware kernels avoid storing (Unit 04, §4.8.1).

**Recomputation** trades FLOPs for memory: store the activations only at some layer boundaries and recompute the rest during the backward pass. With $L$ layers split into segments of $\sqrt L$ layers, peak activation memory falls from $O(L)$ to $O(\sqrt L)$ blocks for one extra forward pass, about a third more FLOPs ([Chen et al. 2016](/references/#chen-2016)). Whether that trade is worth it depends on whether the device is short of memory or of FLOPs — the roofline again.

## 6.3.3 Arithmetic intensity of a matrix product

For $C = AB$ with $A\in\mathbb{R}^{m\times k}$, $B\in\mathbb{R}^{k\times n}$, in $b$-byte numbers, the minimum traffic reads both inputs and writes the output once:

$$
I = \frac{2mnk}{b\,(mk + kn + mn)} .
$$

For square $n\times n$ products in 16-bit numbers, $I = 2n^3/(6n^2) = n/3$: a $4\,096\times4\,096$ product has $I\approx1\,365$ FLOP/byte, far above the A100's 153 — compute-bound. For a **matrix–vector** product ($n = 1$), $I = 2mk/(2(mk + k + m))\approx 1$ FLOP/byte — memory-bound. Everything in between depends on the smallest dimension, which in a Transformer is usually the number of tokens processed together.

## 6.3.4 Why decoding is memory-bound, and what batching buys

During generation with a KV cache (Unit 04, §4.3.10), each step processes one new token per sequence. Every weight matrix is used in a product whose token dimension is the batch size $B$:

$$
I_{\text{decode}} \approx \frac{2BN}{2N + (\text{cache bytes})}\ \text{FLOP/byte}
$$

in 16-bit numbers. With $B = 1$ and a short cache, $I\approx1$; the step time is the time to stream all weights once. Increasing $B$ multiplies the useful FLOPs while the weight traffic stays fixed, so throughput grows almost linearly in $B$ until $I$ reaches the hardware ratio — around $B\approx150$ for an A100 — or until the KV caches fill the memory. The cache itself must also be read every step, and it grows with batch size *and* context length: for long contexts, cache traffic, not weight traffic, becomes the bound ([Pope et al. 2023](/references/#pope-2023)). This is why grouped-query attention (Unit 04, §4.8.2) — which shrinks the cache by sharing keys and values — speeds up generation even though it saves few FLOPs.

## 6.3.5 Utilization

How close does a real training run get to peak? The standard measure is **model FLOPs utilization** ([Chowdhery et al. 2023](/references/#chowdhery-2023)): the FLOPs the model *needs* (the $6N + 12LTd$ per token of Proposition 6.1, excluding recomputation) times tokens per second, divided by peak FLOP/s. Well-tuned large runs report MFU around 40–60% **[O]**; the rest is lost to memory-bound operations (normalization, softmax, elementwise), communication between devices, and pipeline idle time. The spine model, written in NumPy on a CPU, reaches a few percent of its machine's peak — Experiment 6.2 measures it.

## 6.3.6 Energy

Energy follows from time and power. A device drawing $P$ watts at utilization $u$ of peak rate $\pi$ spends

$$
e = \frac{P}{u\,\pi}\ \text{joules per FLOP}.
$$

For an A100 at 400 W, $u = 0.4$: $e\approx3.2\times10^{-12}$ J/FLOP. Multiplying by a training run's FLOPs gives the accelerator energy; data-centre overheads (cooling, power conversion, networking) and the carbon intensity of the electricity turn it into the numbers reported for real runs ([Patterson et al. 2021](/references/#patterson-2021)). Worked Example 6.8 compares the estimate with a published figure and with the thermodynamic minimum.
