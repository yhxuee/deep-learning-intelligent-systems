---
title: "6.8 Modern developments"
description: IO-aware kernels, low-precision arithmetic and quantization, memory management and speculative decoding for inference, mixture of experts, data-constrained scaling, and scaling compute at test time — each tied to a quantity of this unit.
sidebar: { label: "6.8 Modern developments", order: 8 }
---

Labels: **[D]** derivation, **[E]** experimental evidence, **[O]** engineering observation, **[H]** open hypothesis.

## 6.8.1 Moving fewer bytes: IO-aware kernels

**Quantity:** bytes moved, $M$ in $t\ge M/\beta$.

FlashAttention computes exact attention block by block in fast on-chip memory, never writing the $T\times T$ matrix to slow memory (Unit 04, §4.8.1) **[D]/[E]**. The FLOPs are unchanged (slightly increased by recomputation in the backward pass); the time falls because the operation's arithmetic intensity rises ([Dao et al. 2022](/references/#dao-2022)). The same principle — fuse operations so intermediates never leave fast memory — is applied to normalization, activation functions and optimizer updates **[O]**.

## 6.8.2 Fewer bits per number

**Quantity:** bytes per parameter in memory and traffic.

Training in 16-bit formats with 32-bit master weights halves memory traffic relative to 32-bit training ([Micikevicius et al. 2018](/references/#micikevicius-2018)) **[E]**; 8-bit training formats push further. For inference, weights can be quantized after training: 8-bit matrix multiplication with special handling of rare large activations preserves quality at large scale ([Dettmers et al. 2022](/references/#dettmers-2022)), and 3–4-bit weight quantization using second-order information loses little on many benchmarks ([Frantar et al. 2023](/references/#frantar-2023)) **[E]**. Because batch-1 decoding is memory-bound (§6.3.4), 4-bit weights make it up to about four times faster than 16-bit weights **[D]**.

## 6.8.3 Serving: memory management and speculative decoding

**Quantity:** KV-cache bytes; tokens per weight read.

Serving systems allocate KV caches in fixed-size pages, like virtual memory, so that many sequences of unpredictable length share the device without fragmentation ([Kwon et al. 2023](/references/#kwon-2023)) **[E]**; larger effective batches raise arithmetic intensity (§6.3.4). **Speculative decoding** lets a small model propose several tokens that the large model then checks in one forward pass; a modified rejection-sampling rule keeps the output distribution *exactly* that of the large model ([Leviathan et al. 2023](/references/#leviathan-2023)) **[D]**, and because the large model's pass over several tokens costs about the same memory traffic as over one, wall-clock time falls **[E]**. Unit 08 derives the acceptance rule.

## 6.8.4 Parameters without FLOPs: mixture of experts

**Quantity:** the ratio of parameters to FLOPs per token.

A mixture-of-experts layer routes each token to a few of many expert MLPs (Unit 04, §4.8.4), so total parameters grow while FLOPs per token stay nearly fixed **[D]**. Scaling laws must then be stated in terms of both total and active parameters; memory and communication, not FLOPs, become the binding costs **[O]**.

## 6.8.5 When data are limited

**Quantity:** $D$ in $L(N, D)$ when unique data are scarce.

[Muennighoff et al. (2023)](/references/#muennighoff-2023) extended scaling laws to repeated data: up to about four epochs, repeated tokens are worth nearly as much as new ones; beyond that their value decays quickly **[E]**. Generating synthetic training data with models and curating data more aggressively are active responses; training on a model's own outputs repeatedly can degrade the distribution (Unit 07, §7.8) **[E]**.

## 6.8.6 Spending compute at test time

**Quantity:** FLOPs spent per answer rather than per training token.

A fixed model can be made more accurate on some tasks by spending more inference compute: sampling several answers and choosing by a verifier or by majority vote, or generating longer chains of intermediate reasoning. [Snell et al. (2024)](/references/#snell-2024) found that, for some problems and budgets, optimally allocated test-time compute outperforms a model many times larger **[E]**. Training-time and test-time compute have become two axes of one budget; Unit 09 shows how reinforcement learning against a verifier trains models to use the second axis well **[O]**.
