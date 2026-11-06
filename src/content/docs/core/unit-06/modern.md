---
title: "6.8 Modern developments"
description: How IO-aware kernels, lower precision, serving systems, sparse experts, scarce data, and test-time compute change the resource equations—and where each promised speed-up can fail.
sidebar: { label: "6.8 Modern developments", order: 8 }
---

This is a selective literature snapshot checked through **4 November 2026**, not a leaderboard. Hardware, kernels and model recipes change quickly; the durable skill is to ask **which term in the cost model changed**. Labels: **[D]** derivation, **[E]** experimental evidence, **[O]** engineering observation, **[H]** open hypothesis.

## 6.8.1 Moving fewer bytes: IO-aware kernels

Recall the roofline lower bound

$$
t\ge \max\!\left(\frac{F}{\pi},\frac{M}{\beta}\right).
$$

If an operation is limited by bytes moved $M$, reducing its FLOPs may do little. FlashAttention attacks $M$ instead: it computes attention in blocks that fit in fast on-chip memory and avoids materializing the full $T\times T$ score and probability matrices in slower high-bandwidth memory ([Dao et al. 2022](/references/#dao-2022)). The forward pass still computes **exact softmax attention**, up to ordinary floating-point roundoff. Its backward pass reconstructs blocks from saved row statistics, so it may perform extra arithmetic and still finish sooner because it moves far fewer bytes **[D]/[E]**.

That example suggests a general pattern: fuse normalization, activation or optimizer operations so their intermediate arrays stay on chip **[O]**. But “fused” is not a correctness argument. A replacement kernel must still preserve causal masking, stable softmax, gradient agreement and acceptable numerical error. Faster wrong answers are not an optimization.

**Beginner checkpoint.** FlashAttention does not make dense attention linear in sequence length. It changes memory traffic and stored intermediates; the attention arithmetic is still quadratic in $T$.

## 6.8.2 Fewer bits per number

Lower precision can change three different quantities:

1. **stored bytes** for weights, optimizer states and activations;
2. **moved bytes** between memory levels; and
3. **peak arithmetic rate**, because accelerators often execute low-precision matrix products faster.

Mixed-precision training commonly uses 16-bit tensors for many forward and backward operations while retaining selected 32-bit values for stability ([Micikevicius et al. 2018](/references/#micikevicius-2018)) **[E]**. The traffic for eligible tensors can be roughly half that of all-FP32 computation, but total training memory does **not** automatically halve: master weights, optimizer moments, gradients and scaling metadata depend on the recipe. Newer 8-bit and FP8 recipes push the same trade-off further. For example, the DeepSeek-V3 report combines FP8 mixed-precision training with a sparse-expert model ([DeepSeek-AI et al. 2024](/references/#deepseek-v3-2024)) **[O]**; it is a system-level example, not proof that every model trains safely in FP8.

For inference, **post-training quantization** compresses an already trained model. LLM.int8() treats rare large activation dimensions specially ([Dettmers et al. 2022](/references/#dettmers-2022)); GPTQ uses approximate second-order information to choose 3–4-bit weight errors ([Frantar et al. 2023](/references/#frantar-2023)). Both report small quality losses in their tested model-and-benchmark ranges **[E]**. Neither says that four bits are universally lossless.

Suppose batch-1 decoding is purely limited by reading weights. Replacing 16-bit weights by 4-bit weights divides the idealized weight traffic by four, so the **weight-bandwidth ceiling** rises by at most $4\times$ **[D]**. Real speed-up is usually smaller because scales and zero points occupy space, dequantization costs work, KV-cache and activation traffic remain, kernels may not reach peak rate, and some layers lose accuracy unless kept at higher precision.

## 6.8.3 Serving: pages and speculative decoding

Serving has two recurrent wastes: unused cache space and repeated weight reads.

**Paged KV caches.** Requests end at unpredictable lengths. Reserving one contiguous maximum-size cache per request leaves holes. PagedAttention stores cache blocks through a page table, analogous to virtual memory, so blocks can be allocated and shared as needed ([Kwon et al. 2023](/references/#kwon-2023)) **[E]**. This greatly reduces—not literally eliminates—fragmentation and can admit larger effective batches. Larger batches reuse each weight read across more tokens, raising arithmetic intensity.

**Speculative decoding.** A small draft model proposes several tokens. The large target model scores the proposed block in one parallel pass; an acceptance-and-correction rule then returns exactly the target model's distribution ([Leviathan et al. 2023](/references/#leviathan-2023)) **[D]**. Unit 08 derives that rule.

Why might this be faster? During memory-bound decoding, checking several positions can require a similar **target-weight** read to checking one. But total traffic is not identical: the draft model, KV caches, verification tensors and rejected proposals all cost something. Measured speed-up therefore depends on draft acceptance, block length, verification kernels, batching and cache overhead ([Leviathan et al. 2023](/references/#leviathan-2023)) **[E]**. A poor draft can make speculation slower. “Exact” describes the corrected output distribution, not arbitrary greedy acceptance of draft tokens.

## 6.8.4 More total parameters without proportional active FLOPs

A mixture-of-experts (MoE) layer has many expert MLPs but routes each token to only a few. If there are $E$ experts and $k\ll E$ are active, expert capacity can grow roughly with $E$ while expert arithmetic per token grows with $k$ **[D]**. This is why DeepSeek-V3 reports 671 billion total parameters but 37 billion activated per token ([DeepSeek-AI et al. 2024](/references/#deepseek-v3-2024)) **[O]**.

The shortcut “parameters without FLOPs” is memorable but incomplete. Attention, routing, shared layers and communication still cost work; all expert weights still require storage; uneven routing can overflow capacity or leave devices idle. Scaling reports must therefore separate **total parameters**, **active parameters**, **FLOPs/token**, **inter-device traffic** and **measured throughput**. MoE changes the constraint—it does not remove constraints.

## 6.8.5 When unique data run short

The variable $D$ in $L(N,D)$ counts training tokens, but the value of the next token depends on whether it is new, repeated, contaminated, mislabeled or generated. In the regimes tested by [Muennighoff et al. (2023)](/references/#muennighoff-2023), repeating data for up to roughly four epochs caused little loss relative to using the same number of unique tokens; returns then diminished, and the useful amount depended on the compute regime and data mixture **[E]**. “Four epochs are free” is therefore not a universal law.

Three responses target different bottlenecks:

- **deduplicate and curate** to raise information per token;
- **repeat high-quality data** when another unique token is unavailable;
- **generate synthetic data** to cover tasks or provide feedback that the raw corpus lacks.

Synthetic data are not automatically diverse or correct. Repeated self-training can amplify omissions and errors, while verified synthetic mathematics or code may be unusually valuable because answers can be checked. The source model, filtering rule, verifier and mixture weight are part of $D$'s provenance, not implementation trivia.

## 6.8.6 Spending compute after training

Training compute buys a fixed model. **Test-time scaling** spends a variable budget per problem: sample several answers and vote, search with a verifier, revise an answer, or generate a longer reasoning trace. The relevant curve is now

$$
\text{quality}=q(\text{model},\text{problem},C_{\text{test}}),
$$

not simply quality versus parameter count.

[Snell et al. (2024)](/references/#snell-2024) found regimes where allocating test-time compute according to problem difficulty was more efficient than uniform best-of-$N$, and where a smaller model beat a model 14 times larger under a FLOP-matched comparison on problems the smaller model could already solve nontrivially **[E]**. That qualification matters: extra samples cannot rescue a model whose candidates contain no useful signal. A later comparison across eight open models and four reasoning datasets found no universally best strategy; the best choice varied with model type, difficulty and budget ([Agarwal et al. 2025](/references/#agarwal-2025)) **[E]**.

Training can also change how the model uses this budget. DeepSeek-R1 reports reinforcement-learning runs in which longer verification and self-correction patterns emerged on verifiable tasks ([DeepSeek-AI et al. 2025](/references/#deepseek-r1-2025)) **[O]**. This is evidence for one training recipe, not proof that longer text is itself reasoning or that reinforcement learning always produces it. Unit 09 studies the learning signal; this unit asks who pays the inference bill.

## 6.8.7 A decision table, not a bag of tricks

| Intervention | Primarily reduces | What may increase | Typical failure mode |
|---|---|---|---|
| IO-aware fusion | intermediate memory traffic | recomputation, kernel complexity | numerical or masking mismatch |
| lower precision | bytes/value; sometimes time/FLOP | metadata and conversion | accuracy loss or overflow |
| paged KV cache | wasted reserved cache space | address-management overhead | tiny pages or poor batching |
| speculative decoding | target weight reads per accepted token | draft and rejection work | low acceptance rate |
| mixture of experts | active compute per unit of total capacity | storage and communication | load imbalance |
| data repetition/synthesis | need for new raw examples | correlation and filtering burden | error amplification |
| test-time scaling | training needed for a fixed quality target | latency and serving cost | weak verifier or useless candidates |

For a new optimization, do not begin with “How many times faster is it?” Begin with four questions: **Which term—$F$, $M$, capacity, or answer-time budget—changed? What new cost appeared? Under what workload was it measured? Which correctness or quality invariant was checked?** Those questions survive the next paper cycle.
