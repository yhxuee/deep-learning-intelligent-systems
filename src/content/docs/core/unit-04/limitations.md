---
title: "4.7 Limitations and counterexamples"
description: Quadratic cost, the absence of order, length generalization, saturation and attention sinks, rank collapse, what attention weights do not explain, and bugs that survive inspection.
sidebar: { label: "4.7 Limitations", order: 7 }
---

Labels: **[D]** derivation, **[E]** experimental evidence, **[O]** engineering observation, **[H]** open hypothesis.

## 4.7.1 Quadratic cost in sequence length

Scores and mixing cost $4T^2d$ FLOPs per layer and naive storage of the weights $HT^2$ numbers (§4.5.6). Doubling the context quadruples them. **[D]** Long-context work therefore either makes the exact computation cheaper in memory traffic (§4.8.1), reduces the number of key/value heads (§4.8.2), restricts which pairs interact (sparse or local attention), or replaces softmax attention by a recurrence (Unit 05). Each alternative except the first changes the function computed.

## 4.7.2 No order without positions

By Proposition 4.2, unmasked self-attention without positional information gives "dog bites man" and "man bites dog" the same multiset of outputs. **[D]** A causal mask breaks the symmetry partially (each position sees a different prefix), and causal language models trained without any explicit positional encoding have been found to learn positional information this way ([Haviv et al. 2022](/references/#haviv-2022)) **[E]**, but explicit encodings remain standard.

## 4.7.3 Length generalization

A model trained on sequences up to length $T$ often degrades on longer ones. With learned absolute embeddings, positions beyond $T_{\max}$ have no embedding at all **[D]**. With sinusoidal or rotary encodings, longer inputs are well-defined but produce score patterns never seen in training; [Press et al. (2022)](/references/#press-2022) showed that sinusoidal embeddings extrapolate poorly in their experiments and proposed ALiBi, a fixed distance-proportional bias on the scores, which extrapolated better **[E]**. Methods for extending the context of RoPE models after training are an active area **[O]**.

## 4.7.4 Saturation, collapse and attention sinks

If scores grow large — through large weights, missing scaling, or training dynamics — rows of $A$ become nearly one-hot and their gradients vanish (§4.2.8, §4.3.6). **[D]** Trained language models also frequently put a large share of attention on the first token(s) regardless of content; [Xiao et al. (2024)](/references/#xiao-2024) call these **attention sinks**, attribute them to the softmax's requirement that weights sum to one (a head with nothing useful to attend to must still put its weight somewhere), and show that keeping the first tokens in the cache stabilizes streaming generation **[E]**.

## 4.7.5 Rank collapse

Stacks of attention without MLPs and residuals converge toward rank-one outputs — every position the same — doubly exponentially in depth ([Dong et al. 2021](/references/#dong-2021)). **[D]** This is why the block's other components are essential, and a useful check when an ablation "removes the MLP" or "removes the residual".

## 4.7.6 Attention weights are not explanations

It is tempting to read a large weight $A_{ij}$ as "the model used token $j$ for its decision at $i$". [Jain & Wallace (2019)](/references/#jain-2019) found that attention weights often correlate weakly with gradient-based importance and that very different attention distributions can produce the same prediction; [Wiegreffe & Pinter (2019)](/references/#wiegreffe-2019) showed that whether attention can serve as an explanation depends on the definition of explanation and on the test used. **[E]** The weights describe where a head *reads*; whether that information *matters* for the output requires an intervention — the method of Unit 11.

## 4.7.7 Bugs that survive inspection

Five plausible implementation errors — softmax over the wrong axis, a missing $1/\sqrt{d_k}$, masking after the softmax, an off-by-one mask, a wrong head reshape — each produce code that runs, has the right shapes and trains to *some* loss. The lab's bug hunt (`labs/unit-04-attention/buggy_variants.py`) measures which invariant catches each one. One of them, the head reshape, preserves every symmetry tested and is caught only by comparison with an independent reference implementation. **[E]** (measured by `test_bug_hunt.py`). The lesson for working with AI-written code: **reading is not verification; invariants and reference comparisons are.**

## 4.7.8 Numerical range in low precision

Scores $q^{\top}k$ can exceed the range of 16-bit floating point for large $d_k$ or large activations, and $e^{s}$ overflows long before that. Implementations subtract the row maximum before exponentiating (exact, by the softmax's shift invariance) and often accumulate in higher precision. **[D]** for the overflow; **[O]** for specific precision recipes.
