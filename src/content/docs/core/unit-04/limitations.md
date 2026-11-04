---
title: "4.7 Limitations and counterexamples"
description: Quadratic cost, the absence of order, length generalization, saturation and attention sinks, rank collapse, what attention weights do not explain, and bugs that survive inspection.
sidebar: { label: "4.7 Limitations", order: 7 }
---

Labels: **[D]** derivation, **[E]** experimental evidence, **[O]** engineering observation, **[H]** open hypothesis.

The word *limitation* hides several different claims. A quadratic cost follows from counting operations; a missing symmetry follows from the equations; an attention sink is an empirical pattern in trained models; an overflow is an implementation hazard. Keeping these categories separate prevents an observation in one model from being mistaken for a theorem about every Transformer.

## 4.7.1 Quadratic cost in sequence length

For a sequence of length $T$, every query is compared with every key. Computing the scores and then mixing the values costs about $4T^2d$ FLOPs per layer, while an implementation that materializes every attention matrix stores $HT^2$ numbers (§4.5.6). Doubling the context therefore creates four times as many query--key pairs. **[D]** This is an architectural scaling law, not merely an inefficient loop in code.

The remedies solve different parts of the problem. IO-aware exact attention (§4.8.1) computes the same mathematical function without writing the full matrix to slow memory. Multi-query or grouped-query attention (§4.8.2) reduces key/value projections and the KV cache, but it does **not** by itself remove the quadratic number of query--key interactions. Sparse and local attention delete some edges from the interaction graph; recurrent models replace that graph by a compressed state (Unit 05). Those latter choices change the function class or information path, so a faster method is not automatically a drop-in mathematical equivalent.

## 4.7.2 No order without positions

By Proposition 4.2, unmasked self-attention without positional information is permutation equivariant: reordering the input merely reorders the output. Thus "dog bites man" and "man bites dog" produce the same multiset of output vectors, even though their meanings differ. **[D]** Content alone says *what* tokens are present, not *where* they occur.

A causal mask breaks this symmetry only partially. Position $i$ can see $i+1$ tokens, so the size and contents of its prefix provide an indirect position signal. Haviv et al. found that causal language models without an explicit positional encoding learned usable implicit absolute-position information in their experimental settings ([Haviv et al. 2022](/references/#haviv-2022)). **[E]** This is evidence that position can emerge, not a theorem that every architecture, dataset and optimizer will recover it at no cost. Explicit absolute or relative encodings make the intended positional bias available directly and controllably.

## 4.7.3 Length generalization

Two questions must not be confused: **can the model be evaluated at a longer position, and does it still solve the task there?** A learned absolute-embedding table has no entry beyond its configured maximum unless the table is extended. **[D]** Sinusoidal and rotary formulas remain defined at longer positions, but the network may still have learned computations that depend on distances, phases or sequence lengths seen only during training. Being numerically defined is not the same as generalizing.

Press et al. showed that sinusoidal embeddings extrapolated poorly in their experiments and proposed ALiBi, a fixed distance-proportional score bias that extrapolated better in those settings ([Press et al. 2022](/references/#press-2022)). **[E]** Post-training extensions of RoPE likewise trade among rescaling the position variable, modifying frequencies, fine-tuning on longer sequences and preserving short-context behavior. None of these mechanisms guarantees that an algorithm learned on short examples will continue to work on arbitrarily long ones.

## 4.7.4 Saturation, collapse and attention sinks

When the gaps between scores become large, a softmax row approaches a one-hot vector. The entries of the softmax Jacobian then become small, so gradients flowing **through those scores** can shrink (§4.2.8, §4.3.6). **[D]** This is **saturation**. It does not say that every gradient in the whole Transformer is zero: residual paths, value projections and other heads may still carry signal.

An **attention sink** is a different phenomenon. Xiao et al. observed that several trained language models assign disproportionately large attention to initial tokens even when those tokens are not semantically important, and that retaining a few such tokens in a streaming KV cache prevents severe degradation ([Xiao et al. 2024](/references/#xiao-2024)). **[E]** They connect the effect to softmax normalization: a head must distribute total mass one even when no available token is useful. Treat that account as an explanation supported by their experiments, not as the only possible cause in every model. Saturation concerns score sensitivity; a sink concerns where probability mass is routed; rank collapse below concerns representations becoming similar across depth.

## 4.7.5 Rank collapse

Dong et al. proved that, under their assumptions, the residual from a rank-one matrix decreases doubly exponentially with depth in a stack of **pure self-attention layers without skip connections or MLPs** ([Dong et al. 2021](/references/#dong-2021)). **[D]** Informally, token representations become indistinguishable. The scope matters: this is not a theorem that every trained Transformer, every attention variant or every single-component ablation must collapse. It instead explains one pressure that residual paths and nonlinear MLPs can oppose, and tells us what diagnostic to measure when removing them.

## 4.7.6 Attention weights are not explanations

It is tempting to read a large weight $A_{ij}$ as "the model used token $j$ for its decision at $i$". But a head first forms $\sum_j A_{ij}v_j$, then applies an output projection, joins other heads and passes through many later layers. A large weight multiplying an uninformative or nearly zero value can contribute little; many small weights can contribute collectively; later computation can erase or amplify either contribution.

Jain & Wallace found that attention weights often correlate weakly with gradient-based importance and that very different attention distributions can produce the same prediction ([Jain & Wallace 2019](/references/#jain-2019)). Wiegreffe & Pinter showed that the conclusion depends on what counts as an explanation and which test is used ([Wiegreffe & Pinter 2019](/references/#wiegreffe-2019)). **[E]** A weight map is therefore a description of a head's read coefficients, not by itself a causal explanation. To ask whether the read information mattered, intervene: ablate or patch activations, change the attended source under a controlled comparison, and measure the output change (Unit 11).

## 4.7.7 Bugs that survive inspection

Five plausible implementation errors — softmax over the wrong axis, a missing $1/\sqrt{d_k}$, masking after softmax, an off-by-one mask and a wrong head reshape — all produce code that runs, has plausible shapes and trains to *some* loss. Visual inspection therefore supplies weak evidence.

The lab's bug hunt (`labs/unit-04-attention/buggy_variants.py`) records which invariant catches each error. Within this test suite, the wrong head reshape passes every tested symmetry but fails KV-cache equivalence and comparison with an independently written reference implementation. **[E]** (measured by `test_bug_hunt.py`). The claim is deliberately local: another test could catch it too. The general lesson for AI-written as well as human-written code is stronger: **reading is not verification; check mathematical invariants, boundary cases and an independent reference.**

## 4.7.8 Numerical range in low precision

Large activations can make the dot product $q^{\top}k$ large, and exponentiation can overflow at much smaller scores than the dot product format itself can represent. The details depend on the format: `float16` has a much smaller exponent range than `bfloat16`, while optimized kernels often accumulate dot products or evaluate softmax in `float32`. It is therefore too vague to say simply that "16-bit attention overflows."

Stable softmax computes, for each row,
$$
\operatorname{softmax}(s)_j=\frac{e^{s_j-m}}{\sum_\ell e^{s_\ell-m}},\qquad m=\max_\ell s_\ell.
$$
In real arithmetic this is the same function by shift invariance, and its largest exponent is $e^0=1$; finite-precision rounding still remains. **[D]** A separate edge case occurs when every key in a row is masked: subtracting the maximum of an all-$-\infty$ row produces an undefined $-\infty-(-\infty)$. The model or kernel must guarantee at least one visible key or handle the row explicitly. Which operations to promote to higher precision is an implementation choice that must be tested on the target hardware. **[O]**
