---
title: "4.8 Modern developments"
description: IO-aware exact attention, shared KV heads, long-context constraints, sparse and linear variants, mixture-of-experts MLPs, and attention beyond text — with evidence labels.
sidebar: { label: "4.8 Modern developments", order: 8 }
---

Labels: **[D]** derivation, **[E]** experimental evidence, **[O]** engineering observation, **[H]** open hypothesis. Each development is tied to a quantity derived in this unit.

New attention variants are easy to remember as a list of names and hard to understand as design choices. Use three questions instead: **which cost or constraint is being targeted, does the mathematical function change, and what kind of evidence supports the claimed benefit?** The same questions distinguish an exact kernel optimization from a new model architecture.

## 4.8.1 Exact attention with less memory traffic

**Quantity targeted:** the $T\times T$ weight matrix and the memory traffic of reading and writing it (§4.5.6).

The online softmax of Worked Example 4.9 computes one row of attention block by block. It starts from $m=-\infty$, normalizer $\ell=0$ and unnormalized output $o=0$. When a block of keys with scores $s_j$ and values $v_j$ arrives, update

$$
m' = \max\bigl(m, \max_js_j\bigr),\qquad
\ell' = e^{m - m'}\ell + \sum_je^{s_j - m'},\qquad
o' = e^{m - m'}o + \sum_je^{s_j - m'}v_j ,
$$

and after the final block return $o/\ell$. **Claim:** in real arithmetic this equals $\sum_j\mathrm{softmax}_j(s)\,v_j$. *Proof by induction:* after each block, $\ell = \sum_{j\text{ seen}}e^{s_j - m}$ and $o = \sum_{j\text{ seen}}e^{s_j - m}v_j$, with $m$ the largest score seen. Multiplying the old sums by $e^{m-m'}$ merely rewrites them relative to the new maximum before the new terms are added. The common reference factor cancels in $o/\ell$. $\blacksquare$ ([Milakov & Gimelshein 2018](/references/#milakov-2018).) **[D]** Finite-precision evaluation may differ by a few rounding bits because the summation order changes; “exact” here means the intended attention function is not approximated.

**FlashAttention** ([Dao et al. 2022](/references/#dao-2022)) tiles $Q$, $K$ and $V$ into blocks that fit in fast on-chip memory and applies this update. In the backward pass it reconstructs attention blocks from saved row statistics rather than storing the full $A$. The algorithm may recompute values and perform similar or even more arithmetic, yet finish faster because modern accelerators are often limited by transfers to and from slower high-bandwidth memory. It implements the same attention function, uses memory linear rather than quadratic in $T$ for the attention intermediates, and produced large GPU wall-clock speed-ups in the reported experiments. **[E]** A faster kernel does not relax correctness: causal masking, row sums and agreement with a reference remain required invariants.

## 4.8.2 Sharing keys and values across heads

**Quantity targeted:** KV-cache memory, $2\times L\times t\times n_{\text{kv-heads}}\times d_h$ numbers per sequence (§4.3.10).

**Multi-query attention (MQA)** keeps many query heads but uses one key head and one value head shared by all of them ([Shazeer 2019](/references/#shazeer-2019)). **Grouped-query attention (GQA)** is the middle ground: each key/value head serves a group of query heads ([Ainslie et al. 2023](/references/#ainslie-2023)). Query heads can still ask different questions because their $W^Q$ matrices differ; they simply search a shared set of key coordinates and retrieve shared value vectors.

The cache shrinks in proportion to the number of key/value heads. With 80 layers, head dimension 128 and two bytes per stored number,
$$
\text{bytes per token}=\underbrace{2}_{K\text{ and }V}\times80\times n_{\mathrm{kv}}\times128\times\underbrace{2}_{\text{bytes per number}}.
$$
This is 2.62 MB for 64 KV heads, 0.33 MB for 8 and 0.041 MB for 1; at 4,096 tokens the corresponding decimal sizes are about 10.7 GB, 1.34 GB and 0.17 GB per sequence. **[D]** Sharing changes the learned function and can reduce quality, although Ainslie et al. reported a favorable quality--speed trade-off after uptraining, and larger Llama 2 variants use GQA ([Touvron et al. 2023](/references/#touvron-2023)). **[E]**/**[O]** Crucially, MQA and GQA shrink the cache and memory bandwidth; every query head still scores all visible positions, so the sequence-length dependence remains quadratic for dense attention.

## 4.8.3 Changing the function: sparse, local and linear attention

**Quantity targeted:** the $T^2$ pairwise interactions themselves.

If each query sees only a window of $w$ keys, the layer costs $O(Tw)$ rather than $O(T^2)$. A sparse pattern can add selected long-range or global edges. The price is visible in the graph of §4.5.4: a token outside the neighborhood cannot affect the query in one layer, although information may reach it through a path across several layers. **[D]** Thus “sparse” describes both a computational saving and an information-routing assumption.

Another route replaces the exponential similarity by a factorized kernel, $k(q,k)=\phi(q)^{\top}\phi(k)$. In causal attention the sums can then be reassociated into a running matrix state, giving linear sequence cost ([Katharopoulos et al. 2020](/references/#katharopoulos-2020)). This may define a different kernel, or approximate the softmax kernel depending on the chosen features; it is not the exact IO reorganization of §4.8.1. Unit 05, §5.3.12 derives the recurrence and its finite-state capacity limit. **[D]** Whether a particular linear or sparse model matches dense softmax attention is task- and scale-dependent; broad superiority claims require experiments, especially on exact-recall tasks. **[H]**

## 4.8.4 Conditional computation in the MLP: mixture of experts

**Quantity targeted:** the amount of model capacity activated for each token. A dense MLP with expansion factor four has roughly $8d^2$ weight parameters per block, and every token uses all of them. A **mixture-of-experts (MoE)** layer instead provides many expert MLPs plus a learned router that sends each token to only a few experts ([Shazeer et al. 2017](/references/#shazeer-2017)); Switch Transformers use top-1 routing ([Fedus et al. 2022](/references/#fedus-2022)).

If expert width and the number selected per token stay fixed, total parameters can grow with the expert count while active MLP compute per token remains roughly fixed. **[D]** That is conditional computation, not free computation: the router itself costs work, experts must be load-balanced, and distributed experts require communication. Capacity limits may reroute or drop tokens in some implementations. Fixed-compute quality gains reported for Switch Transformers are empirical results, not a consequence of parameter counting alone ([Fedus et al. 2022](/references/#fedus-2022)) **[E]**. The useful analogy is narrow: routing is attention-like *selection* over parameter sets rather than source positions.

## 4.8.5 Attention beyond text

Nothing in §4.2–§4.3 requires tokens to be words. **Vision Transformers** split an image into fixed-size patches, linearly embed each patch, add positional information and apply Transformer blocks ([Dosovitskiy et al. 2021](/references/#dosovitskiy-2021)). A standard ViT does not build in the same local translation-equivariant mixing rule as a convolution (Theorem 4.2). In the original experiments, performance relative to convolutional networks depended strongly on pretraining scale, and large-scale pretraining produced strong results. **[E]** This is a concrete example of a recurring trade-off: a weaker built-in prior offers flexibility but may require more data to learn the useful regularity.

For graphs, the “positions” are nodes and the mask can be the adjacency relation (§4.5.4). For images, audio or multimodal inputs, the attention equations stay recognizable, but tokenization, positional structure and valid symmetries change. Reusing the equation does not make the domains identical.

## 4.8.6 Long context is four different problems

A model that accepts a long input has not necessarily learned to use all of it. Separate four gates:

| gate | question | representative intervention |
|---|---|---|
| position | are unseen or stretched position indices represented sensibly? | RoPE interpolation or retraining |
| computation | can attention be evaluated within the time and device-memory budget? | IO-aware kernels or sequence parallelism |
| cache | can stored keys and values fit and be moved during decoding? | GQA/MQA, quantization or paging |
| use | can the model retrieve, combine and reason over the relevant evidence? | controlled evaluation with distractors and multiple dependencies |

LongRoPE is one example of extending RoPE-based models through non-uniform positional interpolation and limited fine-tuning ([Ding et al. 2024](/references/#longrope-2024)) **[E]**. Ring Attention instead distributes sequence blocks across devices and overlaps movement of key--value blocks with blockwise attention ([Liu et al. 2023](/references/#ring-attention-2023)) **[E]**. They address different gates: the first changes how positions extrapolate; the second changes where exact attention is computed. Neither by itself proves that information near the beginning will influence a difficult answer near the end.

Evaluation must therefore increase both length and task difficulty. RULER extends a simple needle-retrieval test with multiple needles, aggregation and multi-hop tracing; its reported models often deteriorated with length despite high scores on the simplest retrieval test ([Hsieh et al. 2024](/references/#ruler-2024)) **[E]**. A useful engineering checkpoint is immediate: doubling $T$ makes dense attention's pairwise score count about four times larger, while the uncompressed KV cache becomes about twice as large. A single advertised “context length” hides both costs and says nothing about effective recall.

## 4.8.7 What to take from this section

Each development changes one quantity from this unit: **memory traffic** (IO-aware kernels, same intended function), **cache size** (shared KV heads, changed parameter sharing), **position range and sequence placement** (long-context methods), **the interaction graph** (sparse/local attention), **the similarity kernel and state** (linear attention), **active parameters per token** (MoE), or **the domain and its symmetries** (vision and graphs). No single variant “solves attention.” When you meet a new one, identify the changed quantity, recompute the relevant cost, state whether the function changed, and then inspect the evidence for quality—not just the speed headline.
