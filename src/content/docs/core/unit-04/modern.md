---
title: "4.8 Modern developments"
description: IO-aware exact attention with the online softmax, multi-query and grouped-query attention, linear and sparse variants, mixture-of-experts MLPs, and attention beyond text — with evidence labels.
sidebar: { label: "4.8 Modern developments", order: 8 }
---

Labels: **[D]** derivation, **[E]** experimental evidence, **[O]** engineering observation, **[H]** open hypothesis. Each development is tied to a quantity derived in this unit.

## 4.8.1 Exact attention with less memory traffic

**Quantity targeted:** the $T\times T$ weight matrix and the memory traffic of reading and writing it (§4.5.6).

The online softmax of Worked Example 4.9 computes a row of attention block by block, keeping only a running maximum $m$, normalizer $\ell$ and unnormalized output $o$. In general, when a block of keys with scores $s_j$ and values $v_j$ arrives:

$$
m' = \max\bigl(m, \max_js_j\bigr),\qquad
\ell' = e^{m - m'}\ell + \sum_je^{s_j - m'},\qquad
o' = e^{m - m'}o + \sum_je^{s_j - m'}v_j ,
$$

and after the last block the output is $o/\ell$. **Claim:** this equals $\sum_j\mathrm{softmax}_j(s)\,v_j$ exactly. *Proof by induction:* after each block, $\ell = \sum_{j\text{ seen}}e^{s_j - m}$ and $o = \sum_{j\text{ seen}}e^{s_j - m}v_j$ with $m$ the running maximum; the update multiplies the old sums by $e^{m - m'}$, which re-expresses them relative to the new maximum, and adds the new terms. The ratio $o/\ell$ is independent of the reference point $m$. $\blacksquare$ ([Milakov & Gimelshein 2018](/references/#milakov-2018).) **[D]**

**FlashAttention** ([Dao et al. 2022](/references/#dao-2022)) tiles $Q$, $K$ and $V$ into blocks that fit in fast on-chip memory, applies this update, and in the backward pass recomputes the weights from the stored statistics instead of storing $A$. It computes *the same function* — not an approximation — with memory linear in $T$ and substantially less traffic to slow memory, giving large wall-clock speed-ups reported on GPUs **[E]**. Every invariant of the lab must still hold for such a kernel.

## 4.8.2 Sharing keys and values across heads

**Quantity targeted:** KV-cache memory, $2\times L\times t\times n_{\text{kv-heads}}\times d_h$ numbers per sequence (§4.3.10).

**Multi-query attention** keeps many query heads but a *single* key/value head shared by all of them ([Shazeer 2019](/references/#shazeer-2019)); **grouped-query attention** shares each key/value head among a group of query heads ([Ainslie et al. 2023](/references/#ainslie-2023)). The cache shrinks by the ratio of query heads to key/value heads. For a model with 80 layers, head dimension 128 and 64 query heads, in 16-bit precision, the cache per token is 2.62 MB with 64 key/value heads, 0.33 MB with 8 (grouped), and 0.04 MB with 1 (multi-query); at 4 096 tokens, about 10.7 GB, 1.34 GB and 0.17 GB per sequence. **[D]** for the arithmetic. Reported quality loss from grouped sharing is small in the settings tested, which is why it is used in large open models such as the larger Llama 2 variants ([Touvron et al. 2023](/references/#touvron-2023)) **[E]**/**[O]**.

## 4.8.3 Changing the function: sparse, local and linear attention

**Quantity targeted:** the $T^2$ terms themselves. Restricting each query to a local window or a sparse pattern of keys reduces cost to $O(T\cdot w)$ at the price of the complete graph of §4.5.4 (some lookups become impossible within one layer) **[D]**. Replacing $\exp(q^{\top}k)$ by a product of feature maps $\phi(q)^{\top}\phi(k)$ makes attention a recurrence with a matrix-valued state and linear cost ([Katharopoulos et al. 2020](/references/#katharopoulos-2020)) — derived and analysed in Unit 05, §5.3.12, including its capacity limit **[D]**. Whether such models match softmax attention at scale is task-dependent and an **[H]** for general claims; recall-intensive tasks show clear gaps (Unit 05, §5.7.1).

## 4.8.4 Conditional computation in the MLP: mixture of experts

**Quantity targeted:** the $8d^2$ MLP parameters per block, all used for every token. A **mixture-of-experts** layer replaces the single MLP by many expert MLPs and a learned router that sends each token to a few of them ([Shazeer et al. 2017](/references/#shazeer-2017)); Switch Transformers route each token to one expert ([Fedus et al. 2022](/references/#fedus-2022)). Total parameters grow with the number of experts while compute per token stays roughly constant **[D]**; quality gains at fixed compute are reported **[E]**, with engineering costs in load balancing and communication **[O]**. Note the analogy: routing is attention-like *selection* applied to parameters instead of positions.

## 4.8.5 Attention beyond text

Nothing in §4.2–§4.3 is specific to language. **Vision Transformers** split an image into fixed-size patches, embed each as a token with a positional embedding, and apply standard Transformer blocks ([Dosovitskiy et al. 2021](/references/#dosovitskiy-2021)). They lack the translation-equivariance prior of convolutions (Theorem 4.2) and were found to need large training sets to match convolutional networks, after which they performed strongly **[E]** — a direct illustration of the trade-off between built-in symmetry and data. Graph attention (§4.5.4) applies the same mechanism to molecules and networks.

## 4.8.6 What to take from this section

Each development changes one quantity of this unit: **memory traffic** (IO-aware kernels — same function), **cache size** (shared KV heads — slightly different function), **the interaction graph** (sparse/local — different function), **the kernel** (linear attention — different function, bounded memory), **parameters per token** (mixture of experts), or **the domain** (vision, graphs). When you meet a new variant, ask which quantity it changes, whether it computes the same function, and what evidence supports its quality claims.
