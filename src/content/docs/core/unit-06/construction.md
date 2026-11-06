---
title: "6.3 Mathematical construction"
description: Exact FLOP accounting for a Transformer block with a proof that the instrumented matrix-product count equals 6N_mm + 12LTd per token, activation memory and recomputation, arithmetic intensity of matrix products and decoding, the roofline, and the KV cache as a memory-traffic problem.
sidebar: { label: "6.3 Construction", order: 3 }
---

Section 6.2 gave first estimates. We now open one Transformer block and account for every large matrix product. Keep three ledgers separate while reading:

1. **per sequence** versus **per token** versus **per training step**;
2. **matrix-product FLOPs** versus operations omitted from that convention;
3. **stored bytes** versus **bytes moved**.

Most mistakes in systems estimates are not difficult algebra. They are a correct number attached to the wrong ledger.

## 6.3.1 Every product in a Transformer block

Take the pre-LN decoder of Unit 04 (§4.3.9): width $d$, $L$ layers, $h$ heads of width $d_h=d/h$, a $4d$-wide MLP, sequence length $T$, vocabulary size $V$, and tied input/output embeddings. We begin with **one sequence and one block**. The batch factor will be added only after the single-sequence count is understood.

Each table row follows the rule from §6.2.1: multiply the three matrix dimensions and then multiply by 2.

| Product | Shape | FLOPs |
|---|---|---|
| fused $Q, K, V$ projections | $(T\times d)(d\times 3d)$ | $6Td^2$ |
| scores $QK^{\top}$ (all heads) | $h\times(T\times d_h)(d_h\times T)$ | $2T^2d$ |
| mixing $AV$ (all heads) | $h\times(T\times T)(T\times d_h)$ | $2T^2d$ |
| output projection | $(T\times d)(d\times d)$ | $2Td^2$ |
| MLP up | $(T\times d)(d\times 4d)$ | $8Td^2$ |
| MLP down | $(T\times 4d)(4d\times d)$ | $8Td^2$ |

The four weight projections contribute

$$
6Td^2+2Td^2+8Td^2+8Td^2=24Td^2.
$$

The two attention products contribute $4T^2d$. Hence one block costs $24Td^2+4T^2d$ in the forward pass. After $L$ blocks, the vocabulary readout $XE^\top$ adds $2TVd$.

An input embedding **lookup contributes no dense matrix-product FLOPs**, because it selects rows instead of multiplying by the entire table (Unit 02, Proposition 2.1). It is not “free”: it still reads memory, and the embedding table still occupies capacity. This is the first example of why a FLOP ledger cannot replace a byte ledger.

:::note[Check one row yourself]
For the MLP up-projection, there are $T\cdot4d$ output numbers. Each is a length-$d$ dot product, about $2d$ FLOPs. Therefore the cost is $(4Td)(2d)=8Td^2$. Reconstruct at least one other table row this way before using the total.
:::

**Proposition 6.1 (matrix-product count).** Under the $2mnk$ convention, with every listed product's input gradient computed, one training step on $B$ sequences of length $T$ costs exactly
$$
F_{\text{step}} = BT\,\big(6N_{\text{mm}} + 12LTd\big),\qquad N_{\text{mm}} = 12Ld^2 + Vd,
$$
FLOPs **in the listed matrix products**, where $N_{\text{mm}}$ counts weight parameters that enter one product per token. Biases, normalization, softmax, nonlinearities, loss evaluation, optimizer updates and embedding lookup traffic are outside this equality.

*Proof, with the two kinds of product kept separate.*

**1. Weight products, forward.** Across all blocks and the vocabulary readout,

$$
L(24Td^2)+2TVd=2T(12Ld^2+Vd)=2TN_{\text{mm}}.
$$

**2. Activation--activation attention products, forward.** Scores and value mixing cost

$$
L(2T^2d+2T^2d)=4LT^2d.
$$

So the forward total for one sequence is $2TN_{\text{mm}}+4LT^2d$.

**3. Backward.** Every listed matrix product has two differentiable inputs. For a weight product $Y=XW$, backward computes $\bar W=X^\top\bar Y$ and $\bar X=\bar YW^\top$; together they cost twice its forward product. For $S=QK^\top$, backward computes $\bar Q=\bar SK$ and $\bar K=\bar S^\top Q$; for $Y=AV$, it computes $\bar A=\bar YV^\top$ and $\bar V=A^\top\bar Y$. These also cost twice their respective forward products.

Therefore forward plus backward is three times the forward matrix-product count:

$$
6TN_{\text{mm}}+12LT^2d
$$

per sequence. Multiplying by $B$ gives the stated per-step total. Dividing that total by the $BT$ processed tokens gives $6N_{\text{mm}}+12LTd$ per token. $\blacksquare$

The spine model wraps each matrix product in `mm(a, b)`, which adds $2mnk$ to a counter. `test_model.py` asserts that one `loss_and_grads` call adds exactly $BT(6N_{\text{mm}}+12LTd)$. The test verifies that implementation and proposition agree on the **declared products**; it does not claim that the CPU executed no other instructions.

The attention share of the counted training FLOPs is

$$
f_{\text{attn}}=\frac{12LTd}{6N_{\text{mm}}+12LTd}.
$$

If the vocabulary readout is temporarily ignored, $N_{\text{mm}}=12Ld^2$ and

$$
f_{\text{attn}}=\frac{T}{T+6d}.
$$

This fraction exceeds one half only when $T>6d$. That statement is about this dense block, this MLP ratio and matrix-product FLOPs. It is not a universal threshold for wall-clock time: attention may move or materialize enough data to be slow even before it dominates the FLOP count.

## 6.3.2 Activation memory and recomputation

Why not discard every intermediate as soon as the forward pass has used it? Because backward needs local information. To compute $\bar W=X^\top\bar Y$, it needs the old $X$; to differentiate an activation, it needs either the pre-activation or a value from which its derivative can be recovered. Autodiff's “tape” is a collection of such saved values.

For each block, the spine implementation caches $x_{\text{in}}$, normalized inputs, $Q$, $K$, $V$, the attention weights $A$, head outputs, the MLP input, pre-activation and activation. Their sizes have two characteristic forms:

- hidden and projected values contribute terms proportional to $Td$;
- an explicitly materialized attention matrix contributes $hT^2$ values.

Thus, per sequence and layer, a useful first model is

$$
M_{\text{act}} \approx T\,(c\,d) + hT^2
$$

stored **numbers**, with $c\approx16$ for this particular educational layout. To obtain bytes, multiply by batch size $B$, layers $L$, and bytes per stored number $b$:

$$
\text{activation bytes}\approx BLb\,(cTd+hT^2).
$$

Do not treat $c=16$ as architectural law. Fused kernels, which values a derivative reconstructs, padding, dropout and the exact MLP change it; [Korthikanti et al. (2023)](/references/#korthikanti-2023) give detailed counts for production Transformer layouts. The formula is valuable because it exposes scaling before implementation details are known.

The $hT^2$ term is especially dangerous at long context. Doubling $T$ doubles the $Td$ terms but quadruples an explicitly stored attention matrix. IO-aware exact-attention kernels such as the one introduced in Unit 04 (§4.8.1) tile the computation and avoid writing the whole matrix to slow memory. They still compute exact attention; they change the storage and traffic schedule.

**Recomputation**, also called activation checkpointing, deliberately forgets some forward intermediates. Save only selected layer boundaries; during backward, rerun the forward operations between two checkpoints to recreate what is needed.

Why can doing *more* arithmetic make training faster? If the uncheckpointed model does not fit, it cannot run at all. If it fits only with a smaller batch or costly device communication, extra local computation may be cheaper than the avoided memory pressure. In the classical square-root schedule, splitting $L$ layers into about $\sqrt L$ segments reduces peak saved blocks from $O(L)$ to $O(\sqrt L)$ while adding roughly one extra forward pass—about one third on top of a forward-plus-backward matrix-product budget ([Chen et al. 2016](/references/#chen-2016)). Exact overhead depends on what is checkpointed and whether recomputed kernels are themselves memory- or compute-bound.

The decision is therefore not “memory good, FLOPs bad.” It is: which resource is binding on this hardware for this workload?

## 6.3.3 Arithmetic intensity of a matrix product

For $C=AB$ with $A\in\mathbb{R}^{m\times k}$, $B\in\mathbb{R}^{k\times n}$ and $b$ bytes per number, an optimistic traffic lower bound reads both inputs once and writes the output once:

$$
I = \frac{2mnk}{b\,(mk + kn + mn)} .
$$

The numerator is useful work. The denominator inventories values: $mk$ from $A$, $kn$ from $B$, and $mn$ written to $C$. This is a **best case for the chosen memory level**. A poor tiling may reread data; a cache may supply some bytes without returning to high-bandwidth memory; accumulation precision may use a different byte width. Arithmetic intensity is a property of an algorithm *and its data-movement schedule*, not just its mathematical function.

Two limiting cases create the key intuition:

- **Large square product, 16-bit.** For three $n\times n$ matrices, $I=2n^3/(6n^2)=n/3$. At $n=4,096$, $I\approx1,365$ FLOP/byte, above the A100 balance point of 153. A well-tiled kernel can be compute-bound because each loaded tile is reused many times.
- **Matrix--vector product, 16-bit.** Set the output-column dimension to 1: $I=2mk/[2(mk+k+m)]\approx1$ FLOP/byte when $mk$ dominates. Almost every weight is fetched to perform only one multiply and one addition. More arithmetic units do little unless reuse is created.

Transformer operations lie between these cases. The reuse dimension is often the number of token rows processed together—batch times sequence in training, but only batch during one cached decoding step.

## 6.3.4 Why decoding is memory-bound, and what batching buys

Inference has two phases that should not be mixed:

- **prefill** processes all prompt tokens together; its large token dimension creates matrix--matrix products and substantial reuse;
- **cached decoding** processes one new token per active sequence at each step; the token-row dimension of each weight product is only the batch size $B$.

During cached decoding (Unit 04, §4.3.10), a deliberately simplified upper-bound model gives

$$
I_{\text{decode}} \lesssim \frac{2BN_{\mathrm{mm}}}{2N_{\mathrm{mm}} + M_{\mathrm{KV}} + M_{\mathrm{other}}}
\quad\text{FLOPs/byte}
$$

for 16-bit weights. The numerator counts dense-weight arithmetic for $B$ token rows. The denominator includes one idealized read of the weights, KV-cache traffic $M_{\mathrm{KV}}$, and other unavoidable traffic. The inequality reminds us that extra rereads and intermediate writes only reduce realized intensity.

With $B=1$ and a short cache, the weight-only approximation gives $I\approx1$. If $B$ independent sequences use the weights together, the numerator initially grows with $B$ while one weight read is shared. Throughput can then grow almost linearly until one of several limits arrives:

1. arithmetic intensity approaches the hardware balance point—weight-only arithmetic suggests a batch of order 150 on the A100 example, not an exact threshold;
2. the KV caches and activations consume device capacity;
3. requests have different lengths, leaving some batch slots idle;
4. the service's latency target forbids waiting for a large batch.

The KV cache must also be read during attention, and its size grows with batch size, cached length, layer count and KV-head width. At long context, cache traffic can overtake weight traffic ([Pope et al. 2023](/references/#pope-2023)). Grouped-query attention (Unit 04, §4.8.2) shares keys and values across query heads, reducing cache capacity and traffic even when the saved FLOP count is modest. This is a recurring systems lesson: removing bytes can matter more than removing arithmetic.

## 6.3.5 Utilization

Peak specifications become useful only after we ask how much of them the workload sustains. **Model FLOPs utilization (MFU)** is commonly defined as ([Chowdhery et al. 2023](/references/#chowdhery-2023))

$$
\mathrm{MFU}=
\frac{(\text{model FLOPs per token})(\text{tokens per second})}
     {(\text{number of devices})(\text{peak FLOP/s per device})}.
$$

The numerator uses the model's useful forward-and-backward count, usually excluding activation recomputation. Suppose checkpointing adds work but token throughput stays fixed: hardware FLOPs utilization has risen, while MFU as defined above does not. MFU therefore measures useful-model throughput relative to peak, not the fraction of all executed instructions that were busy.

Why is MFU below 100%? Small or awkward products underfill matrix units; normalization, softmax and elementwise kernels move data with few FLOPs; devices communicate gradients and activations; pipeline stages wait; input loading or compilation may stall. Well-tuned large training runs have reported roughly 40–60% MFU **[O]**, but the number is specific to model, precision, device count and accounting convention. A NumPy spine model on a CPU is an educational implementation, not a performance baseline. Experiment 6.2 asks you to measure it rather than inherit a percentage.

## 6.3.6 Energy

Energy is power integrated over time. If a device draws average power $P$ watts and delivers useful model arithmetic at $u\pi$ FLOP/s, where $u$ is the chosen useful-FLOP utilization and $\pi$ is peak rate, then

$$
e = \frac{P}{u\,\pi}\ \text{joules per FLOP}.
$$

The units verify the formula: watts are joules per second, so

$$
\frac{\mathrm{J/s}}{\mathrm{FLOP/s}}=\mathrm{J/FLOP}.
$$

Using the illustrative A100 values $P=400$ W, $u=0.4$ and $\pi=3.12\times10^{14}$ FLOP/s gives $e\approx3.2\times10^{-12}$ J per useful model FLOP. Multiplying by a run's counted model FLOPs estimates **accelerator energy only** under those assumptions.

Three boundaries must accompany that number:

1. 400 W is a nominal device rating, not necessarily measured average power;
2. data-centre energy also includes host processors, memory, networking, cooling and power conversion, often summarized through power usage effectiveness;
3. carbon emissions require the time- and location-dependent carbon intensity of the electricity, not energy alone.

[Patterson et al. (2021)](/references/#patterson-2021) illustrate how hardware, data-centre efficiency and electricity source change reported training emissions. Worked Example 6.8 compares this estimate with a published run and explains why Landauer's thermodynamic limit is not a per-FLOP benchmark.

**Next:** [§6.4 Worked examples](/core/unit-06/worked-examples/) turns every ledger on this page into a concrete calculation before §6.5 uses those costs as a budget constraint.
