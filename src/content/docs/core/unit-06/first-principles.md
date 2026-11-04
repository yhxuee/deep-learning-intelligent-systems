---
title: "6.2 First principles"
description: Counting the arithmetic of a matrix product, the forward and backward pass of a linear layer, the 6N rule, what training must store, and time as the larger of compute time and memory time.
sidebar: { label: "6.2 First principles", order: 2 }
---

## 6.2.1 The unit of work: a multiply–add

Start with one output number, not a whole neural network. A dot product of two length-$k$ vectors is

$$
y=a_1b_1+a_2b_2+\cdots+a_kb_k.
$$

It uses $k$ multiplications and $k-1$ additions. For example, a length-3 dot product uses 3 multiplications and 2 additions. In large-model accounting we round this to $2k$ operations. The dropped one addition is negligible when $k$ is large, and the convention makes formulas composable.

Now multiply an $m\times k$ matrix by a $k\times n$ matrix. The result contains $mn$ numbers, and each is one length-$k$ dot product. Therefore

$$
F_{\mathrm{matmul}}\approx mn\,(2k)=2mnk\ \text{FLOPs}.
$$

The dimensions tell a story: $mn$ counts **how many outputs** are produced, and $k$ counts **how much work contributes to each output**. If you cannot remember the formula, reconstruct it from that sentence.

Most arithmetic in a dense neural network is in such products. Softmax, normalization, nonlinearities and residual additions still cost time, but their operation counts usually grow with the number of activations rather than with the product of two large dimensions. We first count matrix products exactly under our convention, then return to the omitted work when discussing time. “Small in FLOPs” does not mean “free in seconds,” because an elementwise operation may move many bytes for little arithmetic.

Modern hardware often performs $ab+c$ as one fused multiply–add instruction. We nevertheless count it as **two FLOPs**—one multiplication and one addition. This is an accounting convention, not a claim about instruction count. Likewise, **FLOPs** is a quantity of work, while **FLOP/s** is a rate. A GPU datasheet gives a peak rate; a training run has a total count. Dividing work by peak rate gives an optimistic lower bound on time, never the measured time itself (§6.2.5).

:::note[Dimensional check]
If a result is measured in FLOPs, no “per second” should remain. If it is in FLOP/s, some time interval has been divided out. Treating the units as carefully as metres and metres per second prevents the 22,000-token/s mistake from §6.1.
:::

## 6.2.2 One linear layer, forward and backward

Take a linear layer $Y=XW$. Let $X\in\mathbb{R}^{S\times d_{\text{in}}}$ contain $S$ input rows and $W\in\mathbb{R}^{d_{\text{in}}\times d_{\text{out}}}$ contain the weights. In a language model, one row is usually one token position, so a batch of $B$ sequences of length $T$ has $S=BT$ rows after the batch and sequence axes are flattened.

The forward product has dimensions

$$
(S\times d_{\text{in}})(d_{\text{in}}\times d_{\text{out}})ightarrow(S\times d_{\text{out}}),
$$

and therefore costs $2S d_{\text{in}}d_{\text{out}}$ FLOPs. Write $P=d_{\text{in}}d_{\text{out}}$ for the number of matrix weights. Then **forward = $2SP$ FLOPs**, or **$2P$ per token row**. Biases, if present, add only $Sd_{\text{out}}$ scalar additions; we keep them outside the matrix-product count.

The backward pass, derived in Unit 03 (§3.3.5), needs two products:

$$
\bar W = X^{\top}\bar Y \quad(2Sd_{\text{in}}d_{\text{out}}),\qquad \bar X = \bar YW^{\top}\quad(2Sd_{\text{in}}d_{\text{out}}).
$$

The first product accumulates how every input row contributed to the weight gradient. The second sends a gradient to every input row so earlier layers can be trained. Each has the same leading cost as the forward product. Thus **backward = $4SP$ FLOPs**, twice the forward pass, and forward plus backward is $6SP$.

The factor of three is not mystical: it is simply **one product forward + two products backward**. If the input below a layer is frozen and its gradient is not needed, $\bar X$ can be skipped; if weights are shared, sparse, low-rank or conditionally active, the count must be redone for the actual products. The $6P$ rule is a reusable baseline, not permission to stop inspecting shapes.

## 6.2.3 The 6N rule

Add the three products: a dense weight matrix with $P$ entries costs about $6P$ FLOPs per token during training. If the model has $N_{\mathrm{mm}}$ parameters that each enter one such matrix product per token, and training processes $D$ token positions, then

$$
\boxed{\;C_{\text{weights}} \approx 6N_{\mathrm{mm}}D\;}.
$$

This is commonly shortened to $6ND$ ([Kaplan et al. 2020](/references/#kaplan-2020), §2.1). The subscript matters: not every number called a “parameter” necessarily participates in a dense multiply once per token. Normalization scales and biases use elementwise operations; an embedding lookup reads one row rather than multiplying by the whole table; tied input/output embeddings may enter computation differently; a mixture-of-experts model activates only some expert weights for each token. In an ordinary dense Transformer, projection and MLP weights dominate, so $N_{\mathrm{mm}}\approx N$ and the shortcut is useful.

The word “approximately” hides three corrections:

1. **Attention over positions.** Forming attention scores and mixing values multiplies activations by activations. With context length $T$, width $d$ and $L$ layers, their forward-plus-backward cost is $12LTd$ per token under the convention derived in §6.3.1.
2. **Non-matrix operations.** Normalization, softmax, activations, loss evaluation and residual additions are omitted from the matrix-product count. They may be a small FLOP fraction yet a visible time fraction.
3. **The optimizer step.** Adam updates each parameter once per optimization step, not once per token. If a step contains many tokens, this work is amortized; with a tiny batch, it is less negligible.

For the spine implementation, the instrumented matrix products total exactly $6N_{\mathrm{mm}}+12LTd$ per token. “Exactly” refers to that declared set of products and convention, not to every machine instruction executed by Python or the processor (Worked Example 6.1).

Generating text uses a forward pass but no backward pass, so its dense-weight baseline is **about $2N_{\mathrm{mm}}$ FLOPs per generated token**, plus attention over the KV cache and non-matrix work. This explains the arithmetic count; it does not yet explain the elapsed time.

## 6.2.4 What training must store

FLOPs measure work. Memory capacity asks a different yes-or-no question: **can the training state fit on the device at all?** A training step may need all of the following at once:

1. **Parameters** — $N$ numbers.
2. **Gradients** — one per parameter.
3. **Optimizer state** — Adam keeps first- and second-moment estimates per updated parameter (Unit 03, §3.3.8).
4. **A master weight copy, sometimes** — some mixed-precision recipes update a 32-bit copy even when forward and backward use 16-bit values ([Micikevicius et al. 2018](/references/#micikevicius-2018)). Other modern recipes differ.
5. **Activations** — intermediate forward values needed by backward (Unit 03's tape). These grow with batch size, sequence length, width and depth.
6. **Temporary workspaces and fragmentation** — kernels and memory allocators need space not represented by the four conceptual tensors above.

Under one common mixed-precision accounting—16-bit model weights and gradients (2 + 2 bytes), a 32-bit master weight copy (4 bytes), and two 32-bit Adam moments (4 + 4 bytes)—model state costs

$$
2+2+4+4+4=16\ \text{bytes per parameter}.
$$

A 7-billion-parameter model then needs $7\times10^9\times16=112$ GB before activations, workspaces or the CUDA context. That exceeds one 80 GB device and motivates partitioning parameters, gradients and optimizer states across devices (§6.5.5; [Rajbhandari et al. 2020](/references/#rajbhandari-2020)). But **16 bytes is a recipe, not a constant of nature**: lower-precision optimizer states, no master copy, SGD, quantization or sharding change the number. Always write the byte sum before quoting a memory budget.

Activation memory follows a different scaling. Merely storing one 16-bit hidden-state tensor for $B=16$, $T=128$, $d=64$ costs $16\cdot128\cdot64\cdot2=262{,}144$ bytes, about 0.25 MiB. Saving one per layer for three layers is about 0.75 MiB; real backward needs several intermediates per layer, so the true total is larger. This small calculation reveals the levers: doubling batch or context doubles this tensor; doubling width doubles it; doubling layers doubles the number of saved layer tensors. §6.3.2 counts the Transformer-specific intermediates and explains recomputation.

## 6.2.5 Time: compute or memory, whichever is slower

A processor can do arithmetic only after operands reach its arithmetic units. Suppose an operation needs $F$ FLOPs and transfers at least $M$ bytes through the memory level being modelled. Hardware offers peak arithmetic rate $\pi$ FLOP/s and peak bandwidth $\beta$ bytes/s. Even in an ideal implementation,

$$
t \;\ge\; \max\Big(\frac{F}{\pi},\ \frac{M}{\beta}\Big).
$$

Why a maximum rather than a sum? Arithmetic and data movement can overlap. The operation cannot finish before either required time has elapsed, so the slower requirement is a lower bound. Real time may be larger because of kernel launch overhead, imperfect parallelism, communication, cache misses or an algorithm that moves more than the minimum $M$.

The ratio

$$
I=\frac{F}{M}\quad\text{FLOPs per byte}
$$

is **arithmetic intensity**. The hardware balance point is $\pi/\beta$. For the A100 figures used here,

$$
\frac{312\times10^{12}}{2.039\times10^{12}}\approx153\ \text{FLOPs/byte}.
$$

If $I\gg153$, memory can in principle supply operands quickly enough and arithmetic is the likely limit: **compute-bound**. If $I\ll153$, the arithmetic units wait for data: **memory-bound**. The attainable rate is bounded by

$$
\text{performance}\le\min(\pi,\beta I),
$$

the two “roofs” of the **roofline model** ([Williams et al. 2009](/references/#williams-2009)). The boundary belongs to a particular hardware precision and memory level; changing either changes the roofline.

Now return to Question 2. Under a deliberately simple model, batch-1 decoding of a dense 7-billion-parameter model performs $2N=1.4\times10^{10}$ FLOPs and streams one 16-bit copy of the weights, also $2N=1.4\times10^{10}$ bytes. Thus $I\approx1$ FLOP/byte, far below 153. Weight bandwidth alone gives

$$
t\ge\frac{1.4\times10^{10}}{2.039\times10^{12}}\approx6.9\ \text{ms/token},
$$

or at most about 146 tokens/s. This is already over 150 times slower than the compute-only fantasy. Reading the KV cache, launching kernels and failing to sustain peak bandwidth make the observed rate lower still.

Batching changes reuse. If $B$ sequences use the same weights together, roughly $B$ times as much arithmetic is done while the weights need be fetched only once for the batch; intensity initially grows roughly with $B$. That can improve **throughput**, though an individual request may wait to form a batch and therefore see greater **latency**. Systems engineering is the art of choosing which quantity matters for the service (§6.8).

## 6.2.6 What we have so far

| Question | First estimate | Spine (Stage 07 configuration) | Dense 7B illustration |
|---|---|---|---|
| How much training arithmetic? | $6N_{\mathrm{mm}}+12LTd$ per token | 1,198,848 FLOPs/token | $\approx4.2\times10^{10}$ plus attention |
| How much generation arithmetic? | $\approx2N_{\mathrm{mm}}$ plus cache attention | $\approx3\times10^5$ FLOPs/token | $1.4\times10^{10}$ FLOPs/token |
| Does model state fit? | sum bytes for weights, gradients and optimizer | about 2.6 MB under the 16-byte recipe | 112 GB before activations |
| What limits batch-1 decode? | compare $F/\pi$ with $M/\beta$ | measure; tiny products have overheads | at least 6.9 ms/token from weight bandwidth on an A100 |

Before using any row, say what it omits. The next section replaces these first estimates with a matrix-by-matrix Transformer account.
