---
title: "6.2 First principles"
description: Counting the arithmetic of a matrix product, the forward and backward pass of a linear layer, the 6N rule, what training must store, and time as the larger of compute time and memory time.
sidebar: { label: "6.2 First principles", order: 2 }
---

## 6.2.1 The unit of work: a multiply–add

Almost all the arithmetic in a neural network is in matrix products. Multiplying an $m\times k$ matrix by a $k\times n$ matrix produces $mn$ outputs, each a sum of $k$ products: $mnk$ multiplications and (about) $mnk$ additions. We count both: **a matrix product costs $2mnk$ floating-point operations (FLOPs)**. Everything else in a Transformer — softmax, normalization, nonlinearities, additions — costs a number of operations proportional to the size of the activations, which for realistic widths is small next to the products. We ignore it, and check later (§6.3.1) that the count is then exact for the products.

Two words that are easy to confuse: **FLOPs** is a count of operations; **FLOP/s** is a rate. A GPU's datasheet gives a peak rate; a training run has a total count; dividing one by the other gives a lower bound on time, never the time itself (§6.2.5).

## 6.2.2 One linear layer, forward and backward

Take a layer $Y = XW$ with $X\in\mathbb{R}^{B\times d_{\text{in}}}$ (a batch of $B$ inputs) and $W\in\mathbb{R}^{d_{\text{in}}\times d_{\text{out}}}$. The forward pass costs $2Bd_{\text{in}}d_{\text{out}}$ FLOPs. Write $P = d_{\text{in}}d_{\text{out}}$ for the number of weights: **forward = $2P$ FLOPs per example**.

The backward pass, derived in Unit 03 (§3.3.5), needs two products:

$$
\bar W = X^{\top}\bar Y \quad(2Bd_{\text{in}}d_{\text{out}}),\qquad \bar X = \bar YW^{\top}\quad(2Bd_{\text{in}}d_{\text{out}}).
$$

The first gives the gradient for the update; the second passes the signal to the layer below. **Backward = $4P$ FLOPs per example**, twice the forward pass. (The first layer of a network does not need $\bar X$ unless something below it is trained — in a language model the embedding table is, so we keep it.)

## 6.2.3 The 6N rule

Add forward and backward: $6P$ FLOPs per example for every weight matrix. If a model has $N$ parameters that each take part in one multiply–add per token — as the weights of a Transformer's projections and MLPs do — then

$$
\boxed{\;C_{\text{train}} \approx 6ND\;}
$$

FLOPs to train on $D$ tokens. This is the most useful formula in large-scale deep learning ([Kaplan et al. 2020](/references/#kaplan-2020), §2.1). It makes training cost a product of two numbers anyone can state, and it is the constraint under which §6.5 chooses $N$ and $D$.

Two terms are missing from $6ND$, and both have exact forms (§6.3.1). Attention's score and mixing products involve two *activations*, not a weight, so they are not proportional to $N$; per token and layer they cost $12Td$ FLOPs over a training step. And the optimizer update costs a few FLOPs per *parameter* per *step*, not per token — negligible when each step processes thousands of tokens. For the spine model the counted total is exactly $6N + 12LTd$ per token (Worked Example 6.1).

Generating text costs the forward pass only: **about $2N$ FLOPs per generated token**, plus attention over the cache.

## 6.2.4 What training must store

FLOPs measure work; memory decides whether the work fits on the device at all. A training step must hold:

1. **Parameters** — $N$ numbers.
2. **Gradients** — one per parameter.
3. **Optimizer state** — Adam keeps two running moments per parameter (Unit 03, §3.3.8); with mixed precision ([Micikevicius et al. 2018](/references/#micikevicius-2018)) a 32-bit master copy of the weights is kept as well.
4. **Activations** — every intermediate value of the forward pass that the backward pass will need (Unit 03's tape). This grows with batch size, sequence length and depth.

In the common mixed-precision recipe — 16-bit weights and gradients (2 + 2 bytes), 32-bit master weights and two 32-bit Adam moments (4 + 4 + 4 bytes) — items 1–3 cost **16 bytes per parameter** ([Rajbhandari et al. 2020](/references/#rajbhandari-2020)). A 7-billion-parameter model therefore needs 112 GB before storing a single activation — more than one 80 GB GPU holds. That one number explains why large models are trained on many devices with the parameters, gradients and optimizer states partitioned among them (§6.5.5).

## 6.2.5 Time: compute or memory, whichever is slower

A processor can do arithmetic only on numbers that have arrived from memory. If an operation performs $F$ FLOPs and must move $M$ bytes between memory and the arithmetic units, then on hardware with peak rate $\pi$ FLOP/s and memory bandwidth $\beta$ bytes/s,

$$
t \;\ge\; \max\Big(\frac{F}{\pi},\ \frac{M}{\beta}\Big).
$$

The ratio $I = F/M$ is the operation's **arithmetic intensity** (FLOPs per byte). The hardware has its own ratio $\pi/\beta$ — for the A100, $312\times10^{12}/2.039\times10^{12}\approx153$ FLOP/byte. An operation with $I$ above the hardware's ratio is **compute-bound** (it can approach peak FLOP/s); below it, **memory-bound** (it runs at a fraction $I/(\pi/\beta)$ of peak, however fast the arithmetic units are). This is the **roofline model** ([Williams et al. 2009](/references/#williams-2009)).

Now Question 2 of §6.1 answers itself. Generating one token for one user with a 7-billion-parameter model in 16-bit precision performs $2N = 1.4\times10^{10}$ FLOPs but must read all $2N = 1.4\times10^{10}$ bytes of weights: $I\approx1$ FLOP/byte, more than a hundred times below the A100's 153. The time is set by memory: $1.4\times10^{10}/2.039\times10^{12}\approx6.9$ ms per token, at most about 146 tokens per second (Worked Example 6.5). Serving many users at once — batching — reuses each weight for every sequence in the batch and raises $I$ in proportion. This single inequality explains the design of inference systems (§6.8).

## 6.2.6 What we have so far

| Cost | Rule | Spine (Stage 07 model) | 7B model |
|---|---|---|---|
| training FLOPs per token | $6N + 12LTd$ | 1 198 848 | $\approx4.2\times10^{10}$ (+ attention) |
| generation FLOPs per token | $\approx2N$ | $\approx3\times10^{5}$ | $1.4\times10^{10}$ |
| training state | 16 bytes/param (mixed precision, Adam) | 2.6 MB | 112 GB |
| decode time, batch 1 | weight bytes / bandwidth | — | 6.9 ms on an A100 |

The next section derives each entry exactly.
