---
title: "3.8 Modern developments"
description: How current practice extends the unit's principles — variance-preserving initialization, normalization layers, residual connections, learning-rate schedules, large-batch training, the edge of stability, mixed precision and structured preconditioners — with evidence labels.
sidebar: { label: "3.8 Modern developments", order: 8 }
---

Labels: **[D]** derivation, **[E]** experimental evidence from controlled comparisons, **[O]** engineering observation, **[H]** open hypothesis.

## 3.8.1 Initialization: preserve variance forward and backward

Worked Example 3.9 chose weight variance so that pre-activations keep unit variance in the **forward** pass: $\sigma_w^2 = 1/n_{\text{in}}$ for linear-like activations, $2/n_{\text{in}}$ for ReLU. The **backward** pass multiplies by $W^{\top}$, which sums over the $n_{\text{out}}$ outputs, so preserving gradient variance asks for $\sigma_w^2 = 1/n_{\text{out}}$. [Glorot & Bengio (2010)](/references/#glorot-2010) compromise with $\sigma_w^2 = 2/(n_{\text{in}}+n_{\text{out}})$; [He et al. (2015)](/references/#he-2015) use $2/n_{\text{in}}$ for ReLU networks. **[D]** for the variance calculations (under independence assumptions that hold at initialization only); **[E]** for their effect on trainability of deep networks (Figure 3.3).

## 3.8.2 Normalization and residual connections

**Normalization layers** rescale activations during training: batch normalization over the batch dimension ([Ioffe & Szegedy 2015](/references/#ioffe-2015)), layer normalization over the feature dimension ([Ba et al. 2016](/references/#ba-2016)), the latter standard in Transformers. They make deep networks markedly easier to train **[E]**. *Why* is less settled: the original explanation ("reducing internal covariate shift") was challenged by experiments suggesting the main effect is a smoother loss landscape, i.e. better conditioning ([Santurkar et al. 2018](/references/#santurkar-2018)). Treat mechanistic explanations of normalization as **[H]**.

**Residual connections** $x\mapsto x + F(x)$ have Jacobian $I + \partial F/\partial x$; the identity term gives the backward signal a path that is not multiplied by any weight matrix, so it cannot vanish through that path ([He et al. 2016](/references/#he-2016); [residual connection](/atlas/nodes/mech-residual-connection/)). **[D]** for the Jacobian; **[E]** for trainability at depth. They are a structural answer to §3.7.4, and the reason Unit 04's Transformer blocks can be stacked.

## 3.8.3 Learning-rate schedules

Practice rarely uses a constant step. Common schedules: a **warmup** that increases $\eta$ linearly over the first steps, then a **decay** (cosine or linear) toward a small value ([Loshchilov & Hutter 2017](/references/#loshchilov-2017) introduced cosine annealing with restarts). The decay is motivated by §3.5.6: a shrinking step lowers the SGD noise floor. **[D]** Warmup is widely used for Transformers trained with Adam; one analysis links its necessity to the placement of layer normalization ([Xiong et al. 2020](/references/#xiong-2020)). **[E]** for the reported effect; the general mechanism is **[H]**.

## 3.8.4 Large batches and learning-rate scaling

§3.5.6 showed that the noise floor scales like $\eta/B$. [Goyal et al. (2017)](/references/#goyal-2017) trained ImageNet models with batches up to 8 192 by scaling the learning rate linearly with the batch size, plus warmup — the "linear scaling rule". **[E]** for their setting. The rule breaks down beyond some batch size, where noise is no longer the limiting factor and curvature is (the $2/\lambda_{\max}$ limit returns). **[O]** for where that point lies in a given problem.

## 3.8.5 The edge of stability

The quadratic theory predicts divergence if $\eta > 2/\lambda_{\max}$. Experiments on full-batch gradient descent for neural networks found something subtler: the largest Hessian eigenvalue ("sharpness") rises during training until it reaches about $2/\eta$, then *hovers there* while the loss keeps decreasing non-monotonically ([Cohen et al. 2021](/references/#cohen-2021)). **[E]** The training dynamics adapt the curvature to the step size. Why this happens, and its consequences for generalization, are **[H]** — a live example of the $2/\lambda$ condition of this unit being the right quantity even where its assumptions fail.

## 3.8.6 Mixed precision and structured preconditioners

**Mixed precision** computes most operations in 16-bit formats while keeping 32-bit master weights and scaling the loss to avoid gradient underflow ([Micikevicius et al. 2018](/references/#micikevicius-2018)). **[E]** It roughly halves memory traffic — Unit 06 returns to why that matters more than arithmetic.

**Structured preconditioners.** Between diagonal methods (Adam) and full Newton steps lie optimizers that maintain preconditioners with structure, e.g. per-layer Kronecker-factored matrices in Shampoo ([Gupta et al. 2018](/references/#gupta-2018)). They target rotated ill-conditioning that diagonal methods cannot fix (§3.5.8). **[D]** for the construction; reported speed-ups in wall-clock time over well-tuned AdamW vary across settings and are an active area — **[O]**/**[H]**. When you read optimizer comparisons, check whether the baselines were tuned with the same budget (Unit 12).

## 3.8.7 What to take from this section

Every development above targets a quantity derived in this unit: the **variance** of signals through depth (initialization, normalization), the **Jacobian** of the backward path (residuals), the **noise floor** $\beta\eta\sigma^2$ (schedules, batch scaling), the **stability limit** $2/\lambda_{\max}$ (warmup, edge of stability), the **condition number** (normalization, preconditioning), and **finite precision** (mixed precision). New optimizers and architectures can be evaluated by asking which of these quantities they change, and how you would measure it.
