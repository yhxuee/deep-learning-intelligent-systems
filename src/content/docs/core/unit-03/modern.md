---
title: "3.8 Modern developments"
description: How current practice extends the unit's principles — variance-preserving initialization, normalization layers, residual connections, learning-rate schedules, large-batch training, the edge of stability, mixed precision and structured preconditioners — with evidence labels.
sidebar: { label: "3.8 Modern developments", order: 8 }
---

This section translates the quantities derived earlier into design choices. For each development, ask three questions: which quantity is being controlled, under what approximation, and what evidence shows that the intervention helps? Labels mark derivation **[D]**, controlled experimental evidence **[E]**, engineering observation **[O]**, and open hypothesis **[H]**.

## 3.8.1 Initialization: preserve variance forward and backward

Worked Example 3.9 chose weight variance to preserve the scale of pre-activations in the **forward** pass. Under independent, zero-mean approximations,

$$
\operatorname{Var}(z_i)
\approx n_{\text{in}}\sigma_w^2\,\mathbb E[x_j^2].
$$

For a linear-like activation this suggests $\sigma_w^2=1/n_{\text{in}}$; for a symmetric unit-variance input followed by ReLU, half the second moment survives, giving $2/n_{\text{in}}$.

The **backward** calculation has the same structure but sums over $n_{\text{out}}$ downstream units and includes the activation derivative:

$$
\operatorname{Var}(\bar x_j)
\approx n_{\text{out}}\sigma_w^2\,
\mathbb E[\varphi'(z)^2]\operatorname{Var}(\bar z).
$$

Thus a linear activation suggests $1/n_{\text{out}}$, while an idealized ReLU gate suggests $2/n_{\text{out}}$. One variance cannot generally satisfy both forward and backward requirements when fan-in and fan-out differ. [Glorot & Bengio (2010)](/references/#glorot-2010) use the compromise $2/(n_{\text{in}}+n_{\text{out}})$ for symmetric activations; [He et al. (2015)](/references/#he-2015) emphasize the ReLU forward choice $2/n_{\text{in}}$. **[D]** covers the calculation under initialization-time independence approximations; **[E]** covers trainability in the studied architectures, not permanent variance preservation after training.

## 3.8.2 Normalization and residual connections

**Normalization layers** standardize selected axes and then apply learned scale and shift. Batch normalization uses batch statistics during training and running statistics at inference ([Ioffe & Szegedy 2015](/references/#ioffe-2015)); layer normalization uses features within each example and behaves the same way at training and inference ([Ba et al. 2016](/references/#ba-2016)). Layer normalization is standard in Transformers. Both have made many deep architectures easier to optimize **[E]**, but they are not interchangeable: batch size, sequence structure, and train–test behavior matter.

The mechanism is less settled. The original “internal covariate shift” explanation was challenged by experiments showing smoother optimization behavior ([Santurkar et al. 2018](/references/#santurkar-2018)). “Smoother” is not automatically identical to “a smaller Hessian condition number,” and no single explanation covers every normalization variant. Treat broad mechanistic stories as **[H]**.

**Residual connections** $x\mapsto x+F(x)$ have Jacobian

$$
I+\frac{\partial F}{\partial x}.
$$

The identity term creates a direct contribution to the backward signal instead of forcing every contribution through the residual branch's weight matrices. This improves gradient transport in many deep architectures ([He et al. 2016](/references/#he-2016); [residual connection](/atlas/nodes/mech-residual-connection/)). It does **not** mathematically guarantee a nonvanishing product: $I+\partial F/\partial x$ can have small or zero singular values, and many such factors can still contract. **[D]** establishes the Jacobian; **[E]** establishes improved trainability in the studied designs.

## 3.8.3 Learning-rate schedules

Practice rarely uses a constant step. A common pattern is **warmup**, which raises $\eta$ over the first updates, followed by linear or cosine **decay** ([Loshchilov & Hutter 2017](/references/#loshchilov-2017) studied cosine annealing with restarts). Under the stochastic assumptions of §3.5.6, shrinking $\eta$ reduces the nonvanishing noise term in the convergence bound; it also reduces deterministic progress per update. **[D]** Warmup is widely used with Adam and Transformers, but it is not one universal algorithm: duration, peak rate, normalization placement, parameterization, and optimizer interact. [Xiong et al. (2020)](/references/#xiong-2020) analyze one such interaction. Their reported result is **[E]**; a universal mechanism is **[H]**.

## 3.8.4 Large batches and learning-rate scaling

Under independent with-replacement sampling, §3.5.6 gives a variance contribution proportional to $\eta/B$. Keeping that ratio fixed motivates increasing the learning rate with batch size. [Goyal et al. (2017)](/references/#goyal-2017) combined this **linear scaling rule** with warmup to train ImageNet models at batches up to 8,192. **[E]** applies to that recipe and setting, not to every model.

Scaling eventually stops helping: variance may cease to be the bottleneck, hardware efficiency may saturate, and curvature still limits stable steps. Correlated or without-replacement batches also modify the simple $1/B$ model. The useful maximum batch is therefore an empirical property of the workload. **[O]**

## 3.8.5 The edge of stability

Quadratic theory predicts instability when $\eta>2/\lambda_{\max}$. In specified full-batch neural-network experiments, the largest Hessian eigenvalue—one measure of “sharpness”—rose toward roughly $2/\eta$ and then stayed near that scale while loss decreased non-monotonically ([Cohen et al. 2021](/references/#cohen-2021)). **[E]** This co-evolution is called the **edge of stability**. The observation does not by itself prove that the dynamics causally “adapt curvature,” nor that the same behavior holds for minibatch Adam training. Its mechanism and implications for generalization remain **[H]**.

## 3.8.6 Mixed precision and structured preconditioners

**Mixed precision.** The classic FP16 recipe computes many operations in 16 bit, retains higher-precision weight state, accumulates selected reductions in higher precision, and scales the loss when needed ([Micikevicius et al. 2018](/references/#micikevicius-2018)). **[E]** bfloat16 and newer formats have different exponent and precision trade-offs, so “mixed precision” is a family of policies rather than one fixed recipe. Moving 16-bit tensors can halve their byte traffic relative to 32 bit, but total memory and speedups depend on optimizer state, retained activations, conversions, and hardware.

**Structured preconditioners.** Between diagonal methods (Adam) and full Newton steps lie optimizers that maintain preconditioners with structure, e.g. per-layer Kronecker-factored matrices in Shampoo ([Gupta et al. 2018](/references/#gupta-2018)). They target rotated ill-conditioning that diagonal methods cannot fix (§3.5.8). **[D]** for the construction; reported speed-ups in wall-clock time over well-tuned AdamW vary across settings and are an active area — **[O]**/**[H]**. When you read optimizer comparisons, check whether the baselines were tuned with the same budget (Unit 12).

## 3.8.7 What to take from this section

Every development above targets a quantity from this unit: signal **second moments** (initialization), activation statistics (normalization), backward-path **Jacobians** (residuals), the stochastic-bound term $\beta\eta\sigma^2$ (schedules and batches), the quadratic **stability scale** $2/\lambda_{\max}$ (edge-of-stability measurements), effective **conditioning** (preconditioners), and **finite precision** (mixed precision).

When evaluating a new optimizer or architecture, do not stop at its metaphor. Ask which measurable quantity it changes, under which assumptions, what baseline it was compared with, and whether the evidence reports optimization progress, wall-clock cost, memory, or held-out performance.
