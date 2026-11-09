---
title: "6.5 General theory"
description: Empirical scaling laws in parameters, data and compute; the irreducible loss; the closed-form compute-optimal allocation and why its optimum is flat; a heuristic for why power laws appear; and the communication cost of parallel training.
sidebar: { label: "6.5 General theory", order: 5 }
---

## 6.5.1 Loss is a power law in scale

Across many orders of magnitude, the held-out loss of language models trained with a good recipe follows simple laws **[E]**. [Kaplan et al. (2020)](/references/#kaplan-2020) measured, for models not limited by the other resource,

$$
L(N)\approx\Big(\frac{N_c}{N}\Big)^{\alpha_N},\quad L(D)\approx\Big(\frac{D_c}{D}\Big)^{\alpha_D},\quad L(C_{\min})\approx\Big(\frac{C_c}{C_{\min}}\Big)^{\alpha_C},
$$

with $\alpha_N\approx0.076$, $\alpha_D\approx0.095$ and $\alpha_C\approx0.050$, where $N$ counts non-embedding parameters. Earlier work had found power laws across several domains ([Hestness et al. 2017](/references/#hestness-2017)). A power law is a straight line on log–log axes: each doubling of $N$ removes the same *fraction* of the remaining loss.

Pure power laws must eventually fail — loss cannot go below the entropy of the data (Unit 01, Corollary 1.1). [Hoffmann et al. (2022)](/references/#hoffmann-2022) therefore fitted a form with an explicit floor and both resources at once:

$$
L(N, D) = E + \frac{A}{N^{\alpha}} + \frac{B}{D^{\beta}} .
$$

$E$ is the irreducible loss; the second term is what a model of size $N$ loses for being finite when data are unlimited; the third is what $D$ tokens lose for being finite when the model is unlimited. Their published fit ("approach 3") is $E = 1.69$, $A = 406.4$, $B = 410.7$, $\alpha = 0.34$, $\beta = 0.28$, in nats per token on their data. These constants depend on the data, the tokenizer and the training recipe; the *form* is what transfers.

## 6.5.2 Fitting a law and predicting with it

Fitting $L = E + AN^{-\alpha}$ to a handful of measurements is a small nonlinear regression. For a fixed $E$ it is linear in log space: $\ln(L - E) = \ln A - \alpha\ln N$. So one can scan $E$ over a grid, solve ordinary least squares at each value, and keep the $E$ whose fitted curve has the smallest squared error in $L$ itself — the method of `spine/stage06_scaling.py`, which the spine tests check recovers known parameters exactly.

A scaling law earns its keep by **prediction**: fit on small models, predict a larger one *before* training it, then compare. The procedure is the pre-registration of Unit 12 in miniature: the prediction is written to disk with the fit, so it cannot be adjusted after seeing the result. Worked Example 6.6 does this for the spine model; when the prediction misses, the size and direction of the miss is information about where the law's assumptions fail.

:::note[Claim check]
- **Claim.** A power law fitted on small models predicts the held-out loss of a larger one.
- **Assumptions.** The same data distribution and an equally well-tuned recipe at every size, and no change of regime between the fitted range and the target (§6.7.2).
- **Test.** Write the prediction to disk before training the larger model, then compare (Worked Example 6.6; Experiment 6.4).
- **Limitation.** In Worked Example 6.6 the registered prediction missed by 0.12 nats because the fitted points straddled a capacity threshold; and the constants belong to one recipe and one dataset (§6.7.5).
:::


## 6.5.3 The compute-optimal allocation: rigorous derivation of the Chinchilla scaling law

In modern foundation model pre-training, the central engineering decision is: **Given a fixed computational budget of $C$ FLOPs, how should compute be partitioned between larger model parameters $N$ and more training tokens $D$?**

According to the FLOP conservation principle established in §6.2.3, the total training cost per forward and backward pass is $C \approx 6ND$.
Our objective is to find the allocation $(N^\star, D^\star)$ that minimizes the expected test loss $L(N, D) = E + \frac{A}{N^\alpha} + \frac{B}{D^\beta}$ subject to $6ND = C$.

**Proposition 6.2 (closed-form compute-optimal allocation).** Under the budget constraint $6ND = C$, the analytical minimizer of reducible loss $AN^{-\alpha} + BD^{-\beta}$ is:
$$
N^\star = G\Big(\frac{C}{6}\Big)^{\frac{\beta}{\alpha+\beta}},\qquad D^\star = \frac{C}{6N^\star},\qquad G = \Big(\frac{\alpha A}{\beta B}\Big)^{\frac{1}{\alpha+\beta}}.
$$

*Proof.* Rewrite the budget constraint as $D = \frac{C}{6N}$ and substitute directly into the objective, eliminating $D$. Define the single-variable function:
$f(N) = A N^{-\alpha} + B \Big(\frac{C}{6}\Big)^{-\beta} N^{\beta}$.
Differentiating with respect to parameter count $N$:
$f'(N) = -\alpha A N^{-\alpha-1} + \beta B \Big(\frac{C}{6}\Big)^{-\beta} N^{\beta-1}$.
Setting the first derivative to zero $f'(N^\star) = 0$ yields:
$\alpha A (N^\star)^{-\alpha} = \beta B \Big(\frac{C}{6N^\star}\Big)^{-\beta} = \beta B (D^\star)^{-\beta}$.

**The economic principle of marginal parity:**
At the compute-optimal point, **the marginal reduction in loss from investing an additional unit of compute into model parameters $\alpha A N^{-\alpha}$ must strictly equal the marginal reduction from investing into training tokens $\beta B D^{-\beta}$!**

Rearranging terms:
$(N^\star)^{\alpha+\beta} = \frac{\alpha A}{\beta B}\Big(\frac{C}{6}\Big)^\beta \implies N^\star = \Big(\frac{\alpha A}{\beta B}\Big)^{\frac{1}{\alpha+\beta}}\Big(\frac{C}{6}\Big)^{\frac{\beta}{\alpha+\beta}}$.
Then from $D^\star = \frac{C}{6N^\star}$:
$D^\star = \Big(\frac{\beta B}{\alpha A}\Big)^{\frac{1}{\alpha+\beta}}\Big(\frac{C}{6}\Big)^{\frac{\alpha}{\alpha+\beta}}$.
Since the second derivative $f''(N) > 0$ everywhere on $N > 0$, this critical point is the unique global minimum of a strictly convex objective. $\blacksquare$

### Kaplan vs. Chinchilla: a historical correction
The power law exponents govern the scaling trajectory:
- Optimal parameters scale as $N^\star \propto C^{\frac{\beta}{\alpha+\beta}}$;
- Optimal training tokens scale as $D^\star \propto C^{\frac{\alpha}{\alpha+\beta}}$.

**Kaplan et al. (2020) and the under-training regime:**
Early OpenAI research ([Kaplan et al. 2020](/references/#kaplan-2020)) fitted exponents $\alpha \approx 0.076, \beta \approx 0.095$, concluding that $N^\star \propto C^{0.73}$ while $D^\star \propto C^{0.27}$. This led the industry to pour compute almost exclusively into model parameters (e.g. GPT-3 175B trained on only 300B tokens, a ratio of less than 2:1), leaving models severely under-trained relative to their capacity.

**Chinchilla (Hoffmann et al. 2022) correction:**
DeepMind corrected the learning rate schedule and parameter counting methodologies, discovering $\alpha \approx 0.34, \beta \approx 0.28$ (with both exponents exceedingly close to $0.5$ in their other two empirical approaches):
$$
\frac{\beta}{\alpha+\beta} \approx 0.5, \qquad \frac{\alpha}{\alpha+\beta} \approx 0.5.
$$
**The Chinchilla principle:** Parameters and tokens should scale in equal 1:1 proportion with compute!
Doubling training compute requires scaling both model parameters and training tokens by $\sqrt{2} \approx 1.41\times$. This gives the famous **Chinchilla rule of thumb: optimal training requires approximately 20 tokens per model parameter ($D/N \approx 20$)**.

### Why modern foundation models choose deliberate "over-training"
Inspecting the curvature of the loss surface $f(N)$ around the minimum reveals that the **optimum is remarkably flat**:
At the Chinchilla compute scale, halving parameter count while doubling training data increases test loss by less than 0.01 nats (Worked Example 6.7).

Crucially, **training compute is paid once, whereas inference compute is paid on every single user request!**
Serving a 70B model across millions of daily queries uses far less GPU memory, memory bandwidth, and latency than serving a 280B model. Modern frontier open-weights models (such as LLaMA-1/2/3) therefore intentionally violate pure training compute optimality: they train 8B and 70B models on 15+ trillion tokens ($D/N > 200$, more than $10\times$ beyond the Chinchilla recommendation). This trades higher one-time pre-training expenditure for dramatic permanent inference cost savings.

## 6.5.4 ★ Why power laws?

No derivation from first principles covers real language models, but a heuristic explains the form **[H]**. Suppose the data lie near a manifold of intrinsic dimension $d_{\text{int}}$ and the model approximates the target function piecewise, with resolution set by how many pieces it can afford. With $N$ parameters, the piece size scales like $N^{-1/d_{\text{int}}}$; for a smooth target the squared error of a piecewise-linear fit scales like (piece size)$^4$, giving $L - E\propto N^{-4/d_{\text{int}}}$. Higher intrinsic dimension means slower improvement. [Sharma & Kaplan (2022)](/references/#sharma-2022) develop this argument and find measured exponents roughly consistent with independently estimated intrinsic dimensions in several settings **[E]**. The heuristic also explains why the exponents differ between domains and why no single exponent is "the" scaling law.

## 6.5.5 ★ Parallel training and communication

When one device cannot hold or compute a model fast enough, the work is split **[O]**:

- **Data parallelism.** Each of $p$ devices holds a full copy and processes a different slice of the batch; gradients are averaged every step. A ring all-reduce of $G$ bytes sends and receives $2G(p-1)/p$ bytes per device — almost independent of $p$ — so the cost per step is about $2\times$ the gradient size divided by link bandwidth.
- **Sharded optimizer state.** Partitioning parameters, gradients and Adam states across the data-parallel devices removes the redundant copies of the 16 bytes per parameter (§6.2.4) at the price of extra communication ([Rajbhandari et al. 2020](/references/#rajbhandari-2020)).
- **Tensor parallelism** splits individual matrix products across devices; **pipeline parallelism** places consecutive layers on different devices and streams micro-batches through them ([Narayanan et al. 2021](/references/#narayanan-2021)).

Each scheme trades memory per device against communication volume and idle time. **Amdahl's law** bounds the payoff: if a fraction $s$ of each step cannot be parallelized (or is spent communicating), the speedup on $p$ devices is at most $1/(s + (1-s)/p)$, which saturates at $1/s$. Large runs reach 40–60% MFU (§6.3.5) by overlapping communication with computation so that $s$ stays small.

## 6.5.6 Summary

| Result | Says | Status |
|---|---|---|
| Prop. 6.1 | training FLOPs $= BT(6N_{\text{mm}} + 12LTd)$ exactly in matrix products | [D], verified by counter |
| Roofline (§6.2.5) | $t\ge\max(F/\pi, M/\beta)$; decode at batch 1 is memory-bound | [D] |
| Scaling laws (§6.5.1) | $L(N, D) = E + AN^{-\alpha} + BD^{-\beta}$ fits many runs | [E]; constants recipe-dependent |
| Prop. 6.2 | compute-optimal $N^\star\propto C^{\beta/(\alpha+\beta)}$ | [D] given the law |
| Flat optimum (§6.5.3) | $2\times$ off in $N$ costs ~0.006 nats at Chinchilla scale | [D] given the fit |
| Manifold heuristic (§6.5.4) | exponent $\approx 4/d_{\text{int}}$ | [H] with supporting [E] |
