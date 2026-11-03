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


## 6.5.3 The compute-optimal allocation

Given a budget of $C$ FLOPs and the cost $C = 6ND$ (§6.2.3), which $N$ minimizes $L(N, D)$?

**Proposition 6.2.** Minimizing $AN^{-\alpha} + BD^{-\beta}$ subject to $6ND = C$ gives
$$
N^\star = G\Big(\frac{C}{6}\Big)^{\frac{\beta}{\alpha+\beta}},\qquad D^\star = \frac{C}{6N^\star},\qquad G = \Big(\frac{\alpha A}{\beta B}\Big)^{\frac{1}{\alpha+\beta}}.
$$

*Proof.* Substitute $D = C/(6N)$: minimize $f(N) = AN^{-\alpha} + B(C/6)^{-\beta}N^{\beta}$. Then $f'(N) = -\alpha AN^{-\alpha-1} + \beta B(C/6)^{-\beta}N^{\beta-1} = 0$ gives $N^{\alpha+\beta} = \frac{\alpha A}{\beta B}(C/6)^{\beta}$. $f$ is convex in $\ln N$, so this is the minimum. $\blacksquare$

The exponents tell the story. With $\alpha = 0.34$ and $\beta = 0.28$, $N^\star\propto C^{0.45}$ and $D^\star\propto C^{0.55}$: **parameters and data should grow at nearly the same rate**, so the number of tokens per parameter changes slowly with the budget. Hoffmann et al.'s other two estimation approaches gave exponents close to 0.5 each and the widely used rule of thumb of about 20 tokens per parameter at their scale ([Hoffmann et al. 2022](/references/#hoffmann-2022)) **[E]**. Kaplan et al.'s earlier fits had implied $N^\star\propto C^{0.73}$ — spend most of a larger budget on parameters; the difference was traced largely to learning-rate schedules not adjusted to each run's length and to counting non-embedding parameters at small scale.

**The optimum is flat.** Near the minimum, $f$ changes slowly: with the approach-3 constants and Chinchilla's budget $C = 5.76\times10^{23}$, Proposition 6.2 gives $N^\star\approx3.2\times10^{10}$ and $D^\star\approx3.0\times10^{12}$, a predicted loss of 1.931 — while the 70-billion-parameter, 1.4-trillion-token model actually trained is predicted at 1.937 (Worked Example 6.7). A factor of two in $N$ costs 0.006 nats. Two consequences: the "optimal" allocation is poorly determined by noisy fits — a replication found the published approach-3 fit inconsistent with the other two approaches and refitted it ([Besiroglu et al. 2024](/references/#besiroglu-2024)) **[E]** — and practical choices can depart from it cheaply in exchange for other benefits. The commonest departure is to train a *smaller* model on *more* tokens than compute-optimal, because a smaller model is cheaper every time it is used afterwards (§6.7.4).

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
