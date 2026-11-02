---
title: Notation
description: The notation shared by all twelve units, so that a symbol means the same thing from Unit 01 to Unit 12.
sidebar: { order: 3 }
---

Every unit uses the conventions below. Where a symbol is overloaded by long-standing usage (for example $\beta$), the meaning is stated locally and listed here.

## General

| Symbol | Meaning | First used |
|---|---|---|
| $a, x, w$ | scalars and vectors (lower case, not bold); dimensions are stated | 01 |
| $W, A, Q, K, V$ | matrices (upper case); $X\in\mathbb{R}^{T\times d}$ stacks one row per position | 02 |
| $x_i$, $x_t$ | the $i$-th example, or the token / input at position or time $t$ | 01 |
| $\mathbf 1$ | the all-ones vector | 04 |
| $e_y$ | the one-hot (standard basis) vector for class $y$ | 02 |
| $\odot$ | element-wise product | 03 |
| $\langle u, v\rangle = u^{\top}v$ | inner product | 02 |
| $\lVert\cdot\rVert$ | Euclidean norm (vectors), spectral norm (matrices) unless stated | 03 |
| $\ln$ | natural logarithm; all information quantities are in **nats** | 01 |

## Data, models and losses

| Symbol | Meaning | First used |
|---|---|---|
| $\mathcal{D} = \{(x_i, y_i)\}_{i=1}^n$ | a training set of $n$ examples | 01 |
| $p_{\text{data}}$ | the (unknown) data distribution | 01 |
| $\theta$ | all trainable parameters of a model; $N = $ their number | 01, 06 |
| $p_\theta(y\mid x)$ | the model's predicted distribution | 01 |
| $\ell(\theta; x, y)$ | the loss on one example, usually $-\ln p_\theta(y\mid x)$ | 01 |
| $L(\theta)$, $\hat L(\theta)$ | population and empirical (training) loss | 01 |
| $V$ | vocabulary size; $\ln V$ = loss of the uniform predictor | 01 |
| $T$ | sequence length (context) | 04 |
| $D$ | number of training tokens | 06 |
| $H(p)$, $H(p, q)$, $\mathrm{KL}(p\,\Vert\,q)$ | entropy, cross-entropy, Kullback–Leibler divergence | 01 |

## Optimization

| Symbol | Meaning | First used |
|---|---|---|
| $\nabla L$ | gradient | 03 |
| $\bar x = \partial L/\partial x$ | adjoint of an intermediate quantity $x$ (reverse mode) | 03 |
| $J_f$ | Jacobian of $f$ | 03 |
| $\eta$ | step size (learning rate) | 03 |
| $B$ | minibatch size | 03 |
| $\beta_1, \beta_2$ | Adam's moment decay rates | 03 |
| $\lambda$ | weight decay, or an eigenvalue (stated locally) | 03 |
| $\kappa$ | condition number $\lambda_{\max}/\lambda_{\min}$ | 03 |

## Architectures

| Symbol | Meaning | First used |
|---|---|---|
| $E\in\mathbb{R}^{V\times d}$ | embedding table; $d$ = model width | 02 |
| $q, k, v$; $Q, K, V$ | queries, keys, values (rows of $Q, K, V$); $d_k$ key width | 04 |
| $A = \mathrm{softmax}(QK^{\top}/\sqrt{d_k})$ | attention weights, row-wise softmax | 04 |
| $h$ (count) | number of heads; $d_h = d/h$ | 04 |
| $h_t$ (vector) | recurrent state at time $t$ | 05 |
| $a_t$ or $\Lambda$ | state transition (decay); $\rho(\cdot)$ spectral radius | 05 |
| $C$ | training compute in FLOPs ($C\approx 6ND$) | 06 |

## Generation and decision

| Symbol | Meaning | First used |
|---|---|---|
| $\tau$ | sampling temperature | 08 |
| $x_0, x_1$; $t\in[0,1]$ | noise and data endpoints; interpolation time in flow matching | 08 |
| $v_\theta(x, t)$ | learned velocity field | 08 |
| $\pi_\theta(a\mid s)$ | policy; $s$ state, $a$ action | 09 |
| $r$, $R$ | reward, return | 09 |
| $\pi_{\text{ref}}$ | reference (pretrained) policy | 09 |
| $\beta$ | inverse temperature in attention (04); KL coefficient in post-training (09) | 04, 09 |

## Evidence

| Symbol | Meaning | First used |
|---|---|---|
| $\hat\mu$, $\mathrm{se}$ | an estimate and its standard error | 01, 10 |
| $\mathrm{ECE}$ | expected calibration error | 10 |
| $\mathrm{CI}_{95}$ | 95% confidence interval | 10 |

## Typographic conventions

- **Worked examples** are numbered by unit (Worked example 4.3); **propositions, theorems and corollaries** likewise (Theorem 5.2).
- **★ Advanced** marks proofs and generalizations that can be skipped on a first reading.
- **[D] / [E] / [O] / [H]** in "Modern developments" label a claim as derived, experimental evidence, engineering observation or open hypothesis.
- **[AI-off]** marks problems for independent practice and assessment.
