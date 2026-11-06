---
title: Notation
description: The notation shared by all twelve units, so that a symbol means the same thing from Unit 01 to Unit 12.
sidebar: { order: 3 }
---

Notation is compressed language. You are not supposed to recognize every symbol before beginning the book, and this page is a lookup table rather than a prerequisite exam. When an equation feels dense, read it in three passes:

1. **name the objects** — which are scalars, vectors, matrices, probabilities or sets?
2. **check the shapes and range** — can the stated operations legally fit together, and should the result be a number or an array?
3. **say the operation in words** — only then substitute numbers or manipulate the algebra.

Every unit uses the conventions below. Where a symbol is overloaded by long-standing usage (for example $\beta$), the meaning is stated locally and listed here. A local definition always wins.

## Decode one equation before using the tables

Consider the per-example loss

$$
\ell(\theta;x,y)=-\ln p_\theta(y\mid x).
$$

- $x$ is the input and $y$ is its observed target.
- $\theta$ denotes all trainable parameters, not a multiplication factor.
- $p_\theta(y\mid x)$ is one number between 0 and 1: the probability the model assigns to the observed target after seeing $x$.
- $\ln$ is the natural logarithm. The leading minus sign makes confident correct predictions have small loss and assigns large loss to a target given tiny probability.

For example, probability $0.5$ gives $-\ln 0.5\approx0.693$ **nats**, while probability $0.01$ gives about $4.605$ nats. A nat is the amount produced by using the natural logarithm; it is a unit of log-loss, not a new kind of probability. Changing to logarithm base 2 would express the same information in bits.

## Shapes are part of the meaning

This book does not bold every vector, so dimensions carry important information. If $X\in\mathbb R^{T\times d}$ and $W\in\mathbb R^{d\times m}$, then $XW\in\mathbb R^{T\times m}$: the inner dimensions $d$ agree, and the operation transforms each of the $T$ rows from width $d$ to width $m$. An expression such as $WX$ is generally invalid for those shapes. Writing shapes beside each intermediate is one of the fastest ways to find an error.

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

Indices are labels, not powers: $x_i$ means “the item indexed by $i$,” whereas $x^2$ means $x$ squared. A superscript may also label a layer or dataset when explicitly stated; context and the local definition decide.

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

## When the same letter means two things

Mathematics reuses a small alphabet. Do not infer meaning from the letter alone. For each equation, make a tiny local legend and check type and shape. In this book in particular:

- $V$ can be the vocabulary size when it is a scalar, or the value matrix in attention when it has rows and columns;
- $h$ can count attention heads, while $h_t$ is a vector state at time $t$;
- $\beta$ is an inverse temperature in Unit 04 and a KL coefficient in Unit 09;
- $L$ usually denotes a loss, but an architecture table may use $L$ for the number of layers when it says so explicitly.

If the local meaning is missing or two shape-consistent readings remain possible, that is a documentation defect rather than something the reader should guess. Record the equation and report it through the [errata page](/start/errata/).
