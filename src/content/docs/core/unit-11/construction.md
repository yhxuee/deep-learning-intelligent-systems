---
title: "11.3 Mathematical construction"
description: The residual stream as a sum of contributions, the QK and OV circuits of an attention head, composition between layers, an induction circuit built weight by weight, ablation and activation patching defined as interventions, and probes with control tasks.
sidebar: { label: "11.3 Construction", order: 3 }
---

## 11.3.1 The residual stream is a sum

In a Transformer without layer normalization, the input to the read-out is a sum: the token embedding, the position embedding, and the output of every head and MLP that wrote to the residual stream (Unit 04, §4.3.8). For an attention-only model with tied embeddings,

$$
\text{logits}_t = E\Big(e_{x_t} + P_t + \sum_{\ell, h}\mathrm{head}_{\ell, h}(t)\Big),
$$

so each component's **direct contribution** to each logit can be computed exactly: $E\,\mathrm{head}_{\ell,h}(t)$ ([Elhage et al. 2021](/references/#elhage-2021)). Contributions also flow *indirectly*, when a later component reads what an earlier one wrote; this composition is what makes circuits. With layer normalization the sum is rescaled at each read, and direct contributions become approximations.

## 11.3.2 A head is two circuits

A head with query, key, value and output matrices $W_Q, W_K, W_V\in\mathbb{R}^{d\times d_h}$ and $W_O\in\mathbb{R}^{d_h\times d}$ computes, for residual vectors $x_i$ (query position) and $x_j$ (key position),

$$
\text{score}_{ij} = \frac{x_i^{\top}W_QW_K^{\top}x_j}{\sqrt{d_h}},\qquad \text{output}_i = \sum_j A_{ij}\,x_j^{\top}W_VW_O .
$$

Two $d\times d$ matrices determine the head: the **QK circuit** $W_QW_K^{\top}$, which decides *where to look*, and the **OV circuit** $W_VW_O$, which decides *what to move*. Each has rank at most $d_h$. Reading them in the embedding basis — for example $E\,W_QW_K^{\top}E^{\top}$, a $V\times V$ table of which token attends to which — is often the fastest way to understand a head.

## 11.3.3 Composition

A layer-1 head can read what a layer-0 head wrote. If it uses that information in its **keys**, it is *K-composition*: "attend to positions whose layer-0 output says X". In its **queries**: *Q-composition*. In its **values**: *V-composition*. Induction is the standard example of K-composition: layer 0 writes "the token before me" at every position; layer 1 attends to positions whose previous token equals the current token.

## 11.3.4 An induction circuit, weight by weight

Take $K$ ordinary tokens plus a beginning-of-sequence token (vocabulary $V = K+1$), sequences $[\text{BOS}, s, s]$ of length $2K+1$ with $s$ a permutation of the $K$ tokens, and an attention-only model with two layers of two heads. Lay out the residual stream in three blocks of one-hot coordinates, plus one unused coordinate so that the width is even:

$$
x = [\ \underbrace{\text{token}}_{V}\ \mid\ \underbrace{\text{position}}_{2K+1}\ \mid\ \underbrace{\text{previous token}}_{V}\ \mid\ 0\ ],\qquad d = 2V + 2K + 2 .
$$

Embeddings write the token block ($E[v] = $ one-hot $v$) and the position block ($P[t] = $ one-hot $t$). With head width $d_h = d/2\ge2K+1$, set:

- **Head 0.0 — previous-token head.** Query from position $t$: $c\,\mathbf u_t$. Key at position $j$: $\mathbf u_{j+1}$. Score $c[j = t - 1]/\sqrt{d_h}$: with $c$ large, position $t$ attends to $t - 1$. Value: the token block. Output: write it into the **previous-token** block.
- **Head 1.0 — induction head.** Query: the current token, scaled by $c$. Key: the previous-token block. Score is large exactly where "the token before $j$" equals "my token": at $j$ = one past the earlier occurrence. Value: the token block. Output: write it, scaled by $\lambda$, into the token block, which the tied read-out turns into logits.
- **Heads 0.1 and 1.1:** all weights zero.

On the second copy of $s$, position $i$ holds $s_k$; its earlier occurrence is at $k+1$ and the token after it, $s_{k+1}$, is the next token. Head 1.0 attends there and adds $\lambda$ to the logit of $s_{k+1}$, while the embedding adds 1 to the logit of the current token. With $c = 60$, $\lambda = 10$ and $K = 8$ ($d = 36$, $d_h = 18$), the correct token receives probability 0.9996 and the loss on the predictable part of the repeat is 0.0004 nats (Worked Example 11.2). The BOS token matters: without it, position 0's previous-token block would hold its own token, creating a spurious second match.

`stage11_induction.build_induction_circuit` constructs exactly these weights; the spine tests check every claim of §11.2.4 on it.

## 11.3.5 Interventions, defined

Let $f(x)$ be the model's output and $h_c(x)$ the activation of a component $c$ on input $x$. Write $f(x\mid h_c\leftarrow a)$ for the output when the activation of $c$ is forced to $a$ while everything upstream is computed normally — the **do-operator** of causal inference ([Pearl 2009](/references/#pearl-2009)).

- **Zero ablation:** $f(x\mid h_c\leftarrow 0)$. **Mean ablation:** $a = \mathbb{E}_{x'}[h_c(x')]$ over a reference distribution — keeps the activation in a typical range. **Resample ablation:** $a = h_c(x')$ for a random $x'$.
- **Activation patching:** given a clean input $x_{\text{c}}$ and a corrupted input $x_{\text{x}}$ that differ in one controlled respect, compute $f(x_{\text{x}}\mid h_c\leftarrow h_c(x_{\text{c}}))$. If the patched output recovers the clean behaviour, component $c$ carries the information that distinguishes the two inputs ([Zhang & Nanda 2024](/references/#zhang-nanda-2024); [Heimersheim & Nanda 2024](/references/#heimersheim-2024)).

Two invariants check an implementation (both in `test_model.py` and `test_spine.py`): patching a run with its *own* activations changes nothing, and zero ablation equals patching with zeros.

## 11.3.6 Probes and their controls

A **probe** is a simple classifier (usually linear) trained to predict a property from a hidden state. High probe accuracy shows the property is *decodable*. It does not show the model uses it, and with expressive probes it may not even show the representation encodes it simply — the probe may learn the task itself. **Control tasks** address the second problem: train the same probe to predict a random relabelling of the same inputs; the difference between real and control accuracy, the *selectivity*, measures how much the representation, rather than the probe, carries ([Hewitt & Liang 2019](/references/#hewitt-2019)). The first problem needs an intervention: change the decodable information and see if behaviour changes.

## 11.3.7 Gradient attribution

The simplest attribution of an output $f$ to an input or activation $z$ is $z\odot\nabla_zf$ ("gradient × input"). **Integrated gradients** averages the gradient along a straight path from a baseline $z_0$ to $z$ and multiplies by $z - z_0$; it satisfies a completeness axiom — attributions sum to $f(z) - f(z_0)$ ([Sundararajan et al. 2017](/references/#sundararajan-2017)). Attributions are first-order approximations of interventions; they can be misleading where the function is far from linear (saturated softmaxes, as in the hand-built circuit) and some popular saliency methods produce similar maps for trained and randomly initialized networks ([Adebayo et al. 2018](/references/#adebayo-2018)) — a failure their own sanity checks were designed to expose.
