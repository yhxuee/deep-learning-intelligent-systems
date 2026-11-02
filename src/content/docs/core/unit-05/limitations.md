---
title: "5.7 Limitations and counterexamples"
description: Where each memory design fails — with explicit constructions, the evidence for each failure, and what the failure does and does not imply.
sidebar: { label: "5.7 Limitations", order: 7 }
---

A mechanism is understood when you can say exactly where it breaks. Each item below states a failure, gives a construction or evidence, and classifies the claim (derivation, experimental evidence, engineering observation, or open hypothesis — see [Governance](/project/governance/)).

## 5.7.1 Fixed-size states cannot recall arbitrary content

**Claim.** A recurrent model with an $N$-dimensional state cannot recall arbitrarily many independent items exactly. *Derivation* (Theorem 5.2; the rank bound of §5.5.4).

**Construction.** Feed $n$ random key–value token pairs followed by one of the keys; ask for its value (*associative recall*). With a linear readout, the past-to-future block of the layer's matrix has rank at most $N$, so for $n > N$ some pairs cannot be distinguished.

**Evidence at scale.** On multi-query associative recall, attention-free models trail attention at matched size, and the gap grows with the number of pairs ([Arora et al. 2024](/references/#arora-2024)). For copying long strings, two-layer Transformers can provably copy sequences exponentially long in their size, whereas models with a fixed-size state are limited by that state; experiments agree ([Jelassi et al. 2024](/references/#jelassi-2024)). *Experimental evidence plus theorem.*

**What it does not imply.** It does not say recurrent models are worse on average language-modelling loss, nor that they cannot recall the *particular* items a task makes important — selectivity (§5.3.10) exists exactly to spend limited capacity well.

## 5.7.2 ρ(A) < 1 does not mean gentle forgetting; ρ(W) ≥ 1 does not mean explosion

**Construction 1.** $A = \begin{pmatrix}0.9&5\\0&0.9\end{pmatrix}$ amplifies inputs almost twentyfold for about ten steps (§5.3.5, Worked Example 5.5). *Derivation.*

**Construction 2.** A tanh RNN whose weight matrix has spectral radius 1.5 still shows *vanishing* gradients over 60 steps (Figure 5.5), because saturation multiplies each Jacobian by $\mathrm{diag}(1-h^2)$. *Experimental evidence (simulation in `figures/make_figures.py`).*

**Lesson.** Eigenvalues describe the linear, long-run regime. Short-run and nonlinear behaviour must be measured or analysed with the full matrix.

## 5.7.3 Discretization can create or hide instability

Forward Euler turns the stable system $\tau\dot h = x-h$ into an exploding recurrence for $\Delta t > 2\tau$ (§5.2.8). Conversely, exact discretization of an *unstable* continuous system (eigenvalue with positive real part) gives $|\bar\lambda| = e^{\Delta\,\mathrm{Re}\lambda} > 1$ for every $\Delta$: no choice of step can make a growing mode decay. Parameterizations used in state-space models therefore constrain $\mathrm{Re}\,\lambda < 0$ explicitly. *Derivation.*

## 5.7.4 Gates can get stuck

If a forget gate saturates at $f\approx1$ (or $0$), the derivative of the logistic function, $\sigma'(z) = \sigma(z)(1-\sigma(z))$, is near zero, so gradients with respect to the gate's own parameters vanish: a gate can become unable to learn to open or close. Initialization of gate biases (§5.3.8) and normalization of gate inputs are engineering responses. *Derivation for the mechanism; the size of the effect in practice is an engineering observation.*

## 5.7.5 Backpropagation through time is expensive or biased

Exact BPTT stores every state $h_1,\dots,h_T$: memory $O(TN)$. Truncated BPTT bounds memory but returns a gradient that ignores dependencies longer than the truncation window — a *biased* estimate that cannot teach those dependencies at all. Checkpointing (Unit 03, §3.7.6) trades recomputation for memory. *Derivation.*

## 5.7.6 Attention forgets abruptly

The KV cache never forgets *inside* the context window and forgets *completely* outside it: a token one position beyond the window has exactly zero influence, unlike the graceful geometric decay of a recurrence. And the cache's memory grows linearly with context: for GPT-2 small it is 37.7 MB per sequence at 1 024 tokens in 16-bit precision ([KV cache](/atlas/nodes/mech-kv-cache/)). Whether attention generalizes to sequences longer than those seen in training depends heavily on the positional encoding (Unit 04, §4.7.3; [Press et al. 2022](/references/#press-2022)). *Derivation (cut-off); experimental evidence (length generalization).*

## 5.7.7 Associative memories interfere

Outer-product storage is exact only for orthonormal keys; with $n$ random keys in $d_k$ dimensions the relative error grows like $\sqrt{(n-1)/d_k}$ (§5.5.3). Correlated keys — common in real data, where similar contexts produce similar keys — interfere more than random ones. The delta rule removes the orthogonality requirement but not the dimension limit $n\le d_k$ (Theorem 5.3). *Derivation.*

## 5.7.8 Capacity claims need stated assumptions

A single real number can store an entire history if arithmetic is exact (§5.5.2). Any claim of the form "a state of size $d$ can hold $O(d)$ items" assumes finite precision, noise, or restricted dynamics. When you read such a claim, identify the assumption; when you make one, state it. *Derivation (counterexample).*

## 5.7.9 Benchmarks may not test memory at all

A model can score well on a long-context benchmark without using long-range information if the answer is predictable from nearby text. Recall claims need controls: the same task with the relevant information removed (performance should drop to chance), and with the distance varied systematically. Unit 10 develops these controls. *Methodological; see [Unit 10](/core/unit-10/).*
