---
title: "5.7 Limitations and counterexamples"
description: Where each memory design fails — with explicit constructions, the evidence for each failure, and what the failure does and does not imply.
sidebar: { label: "5.7 Limitations", order: 7 }
---

A mechanism is understood when you can say exactly where it breaks. Each item below states a failure, gives a construction or evidence, and classifies the claim (derivation, experimental evidence, engineering observation, or open hypothesis — see [Governance](/project/governance/)).

## 5.7.1 Fixed-size states face a precision or rank bottleneck

**Claim.** Two precise statements replace the careless slogan “an $N$-dimensional state remembers at most $N$ items.” A state with $b$ physical bits cannot losslessly distinguish more than $2^b$ histories (Theorem 5.2). A conditionally linear recurrence with an $N$-dimensional state cannot transmit more than $N$ independent linear directions across a time cut (§5.5.4). Neither statement, by itself, is a theorem about arbitrary infinite-precision nonlinear recurrences. *Derivation.*

**Construction.** Feed $n$ independent key--value pairs followed by queries for their values (*multi-query associative recall*). If the layer is conditionally linear and the answers require $n$ independent output directions across one cut, its past-to-future block needs rank at least $n$; an $N<n$ state cannot provide that linear map. A nonlinear finite-precision model is instead covered by the bit-counting argument, with a bound that depends on alphabet size and required error.

**Evidence at scale.** In the architectures and training setup studied by [Arora et al. (2024)](/references/#arora-2024), attention substantially outperforms gated convolutions on multi-query associative recall, and the required gated-convolution width grows with sequence length. Under the formal model and precision assumptions of [Jelassi et al. (2024)](/references/#jelassi-2024), a two-layer Transformer can copy strings exponentially long in its size while generalized fixed-state models face a state bottleneck; their synthetic and pretrained-model experiments show the same qualitative gap. These results support a scoped comparison, not “every attention model beats every recurrence.” *Theorem plus experimental evidence.*

**What it does not imply.** It does not say recurrent models are worse on average language-modeling loss, nor that they cannot recall the *particular* items a task makes important. Selectivity (§5.3.10), structure in the data and external memory can all change the task. Capacity is a claim about a specified distribution, precision and error tolerance.

## 5.7.2 ρ(A) < 1 does not mean gentle forgetting; ρ(W) ≥ 1 does not mean explosion

**Construction 1.** $A = \begin{pmatrix}0.9&5\\0&0.9\end{pmatrix}$ amplifies inputs almost twentyfold for about ten steps (§5.3.5, Worked Example 5.5). *Derivation.*

**Construction 2.** In the numerical example behind Figure 5.5, a tanh RNN whose weight matrix has spectral radius 1.5 still shows *vanishing* Jacobian products over 60 steps, because saturation contributes $\mathrm{diag}(1-h^2)$ at every step. This is an illustrative simulation, not evidence that every RNN with $\rho(W)=1.5$ behaves that way. *Computed counterexample (`figures/make_figures.py`).*

**Lesson.** Eigenvalues describe the linear, long-run regime. Short-run and nonlinear behaviour must be measured or analysed with the full matrix.

## 5.7.3 Discretization can create or hide instability

Forward Euler turns the stable system $\tau\dot h=x-h$ into an exploding recurrence for $\Delta t>2\tau$ (§5.2.8). Conversely, exact discretization of an *unstable* continuous mode with positive real part gives $|\bar\lambda|=e^{\Delta\,\mathrm{Re}\lambda}>1$ for every positive $\Delta$: exact sampling cannot turn growth into decay. Many SSM parameterizations intended to provide stable fading memory therefore constrain $\mathrm{Re}\,\lambda<0$ explicitly; other designs may intentionally use marginal modes and need different controls. *Derivation.*

## 5.7.4 Gates can get stuck

If a forget gate saturates at $f\approx1$ (or $0$), the logistic derivative $\sigma'(z)=\sigma(z)(1-\sigma(z))$ is near zero. The direct gradient through that gate logit is then small, so changing the gate can become slow. Other loss paths, later states and optimizer history may still move its parameters; “stuck” is a conditioning warning, not an impossibility theorem. Gate-bias initialization (§5.3.8) and normalization are engineering responses. *Derivation for the local mechanism; practical severity is an engineering observation.*

## 5.7.5 Backpropagation through time is expensive or biased

A straightforward exact BPTT implementation stores $h_1,\dots,h_T$, using $O(TN)$ activation memory. Truncation bounds that memory and removes direct gradient paths longer than its window; relative to the full-sequence gradient it is generally *biased*. It does **not** impose the same hard limit on behavior: carried state, overlapping chunks or local losses can still teach dynamics that persist longer. Checkpointing (Unit 03, §3.7.6) trades recomputation for memory. *Derivation.*

## 5.7.6 Attention forgets abruptly

Ordinary full causal attention retains the **encoded** key and value for each token still in its cache. If an older token is removed and no other state summarizes it, that direct path becomes exactly zero—an abrupt cutoff rather than geometric decay. “Encoded” matters: the cache is not a lossless copy of the raw token or all activations, and finite precision still applies. Retention is also not recall. A stored key can be diluted, misweighted or ignored (Unit 04, §4.7), so effective retrieval distance is an empirical property of the trained model. Serving systems may evict, quantize or compress entries, and sliding-window architectures deliberately change the boundary.

The cache also grows linearly with retained context: for the stated GPT-2-small configuration it is 37.7 decimal MB per sequence at 1,024 tokens and 16-bit elements ([KV cache](/atlas/nodes/mech-kv-cache/)). Whether performance generalizes beyond training lengths depends on position handling, data and learned computation—not position encoding alone (Unit 04, §4.7.3; [Press et al. 2022](/references/#press-2022)). *Derivation for the storage and cutoff; experimental question for retrieval and length generalization.*

## 5.7.7 Associative memories interfere

Outer-product storage is exact for **all possible values** only for orthonormal keys. With the random-key assumptions of §5.5.3, the ratio of cross-talk RMS to signal RMS is $\sqrt{(n-1)/d_k}$. Correlated keys can interfere more strongly, although special correlations among values can also cancel error. An ideal linear solver—and a delta rule under suitable convergence conditions—removes the orthogonality requirement, but arbitrary exact associations still require independent keys and $n\le d_k$ (Theorem 5.3). *Derivation under stated assumptions.*

## 5.7.8 Capacity claims need stated assumptions

A single real number can encode an entire finite history if arithmetic is exact, although the code is exponentially fragile (§5.5.2). Any claim of the form “a state of size $d$ can hold $O(d)$ items” needs assumptions such as finite precision, noise tolerance, Lipschitz/robust decoding or restricted dynamics. When reading a capacity result, write down four things: the input family, state precision, allowed error and decoder class. *Derivation (counterexample).*

## 5.7.9 Benchmarks may not test memory at all

A model can score well on a long-context benchmark without using long-range information if nearby text or dataset artifacts predict the answer. Recall claims need matched controls: remove or replace the distant evidence while preserving other cues; move the evidence across distances without changing content; and compare against a clearly defined no-memory baseline. Performance should approach that baseline when the only valid evidence is removed—not an undefined universal “chance.” Unit 10 develops these controls. *Methodological; see [Unit 10](/core/unit-10/).*
