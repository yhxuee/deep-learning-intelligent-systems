---
title: "5.9 Experiments"
description: Eight experiments that turn the unit's derivations into measurements, each with a prediction to write first, a procedure, and what to look for.
sidebar: { label: "5.9 Experiments", order: 9 }
---

Each experiment tests a specific result of this unit. **Write your prediction before running anything** — a number, a sign, or a curve shape, with one sentence of reasoning. The derivations you need are referenced in each entry.

Code lives in [`labs/unit-05-memory`](https://github.com/yhxuee/deep-learning-intelligent-systems/tree/main/labs/unit-05-memory) (AI-off construction: `memory.py`, tested by `test_memory.py`) and in [`figures/make_figures.py`](https://github.com/yhxuee/deep-learning-intelligent-systems/tree/main/figures), which regenerates every figure in the unit. AI assistance is appropriate for plotting, sweeps and boilerplate in experiments 5–8; the predictions and interpretations are yours.

## Experiment 5.1 — The noise–lag trade-off

**Tests:** §5.2.6. **Predict:** the measured variance of the EMA output on pure noise, for $\alpha \in \{0.5, 0.9, 0.99\}$, as a multiple of $\sigma^2$.

**Procedure.** Generate $10^5$ samples of $\mathcal N(0,1)$ noise; filter with each $\alpha$; discard the first $10/(1-\alpha)$ samples; measure the variance. Then filter a noiseless step and measure the time to reach 95%.

**Look for:** variance within a few percent of $(1-\alpha)/(1+\alpha)$; response times matching $\ln 0.05/\ln\alpha$. Then reproduce Figure 5.1.

## Experiment 5.2 — When discretization explodes

**Tests:** §5.2.8. **Predict:** for forward Euler with $\Delta t/\tau \in \{0.5, 1.5, 1.99, 2.01, 2.5\}$, which runs stay bounded over 200 steps?

**Procedure.** Iterate $h_t = (1-\Delta t/\tau)h_{t-1} + (\Delta t/\tau)x_t$ with a constant input $x_t = 1$ and $h_0 = 0$; also iterate the exact update with $\alpha = e^{-\Delta t/\tau}$.

**Look for:** bounded oscillation for $1 < \Delta t/\tau < 2$ (negative $\alpha$, decaying), growth beyond 2, and an exact update that converges to 1 in every case.

## Experiment 5.3 — Eigenvalues as memory modes

**Tests:** §5.3.3. **Predict:** the impulse response of $A = V\,\mathrm{diag}(0.5, 0.95)\,V^{-1}$ for a random invertible $V$ — which mode dominates after 10 steps, and roughly by what factor?

**Procedure.** Implement `convolution_kernel(A, B, length)` in the lab, compare with `recurrence_loop` on a random input (test `test_convolution_equivalence_with_zero_initial_state`), and plot $CA^jB$.

**Look for:** a late-time decay rate of $0.95^j$ regardless of $V$, and an early shape that depends on $V$, $B$ and $C$ (Worked Example 5.3).

## Experiment 5.4 — Transient growth

**Tests:** §5.3.5. **Predict:** for $A_c = \begin{pmatrix}0.9 & c\\0&0.9\end{pmatrix}$ with $c \in \{0.5, 5, 50\}$, the peak of $\|A_c^t\|_2$ and the step at which it occurs.

**Procedure.** Use `influence_norms(A, 200)`.

**Look for:** a peak near $t\approx 9.5$ for all $c$ (the location does not depend on $c$ — why?) with height approximately $c\cdot t^*\,0.9^{t^*-1}$ when $c$ is large.

## Experiment 5.5 — Gradients through time

**Tests:** §5.3.7. **Predict:** the slope of $\log_{10}\|\partial h_t/\partial h_k\|$ against $t-k$ for a *linear* recurrence with $\rho(A) = 0.9$ — then for a tanh RNN with $\rho(W) = 0.9$. Which is steeper, and why?

**Procedure.** Accumulate the Jacobian products as in `u05_bptt_gradient_decay` in `figures/make_figures.py`; repeat with $W$ a scaled random orthogonal matrix (all singular values equal).

**Look for:** a slope of $\log_{10}0.9 \approx -0.046$ per step for the linear case; a steeper slope for tanh (saturation); and much slower decay for orthogonal $W$ at the same spectral radius — the singular values, not only the eigenvalues, govern the products.

## Experiment 5.6 — Gates make long dependencies learnable

**Tests:** §5.3.8. **Predict:** the largest delay $D$ at which a vanilla RNN and an LSTM (same number of parameters) learn to output, at the last step, the first token of a random sequence of length $D$.

**Procedure** (framework autograd allowed). Vocabulary of 8 symbols; sequences of length $D \in \{5, 10, 20, 50, 100\}$; train each model to 99% accuracy or 2 000 steps; record success. Repeat with three seeds.

**Look for:** the RNN failing at modest $D$ while the LSTM succeeds much further — and whether initializing the LSTM forget-gate bias to 1 changes the threshold. Report seed variance; a single run is not evidence (Unit 12).

## Experiment 5.7 — Capacity of an associative memory

**Tests:** §5.5.3, Theorem 5.3. **Predict:** the relative recall error at $n = d_k$ and at $n = 4d_k$ for Hebbian storage with random keys; then for the delta rule.

**Procedure.** The following computes the Hebbian case; extend it with the delta rule $S \leftarrow S + (v - Sk)k^\top$ applied for several passes over the pairs.

```python
import numpy as np
rng = np.random.default_rng(0)
d_k, d_v = 16, 8
for n in [4, 16, 32, 64]:
    K = rng.normal(size=(n, d_k)); K /= np.linalg.norm(K, axis=1, keepdims=True)
    V = rng.normal(size=(n, d_v))
    S = V.T @ K                          # sum of outer products v_j k_j^T
    err = np.linalg.norm(K @ S.T - V, axis=1) / np.linalg.norm(V, axis=1)
    print(n, err.mean().round(3), np.sqrt((n - 1) / d_k).round(3))
```

**Look for:** Hebbian error near $\sqrt{(n-1)/d_k}$ (Figure 5.6). Delta-rule error at machine precision for $n$ well below $d_k$ (independent keys suffice, Theorem 5.3 part 2). At $n = d_k$, expect the delta rule to be *slow*, not wrong: with 200 passes we measured a mean error of about 0.30 at $n = d_k = 16$, because random keys in that regime are nearly dependent (the key matrix is ill-conditioned). Beyond $d_k$ the error cannot reach zero. Explain the slowdown using the condition number of $K$ — the same quantity that slowed gradient descent in Unit 03.

## Experiment 5.8 — Spine Stage 05: the KV cache and a recall stress test

**Tests:** §5.3.11 and the [spine project](/spine/). **Predict:** the speed-up in tokens per second from KV-cache decoding at context 256, and the distance beyond which the Stage 04 model fails a recall test.

**Procedure.** Implement cached decoding for the spine model. Verify that cached logits equal full recomputation at every prefix (the invariant tested in `labs/unit-04-attention`). Then build sequences "`… key value … key`" with the distance between the two occurrences varied from 4 to the context length, and measure accuracy on the value.

**Look for:** exact equality of logits (to float tolerance) — if not, check the position index (the classic bug, [KV cache](/atlas/nodes/mech-kv-cache/)); a recall curve that stays high within the trained context and collapses beyond it (§5.7.6). Include the control "same sequence with the first occurrence removed" (§5.7.9).
