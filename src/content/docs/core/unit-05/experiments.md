---
title: "5.9 Experiments"
description: Nine reproducible experiments that turn the unit's derivations into measurements, with predictions, controls, metrics and interpretation boundaries.
sidebar: { label: "5.9 Experiments", order: 9 }
---

Each experiment tests a specific result of this unit. **Write your prediction before running anything**—a number, sign or curve shape, with one sentence of reasoning—but do not turn that prediction into a required result. Record the seed, software version, dtype and hardware; keep raw measurements; report uncertainty across repetitions. The derivations you need are referenced in each entry.

Code lives in [`labs/unit-05-memory`](https://github.com/yhxuee/deep-learning-intelligent-systems/tree/main/labs/unit-05-memory) (AI-off construction: `memory.py`, tested by `test_memory.py`), [`labs/beyond_language.py`](https://github.com/yhxuee/deep-learning-intelligent-systems/blob/main/labs/beyond_language.py) for the sensor experiment, and [`figures/make_figures.py`](https://github.com/yhxuee/deep-learning-intelligent-systems/tree/main/figures), which regenerates the figures. AI assistance is appropriate for plotting, sweeps and boilerplate; predictions, controls and interpretations remain yours.

## Experiment 5.1 — The noise–lag trade-off

**Tests:** §5.2.6. **Predict:** the measured variance of the EMA output on pure noise, for $\alpha \in \{0.5, 0.9, 0.99\}$, as a multiple of $\sigma^2$.

**Procedure.** Generate $10^5$ samples of $\mathcal N(0,1)$ noise; filter with each $\alpha$; discard the first $10/(1-\alpha)$ samples; measure the variance. Then filter a noiseless step and measure the time to reach 95%.

**Look for:** variance within a few percent of $(1-\alpha)/(1+\alpha)$; response times matching $\ln 0.05/\ln\alpha$. Then reproduce Figure 5.1.

## Experiment 5.2 — When discretization explodes

**Tests:** §5.2.8. **Predict:** for forward Euler with $\Delta t/\tau \in \{0.5, 1.5, 1.99, 2.01, 2.5\}$, which runs stay bounded over 200 steps?

**Procedure.** Iterate $h_t = (1-\Delta t/\tau)h_{t-1} + (\Delta t/\tau)x_t$ with a constant input $x_t = 1$ and $h_0 = 0$; also iterate the exact update with $\alpha = e^{-\Delta t/\tau}$.

**Look for:** bounded oscillation for $1 < \Delta t/\tau < 2$ (negative $\alpha$, decaying), growth beyond 2, and an exact update that converges to 1 in every case.

## Experiment 5.3 — Eigenvalues as memory modes

**Tests:** §5.3.3. **Predict:** the impulse response of $A=V\,\mathrm{diag}(0.5,0.95)\,V^{-1}$ for a fixed random invertible $V$, with $B$ and $C$ chosen so both eigenmodes have nonzero coefficients. Which mode dominates after 10 steps, and by what factor in your chosen coordinates?

**Procedure.** Implement `convolution_kernel(A, B, length)` in the lab, compare with `recurrence_loop` on a random input (test `test_convolution_equivalence_with_zero_initial_state`), and plot $CA^jB$.

**Look for:** when the slow mode is both excited by $B$ and observed by $C$, the asymptotic rate is $0.95^j$; the early shape and coefficient depend on $V,B,C$. As a counter-control, choose $B$ or $C$ to cancel the slow mode and verify that $0.5^j$ then dominates. Eigenvalues present in $A$ need not appear in an input--output kernel.

## Experiment 5.4 — Transient growth

**Tests:** §5.3.5. **Predict:** for $A_c = \begin{pmatrix}0.9 & c\\0&0.9\end{pmatrix}$ with $c \in \{0.5, 5, 50\}$, the peak of $\|A_c^t\|_2$ and the step at which it occurs.

**Procedure.** Use `influence_norms(A, 200)`.

**Look for:** integer peaks near steps 9--10 for these values. The off-diagonal approximation $|c|\,t\,0.9^{t-1}$ peaks at $t^*=-1/\ln0.9\approx9.49$, independent of $c$; the **exact spectral norm** also contains diagonal terms, so its maximizing step can depend slightly on $c$. Report the exact discrete peak rather than claiming invariance.

## Experiment 5.5 — Gradients through time

**Tests:** §5.3.7. **Predict:** first use the controlled linear recurrence $A=0.9Q$ with orthogonal $Q$. Predict the slope of $\log_{10}\|\partial h_t/\partial h_k\|_2$ against $t-k$. Then use $W=0.9Q$ in a tanh RNN under (a) small inputs that keep it near zero and (b) large inputs that saturate it.

**Procedure.** Accumulate Jacobian products as in `u05_bptt_gradient_decay` in `figures/make_figures.py`. Use the same $Q$, initial state and input sequence when comparing cases; repeat over at least five input seeds. Add a non-normal matrix rescaled to spectral radius 0.9 as a counterexample to any spectral-radius-only prediction.

**Look for:** exactly $\log_{10}0.9\approx-0.046$ per step for $A=0.9Q$. The near-zero tanh trajectory should approach that slope; saturation should make it steeper. The non-normal case may show transients despite the same spectral radius. Singular values give one-step bounds, while the aligned product of trajectory-dependent Jacobians determines the measured curve.

## Experiment 5.6 — Gates make long dependencies learnable

**Tests:** §5.3.8. **Predict:** the largest delay $D$ at which a vanilla RNN and an LSTM, matched to within 5% in trainable parameters, learn to output the first token of a random sequence at the last step. Treat “LSTM goes farther” as a hypothesis to test, not the acceptance criterion.

**Procedure** (framework autograd allowed). Use a vocabulary of 8 symbols and $D\in\{5,10,20,50,100\}$. Generate fresh training batches but freeze a separate 10,000-example test set for each $D$ and seed. Predeclare optimizer, batch size, clipping and a budget of 2,000 updates; success means at least 99% **test** accuracy. Choose hidden sizes to match parameter counts, and run at least five seeds. Repeat the LSTM with forget-gate bias 0 and 1.

**Look for:** plot success probability and learning curves against $D$, with seed-level points. If the LSTM does not go farther, inspect saturation, optimization budget and parameter matching before changing the hypothesis. Report negative as well as positive results; a single run is not evidence (Unit 12).

## Experiment 5.7 — Capacity of an associative memory

**Tests:** §5.5.3, Theorem 5.3. **Predict:** the **ratio of error RMS to value RMS** at $n=d_k$ and $n=4d_k$ for Hebbian storage with random keys; then predict how the delta rule changes convergence below, at and above $d_k$.

**Procedure.** The following computes the Hebbian case; extend it with the delta rule $S \leftarrow S + (v - Sk)k^\top$ applied for several passes over the pairs.

```python
import numpy as np
rng = np.random.default_rng(0)
d_k, d_v = 16, 8
for n in [4, 16, 32, 64]:
    K = rng.normal(size=(n, d_k)); K /= np.linalg.norm(K, axis=1, keepdims=True)
    V = rng.normal(size=(n, d_v))
    S = V.T @ K                          # sum of outer products v_j k_j^T
    E = K @ S.T - V
    rms_ratio = np.sqrt(np.mean(np.sum(E**2, axis=1))) / np.sqrt(np.mean(np.sum(V**2, axis=1)))
    print(n, rms_ratio.round(3), np.sqrt((n - 1) / d_k).round(3))
```

**Look for:** the RMS ratio, not the mean of per-item relative errors, should follow $\sqrt{(n-1)/d_k}$ in expectation. For the delta rule, record residual versus pass and $\kappa(K)$ across many seeds. When $n<d_k$ and keys are independent it can converge to numerical precision; near $n=d_k$, random square key matrices are often ill-conditioned, so convergence can be very slow and highly variable. For $n>d_k$, arbitrary random values are inconsistent with exact linear recall and a nonzero residual remains. Do not hard-code one error after 200 passes as a theorem.

## Experiment 5.8 — Spine Stage 05: the KV cache and a recall stress test

**Tests:** §5.3.11 and the [spine project](/spine/). **Predict:** a range—not a universal number—for KV-cache speed-up at context 256 on your recorded hardware, and a recall-accuracy curve versus distance for the Stage 04 model.

**Procedure.** Implement cached decoding for the spine model. In evaluation mode, verify cached and full-recomputation logits at every prefix to a declared tolerance (the invariant tested in `labs/unit-04-attention`). Benchmark after warm-up at fixed batch size, dtype and output length, reporting median and dispersion over repeated runs. Then build sequences “`… key value … key`” with distance varied from 4 to the supported context length. Use held-out keys and matched controls: remove the first pair, permute its value, and keep local token statistics fixed.

**Look for:** agreement of logits within the declared numerical tolerance—if not, first check position indices and cache ordering ([KV cache](/atlas/nodes/mech-kv-cache/)). Interpret the recall curve only relative to controls. It may decline early, remain flat or never rise above baseline; do not assume “high inside the window.” Testing beyond the architectural context limit is a different intervention and may require changing position handling, so report it separately (§5.7.6).


## Experiment 5.9 — Beyond language: a noisy sensor

**Tests:** §5.2.6, §5.6.2 (a steady-state random-walk Kalman filter becomes an EMA). **Predict:** the EMA coefficient $\alpha$ that minimizes mean squared error for a random-walk temperature with process-noise variance $q=0.01$ and measurement-noise variance $r=0.25$.

**Procedure.** Generate 5,000 steps with `sensor_stream`; compute `steady_state_kalman_alpha(q,r)`; sweep $\alpha$ and plot test MSE over at least 50 independent streams with confidence intervals. Then change the true transition to $\mu_t=1.01\mu_{t-1}+w_t$. Compare (i) the old random-walk-tuned EMA, (ii) the raw observation $\alpha=0$, and (iii) a Kalman filter using the correct transition 1.01. Report both absolute MSE and error normalized by signal power.

**Look for:** under the random-walk model, the population optimum is $\alpha\approx0.819$, the steady-state Kalman value; finite sweeps should fluctuate around it. Under the changed dynamics, the old EMA develops model-mismatch lag, while the raw observation retains measurement noise but does not necessarily diverge in error. The correctly specified Kalman filter is the relevant comparison. The lesson is not “every fixed filter fails,” but that a coefficient optimal under one transition model need not remain optimal after the dynamics change.
