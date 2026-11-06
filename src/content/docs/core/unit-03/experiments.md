---
title: "3.9 Experiments"
description: Eight experiments that test the unit's results — divergence thresholds, condition numbers, momentum, SGD noise floors, finite-difference error, depth and initialization, the hand-built autodiff engine, and spine Stage 03.
sidebar: { label: "3.9 Experiments", order: 9 }
---

Write a prediction — a number or a curve shape, with one sentence of reasoning — **before** running each experiment. Code: [`labs/unit-03-autodiff`](https://github.com/yhxuee/deep-learning-intelligent-systems/tree/main/labs/unit-03-autodiff), [`spine/`](https://github.com/yhxuee/deep-learning-intelligent-systems/tree/main/spine), and [`figures/make_figures.py`](https://github.com/yhxuee/deep-learning-intelligent-systems/tree/main/figures). Experiment 3.7 is AI-off; for the others, AI may write plotting and sweep code, while predictions and interpretations stay yours.

## Experiment 3.1 — Where gradient descent diverges

**Tests:** §3.5.4. **Predict:** for $L(\theta) = \tfrac12\theta^{\top}H\theta$ with $H = \mathrm{diag}(1, 4, 25)$, the largest stable learning rate, and the behaviour just above it.

**Procedure.** Run 200 steps from $\theta_0 = (1,1,1)$ for $\eta$ on a fine grid around your prediction; record $\|\theta_{200}\|$.

**Look for:** a sharp transition at $\eta = 2/25 = 0.08$; oscillation in the third coordinate just below it; divergence of that coordinate alone just above.

## Experiment 3.2 — Iterations grow with the condition number

**Tests:** §3.5.4. **Predict:** the number of steps to reduce the error by $10^6$ with the best fixed step, for $\kappa\in\{10, 100, 1000\}$.

**Procedure.** Quadratics $\mathrm{diag}(1,\kappa)$; use $\eta^* = 2/(1+\kappa)$; count steps. Then run gradient descent on the Rosenbrock function (§3.7.3) from $(-1.2, 1)$ with the largest stable fixed step you can find, and count steps to reach $\|\theta - (1,1)\| < 10^{-3}$.

**Look for:** step counts close to $\frac\kappa2\ln10^6\approx6.9\kappa$ on the quadratics; thousands of steps on Rosenbrock.

## Experiment 3.3 — Momentum and √κ

**Tests:** §3.5.7. **Predict:** the speed-up of tuned heavy-ball momentum over gradient descent for the same three values of $\kappa$.

**Procedure.** Use the $\eta,\beta$ formulas of §3.5.7. Then deliberately mis-tune $\beta$ by ±0.1 and observe.

**Look for:** speed-ups near $\sqrt\kappa$ (about 3, 10, 30); sensitivity to mis-tuning that grows with $\kappa$.

## Experiment 3.4 — The SGD noise floor

**Tests:** §3.5.6. **Predict:** how the final loss plateau of constant-step SGD on a least-squares problem changes when (a) the step halves, (b) the batch size quadruples.

**Procedure.** Linear regression with 10,000 examples and 20 features plus label noise. Compute the full-batch optimum first. Run SGD with batch sizes 1, 4, 16, 64 and steps $\eta, \eta/2$; average the **excess over the optimum training loss** during the last 1,000 steps and across at least three seeds, with uncertainty bars. Then try $\eta_k\propto1/k$ using the same update budget.

**Look for:** under the independent-batch approximation, plateau excess roughly proportional to $\eta/B$ over the stable range. Treat deviations as data: finite-population sampling, curvature, and transient bias all break the simple scaling. The decreasing schedule should lower late excess while usually slowing early progress.

## Experiment 3.5 — Finite-difference error

**Tests:** Worked Example 3.6, §3.7.5. **Predict:** the optimal $h$ and the minimum error for central differences of $e^x$ at $x = 0$ in float64, and in float32.

**Procedure.** Sweep $h$ over $10^{-14}$ to $10^{-1}$; plot the error on log–log axes (compare Figure 3.4).

**Look for:** the V shape; slopes of 2 (central) and 1 (forward) on the right; slope $-1$ on the left; an optimum that moves to much larger $h$ in float32 (where $u\approx6\times10^{-8}$).

## Experiment 3.6 — Depth and initialization

**Tests:** §3.7.4, §3.8.1. **Predict:** the gradient norm at the first layer of a 50-layer, width-256 MLP for sigmoid, tanh and ReLU, each with $\sigma_w^2 \in\{1/n, 2/n\}$.

**Procedure.** Adapt `u03_vanishing_gradients` in `figures/make_figures.py`. Run at least 10 seeds and plot a median plus an interval. Then compare plain layers with $h_\ell=h_{\ell-1}+\alpha\varphi(W_\ell h_{\ell-1})$ for both $\alpha=1$ and the depth-scaled choice $\alpha=1/\sqrt{50}$.

**Look for:** collapse for sigmoid; better scale preservation for the matched pairs; and geometric decay for ReLU with $1/n$ under the independence approximation. Do not assume every residual curve is flatter: an unscaled branch can accumulate or explode. Test whether depth scaling makes the identity contribution dominate more reliably.

## Experiment 3.7 — Build and check your engine (AI-off)

**Tests:** §3.3.3–§3.3.9. **Predict:** which of the lab tests your first implementation will fail.

**Procedure.** Implement `labs/unit-03-autodiff/engine.py` from the description in §3.3.9, then run `python -m pytest` in that directory. Fix failures by reasoning from the failing invariant, not by reading `reference.py`.

**Look for:** all tests passing, including the worked example (§3.3.3), fan-out accumulation (Worked Example 3.2), central-difference agreement on random graphs, linearity, a 10 000-node deep graph, and accumulation across graphs.

## Experiment 3.8 — Spine Stage 03: a neural bigram from your own gradient

**Tests:** §3.3.5(c), §3.3.8. **Predict:** the initial loss of a neural bigram with $W = 0$ on the spine corpus, and the training loss after 1 000 Adam steps relative to the count-based maximum-likelihood bigram.

**Procedure.** Run `python spine/stage03_neural_bigram.py`; read `loss_and_grad` and compare it line by line with $\bar s = p - e_y$ and the scatter-add rule for the rows of $W$ (fan-out). Run `python -m pytest spine`.

**Look for:** an initial loss of exactly $\ln V$ (Unit 01's uniform baseline); a training loss that never goes below the count-based optimum and approaches it within 0.01 nats; a finite-difference check that passes.
