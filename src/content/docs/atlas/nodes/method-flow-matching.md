---
title: Flow matching
description: Learn a velocity field by regressing on x1 − x0 along straight paths between noise and data; sample by integrating the field as an ODE.
atlas:
  id: method-flow-matching
  type: method
  mastery: understand-and-verify
  core_units: [8]
  prerequisites: [mech-cross-entropy]
  related: [mech-linear-recurrence]
  references: [lipman-2023, liu-2023, albergo-2023]
  last_reviewed: 2026-11-02
---

## In one paragraph

Pair noise $x_0\sim N(0, I)$ with data $x_1$, interpolate $x_t = (1-t)x_0 + tx_1$, and train $v_\theta(x_t, t)$ to predict $x_1 - x_0$ by least squares. The optimum is the marginal field $v^\star(x, t) = \mathbb{E}[x_1 - x_0\mid x_t = x]$, and integrating $\dot x = v^\star(x, t)$ from noise at $t = 0$ produces samples of the data at $t = 1$ (Unit 08, Proposition 8.2). The training loss does not go to zero: crossing paths leave an irreducible conditional variance.

## Invariants

- For $x_1\sim N(\mu, \sigma^2I)$, the exact field is $v^\star(x,t) = \mu + \frac{t\sigma^2 - (1-t)}{(1-t)^2 + t^2\sigma^2}(x - t\mu)$ and $x(t) = t\mu + s_tx_0$ solves the ODE (spine `test_stage08_exact_gaussian_velocity_transports_noise_to_data`).
- The hand-derived gradient of the regression loss matches central differences (spine `test_stage08_flow_matching_gradient_matches_finite_differences`).
- One Euler step from $t = 0$ with the exact Gaussian field maps every sample to $\mu$: few steps shrink the spread.

## Known values

| Quantity | Value |
|---|---|
| $v^\star(0, 0.5)$ for $\mu = (1, -2)$, $\sigma = 0.5$ | $(1.6, -3.2)$ |
| Sample std after 1 / 2 / 5 / 20 / 200 Euler steps (target 0.5) | 0 / 0.20 / 0.37 / 0.47 / 0.50 |
| 2-D ring of 8 Gaussians, 3 000 steps: loss | about 5.3 → 3.8 |
| Same model, 100 Euler steps: samples within 0.5 of a mode; modes covered | about 96%; 8 of 8 |

## Common bugs

- Regressing on $x_1$ or $x_0$ instead of the velocity $x_1 - x_0$ (or mixing conventions about which end is noise).
- Sampling $t$ only on a grid seen in training and integrating at other times.
- Integrating from $t = 1$ to $t = 0$ with the sign of the field unchanged.
- Expecting the loss to reach zero and tuning until it does — that overfits.
