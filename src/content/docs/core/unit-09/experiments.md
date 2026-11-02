---
title: "9.9 Experiments"
description: Six experiments — the estimator on a bandit, the tilted optimum, post-training the spine model, the KL coefficient, DPO versus REINFORCE, and finding a reward hack before the policy does.
sidebar: { label: "9.9 Experiments", order: 9 }
---

Write the prediction first. Code: [`spine/stage09_post_training.py`](https://github.com/yhxuee/deep-learning-intelligent-systems/tree/main/spine) and its tests in `spine/test_spine.py`.

## Experiment 9.1 — The estimator on a bandit

**Tests:** Theorem 9.1, Worked Examples 9.1–9.2. **Predict:** the variance of the single-sample REINFORCE estimate with no baseline, with the mean-reward baseline, and with the optimal baseline of Proposition 9.2, for a 10-armed bandit with random rewards.

**Procedure.** Implement the estimator (AI-off); estimate the variance from 100 000 samples each; check all three means against the exact gradient.

**Look for:** identical means; variance ordered none > mean > optimal, with the last two close.

## Experiment 9.2 — Converging to the tilted reference

**Tests:** Proposition 9.1, Worked Example 9.3. **Predict:** the final distribution of exact gradient ascent on $J_\beta$ for $\beta\in\{0.1, 0.5, 2\}$.

**Procedure.** Run `test_stage09_kl_regularized_optimum_is_the_tilted_reference` and extend it to the three values; then replace exact gradients with REINFORCE estimates of batch 16 and plot the distance to $\pi^\star$ over steps.

**Look for:** exact convergence with exact gradients; noisy convergence to a neighbourhood with estimates, whose size shrinks with batch size.

## Experiment 9.3 — Post-training the spine model

**Tests:** §9.3.2, §9.3.7, Worked Example 9.6. **Predict:** greedy accuracy after 50, 150 and 400 steps.

**Procedure.** `python stage09_post_training.py --steps N`; record training reward, held-out accuracy, $p$(correct) and KL at each checkpoint; repeat with three seeds.

**Look for:** rapid improvement in the first 100 steps, then a plateau below 1.0; KL rising with accuracy. Report seed-to-seed spread (Unit 10).

## Experiment 9.4 — How tight should the leash be?

**Tests:** §9.2.5, §9.5.4. **Predict:** held-out accuracy and KL for $\beta\in\{0, 0.05, 0.5, 5\}$.

**Procedure.** Change `beta` in `post_train`; also measure held-out *story* NLL (not just arithmetic) before and after, to see what post-training costs elsewhere.

**Look for:** $\beta = 0$ reaches high accuracy fastest and degrades story NLL most; $\beta = 5$ barely moves. Locate the trade-off.

## Experiment 9.5 — DPO versus REINFORCE

**Tests:** §9.3.6, Worked Example 9.6. **Predict:** which method reaches 0.75 accuracy in fewer steps, and which changes the rest of the distribution more.

**Procedure.** Run both with matched numbers of prompts per step; compare accuracy curves and the KL to the reference on non-arithmetic prompts.

**Look for:** DPO is cheaper per step (no sampling) and needs a source of pairs; REINFORCE learns from its own samples. Explain which would scale to tasks without a checker.

## Experiment 9.6 — Find the hack before the policy does

**Tests:** §9.7.1, Worked Example 9.7. **Predict:** for each of three candidate checkers — (a) `answer >= a+b`, (b) `answer in "0123456789"`, (c) `abs(answer - (a+b)) <= 1` — what a policy would converge to.

**Procedure.** Before training, write down the policy that maximizes each reward. Then run REINFORCE with each; compare your prediction with the result and with true accuracy.

**Look for:** (a) collapse to "9"; (b) no pressure toward correctness at all; (c) a policy that is often off by one. Design a test that would have caught each bug before training.
