---
title: "9.4 Worked examples"
description: Eight examples — an exact policy gradient, what a baseline does to variance, the tilted optimum at three temperatures, a two-state Bellman equation, Bradley–Terry and DPO in numbers, post-training the spine model, reward hacking, and reading a KL.
sidebar: { label: "9.4 Worked examples", order: 4 }
---

Exact numbers are recomputed by `labs/test_worked_examples_u09_u12.py`; post-training results come from `spine/stage09_post_training.py` (150 steps of 64 prompts, $\beta = 0.05$, Adam at $10^{-3}$) and are quoted to two decimals.

## Worked Example 9.1 — An exact policy gradient

**Setting.** Three actions with logits $z = (0, 0, 0)$, so $\pi = (\tfrac13, \tfrac13, \tfrac13)$; rewards $r = (1, 0, 0)$.

**Gradient.** $\partial J/\partial z_k = \pi_k(r_k - \pi^{\top}r)$ with $\pi^{\top}r = \tfrac13$: $(\tfrac13\cdot\tfrac23,\ \tfrac13\cdot(-\tfrac13),\ \tfrac13\cdot(-\tfrac13)) = (0.2222, -0.1111, -0.1111)$.

**Reading.** The rewarded action's logit goes up by twice as much as each other logit goes down, and the components sum to zero — adding a constant to all logits changes nothing. The spine test draws 400 000 actions and checks that the REINFORCE average reproduces this vector to within 0.004.

## Worked Example 9.2 — What a baseline does

**Setting.** Worked Example 9.1, one sample per estimate, component 1 of $\hat g = (r_a - b)(e_a - \pi)$.

| baseline $b$ | $\hat g_1$ if $a = 1$ | $\hat g_1$ otherwise | mean | variance |
|---|---|---|---|---|
| 0 | $1\cdot\tfrac23 = 0.667$ | $0$ | 0.2222 | 0.0988 |
| $\pi^{\top}r = \tfrac13$ | $\tfrac23\cdot\tfrac23 = 0.444$ | $(-\tfrac13)(-\tfrac13) = 0.111$ | 0.2222 | 0.0247 |

**Reading.** Same mean, a quarter of the variance. Without the baseline, unrewarded samples carry no signal; with it, they push their own action down, which is information too.

## Worked Example 9.3 — The tilted optimum

**Setting.** Reference $\pi_{\text{ref}} = (0.2, 0.5, 0.3)$, rewards $r = (1, 0, 0)$. Proposition 9.1 gives $\pi^\star\propto\pi_{\text{ref}}e^{r/\beta}$.

| $\beta$ | $\pi^\star$ | optimal value $\beta\ln Z$ |
|---|---|---|
| 0.1 | (0.9998, 0.0001, 0.0001) | 0.839 |
| 0.5 | (0.6488, 0.2195, 0.1317) | 0.412 |
| 2.0 | (0.2919, 0.4426, 0.2655) | 0.244 |

**Reading.** At $\beta = 0.5$ the rewarded action's odds are multiplied by $e^{2} = 7.39$; the unrewarded actions keep their 5 : 3 ratio from the reference. Large $\beta$ barely moves the reference; small $\beta$ approaches the argmax of the reward and forgets everything else the reference knew.

## Worked Example 9.4 — A Bellman equation by hand

**Setting.** From state A the only action leads to state B with reward 0; from B the episode ends with reward 1. Discount $\gamma = 0.9$.

**Values.** $V(B) = 1$; $V(A) = 0 + 0.9\cdot V(B) = 0.9$. If A also had a "stay" action with reward 0 returning to A, the optimality equation $V(A) = \max(0.9\,V(B),\ 0.9\,V(A))$ still gives $V(A) = 0.9$: waiting forever is worth $0.9^k\to0$, never more than going.

**Reading.** Discounting makes delayed reward worth less, and the Bellman equation propagates value backward one step at a time — the dynamic-programming counterpart of backpropagation through time (Unit 05).

## Worked Example 9.5 — Preferences in numbers

**Bradley–Terry.** If $r(y_w) - r(y_l) = 1$, then $P(y_w\succ y_l) = \sigma(1) = 0.7311$ and the reward-model loss for that pair is $-\ln0.7311 = 0.3133$.

**DPO at initialization.** With $\pi = \pi_{\text{ref}}$ the margin is $u = 0$ and the loss is $\ln2 = 0.6931$ for every pair.

**One step later.** If the policy has doubled $\pi(y_w)/\pi_{\text{ref}}(y_w)$ and halved $\pi(y_l)/\pi_{\text{ref}}(y_l)$, then with $\beta = 0.1$, $u = 0.1(\ln2 + \ln2) = 0.1386$ and the loss is $\ln(1 + e^{-0.1386}) = 0.6262$.

**Reading.** DPO's implied reward of a response is $\beta$ times its log-probability ratio to the reference; the loss only cares about the *difference* between the two responses' ratios.

## Worked Example 9.6 — Post-training the spine model

**Setting.** 400 held-out arithmetic prompts ("… Now Ben has "), single-digit answers.

| model | greedy accuracy | mean $p$(correct) | KL to reference at the answer |
|---|---|---|---|
| pretrained (Stage 07) | 0.32 | 0.20 | 0 |
| REINFORCE, reward $1[\text{ans} = a+b]$ | **0.79** | **0.72** | 1.34 |
| DPO on (correct, random incorrect) pairs | 0.78 | 0.76 | — |

**Reading.** 150 steps with a verifiable reward move the model from imitating the corpus's 30%-wrong arithmetic to answering correctly four times in five. DPO, trained on pairs constructed from the same checker, reaches nearly the same accuracy without sampling. (These are single runs. With the biased batch-mean baseline of v2.0, the REINFORCE run reached 0.82 and 0.80; the correction rescales the advantages by $64/63$ and leaves the expected gradient's direction unchanged, yet moved the result by 0.03 — run-to-run variation of the size Unit 10 and Unit 12 warn about.) Neither reaches 100%: the KL anchor holds the policy near a reference that is often confidently wrong, and a 3-layer model trained on 1 187 stories has not learned addition well enough to be reliably right — post-training can only reweight what the model can already produce (§9.5.3).

## Worked Example 9.7 — Reward hacking with a one-character bug

**Setting.** Identical to Worked Example 9.6, but the checker returns 1 whenever the answer is *at least* $a + b$ — a `>=` where `==` was meant.

| | training reward (last 20 steps) | true accuracy on training samples | held-out greedy accuracy | mean $p$(correct) |
|---|---|---|---|---|
| buggy checker | **0.99** | 0.07 | **0.09** | 0.11 |

**Reading.** Answering "9" satisfies "at least $a+b$" for every prompt in the task (the sums never exceed 9). The policy found this in 150 steps: its reward is nearly perfect and its arithmetic is far worse than before post-training (0.09 versus 0.32). Every monitoring number computed from the buggy checker says the run is a success. Only an *independent* measurement — the true accuracy on held-out prompts — reveals it. This is the bridge to Unit 10.

## Worked Example 9.8 — Reading a KL

**Setting.** The REINFORCE run of Worked Example 9.6 ends with $\mathrm{KL}(\pi\,\Vert\,\pi_{\text{ref}}) = 1.34$ nats at the answer position.

**A lower reference point.** Moving 0.8 of the probability onto the correct digit from a reference that gave it 0.2, and leaving the rest proportional, costs $0.8\ln\frac{0.8}{0.2} + 0.2\ln\frac{0.2}{0.8} = 0.832$ nats.

**Reading.** The measured 1.34 is larger: the policy also reshaped how it distributes the remaining probability among wrong digits. The KL is a total budget of change — it does not say *which* change was made, which is why the buggy run (KL 1.43) and the correct run (KL 1.34) look alike on this number. Unit 11 asks what changed inside.
