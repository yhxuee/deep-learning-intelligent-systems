---
title: "9.6 Connections"
description: Goals and feedback in optimal control, economics and mechanism design, behavioural psychology, the neuroscience of reward prediction errors, and statistics — and within the book.
sidebar: { label: "9.6 Connections", order: 6 }
---

## 9.6.1 Optimal control and dynamic programming

The Bellman equation came from control and operations research ([Bellman 1957](/references/#bellman-1957)): choose actions over time to minimize a cost, given a model of the dynamics. When the dynamics are known, dynamic programming solves the problem exactly for small state spaces; linear dynamics with quadratic cost give the linear–quadratic regulator, the sibling of the Kalman filter met in Unit 05 (§5.6.2). Reinforcement learning is the same problem without a known model — the dynamics and rewards are learned from interaction — which is why its methods trade the certainty of planning for the variance of sampling.

## 9.6.2 Economics: utility, incentives and mechanism design

A reward function is a utility function, and the policy is an agent maximizing expected utility. Economics has long studied what happens when an agent optimizes a *contract* — a measurable proxy for what the principal wants: workers paid by the piece cut quality, managers paid on quarterly numbers sacrifice the long run. Reward hacking is the principal–agent problem with a very capable, very literal agent. Mechanism design asks how to construct incentives so that the agent's optimal behaviour is the behaviour the designer wants; reward design for AI systems is the same question, and its partial answers (shaping that preserves optimal policies, [Ng et al. 1999](/references/#ng-1999); multiple independent checks; penalties on deviation) have economic analogues.

## 9.6.3 Psychology: reinforcement and preference

Operant conditioning — behaviours followed by reward become more frequent — is the behavioural form of the policy-gradient update. Pairwise comparison as a measurement method also comes from psychology: Thurstone's law of comparative judgement and the Bradley–Terry model were developed to scale preferences that people cannot express as numbers but can express as choices. RLHF inherits their assumptions — preferences are consistent and noisy around a latent utility — and their known failures (intransitive and context-dependent judgements).

## 9.6.4 Neuroscience: reward prediction errors

Dopamine neurons in the midbrain fire in proportion to the *difference* between received and expected reward, not to reward itself — a reward prediction error. This is the advantage $r - b$ of §9.2.3, and temporal-difference learning, which uses $r_t + \gamma V(s_{t+1}) - V(s_t)$ as its learning signal, became a leading model of how this signal could train behaviour ([Dayan & Abbott 2001](/references/#dayan-2001)). The correspondence is one of the clearest bridges between machine learning and neuroscience, and like all such bridges it is a model, not an identity.

## 9.6.5 Statistics: importance sampling

The PPO ratio $\pi_\theta/\pi_{\text{old}}$ (§9.3.5) is an importance weight: it corrects an expectation estimated with samples from the wrong distribution. Importance sampling is unbiased and high-variance when the two distributions differ much — the statistical reason for PPO's clipping and for keeping successive policies close. Off-policy evaluation (estimating how a new policy would perform from logs of an old one) is the same problem and appears again in Unit 10 as an evaluation question.

## 9.6.6 Inside the book

| This unit | Returns in |
|---|---|
| reward hacking | Unit 10 (when is a score evidence?), Unit 12 (pre-registration against self-deception) |
| KL divergence as an anchor | Unit 01 (KL as excess loss), Unit 08 (VAE's KL term) |
| variance of gradient estimators | Unit 03 (minibatch noise) |
| sampling cost of RL | Unit 06 (inference FLOPs; test-time compute) |
| the post-trained model | Unit 10 (evaluating it), Unit 11 (what changed inside it) |
