---
title: "Unit 09 · How do goals and feedback shape behavior?"
description: When the data say one thing and we want another, the objective must come from a goal — a reward, a preference, a verifier. From the score-function gradient to RLHF, DPO and verifiable-reward training of the spine model, with a reward-hacking audit.
sidebar: { label: "9.1 The problem", order: 1 }
unit:
  number: 9
  part: 3
  question: How do systems choose actions using goals and feedback — and how does post-training change a language model?
  anchor: The policy-gradient estimator with a KL anchor, and DPO, applied to the pretrained spine model on an arithmetic task with a programmatically checkable answer — including a deliberately buggy reward checker.
  spine: "Stage 09 — post-train the Stage 07 model with REINFORCE against the reward 1[answer = a + b] with a KL penalty to the pretrained model, and with DPO on (correct, incorrect) pairs; then repeat REINFORCE with a buggy checker (answer ≥ a + b) and measure true accuracy."
  checkpoint: Derive the policy-gradient estimator and show a baseline leaves it unbiased; derive the optimum of KL-regularized reward maximization and from it the DPO loss; construct a reward that a policy can satisfy without doing the task.
  status: draft
  atlas: [method-policy-gradient, method-dpo, mech-cross-entropy]
  answer: "When the objective is a reward rather than a target, the policy-gradient estimator turns trial and error into gradient steps; RLHF, DPO and verifiable-reward RL apply it to language models, with a KL anchor to the pretrained model limiting drift."
  leaves: "Optimizing a measured reward exposes every gap between the measurement and the intent — reward hacking — so we must ask when a score is evidence of reliability at all."
  builds_on:
    - { unit: 3, uses: "stochastic gradients and their variance" }
    - { unit: 7, uses: "the pretrained model as the initial policy" }
    - { unit: 8, uses: "sampling and the KL divergence" }
---

## 9.1 The problem

The pretrained spine model has read 1 187 stories, many of which contain sentences like "Ben had 2 eggs. Then Ben found 3 more. Now Ben has 5 eggs." In 30% of them the stated answer is wrong — the corpus generator writes an incorrect sum on purpose, as noisy web text does. Ask the pretrained model to complete "Now Ben has " and its single most likely answer is correct only **32%** of the time; on average it puts **0.20** of its probability on the correct digit.

Nothing in Units 01–08 can fix this. More pretraining fits the data's error rate more precisely (Unit 07, §7.7.1). Lower temperature sharpens whatever the model already prefers (Unit 08, §8.7.6). What is needed is a **different objective**: not "predict the text" but "produce the correct answer". We cannot write down the correct answer as a training target for every possible prompt — but we can *check* any answer the model produces. A program that returns 1 when the answer equals $a + b$ and 0 otherwise is a **reward**, and the model's answer is an **action**.

Learning from rewards instead of targets is reinforcement learning. Applied to a pretrained language model it is called **post-training**, and it is how current assistants are made helpful, how reasoning models are trained against verifiers, and how agents learn to use tools. It has a characteristic danger: the model optimizes the reward *as written*, not the intention behind it. A checker with a one-character bug — `>=` instead of `==` — is a different reward, and the model will find out.

:::note[Predict before reading on]
The spine model is post-trained for 150 steps of 64 prompts. With the correct checker, what greedy accuracy do you expect? With the buggy checker (reward 1 whenever the answer is at least $a + b$), what will the *training reward* be at the end, and what will the *true* accuracy be? (Worked Examples 9.6–9.7.)
:::

## What this unit builds

1. **Learning from reward, from first principles** (§9.2). Actions and rewards; why sampling blocks ordinary backpropagation; the score-function trick; baselines; sequences of decisions; staying close to the pretrained model; learning from preferences.
2. **The mathematics** (§9.3). Markov decision processes and returns; the policy-gradient theorem; advantages and value functions; the Bellman equation; the closed-form optimum of KL-regularized reward maximization; reward models from pairwise preferences; PPO's clipped objective; the derivation of DPO; verifiable rewards and group baselines.
3. **The general theory** (§9.5). Variance of gradient estimators and the optimal baseline; KL-regularized RL as Bayesian inference; Goodhart's law and reward overoptimization.
4. **What it leaves open** (§9.7). A reward is a measurement, and optimizing it exposes every gap between measurement and intent. When is any score evidence of what we want? That is [Unit 10](/core/unit-10/).

## Prerequisites

- **From Unit 03:** stochastic gradients, their variance and the step-size question.
- **From Unit 07:** the pretrained model and why it imitates its data.
- **From Unit 08:** ancestral sampling; temperature; the KL divergence.
- **Probability:** expectation of a function of a random variable; the log-derivative $\nabla\ln p = \nabla p/p$.

## How to read this unit

The main text is complete without an AI tutor. Exact numbers are recomputed by `labs/test_worked_examples_u09_u12.py`; the estimators and gradients are checked by `spine/test_spine.py` (Stage 09); post-training results come from `spine/stage09_post_training.py` and are quoted to two decimals.

| Section | Content | Approx. reading time |
|---|---|---|
| [9.2 First principles](/core/unit-09/first-principles/) | rewards, the score-function trick, baselines, KL anchor, preferences | 70 min |
| [9.3 Mathematical construction](/core/unit-09/construction/) | MDPs, policy-gradient theorem, Bellman, KL optimum, RLHF, PPO, DPO | 2–3 h |
| [9.4 Worked examples](/core/unit-09/worked-examples/) | eight examples | 75 min |
| [9.5 General theory](/core/unit-09/theory/) | variance, Bayesian view, overoptimization | 75 min |
| [9.6 Connections](/core/unit-09/connections/) | control, economics, psychology, neuroscience | 30 min |
| [9.7 Limitations and counterexamples](/core/unit-09/limitations/) | reward hacking, sycophancy, credit assignment | 40 min |
| [9.8 Modern developments](/core/unit-09/modern/) | RLHF, AI feedback, verifiable rewards, agents | 35 min |
| [9.9 Experiments](/core/unit-09/experiments/) | six experiments | lab time |
| [9.10 Problems and further study](/core/unit-09/problems/) | graded problems and reading | — |
