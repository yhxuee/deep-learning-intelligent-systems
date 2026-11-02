---
title: "9.2 First principles"
description: Actions, rewards and policies; why sampling blocks backpropagation and how the score-function trick gets around it; baselines; sequences of decisions; the KL anchor to the pretrained model; and learning from preferences instead of rewards.
sidebar: { label: "9.2 First principles", order: 2 }
---

## 9.2.1 From targets to rewards

In Units 01–08 every example came with a target: the next character, the masked token, the matching pair. The loss compared the model's prediction with the target, and its gradient told every parameter how to move. Now there is no target. There is a **policy** — the model's distribution $\pi_\theta(a\mid s)$ over **actions** $a$ given a **state** (prompt) $s$ — and a **reward** $r(s, a)$ that scores whatever action was taken. The goal is to maximize the expected reward

$$
J(\theta) = \mathbb{E}_{s}\,\mathbb{E}_{a\sim\pi_\theta(\cdot\mid s)}\big[r(s, a)\big].
$$

For the spine model's arithmetic, $s$ is the prompt ending in "Now Ben has ", $a$ is the next character, and $r = 1$ if $a$ is the digit $a+b$, else 0. The reward is a black box: we can evaluate it, not differentiate it.

## 9.2.2 Why backpropagation stops at the sample

To improve $J$ by gradient ascent we need $\nabla_\theta J$. The obvious route — compute an action, compute its reward, backpropagate — fails twice. The action is *sampled*: drawing $a$ from $\pi_\theta$ is not a differentiable operation of $\theta$, because a small change in $\theta$ changes *which* action is drawn, discontinuously. And the reward is a program with no derivative (an equality test is a step function).

The way out is to differentiate the *probabilities* instead of the samples. For a discrete action space,

$$
\nabla_\theta J = \nabla_\theta\sum_a\pi_\theta(a)\,r(a) = \sum_a r(a)\,\nabla_\theta\pi_\theta(a) = \sum_a\pi_\theta(a)\,r(a)\,\nabla_\theta\ln\pi_\theta(a) = \mathbb{E}_{a\sim\pi_\theta}\big[r(a)\nabla_\theta\ln\pi_\theta(a)\big],
$$

using $\nabla\pi = \pi\nabla\ln\pi$. The last expression is an *expectation over the policy's own samples*: draw actions, weight each one's log-probability gradient by its reward, average. This is the **score-function** or **REINFORCE** estimator ([Williams 1992](/references/#williams-1992)). It needs only that we can sample actions, evaluate rewards, and differentiate $\ln\pi_\theta(a)$ — which for a language model is exactly the cross-entropy gradient of Unit 03, applied to the sampled token as if it were the target.

Read it as **weighted imitation of the model's own behaviour**: train on your own samples as if they were correct answers, with weight proportional to how good they turned out to be.

## 9.2.3 Baselines: compare with what was expected

If every reward is positive, every sampled action is pushed up — the good ones more than the bad ones, but all of them. The estimate is right on average and very noisy. Subtract a **baseline** $b$ that does not depend on the action:

$$
\nabla_\theta J = \mathbb{E}_{a\sim\pi_\theta}\big[(r(a) - b)\nabla_\theta\ln\pi_\theta(a)\big].
$$

The expectation is unchanged, because $\mathbb{E}_a[\nabla\ln\pi_\theta(a)] = \sum_a\nabla\pi_\theta(a) = \nabla\sum_a\pi_\theta(a) = \nabla 1 = 0$. But the variance can fall dramatically (Worked Example 9.2: a factor of four in a three-armed problem). With $b$ equal to the average reward, actions better than average are pushed up and worse ones down — the quantity $r - b$ is called the **advantage**. The spine uses the mean reward of each batch of 64 prompts as $b$; group-relative methods for language models use the mean reward of several answers to the *same* prompt ([Shao et al. 2024](/references/#shao-2024)).

## 9.2.4 Sequences of decisions

A language model's answer is usually many tokens, and the reward arrives only at the end. A **trajectory** $\tau = (s_0, a_0, s_1, a_1, \dots)$ has probability $\prod_t\pi_\theta(a_t\mid s_t)$ times the environment's transition probabilities, which do not depend on $\theta$. The same log-derivative argument gives

$$
\nabla_\theta J = \mathbb{E}_\tau\Big[\sum_t\nabla_\theta\ln\pi_\theta(a_t\mid s_t)\,(R - b)\Big],
$$

where $R$ is the total reward: every token of a good answer is reinforced, every token of a bad one discouraged. Assigning credit to *the tokens that mattered* — the **credit-assignment problem** — is what value functions and advantage estimates address (§9.3.3). For the spine's one-character answer the problem disappears, which is why it is a clean exemplar.

## 9.2.5 An anchor to what the model already knows

A policy optimized for reward alone can drift arbitrarily far from the pretrained model — forgetting how to write, collapsing onto a few high-reward outputs, or exploiting quirks of the reward. Post-training therefore maximizes reward *minus* a penalty for moving away from the pretrained **reference** policy $\pi_{\text{ref}}$:

$$
J_\beta(\theta) = \mathbb{E}_{a\sim\pi_\theta}[r(a)] - \beta\,\mathrm{KL}\big(\pi_\theta\,\Vert\,\pi_{\text{ref}}\big).
$$

This objective has a closed-form optimum (§9.3.4): $\pi^\star(a)\propto\pi_{\text{ref}}(a)\,e^{r(a)/\beta}$ — the reference distribution, **tilted** toward high reward. Small $\beta$ trusts the reward; large $\beta$ trusts the reference. The KL term is the same quantity as Unit 01's excess loss, now used as a leash.

## 9.2.6 When there is no reward: preferences

For "is this answer correct?" a program can decide. For "is this answer helpful, honest and harmless?" none can. What people *can* do reliably is compare: shown two responses, say which is better. Pairwise preferences can be turned into a reward model by fitting a **Bradley–Terry** model, $P(y_w\succ y_l) = \sigma(r(y_w) - r(y_l))$ ([Bradley & Terry 1952](/references/#bradley-1952)), and the policy is then optimized against that learned reward — **reinforcement learning from human feedback** ([Christiano et al. 2017](/references/#christiano-2017); [Ouyang et al. 2022](/references/#ouyang-2022)). Or, as §9.3.6 derives, the reward model can be skipped: **direct preference optimization** trains the policy on the preferences directly, using the closed-form optimum of §9.2.5 ([Rafailov et al. 2023](/references/#rafailov-2023)).

## 9.2.7 What we have so far

| Signal | Example | Method | Spine |
|---|---|---|---|
| target | the next character | cross-entropy (Units 01–07) | pretraining: 32% correct arithmetic |
| programmatic reward | 1[answer = a+b] | policy gradient + KL anchor | Worked Example 9.6 |
| pairwise preference | correct ≻ incorrect | DPO | Worked Example 9.6 |
| buggy reward | 1[answer ≥ a+b] | policy gradient | Worked Example 9.7 |
