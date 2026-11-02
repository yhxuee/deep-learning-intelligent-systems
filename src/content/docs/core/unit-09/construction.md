---
title: "9.3 Mathematical construction"
description: Markov decision processes, the policy-gradient theorem with baselines and advantages, value functions and the Bellman equation, the closed-form optimum of KL-regularized reward maximization, reward models from preferences, PPO's clipped objective, the derivation of DPO, and verifiable rewards with group baselines.
sidebar: { label: "9.3 Construction", order: 3 }
---

## 9.3.1 Markov decision processes

A **Markov decision process** has states $s$, actions $a$, a transition distribution $P(s'\mid s, a)$, a reward $r(s, a)$ and a discount $\gamma\in[0, 1]$ ([Sutton & Barto 2018](/references/#sutton-2018)). A policy $\pi_\theta(a\mid s)$ generates trajectories; the **return** from time $t$ is $G_t = \sum_{k\ge0}\gamma^kr_{t+k}$, and the objective is $J(\theta) = \mathbb{E}_{\tau\sim\pi_\theta}[G_0]$.

Text generation is an MDP with a particularly simple structure: the state is the prompt plus the tokens generated so far, the action is the next token, the transition appends it deterministically, and the reward typically arrives once, at the end. The spine's arithmetic task is a one-step MDP — a *contextual bandit*.

## 9.3.2 The policy-gradient theorem

**Theorem 9.1 (policy gradient).** For any baseline $b(s_t)$ that does not depend on $a_t$,
$$
\nabla_\theta J(\theta) = \mathbb{E}_{\tau\sim\pi_\theta}\Big[\sum_{t}\nabla_\theta\ln\pi_\theta(a_t\mid s_t)\,\big(G_t - b(s_t)\big)\Big].
$$

*Proof.* The trajectory probability is $p_\theta(\tau) = p(s_0)\prod_t\pi_\theta(a_t\mid s_t)P(s_{t+1}\mid s_t, a_t)$; only the policy factors depend on $\theta$, so $\nabla\ln p_\theta(\tau) = \sum_t\nabla\ln\pi_\theta(a_t\mid s_t)$. Then $\nabla J = \nabla\int p_\theta(\tau)G_0(\tau) = \mathbb{E}[G_0\sum_t\nabla\ln\pi_\theta(a_t\mid s_t)]$. Two refinements: a reward received before time $t$ is independent of $a_t$ given the past, so its product with $\nabla\ln\pi(a_t\mid s_t)$ has zero expectation and $G_0$ may be replaced by $G_t$ (causality); and $\mathbb{E}_{a_t}[b(s_t)\nabla\ln\pi_\theta(a_t\mid s_t)\mid s_t] = b(s_t)\nabla\sum_{a}\pi_\theta(a\mid s_t) = 0$, so any action-independent baseline may be subtracted. $\blacksquare$

For a softmax policy with logits $z$ over a finite action set, $\nabla_z\ln\pi(a) = e_a - \pi$ (Unit 03, §3.3.5). The single-sample loss whose gradient is the REINFORCE estimate is therefore $\ell = -(G - b)\ln\pi(a)$, with $\partial\ell/\partial z = (G - b)(\pi - e_a)$ — cross-entropy toward the sampled action, scaled by its advantage. `spine/stage09_post_training.py` implements exactly this line and checks it against finite differences through the whole Transformer.

## 9.3.3 Value functions and the Bellman equation

The **value** of a state under $\pi$ is $V^\pi(s) = \mathbb{E}_\pi[G_t\mid s_t = s]$, and the **action value** is $Q^\pi(s, a) = \mathbb{E}_\pi[G_t\mid s_t = s, a_t = a]$. They satisfy the **Bellman equation** ([Bellman 1957](/references/#bellman-1957)):

$$
V^\pi(s) = \sum_a\pi(a\mid s)\Big[r(s, a) + \gamma\sum_{s'}P(s'\mid s, a)V^\pi(s')\Big],
$$

a linear system for a fixed policy, and with $\max_a$ in place of the average, the optimality equation solved by dynamic programming. $V^\pi(s_t)$ is the variance-minimizing baseline in expectation among functions of the state, and the **advantage** $A^\pi(s, a) = Q^\pi(s, a) - V^\pi(s)$ measures how much better an action is than the policy's average. Actor–critic methods learn $V$ with a second network and use $r_t + \gamma V(s_{t+1}) - V(s_t)$ or a smoothed version ([Schulman et al. 2016](/references/#schulman-2016)) as the advantage — trading a little bias for much lower variance than the raw return.

## 9.3.4 The KL-regularized objective and its optimum

Fix a state and write $\pi$ for $\pi(\cdot\mid s)$. Consider

$$
J_\beta(\pi) = \sum_a\pi(a)r(a) - \beta\sum_a\pi(a)\ln\frac{\pi(a)}{\pi_{\text{ref}}(a)} .
$$

**Proposition 9.1.** The maximizer of $J_\beta$ over distributions is $\pi^\star(a) = \pi_{\text{ref}}(a)\,e^{r(a)/\beta}/Z$, with $Z = \sum_a\pi_{\text{ref}}(a)e^{r(a)/\beta}$, and the maximum is $\beta\ln Z$.

*Proof.* Rewrite $J_\beta(\pi) = -\beta\sum_a\pi(a)\ln\frac{\pi(a)}{\pi_{\text{ref}}(a)e^{r(a)/\beta}} = -\beta\,\mathrm{KL}(\pi\,\Vert\,\pi^\star) + \beta\ln Z$. By Gibbs' inequality (Unit 01, Theorem 1.1) the KL term is non-negative and zero exactly at $\pi = \pi^\star$. $\blacksquare$

The tilted distribution keeps the reference's relative preferences among actions of equal reward and multiplies the odds of an action by $e^{\Delta r/\beta}$ for each unit $\Delta r$ of extra reward. The spine test checks that gradient ascent on $J_\beta$ converges to $\pi^\star$ to six decimals. Inverting the relation gives the step DPO needs:

$$
r(a) = \beta\ln\frac{\pi^\star(a)}{\pi_{\text{ref}}(a)} + \beta\ln Z .
$$

## 9.3.5 Reward models and PPO

**Reward model.** Collect prompts with pairs of responses and a human (or AI) label of which is better. Fit $r_\phi$ by maximizing the Bradley–Terry log-likelihood $\sum\ln\sigma\big(r_\phi(y_w) - r_\phi(y_l)\big)$ — logistic regression on reward differences ([Christiano et al. 2017](/references/#christiano-2017); [Stiennon et al. 2020](/references/#stiennon-2020)). Only differences are identified: adding a constant to every reward changes nothing.

**PPO.** The policy is then optimized against $r_\phi$ with a KL penalty. Because each batch of samples is reused for several gradient steps, the policy that is being updated differs from the one that generated the samples; Proximal Policy Optimization corrects with the importance ratio $\rho = \pi_\theta(a\mid s)/\pi_{\text{old}}(a\mid s)$ and **clips** it to $[1-\epsilon, 1+\epsilon]$ in the objective $\mathbb{E}[\min(\rho A, \mathrm{clip}(\rho, 1-\epsilon, 1+\epsilon)A)]$, so no single update moves the policy far ([Schulman et al. 2017](/references/#schulman-2017)). This three-stage pipeline — supervised fine-tuning, reward model, PPO with KL — produced the first widely used instruction-following assistants ([Ouyang et al. 2022](/references/#ouyang-2022)).

## 9.3.6 Direct preference optimization

Substitute the inverted optimum of §9.3.4 into the Bradley–Terry model. The $\beta\ln Z$ terms are equal for both responses to the same prompt and cancel:

$$
P(y_w\succ y_l) = \sigma\Big(\beta\ln\frac{\pi(y_w)}{\pi_{\text{ref}}(y_w)} - \beta\ln\frac{\pi(y_l)}{\pi_{\text{ref}}(y_l)}\Big).
$$

Maximizing the likelihood of the observed preferences under this model gives the **DPO loss** ([Rafailov et al. 2023](/references/#rafailov-2023)):

$$
\mathcal L_{\text{DPO}} = -\mathbb{E}\Big[\ln\sigma\big(u\big)\Big],\qquad u = \beta\Big[\big(\ln\pi(y_w) - \ln\pi_{\text{ref}}(y_w)\big) - \big(\ln\pi(y_l) - \ln\pi_{\text{ref}}(y_l)\big)\Big].
$$

No reward model, no sampling during training, no PPO: it is a classification loss on pairs. Its gradient with respect to the log-probabilities is $-\beta\sigma(-u)\big(\nabla\ln\pi(y_w) - \nabla\ln\pi(y_l)\big)$ — increase the preferred response, decrease the dispreferred one, weighted by how wrongly the current model orders them. For single-token answers, the softmax terms of the two log-probability gradients cancel, leaving $-\beta\sigma(-u)(e_{y_w} - e_{y_l})$ on the logits; the spine checks it by finite differences.

## 9.3.7 Verifiable rewards and group baselines

When correctness can be checked by a program — arithmetic, unit tests, a proof checker — the reward needs no human and no learned model. Sampling $G$ answers to the same prompt and using their mean reward as the baseline gives an advantage $A_i = r_i - \bar r$ (optionally divided by the group's standard deviation) without a value network ([Shao et al. 2024](/references/#shao-2024)). Training large models this way on mathematics and code, with long chains of reasoning before the answer, produced strong reasoning performance ([DeepSeek-AI 2025](/references/#deepseek-2025)) **[E]**. The spine's Stage 09 is the one-token version of this recipe, with a batch-mean baseline; its buggy-checker run is the cautionary version (§9.7.1).
