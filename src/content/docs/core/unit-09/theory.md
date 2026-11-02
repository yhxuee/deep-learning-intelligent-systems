---
title: "9.5 General theory"
description: The variance of score-function estimators and the optimal baseline, KL-regularized reinforcement learning as Bayesian inference, exploration, and Goodhart's law with the empirical laws of reward-model overoptimization.
sidebar: { label: "9.5 General theory", order: 5 }
---

## 9.5.1 Variance is the price of not differentiating the reward

The score-function estimator is unbiased for any reward, differentiable or not — but its variance can be enormous. For one sample, $\hat g = (r(a) - b)\nabla\ln\pi(a)$, and

$$
\mathrm{Var}(\hat g_i) = \mathbb{E}\big[(r - b)^2(\partial_i\ln\pi)^2\big] - (\partial_iJ)^2 .
$$

**Proposition 9.2 (optimal constant baseline).** Among constant baselines, the variance of component $i$ is minimized by
$$
b_i^\star = \frac{\mathbb{E}\big[r\,(\partial_i\ln\pi)^2\big]}{\mathbb{E}\big[(\partial_i\ln\pi)^2\big]} .
$$

*Proof.* The variance is a quadratic in $b$: $\mathbb{E}[r^2s^2] - 2b\,\mathbb{E}[rs^2] + b^2\mathbb{E}[s^2] - (\partial_iJ)^2$ with $s = \partial_i\ln\pi$. Set its derivative to zero. $\blacksquare$

The optimal baseline is the average reward *weighted by the squared score*; the plain average reward is a good, simple approximation. Variance also grows with the length of the trajectory, because the sum $\sum_t\nabla\ln\pi(a_t\mid s_t)$ accumulates noise from every step — the quantitative reason learned value functions and per-token advantages matter for long generations, and why one-step tasks like the spine's train in a few hundred steps.

Compare with Unit 03's stochastic gradients: there, the noise came from sampling *examples* and shrank as $1/B$. Here, it comes from sampling *actions* and also shrinks as $1/B$, but with a constant that can be orders of magnitude larger.

## 9.5.2 KL-regularized RL as Bayesian inference

Proposition 9.1 has a second reading **[D]**. Treat $\pi_{\text{ref}}$ as a *prior* over outputs and $e^{r(a)/\beta}$ as a *likelihood* of "this output is good". Then $\pi^\star\propto\pi_{\text{ref}}e^{r/\beta}$ is the **posterior**, and maximizing $J_\beta$ is variational inference: it minimizes $\mathrm{KL}(\pi\,\Vert\,\pi^\star)$ ([Korbak et al. 2022](/references/#korbak-2022)). The view explains three practical facts. Post-training cannot make probable what the prior gives (nearly) zero probability — the pretrained model must already be able to produce good answers occasionally. The KL coefficient $\beta$ is a temperature on the evidence. And because KL($\pi\Vert\pi^\star$) is *mode-seeking*, the post-trained model concentrates on some high-reward modes of the posterior rather than covering all of them — a reduction in diversity that is a property of the objective, not a bug of the implementation.

## 9.5.3 Exploration

A policy can only learn from actions it tries. If the pretrained model never produces a correct answer to some prompt, the policy gradient for that prompt is zero whatever the reward **[D]**. Exploration is therefore not optional: it comes from the stochasticity of sampling (temperature), from the diversity of the prompts, and, in classical RL, from deliberate bonuses for novelty. For language models, pretraining is the exploration strategy — it puts non-zero probability on a vast space of sensible outputs, which post-training then reweights. This is why reinforcement learning from scratch on language is hopeless and from a pretrained model is effective.

## 9.5.4 Goodhart's law and overoptimization

"When a measure becomes a target, it ceases to be a good measure." Any reward $r$ is a proxy for an intended objective $r^\dagger$. Optimizing $r$ improves $r^\dagger$ only as long as the two are correlated *on the distribution the policy produces* — and optimization moves that distribution toward places where they diverge **[D]**. [Manheim & Garrabrant (2018)](/references/#manheim-2018) distinguish regressional, extremal, causal and adversarial variants; [Skalse et al. (2022)](/references/#skalse-2022) formalize "hackable" reward pairs.

Reward hacking is also measurable. Training against a learned reward model and scoring the result with a larger "gold" reward model, [Gao et al. (2023)](/references/#gao-2023) found that the gold reward first rises and then falls as optimization proceeds, with the peak and the decline following smooth functions of $\sqrt{\mathrm{KL}(\pi\Vert\pi_{\text{ref}})}$ **[E]**. The KL anchor is thus a direct control on how far one travels along the proxy before it turns against the goal. The spine's buggy checker (§9.7.1) is the extreme case: proxy and gold agree on correct answers and disagree everywhere else, and the policy goes straight to the disagreement.

## 9.5.5 Summary

| Result | Says | Status |
|---|---|---|
| Thm 9.1 | $\nabla J = \mathbb{E}[\sum_t\nabla\ln\pi(a_t\mid s_t)(G_t - b)]$ | [D] |
| Prop 9.1 | KL-regularized optimum $= \pi_{\text{ref}}e^{r/\beta}/Z$ | [D] |
| DPO (§9.3.6) | preference likelihood in terms of log-ratios; $Z$ cancels | [D] given Bradley–Terry |
| Prop 9.2 | optimal baseline = score-weighted mean reward | [D] |
| Bayesian view (§9.5.2) | post-training = posterior with prior $\pi_{\text{ref}}$ | [D] |
| Overoptimization (§9.5.4) | gold reward rises then falls with $\sqrt{\mathrm{KL}}$ | [E] |
