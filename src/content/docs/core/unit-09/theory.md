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

The optimal baseline is the average reward *weighted by the squared score* — the scalar case of the trace-minimizing baseline $b^\star(s) = \mathbb{E}_a[Q^\pi(s,a)\Vert\nabla\ln\pi(a\mid s)\Vert^2]/\mathbb{E}_a[\Vert\nabla\ln\pi(a\mid s)\Vert^2]$ ([Greensmith et al. 2004](/references/#greensmith-2004)). The plain average reward $V^\pi$ is a simple, often effective choice, but it can be far from optimal. *Counterexample:* a Bernoulli policy on a logit, $p = 0.8$, reward 1 for action 1 and 0 for action 0. The scores are $0.2$ and $-0.8$; $b^\star = (0.8\cdot1\cdot0.04)/(0.8\cdot0.04 + 0.2\cdot0.64) = 0.2$, while $V = 0.8$. With $b^\star$ the one-sample estimate equals $0.16$ for *either* action — zero variance — while with $b = V$ its variance is $0.0576$ (`test_stage09_value_baseline_is_not_the_variance_optimal_baseline`). Variance also grows with the length of the trajectory, because the sum $\sum_t\nabla\ln\pi(a_t\mid s_t)$ accumulates noise from every step — the quantitative reason learned value functions and per-token advantages matter for long generations, and why one-step tasks like the spine's train in a few hundred steps.

Compare with Unit 03's stochastic gradients: there, the noise came from sampling *examples* and shrank as $1/B$. Here, it comes from sampling *actions* and also shrinks as $1/B$, but with a constant that can be orders of magnitude larger.

## 9.5.2 KL-regularized RL as Bayesian inference: variational inference and mode-seeking

Proposition 9.1 gives the optimal policy $\pi^\star(a) \propto \pi_{\text{ref}}(a)\,e^{r(a)/\beta}$. Far from a mere algebraic artifact of Lagrange multipliers, this result embodies a deep **duality of control and probabilistic inference** ([Korbak et al. 2022](/references/#korbak-2022)) **[D]**.

### The Bayesian duality framework
Re-interpreting the components of the optimal policy:
- **Prior distribution:** The pre-trained base model $\pi_{\text{ref}}(y \mid x)$ serves as the prior, encapsulating the vast linguistic syntax, world knowledge, and reasoning primitives learned across the web.
- **Likelihood function:** The exponentiated reward $p(\text{high quality} \mid y, x) \propto \exp\big(\frac{r(x, y)}{\beta}\big)$ acts as the observation likelihood that a given response satisfies human preference or a task verifier, with temperature $\beta$ governing evidence uncertainty.
- **Posterior distribution:** The theoretical optimal policy $\pi^\star(y \mid x)$ is precisely the **Bayesian posterior conditioned on high quality**!

Mathematically, maximizing the KL-regularized objective $J_\beta(\pi) = \mathbb{E}_\pi[r] - \beta \mathrm{KL}(\pi \parallel \pi_{\text{ref}})$ is strictly equivalent to minimizing the reverse Kullback–Leibler divergence to this posterior:
$$
\max_\pi J_\beta(\pi) \iff \min_\pi \mathrm{KL}\big(\pi \,\Vert\, \pi^\star\big).
$$
In other words: **alignment and post-training are variational inference executed over the support of the pre-trained prior!**

### Why aligned models collapse in output diversity
In production, aligned models often exhibit severe mode collapse, sounding homogeneous, safe, and repetitive compared to the creative diversity of the pre-trained base model. This phenomenon is driven by three distinct mathematical and physical mechanisms:

1. **Exponential tilting:**
   The optimal posterior $\pi^\star \propto \pi_{\text{ref}} e^{r/\beta}$ stretches probability mass along reward gradients. As $\beta \to 0$, probability concentrates exponentially on the highest-reward outputs. However, as long as $\pi_{\text{ref}}$ has full support and rewards are bounded, $\pi^\star$ theoretically retains full support.
2. **Reverse KL and mode-seeking:**
   The policy optimization minimizes the reverse divergence $\mathrm{KL}(\pi_\theta \,\Vert\, \pi^\star) = \sum_y \pi_\theta(y) \ln \frac{\pi_\theta(y)}{\pi^\star(y)}$.
   - If the true target posterior $\pi^\star$ has multiple distinct modes (e.g., several equally valid creative responses), but the parameterized network $\pi_\theta$ lacks the capacity to cover all modes simultaneously;
   - Reverse KL inflicts an **infinite penalty** wherever $\pi_\theta(y) > 0$ and $\pi^\star(y) \approx 0$ (hallucination / out-of-distribution), while imposing zero penalty when $\pi_\theta(y) = 0$ and $\pi^\star(y) > 0$!
   - Consequently, **the policy deliberately abandons legitimate alternative modes and collapses all its probability mass onto the single safest mode (mode-seeking)**. This information-theoretic asymmetry explains why aligned models coalesce around a stereotypical "assistant voice."
3. **Stochastic optimization lock-in:**
   Under small-batch Monte Carlo policy gradients and high learning rates, the moment a particular response format receives strong early positive reward signals, momentum in adaptive optimizers (AdamW) rapidly locks in that trajectory, causing the policy to collapse onto a narrow subspace.

## 9.5.3 Exploration

A policy can only learn from actions it tries. Two cases must be distinguished. If the policy gives a correct answer *exactly zero* probability, the population policy gradient toward it is zero whatever the reward **[D]** — a support limitation. A softmax policy with finite logits never does that: every token has positive probability, so the population gradient need not vanish. But if a correct answer has probability $10^{-6}$, a batch of 64 samples almost surely contains no success, every reward is 0, and the *sampled* gradient is zero — an exploration and sample-efficiency failure rather than a mathematical one **[D]**. Exploration is therefore not optional: it comes from the stochasticity of sampling (temperature), from the diversity of the prompts, and, in classical RL, from deliberate bonuses for novelty. For language models, pretraining is the exploration strategy — it puts non-zero probability on a vast space of sensible outputs, which post-training then reweights. This is why reinforcement learning from scratch on language is hopeless and from a pretrained model is effective.

## 9.5.4 Goodhart's law and overoptimization

"When a measure becomes a target, it ceases to be a good measure." Any reward $r$ is a proxy for an intended objective $r^\dagger$. Optimizing $r$ improves $r^\dagger$ only as long as the two are correlated *on the distribution the policy produces* — and optimization moves that distribution toward places where they diverge **[D]**. [Manheim & Garrabrant (2018)](/references/#manheim-2018) distinguish regressional, extremal, causal and adversarial variants; [Skalse et al. (2022)](/references/#skalse-2022) formalize "hackable" reward pairs.

Reward hacking is also measurable. Training against a learned reward model and scoring the result with a larger "gold" reward model, [Gao et al. (2023)](/references/#gao-2023) found that the gold reward first rises and then falls as optimization proceeds, with the peak and the decline following smooth functions of $\sqrt{\mathrm{KL}(\pi\Vert\pi_{\text{ref}})}$ **[E]**. The KL anchor is thus a direct control on how far one travels along the proxy before it turns against the goal. The spine's buggy checker (§9.7.1) is the extreme case: proxy and gold agree on correct answers and disagree everywhere else, and the policy goes straight to the disagreement.

## 9.5.5 Summary

| Result | Says | Status |
|---|---|---|
| Thm 9.1 | $\nabla J = \mathbb{E}[\sum_t\gamma^t\nabla\ln\pi(a_t\mid s_t)(G_t - b)]$ | [D] |
| Prop 9.1 | KL-regularized optimum $= \pi_{\text{ref}}e^{r/\beta}/Z$ | [D] |
| DPO (§9.3.6) | preference likelihood in terms of log-ratios; $Z$ cancels | [D] given Bradley–Terry |
| Prop 9.2 | optimal baseline = score-weighted mean reward | [D] |
| Bayesian view (§9.5.2) | post-training = posterior with prior $\pi_{\text{ref}}$ | [D] |
| Overoptimization (§9.5.4) | gold reward rises then falls with $\sqrt{\mathrm{KL}}$ | [E] |
