---
title: "9.3 Mathematical construction"
description: Markov decision processes, the policy-gradient theorem with baselines and advantages, value functions and the Bellman equation, the closed-form optimum of KL-regularized reward maximization, reward models from preferences, PPO's clipped objective, the derivation of DPO, and verifiable rewards with group baselines.
sidebar: { label: "9.3 Construction", order: 3 }
---

## 9.3.1 Markov decision processes

A **Markov decision process** (MDP) formalizes decisions whose actions affect what happens next ([Sutton & Barto 2018](/references/#sutton-2018)):

$$
(\mathcal S,\mathcal A,P,r,\gamma).
$$

| Component | Meaning | Text-generation instance |
|---|---|---|
| state $s_t$ | information available before acting | prompt plus generated prefix |
| action $a_t$ | one choice | next token |
| transition $P(s_{t+1}\mid s_t,a_t)$ | how the world changes | append token; or execute a tool |
| reward $r_t$ | measured consequence | verifier score, preference-model score, task outcome |
| discount $\gamma$ | relative weight of later reward | $1$ for a short finite answer, often $<1$ for continuing tasks |

The **Markov property** says the state contains enough information that, conditional on $(s_t,a_t)$, the distribution of the next state does not require the earlier history. It does not say the world has no memory. If history matters, a valid state must summarize that history; a language-model prefix does so by containing all earlier tokens.

A policy $\pi_\theta(a\mid s)$ and the transition rule generate a trajectory. The return from time $t$ is

$$
G_t
=r_t+\gamma r_{t+1}+\gamma^2r_{t+2}+\cdots
=\sum_{k=0}^{T-t-1}\gamma^k r_{t+k},
$$

and the episodic objective is

$$
J(\theta)
=\mathbb E_{\tau\sim p_\theta(\tau)}[G_0].
$$

The expectation includes both policy randomness and any environment randomness. For finite episodes, $\gamma=1$ is mathematically allowed. For an infinite horizon, $\gamma<1$ keeps bounded rewards summable and expresses a time preference.

Text generation has deterministic token transitions: appending a chosen token fixes the next prefix. Tool use is less simple because a query can return a stochastic or changing observation. The spine arithmetic task ends after one token, so no action affects a later decision. It is therefore a **contextual bandit**, the one-step edge case of an MDP.

:::note[Bandit or MDP?]
If every prompt is drawn independently and one answer receives one score, use the bandit model. If an action changes the next observation or later options, the sequential MDP structure matters. Calling every post-training problem “RL” does not remove this distinction.
:::

## 9.3.2 The policy-gradient theorem

Before differentiating, identify what depends on $\theta$:

- the initial-state distribution is fixed;
- environment transitions and rewards do not directly depend on $\theta$;
- the policy is differentiable and assigns positive probability wherever a log-probability is used;
- differentiation may be exchanged with the finite sum or integral over trajectories.

These assumptions are invisible in the final estimator but essential to its meaning. If the reward model is updated jointly, the environment changes with the policy, or data come from another policy, extra terms or corrections may be required.

**Theorem 9.1 (policy gradient, episodic discounted objective).** Let $J(\theta) = \mathbb{E}_{\tau\sim\pi_\theta}[G_0]$ with $G_t = \sum_{k\ge0}\gamma^kr_{t+k}$ for episodes of finite length. For any baseline $b(s_t)$ that does not depend on $a_t$,
$$
\nabla_\theta J(\theta) = \mathbb{E}_{\tau\sim\pi_\theta}\Big[\sum_{t}\gamma^t\,\nabla_\theta\ln\pi_\theta(a_t\mid s_t)\,\big(G_t - b(s_t)\big)\Big].
$$

*Proof, step 1: differentiate trajectories.* The trajectory probability is

$$
p_\theta(\tau)
=p(s_0)\prod_t
\pi_\theta(a_t\mid s_t)
P(s_{t+1}\mid s_t,a_t).
$$

Only policy factors depend on $\theta$, hence

$$
\nabla_\theta\log p_\theta(\tau)
=\sum_t\nabla_\theta
\log\pi_\theta(a_t\mid s_t).
$$

The score-function identity now gives

$$
\nabla_\theta J
=\mathbb E_\tau\left[
G_0\sum_t\nabla_\theta
\log\pi_\theta(a_t\mid s_t)
\right].
$$

*Step 2: remove rewards from the past.* A reward received before time $t$ is determined before $a_t$ is drawn. Conditional on $s_t$, its expected product with the action score is zero. What remains of $G_0=\sum_k\gamma^k r_k$ in the term for step $t$ is

$$
\sum_{k\ge t}\gamma^k r_k
=\gamma^tG_t.
$$

This is the causality or reward-to-go step.

*Step 3: subtract a baseline.* Conditional on $s_t$,

$$
\mathbb E_{a_t}\left[
b(s_t)\nabla_\theta\log\pi_\theta(a_t\mid s_t)
\right]
=b(s_t)\nabla_\theta\sum_a\pi_\theta(a\mid s_t)
=0.
$$

Multiplying by $\gamma^t$ does not change zero, so the action-independent baseline may be subtracted. $\blacksquare$

The theorem is an exact expectation identity. REINFORCE replaces that expectation with sampled trajectories. The theorem therefore establishes unbiasedness under its assumptions; it does not promise low variance, safe exploration, or a correct reward.

**The factor $\gamma^t$.** It is easy to drop, and many implementations drop it deliberately: $\sum_t\nabla\ln\pi(a_t\mid s_t)(G_t - b)$ is the gradient of a *different* objective, one that weights every visited state as if an episode started there. With $\gamma = 1$ the two coincide. *Example:* a two-step episode with a single action at $t = 0$ and, at $t = 1$, a Bernoulli action with logit $\theta = 0$ that earns reward 1 if it is 1; with $\gamma = 0.5$, $J = 0.5\,\sigma(\theta)$ and $\partial J/\partial\theta = 0.125$. The theorem gives 0.125; the expression without $\gamma^t$ gives 0.25. The spine's one-step task has only $t = 0$, where $\gamma^0 = 1$, so it cannot reveal the difference; `test_stage09_discounted_policy_gradient_carries_gamma_t` checks the two-step case.

For a softmax policy with logits $z$ over a finite action set, $\nabla_z\ln\pi(a) = e_a - \pi$ (Unit 03, §3.3.5). The single-sample loss whose gradient is the REINFORCE estimate is therefore $\ell = -(G - b)\ln\pi(a)$, with $\partial\ell/\partial z = (G - b)(\pi - e_a)$ — cross-entropy toward the sampled action, scaled by its advantage. `spine/stage09_post_training.py` implements exactly this line and checks it against finite differences through the whole Transformer.

When implementing gradient **descent**, the loss carries the minus sign above. When describing gradient **ascent** on expected return, the estimator is positive. Mixing these sign conventions is a common source of code that reliably learns the opposite policy.

## 9.3.3 Value functions and the Bellman equation

The **value** of a state under $\pi$ is $V^\pi(s) = \mathbb{E}_\pi[G_t\mid s_t = s]$, and the **action value** is $Q^\pi(s, a) = \mathbb{E}_\pi[G_t\mid s_t = s, a_t = a]$. They satisfy the **Bellman equation** ([Bellman 1957](/references/#bellman-1957)):

$$
V^\pi(s) = \sum_a\pi(a\mid s)\Big[r(s, a) + \gamma\sum_{s'}P(s'\mid s, a)V^\pi(s')\Big],
$$

a linear system for a fixed policy. The meaning becomes clearer by unpacking one recursion.

The equation is a one-step decomposition, not a mysterious fixed-point slogan:

$$
\text{value now}
=\mathbb E[
\text{reward now}
+\gamma\,\text{value next}
].
$$

For example, suppose state $s$ has only one action, gives no immediate reward, then moves deterministically to a terminal-predecessor state $u$ whose value is 1. With $\gamma=0.9$,

$$
V(s)=0+0.9V(u)=0.9.
$$

Repeated substitution expands the recursion back into the expected discounted sum of rewards. For a fixed $\pi$, this is **policy evaluation**. Replacing the policy-weighted average with $\max_a$ asks for the best action and produces the Bellman optimality equation; that is **control**. Mixing these two equations silently changes the problem.

$V^\pi(s_t)$ is the most widely used state-dependent baseline. It is *not* in general the variance-minimizing one: §9.5.1 derives the optimum, an average of action values weighted by the squared score, which can be far from $V^\pi$. For a Bernoulli policy with $p=0.8$ and reward 1 for action 1, $V=0.8$, while the optimal constant baseline is 0.2 and makes this one-dimensional estimator have zero variance ([Greensmith et al. 2004](/references/#greensmith-2004)).

The **advantage**

$$
A^\pi(s,a)=Q^\pi(s,a)-V^\pi(s)
$$

measures how much better an action is than the policy's own average at that state. Actor–critic methods learn a value approximation with a **critic** and update the policy **actor**. The temporal-difference residual

$$
\delta_t
=r_t+\gamma V_\phi(s_{t+1})-V_\phi(s_t)
$$

has the correct conditional mean advantage when $V_\phi=V^\pi$. With an approximate critic it can be biased. Generalized advantage estimation combines several temporal-difference horizons to trade variance against bias through a parameter $\lambda$ ([Schulman et al. 2016](/references/#schulman-2016)); it does not universally “add a little bias” by a fixed amount.

## 9.3.4 The KL-regularized objective and its optimum

Fix a state and write $\pi$ for $\pi(\cdot\mid s)$. Consider

$$
J_\beta(\pi) = \sum_a\pi(a)r(a) - \beta\sum_a\pi(a)\ln\frac{\pi(a)}{\pi_{\text{ref}}(a)} .
$$

Assume $\beta>0$, rewards are finite, and optimization is restricted to the support of $\pi_{\text{ref}}$. Otherwise the forward KL is infinite whenever $\pi$ assigns positive mass where the reference assigns zero.

**Proposition 9.1.** The maximizer of $J_\beta$ over supported distributions is

$$
\pi^\star(a)
=\frac{\pi_{\text{ref}}(a)e^{r(a)/\beta}}{Z},
\qquad
Z=\sum_a\pi_{\text{ref}}(a)e^{r(a)/\beta},
$$

and the maximum is $\beta\ln Z$.

*Proof.* Since $\pi_{\text{ref}}(a)e^{r(a)/\beta}=Z\pi^\star(a)$,

$$
\begin{aligned}
J_\beta(\pi)
&=-\beta\sum_a\pi(a)
\log\frac{\pi(a)}
{\pi_{\text{ref}}(a)e^{r(a)/\beta}}\\
&=-\beta\sum_a\pi(a)
\log\frac{\pi(a)}
{Z\pi^\star(a)}\\
&=-\beta\,\mathrm{KL}(\pi\Vert\pi^\star)
+\beta\log Z.
\end{aligned}
$$

By Gibbs' inequality (Unit 01, Theorem 1.1), the KL is non-negative and is zero exactly at $\pi=\pi^\star$. $\blacksquare$

The normalization constant is more than bookkeeping. It makes every probability sum to one and gives the optimal objective value. The odds form is often easiest to interpret:

$$
\frac{\pi^\star(a)}{\pi^\star(b)}
=\frac{\pi_{\text{ref}}(a)}
       {\pi_{\text{ref}}(b)}
\exp\!\left(
\frac{r(a)-r(b)}{\beta}
\right).
$$

Equal-reward actions retain their reference odds. A reward advantage $\Delta r$ multiplies the odds by $e^{\Delta r/\beta}$. If $\Delta r=1$ and $\beta=0.5$, that multiplier is $e^2\approx7.39$; it does not mean the preferred action gets probability 1, because the reference odds and all other actions still matter.

The spine test checks that gradient ascent on $J_\beta$ converges to $\pi^\star$ to six decimals. Inverting the relation gives the step DPO needs:

$$
r(a) = \beta\ln\frac{\pi^\star(a)}{\pi_{\text{ref}}(a)} + \beta\ln Z .
$$

The additive $\beta\log Z$ depends on the state but not the action. Preference comparisons between two responses to the same prompt will cancel it.

## 9.3.5 Reward models and PPO

The classic language-model RLHF pipeline separates three datasets and objectives:

| Stage | Data | Objective |
|---|---|---|
| supervised fine-tuning | prompt–response demonstrations | imitate a useful response distribution |
| reward modelling | prompt with preferred and dispreferred responses | predict pairwise preferences |
| policy optimization | samples from the changing policy | raise learned reward while controlling drift |

Keeping the stages separate makes their failure modes visible. A good reward-model validation score does not prove that optimizing the model preserves human preferences out of distribution.

**Reward model.** Collect prompt $x$ with responses $(y_w,y_l)$ and a human or AI label that $y_w$ is preferred. Fit $r_\phi$ by minimizing

$$
\mathcal L_{\text{RM}}
=-\mathbb E\left[
\log\sigma\!\left(
r_\phi(x,y_w)-r_\phi(x,y_l)
\right)
\right],
$$

which is logistic regression on reward differences ([Christiano et al. 2017](/references/#christiano-2017); [Stiennon et al. 2020](/references/#stiennon-2020)). Only differences are identified: adding the same state-dependent constant to every response reward changes no comparison. Reward scale is also consequential during policy optimization because it changes the trade-off against KL.

**PPO.** Suppose a batch was sampled from an old policy $\pi_{\text{old}}$, but optimization has already moved the current policy to $\pi_\theta$. The per-action importance ratio

$$
\rho_t(\theta)
=\frac{\pi_\theta(a_t\mid s_t)}
       {\pi_{\text{old}}(a_t\mid s_t)}
$$

corrects the first-order surrogate for that distribution change. Proximal Policy Optimization uses

$$
L_{\text{clip}}(\theta)
=\mathbb E\left[
\min\!\left(
\rho_t A_t,\;
\operatorname{clip}(\rho_t,1-\epsilon,1+\epsilon)A_t
\right)
\right]
$$

([Schulman et al. 2017](/references/#schulman-2017)).

Read the sign of $A_t$ before interpreting the minimum:

- if $A_t>0$, increasing the sampled action's ratio above $1+\epsilon$ brings no further surrogate gain;
- if $A_t<0$, decreasing the ratio below $1-\epsilon$ brings no further surrogate gain.

Clipping removes an *incentive* in the sampled surrogate. It is not a hard constraint on parameters, unsampled actions, or total KL. A shared neural-network update can move many other probabilities. Implementations therefore often monitor an estimated KL, stop an epoch when it grows too much, and separately add a reference-policy KL penalty.

The supervised-fine-tuning, reward-model, and PPO-with-KL pipeline was used in influential instruction-following systems ([Ouyang et al. 2022](/references/#ouyang-2022)) **[E]**. That historical result supports the pipeline's feasibility, not the claim that PPO is necessary for every preference-learning problem.

## 9.3.6 Direct preference optimization

For a response sequence $y=(y_1,\ldots,y_T)$,

$$
\log\pi_\theta(y\mid x)
=\sum_{t=1}^{T}
\log\pi_\theta(y_t\mid x,y_{<t}).
$$

DPO uses this full sequence log-probability, normally with an attention mask that excludes prompt and padding tokens. It is not a comparison of only the final token.

Now substitute the inverted optimum of §9.3.4 into the Bradley–Terry model. The $\beta\log Z(x)$ terms are equal for both responses to the same prompt and cancel:

$$
P(y_w\succ y_l) = \sigma\Big(\beta\ln\frac{\pi(y_w)}{\pi_{\text{ref}}(y_w)} - \beta\ln\frac{\pi(y_l)}{\pi_{\text{ref}}(y_l)}\Big).
$$

Maximizing the likelihood of the observed preferences under this model gives the **DPO loss** ([Rafailov et al. 2023](/references/#rafailov-2023)):

$$
\mathcal L_{\text{DPO}} = -\mathbb{E}\Big[\ln\sigma\big(u\big)\Big],\qquad u = \beta\Big[\big(\ln\pi(y_w) - \ln\pi_{\text{ref}}(y_w)\big) - \big(\ln\pi(y_l) - \ln\pi_{\text{ref}}(y_l)\big)\Big].
$$

There is no separately trained reward model, no on-policy sampling inside each optimization step, and no PPO: the update is a classification loss on already collected pairs. Its gradient with respect to the log-probabilities is $-\beta\sigma(-u)\big(\nabla\ln\pi(y_w) - \nabla\ln\pi(y_l)\big)$ — increase the preferred response, decrease the dispreferred one, weighted by how wrongly the current model orders them. For single-token answers, the softmax terms of the two log-probability gradients cancel, leaving $-\beta\sigma(-u)(e_{y_w} - e_{y_l})$ on the logits; the spine checks it by finite differences.

At initialization $\pi=\pi_{\text{ref}}$, so $u=0$, the loss is $\log2\approx0.6931$, and the logistic weight is $\sigma(0)=1/2$. If the current policy has already increased the winner's relative log-odds, $u>0$ and the pair receives less weight. If it ranks the pair in the wrong direction, $u<0$ and the weight grows.

“No reward model” here means no separately trained scalar network is evaluated during policy optimization. The derivation still assumes:

1. preferences follow a Bradley–Terry form for some latent reward;
2. the target policy is related to a fixed reference by the KL-regularized optimum;
3. winner and loser share the same prompt, so the partition term cancels;
4. the offline preference pairs cover the behaviour one wishes to improve.

Violating these assumptions does not necessarily make optimization numerically fail; it changes what the loss can be interpreted as estimating. DPO is also offline with respect to its fixed dataset, so it does not automatically collect feedback on new behaviours created by the updated policy.

## 9.3.7 Verifiable rewards and group baselines

When correctness can be checked by a program — arithmetic, unit tests, a proof checker — the reward needs no human and no learned model. Sampling $G$ answers to the same prompt and using their mean reward as the baseline gives an advantage $A_i = r_i - \bar r$ (optionally divided by the group's standard deviation) without a value network ([Shao et al. 2024](/references/#shao-2024)). Because $\bar r$ contains $r_i$ itself, this baseline is not independent of $a_i$: in expectation it scales the gradient by $(G-1)/G$. The **leave-one-out** baseline $b_i = \frac{1}{G-1}\sum_{j\ne i}r_j$ is independent of $a_i$ and keeps the estimator unbiased; it equals the group mean up to the factor $G/(G-1)$ on the advantages. Dividing by the group's standard deviation introduces a further, data-dependent rescaling. Training large models this way on mathematics and code, with long chains of reasoning before the answer, produced strong reasoning performance ([DeepSeek-AI 2025](/references/#deepseek-2025)) **[E]**. The spine's Stage 09 is the one-token version of this recipe, with a leave-one-out baseline over the batch; its buggy-checker run is the cautionary version (§9.7.1). Because the 64 prompts in a batch differ, that baseline is a constant, not a function of the state — unbiased, but less variance-reducing than a per-prompt baseline. The test `test_stage09_batch_mean_baseline_is_biased_and_leave_one_out_is_not` checks both the ordinary and leave-one-out batch baselines.

The scaling can be seen without probability theory. Since

$$
\bar r=\frac1G\left(r_i+\sum_{j\ne i}r_j\right),
$$

the group-mean advantage is

$$
r_i-\bar r
=\frac{G-1}{G}
\left(
r_i-\frac1{G-1}\sum_{j\ne i}r_j
\right)
=\frac{G-1}{G}(r_i-b_i).
$$

Thus group-mean and leave-one-out advantages point in the same direction for a fixed group, but differ by a known factor. In an optimizer, scale interacts with learning rate, clipping and normalization; “same direction” does not make every training run identical.

The group also needs information:

- if every sampled response fails, all binary rewards are 0 and every centred advantage is 0;
- if every response passes, all rewards are 1 and every centred advantage is again 0;
- learning occurs when the group contains reward variation, unless another signal is added.

If advantages are divided by group standard deviation, an all-equal group also has zero denominator; implementations must add an explicit convention such as skipping normalization or using an $\varepsilon$ floor. This creates an exploration requirement. A verifier can tell good from bad only after the policy samples both. If correct solutions have essentially zero probability, a relative baseline cannot invent them.

:::note[Claim check]
- **Claim.** The leave-one-out baseline leaves the policy-gradient estimator unbiased.
- **Assumptions.** The $G$ samples are drawn independently, so that $b_i$ does not depend on $a_i$.
- **Test.** Compute the expected gradient exactly on a small problem and compare it with the expectation of the estimator under each baseline (`test_stage09_batch_mean_baseline_is_biased_and_leave_one_out_is_not`).
- **Limitation.** Unbiased is not minimum-variance: the variance-minimizing baseline is the score-weighted mean reward (Proposition 9.2), which in general differs from both the group mean and $V^\pi$.
:::

