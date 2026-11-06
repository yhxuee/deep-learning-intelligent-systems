---
title: "9.5 General theory"
description: The variance of score-function estimators and the optimal baseline, KL-regularized reinforcement learning as Bayesian inference, exploration, and Goodhart's law with the empirical laws of reward-model overoptimization.
sidebar: { label: "9.5 General theory", order: 5 }
---

## 9.5.1 Variance is the price of not differentiating the reward

Under the assumptions of Theorem 9.1, the score-function estimator does not require a differentiable reward. That robustness comes with Monte Carlo noise. For one fixed state and one parameter component, write

$$
s_i(a)=\partial_i\log\pi(a),
\qquad
\widehat g_i=(r(a)-b)s_i(a).
$$

Its mean is $\partial_iJ$, so

$$
\mathrm{Var}(\hat g_i) = \mathbb{E}\big[(r - b)^2(\partial_i\ln\pi)^2\big] - (\partial_iJ)^2 .
$$

Large rewards, rare rewards, long trajectories and large score norms can all increase the first term. Unbiasedness alone says nothing about whether a feasible batch gives a useful direction.

**Proposition 9.2 (optimal constant baseline).** Among constant baselines, the variance of component $i$ is minimized by
$$
b_i^\star = \frac{\mathbb{E}\big[r\,(\partial_i\ln\pi)^2\big]}{\mathbb{E}\big[(\partial_i\ln\pi)^2\big]} .
$$

*Proof.* Expand the only part that depends on $b$:

$$
\mathbb E[(r-b)^2s_i^2]
=\mathbb E[r^2s_i^2]
-2b\,\mathbb E[rs_i^2]
+b^2\mathbb E[s_i^2].
$$

Its derivative is

$$
-2\mathbb E[rs_i^2]
+2b\mathbb E[s_i^2].
$$

Setting it to zero gives the stated ratio; the second derivative
$2\mathbb E[s_i^2]\ge0$ confirms a minimum. $\blacksquare$

The optimal baseline is an average reward weighted by the **squared score**. Different parameter components can have different component-wise optima. If one scalar baseline must minimize the sum of component variances—the trace of the covariance—the corresponding state-dependent choice is

$$
b^\star(s)
=\frac{
\mathbb E_a[
Q^\pi(s,a)
\lVert\nabla\log\pi(a\mid s)\rVert^2]}
{\mathbb E_a[
\lVert\nabla\log\pi(a\mid s)\rVert^2]}
$$

([Greensmith et al. 2004](/references/#greensmith-2004)).

The plain value $V^\pi(s)$ is simpler and often effective, but need not be optimal. For a Bernoulli policy parameterized by one logit, let $p=0.8$ and reward action 1 with 1. The two scores are $0.2$ and $-0.8$, so

$$
b^\star
=\frac{0.8(1)(0.2)^2}
{0.8(0.2)^2+0.2(-0.8)^2}
=0.2,
$$

whereas $V=0.8$. With $b^\star$, either sampled action produces gradient estimate $0.16$, giving zero variance. With $b=V$, the variance is $0.0576$. The executable test checks both.

For independent estimates with finite variance, averaging batch size $B$ divides covariance by $B$. Policy-gradient noise includes prompt sampling, action sampling, environment randomness and critic error, so its constant can be much larger than ordinary supervised minibatch noise. Correlated trajectories or reused groups do not receive the full independent-sample $1/B$ reduction automatically.

Long trajectories can add many noisy score terms, but variance does not obey one universal law in horizon length: correlations, discounting, reward placement and baselines matter. This is why one-step spine results should not be extrapolated directly to long-form agents.

## 9.5.2 KL-regularized RL as Bayesian inference

Proposition 9.1 permits an inference reading **[D]**:

$$
\pi^\star(a)
\propto
\underbrace{\pi_{\text{ref}}(a)}_{\text{prior-like}}
\underbrace{e^{r(a)/\beta}}_{\text{likelihood-like}}.
$$

If $e^{r/\beta}$ is interpreted as soft evidence that an output is good, the tilted policy resembles a posterior. Maximizing $J_\beta$ is equivalent to minimizing
$\mathrm{KL}(\pi\Vert\pi^\star)$ ([Korbak et al. 2022](/references/#korbak-2022)).

This is a mathematical analogy, not automatically a calibrated Bayesian model. A literal likelihood must describe a normalized observation process; $e^{r/\beta}$ is merely a positive potential unless such a process is specified. Adding a constant to every reward leaves the tilted distribution unchanged, while rescaling rewards changes the effective temperature.

The analogy makes two exact support statements:

- if $\pi_{\text{ref}}(a)=0$, finite reward cannot give $\pi^\star(a)>0$;
- if the reference has full support and rewards are finite, so does $\pi^\star$.

It does **not** imply that low-reference-probability outputs always grow slowly. A sufficiently large reward difference relative to $\beta$ can change their odds sharply. In sampled training, however, discovering a rare rewarded output may still take a long time; that is the exploration issue of §9.5.3.

Three diversity mechanisms should remain separate:

1. **Objective tilting.** Lower $\beta$ concentrates the exact target on higher reward. This is intended by the objective.
2. **Approximation.** A restricted policy family may not represent all modes of $\pi^\star$. Exclusive-KL projections often prefer covering some separated modes well rather than placing mass between them, but “exclusive KL always drops modes” is not a theorem for every family.
3. **Optimization and sampling.** Finite data, large steps, noisy advantages or decoder choices can collapse observed outputs even when the exact target retains support.

Diagnosing diversity loss therefore requires comparing the target objective, representational family, optimizer and sampling procedure—not attributing every collapse to the KL direction.

## 9.5.3 Exploration

A sampled policy-gradient update can reinforce only outcomes represented in its samples. Separate three questions:

### Is the action inside the policy's support?

If $\pi_\theta(a\mid s)=0$ throughout the parameterization around the current point, no on-policy sample contains $a$ and its reward supplies no direct score-function update **[D]**. A finite-logit softmax gives every token positive probability, so token-level support is formally full. Hard masks, truncation, invalid-action filters and deterministic decoders can remove support.

### Is the action likely enough to be observed?

If one success has probability $p$ and a batch contains $B$ independent attempts, the chance of seeing at least one is

$$
1-(1-p)^B.
$$

For $p=10^{-6}$ and $B=64$, this is about $6.4\times10^{-5}$. More than 99.99% of batches contain no success. With binary reward and a centred within-group baseline, an all-failure batch may produce exactly zero advantage.

This is not a contradiction: the **population** gradient can be nonzero while the finite-sample estimator is almost always zero and occasionally huge. Increasing temperature may improve coverage but can lower the probability of already-promising actions; it is a trade-off, not a universal exploration knob.

### Does trying the action reveal the right consequence?

In a long-horizon environment, a useful early action may receive reward only if many later choices also succeed. Exploration then concerns trajectories, not isolated tokens. Credit assignment and exploration interact.

Pretraining often provides a strong initial proposal distribution over grammatical and task-relevant outputs, which can make post-training vastly more sample-efficient **[E]**. It is better described as initialization and prior coverage than as “the exploration strategy.” Reinforcement learning from scratch is possible in bounded language-like environments; whether it is practical depends on reward density, horizon, model and compute. The spine supports only the narrower observation that its pretrained policy already assigns positive mass to every correct digit.

## 9.5.4 Goodhart's law and overoptimization

Goodhart's warning is often summarized as: “When a measure becomes a target, it ceases to be a good measure.” Let $r$ be the optimized proxy and $r^\dagger$ the intended objective. Improvement in $r$ transfers only where the relationship between them continues to hold under the policy-induced distribution. Optimization changes that distribution, often toward high-proxy regions with little original data.

[Manheim & Garrabrant (2018)](/references/#manheim-2018) distinguish several mechanisms:

| Mechanism | What breaks? | Simple post-training example |
|---|---|---|
| regressional | extreme proxy values contain more noise | selecting only the highest reward-model scores |
| extremal | policy leaves the evaluator's training region | unusual response style gets unreliable scores |
| causal | intervention changes the proxy–goal relation | making an answer longer raises a length-correlated score |
| adversarial | optimizer searches for evaluator weaknesses | formatting that fools a checker |

[Skalse et al. (2022)](/references/#skalse-2022) formalize when reward functions can disagree under optimization. The taxonomy is diagnostic: different failures require better data, causal redesign, adversarial testing or a different objective—not merely another optimizer.

Overoptimization can be measured empirically, but the measurement also needs scrutiny. [Gao et al. (2023)](/references/#gao-2023) optimized policies against one learned reward model and evaluated them with a larger held-out “gold” reward model. In their studied settings, gold reward first rose and then fell, and the fitted relationships used $\sqrt{\mathrm{KL}(\pi\Vert\pi_{\text{ref}})}$ as the optimization-distance variable **[E]**.

The bounded claim is important:

- the rise-and-fall pattern is an empirical result for particular models, data and evaluators;
- the “gold” reward model is a stronger proxy, not ground-truth human utility;
- KL is a useful x-axis and control variable, not a universal point at which every proxy turns bad.

The spine's buggy checker is a transparent extreme. The proxy and intended equality agree whenever the answer is correct, but the proxy also rewards many wrong answers. Training moves directly into that disagreement. Unlike a learned reward model, the bug can be enumerated exactly, letting us see the mechanism rather than infer it from curves.

### Optimization pressure is an experimental variable

A sound study varies step count, KL coefficient or target KL and reports both proxy and independent metrics along the path. Reporting only the final checkpoint can hide a peak followed by decline. Choosing the best checkpoint after inspecting the test metric, however, leaks evaluation information; the stopping rule needs a validation set or preregistration.

## 9.5.5 Summary

| Result | Says | Status |
|---|---|---|
| Thm 9.1 | $\nabla J = \mathbb{E}[\sum_t\gamma^t\nabla\ln\pi(a_t\mid s_t)(G_t - b)]$ | [D] |
| Prop 9.1 | KL-regularized optimum $= \pi_{\text{ref}}e^{r/\beta}/Z$ | [D] |
| DPO (§9.3.6) | preference likelihood in terms of log-ratios; $Z$ cancels | [D] given Bradley–Terry |
| Prop 9.2 | variance-optimal scalar baseline = squared-score-weighted mean return | [D] |
| Inference view (§9.5.2) | exponential reward tilt has prior-times-potential form | [D]; literal Bayesian likelihood needs more assumptions |
| Exploration calculation (§9.5.3) | success probability in $B$ independent attempts is $1-(1-p)^B$ | [D] |
| Overoptimization (§9.5.4) | held-out reward first rose then fell in the studied model settings | [E], not a universal KL law |

The derivations say how an update behaves for a stated objective and data-generating process. They do not establish that the reward expresses the goal, the samples cover important cases, or one checkpoint generalizes. Those are empirical and measurement questions carried into Units 10 and 12.
