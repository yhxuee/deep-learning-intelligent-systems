---
title: "9.6 Connections"
description: Goals and feedback in optimal control, economics and mechanism design, behavioural psychology, the neuroscience of reward prediction errors, and statistics — and within the book.
sidebar: { label: "9.6 Connections", order: 6 }
---

## 9.6.1 Optimal control and dynamic programming

The Bellman equation came from control and operations research ([Bellman 1957](/references/#bellman-1957)). Control usually writes a **cost** $c$ to minimize rather than reward $r=-c$ to maximize:

$$
V(s)
=\min_a
\left[
c(s,a)
+\gamma\,
\mathbb E[V(S')\mid s,a]
\right].
$$

The mathematics is the same after changing sign. When transitions and costs are known and the state space is small, dynamic programming can solve the recursion exactly. Linear dynamics

$$
x_{t+1}=Ax_t+Bu_t
$$

with quadratic cost

$$
x_t^\top Qx_t+u_t^\top Ru_t
$$

lead to the linear–quadratic regulator. It is the control-side partner of the Kalman filter in Unit 05: the filter estimates state; the regulator chooses actions.

Reinforcement learning does not simply mean “control with no model.” Several cases exist:

| Dynamics | Reward | Typical route |
|---|---|---|
| known | known | planning or optimal control |
| unknown, observed through interaction | specified | model-free value or policy learning |
| learned dynamics model | specified | model-based RL and planning |
| known or unknown | learned proxy | preference-based post-training |

Sampling replaces some exact expectations with estimates, introducing variance; learning a model can recover planning but introduces model error. This is the same bias–variance–computation triangle seen elsewhere in the book.

## 9.6.2 Economics: utility, incentives and mechanism design

Expected-reward maximization has the mathematical form of expected-utility maximization, but the written reward need not equal the designer's true utility. That gap is exactly the issue.

In a **principal–agent problem**:

| Economics | Post-training |
|---|---|
| principal has a goal | designer or user has an intent |
| agent chooses hidden or complex actions | policy produces responses or tool actions |
| contract pays measurable outcomes | reward scores observable outputs |
| agent optimizes the contract | training increases expected reward |

If piece-rate pay counts units but not quality, output quantity can rise while quality falls. If a response reward correlates with length, a policy may become verbose without becoming more useful. Neither agent must be malicious; the incentive is incomplete.

**Mechanism design** asks for rules under which individually optimal actions produce a desired collective outcome. Reward design is a related but not identical problem: the designer chooses what the learner observes and optimizes, under limited measurability.

One precise result is **potential-based shaping**. Replace reward by

$$
r'(s,a,s')
=r(s,a,s')
+\gamma\Phi(s')-\Phi(s).
$$

Under the standard discounted MDP assumptions, the added terms telescope along a trajectory and preserve optimal policies ([Ng et al. 1999](/references/#ng-1999)). They can make intermediate progress easier to learn without changing which completed policy is optimal.

The conditions matter. Arbitrary “helpful partial credit” can change the optimum. Multiple checks and KL penalties may reduce some exploits, but they do not prove incentive compatibility with an unobserved human goal.

## 9.6.3 Psychology: reinforcement and preference

Operant conditioning studies how consequences change the frequency of behaviour. That is conceptually adjacent to reinforcement learning, but a policy-gradient equation is not a biological law. Machine-learning policies, experimental subjects and human learning differ in representation, memory and feedback.

The closer technical connection is **measurement by comparison**. People may struggle to assign an absolute score but can often choose between two stimuli. Models such as Thurstone's comparative judgement and Bradley–Terry convert repeated choices into a latent scale:

$$
P(i\succ j)
=\sigma(u_i-u_j).
$$

The equation assumes one scalar utility difference plus a particular noise link. It implies stochastic transitivity under a fixed context. Real judgements can violate that structure:

- preference can depend on which alternatives are shown;
- $A\succ B$, $B\succ C$, and $C\succ A$ can all occur;
- annotators can use different criteria;
- order, length and framing can change the label.

RLHF inherits this as a measurement-model problem. More labels reduce sampling noise around the chosen model; they do not make a misspecified one-dimensional utility model true.

:::note[Why pairs can still help]
Pairwise data avoid requiring annotators to calibrate a shared numeric scale. That is a practical advantage, not proof that comparisons are consistent or unbiased.
:::

## 9.6.4 Neuroscience: reward prediction errors

In classic conditioning experiments, responses of some midbrain dopamine neurons resemble a **reward prediction error**: unexpected reward produces a positive response; after learning, a predictive cue can acquire that response; omission of an expected reward can produce a dip. This pattern motivated comparison with the temporal-difference error

$$
\delta_t
=r_t+\gamma V(s_{t+1})-V(s_t)
$$

([Dayan & Abbott 2001](/references/#dayan-2001)).

The connection is useful because $\delta_t$ is zero when outcome and updated expectation agree, positive when things are better than predicted, and negative when worse. In a one-step terminal task it reduces to $r_t-V(s_t)$, resembling the baseline-centred signal of §9.2.3.

But the correspondence has boundaries:

- not every dopamine neuron has the same response;
- neural signals encode salience, movement and other variables as well as reward-related quantities;
- $V$, state representation and learning dynamics are model constructs, not directly observed labels in the brain;
- a TD error trains a critic, whereas a policy advantage enters an actor update.

This is a productive computational model and source of hypotheses, not evidence that brains literally implement the textbook algorithm line for line.

## 9.6.5 Statistics: importance sampling

Suppose actions were logged under behaviour policy $\mu$, but we want an expectation under target policy $\pi$. If $\mu(a\mid s)>0$ wherever $\pi(a\mid s)>0$, then

$$
\mathbb E_{a\sim\pi}[f(a)]
=\mathbb E_{a\sim\mu}
\left[
\frac{\pi(a\mid s)}{\mu(a\mid s)}
f(a)
\right].
$$

The ratio is an **importance weight**. It is exact in expectation under support and integrability conditions. Variance can be enormous when target probability is concentrated where behaviour probability is small.

PPO's ratio $\pi_\theta/\pi_{\text{old}}$ uses this idea inside a local policy surrogate. Clipping deliberately introduces bias to limit the incentive from extreme sampled ratios; it is not an unbiased importance-sampling estimator after clipping.

Off-policy evaluation asks a harder version: estimate a new policy's return using logs from an old one. In long trajectories, products of per-step ratios can explode or vanish:

$$
w_{0:T}
=\prod_{t=0}^{T}
\frac{\pi(a_t\mid s_t)}
     {\mu(a_t\mid s_t)}.
$$

Weighted estimators, per-decision ratios, models and doubly robust methods trade bias and variance differently. Unit 10 returns to the evaluation question: without coverage in the logged policy, no reweighting can reveal outcomes for actions never taken.

## 9.6.6 Inside the book

| This unit | Returns in |
|---|---|
| reward hacking | Unit 10 (when is a score evidence?), Unit 12 (pre-registration against self-deception) |
| KL divergence as an anchor | Unit 01 (KL as excess loss), Unit 08 (VAE's KL term) |
| variance of gradient estimators | Unit 03 (minibatch noise) |
| sampling cost of RL | Unit 06 (inference FLOPs; test-time compute) |
| the post-trained model | Unit 10 (evaluating it), Unit 11 (what changed inside it) |

The same mathematical object can play different roles across units. KL is a discrepancy in Unit 01, a latent regularizer in the VAE of Unit 08, a policy anchor here, and an optimization-distance diagnostic in Unit 10. Reusing the formula does not make those interpretations interchangeable; always ask which two distributions are compared, in which direction, and under which averaging data.

**Connection check.** When importing an idea from another field, identify:

1. the mapped objects—for example, contract $\leftrightarrow$ reward;
2. the shared equation or causal structure;
3. what does *not* map—for example, human institutional constraints versus a token policy;
4. whether the connection is a theorem, an analogy, or empirical evidence.
