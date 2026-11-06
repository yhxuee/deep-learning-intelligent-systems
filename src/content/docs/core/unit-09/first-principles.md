---
title: "9.2 First principles"
description: Actions, rewards and policies; why sampling blocks backpropagation and how the score-function trick gets around it; baselines; sequences of decisions; the KL anchor to the pretrained model; and learning from preferences instead of rewards.
sidebar: { label: "9.2 First principles", order: 2 }
---

## 9.2.1 From targets to consequences

Supervised learning supplies a desired output $y$ before the prediction is made. If the model assigns probability $p_\theta(y\mid x)$, cross-entropy gives the per-example loss $-\log p_\theta(y\mid x)$. The target tells the learner which output to increase.

Reward feedback has a different interface. The learner first chooses, then observes what the choice was worth:

| Symbol | Name | Spine arithmetic example |
|---|---|---|
| $s$ | state or context | the prompt ending in “Now Ben has ” |
| $a$ | action | one sampled next character |
| $\pi_\theta(a\mid s)$ | policy | the model's distribution over characters |
| $r(s,a)$ | reward | $1$ if the character is the digit $x+y$, otherwise $0$ |

For now, prompts come from a fixed distribution $d(s)$ and one action ends the episode. The objective is

$$
J(\theta)
=\mathbb E_{s\sim d}
  \mathbb E_{a\sim\pi_\theta(\cdot\mid s)}
  [r(s,a)].
$$

The two expectations matter. The outer expectation says which prompts count; the inner expectation averages over what the current policy might do. A policy that gets reward 1 on one lucky draw has not achieved $J=1$.

The reward need not be differentiable. It may contain an equality test, execute a compiler, run a game, or ask a person for a comparison. The challenge is that $\theta$ changes the *probability of choosing an action*, while the observed reward is attached to the discrete action that was actually chosen.

:::tip[Do not manufacture a distinction]
If a program can cheaply produce the unique correct answer before training, it can also provide a supervised target. This toy still uses reward learning because it makes the estimator auditable. The stronger motivation appears when outcomes are delayed, many action sequences can succeed, or feedback gives only a ranking.
:::

## 9.2.2 Why backpropagation stops at the sample

To improve $J$ by gradient ascent, we need $\nabla_\theta J$. A tempting computation graph is

$$
\theta\longrightarrow \pi_\theta
\longrightarrow \text{sample }a
\longrightarrow r(s,a).
$$

Ordinary backpropagation cannot pass through the discrete sampling decision. A tiny parameter change usually leaves the sampled symbol unchanged, then occasionally changes it abruptly. The reward program may also be non-differentiable. Treating the sampled token ID as a smooth number would invent a derivative that the problem does not have.

The way out is to differentiate the *probability of each possible action*. For one fixed state, assume the finite action set and reward function do not themselves depend on $\theta$:

$$
\begin{aligned}
\nabla_\theta J_s
&=\nabla_\theta\sum_a\pi_\theta(a\mid s)r(s,a)\\
&=\sum_a r(s,a)\nabla_\theta\pi_\theta(a\mid s)\\
&=\sum_a\pi_\theta(a\mid s)r(s,a)
   \nabla_\theta\log\pi_\theta(a\mid s)\\
&=\mathbb E_{a\sim\pi_\theta}
\left[r(s,a)\nabla_\theta\log\pi_\theta(a\mid s)\right].
\end{aligned}
$$

The identity in the third line is

$$
\nabla_\theta \pi_\theta
=\pi_\theta\nabla_\theta\log\pi_\theta,
$$

which follows from $\nabla\log p=(\nabla p)/p$ for $p>0$.

This is the **score-function** or **REINFORCE** identity ([Williams 1992](/references/#williams-1992)). Here “score” means the probability score $\nabla_\theta\log\pi_\theta(a\mid s)$; it does **not** mean the reward score. One sampled gradient

$$
\widehat g
=r(s,a)\nabla_\theta\log\pi_\theta(a\mid s)
$$

is noisy, but its expectation equals the exact gradient. Averaging independent samples reduces Monte Carlo noise.

### A three-action calculation

Let $\pi=(1/3,1/3,1/3)$ and rewards be $r=(1,0,0)$. Then $J=1/3$. If the policy is a softmax over logits $z$, Unit 03 gives

$$
\frac{\partial J}{\partial z_i}
=\pi_i(r_i-J),
$$

so

$$
\nabla_z J
=\left(\frac{2}{9},-\frac{1}{9},-\frac{1}{9}\right).
$$

Increasing the rewarded action's logit must be balanced by decreasing the others because probabilities sum to one. REINFORCE estimates exactly this vector without enumerating every action. For a huge vocabulary or a long trajectory, avoiding enumeration is the point.

It is useful—but incomplete—to read REINFORCE as **reward-weighted imitation of the policy's own samples**. It is incomplete because negative advantages can explicitly reduce a sampled action, sequential data depend on the policy, and off-policy samples require correction.

## 9.2.3 Baselines: compare with what was expected

The estimator can be unbiased and still be too noisy to learn efficiently. Suppose every reward is near 100, while the meaningful difference between actions is only 0.1. Multiplying score vectors by roughly 100 adds large fluctuations that say little about which action is better.

Subtract a **baseline** $b(s)$ that may depend on the state but not on the sampled action:

$$
\nabla_\theta J_s
=\mathbb E_{a\sim\pi_\theta}
\left[(r(s,a)-b(s))
\nabla_\theta\log\pi_\theta(a\mid s)\right].
$$

Why is the expectation unchanged? Condition on $s$:

$$
\begin{aligned}
\mathbb E_a\left[
b(s)\nabla_\theta\log\pi_\theta(a\mid s)
\right]
&=b(s)\sum_a\pi_\theta(a\mid s)
  \nabla_\theta\log\pi_\theta(a\mid s)\\
&=b(s)\nabla_\theta\sum_a\pi_\theta(a\mid s)\\
&=b(s)\nabla_\theta 1=0.
\end{aligned}
$$

Therefore the baseline changes the estimator's variance, not its desired mean. The difference $r-b$ is an **advantage estimate**: positive means better than the comparison point; negative means worse.

Return to the three-action example. With baseline $b=0$, the first logit's single-sample gradient has variance about $0.0988$. With $b=1/3$, its mean remains $2/9$, but variance falls to $0.0247$—one quarter as large. Worked Example 9.2 shows every outcome.

Three cautions prevent common mistakes:

1. **Action independence is conditional.** A learned value $V(s)$ is allowed because it uses the state, but a baseline that peeks at the sampled action can change the mean.
2. **The value is convenient, not always variance-optimal.** The minimum-variance constant depends on the squared norm of the score as well as reward; §9.5 derives it.
3. **A batch mean can contain the sample itself.** For batch size $B$, subtracting the ordinary batch mean of rewards scales the expected gradient by $(B-1)/B$. The spine instead gives each sample the mean reward of the other $B-1$ samples—a leave-one-out baseline. Group-relative methods may compare multiple answers to the same prompt ([Shao et al. 2024](/references/#shao-2024)).

:::note[Sanity check]
Subtracting a baseline is not “changing bad rewards into negative rewards.” It is centring the gradient estimator. The underlying objective $J$ still contains the original reward.
:::

## 9.2.4 Sequences of decisions

A language model answer is usually many tokens. A tool-using agent also changes the next state: a search query determines which page is observed, and that observation affects the next action. A **trajectory**

$$
\tau=(s_0,a_0,r_0,s_1,a_1,r_1,\ldots)
$$

has probability

$$
p_\theta(\tau)
=p(s_0)\prod_t
\pi_\theta(a_t\mid s_t)
P(s_{t+1}\mid s_t,a_t).
$$

Assume the environment transition $P$ does not depend on $\theta$. Only policy factors contribute to the log derivative:

$$
\nabla_\theta\log p_\theta(\tau)
=\sum_t\nabla_\theta\log\pi_\theta(a_t\mid s_t).
$$

For discounted return

$$
R(\tau)=\sum_{t=0}^{T-1}\gamma^t r_t,
\qquad 0\le\gamma\le1,
$$

the score-function identity first gives $\mathbb E[R\sum_t\nabla\log\pi_t]$. Rewards that occurred before action $a_t$ cannot be caused by that action, so their expected score contribution is zero. Removing them yields the return-to-go form

$$
\nabla_\theta J
=\mathbb E_\tau\left[
\sum_{t=0}^{T-1}
\gamma^t
\nabla_\theta\log\pi_\theta(a_t\mid s_t)
\bigl(G_t-b(s_t)\bigr)
\right],
$$

where

$$
G_t=\sum_{k=t}^{T-1}\gamma^{k-t}r_k.
$$

The outside factor $\gamma^t$ belongs there for the objective as written. Omitting it changes the weighting of later decisions. Some texts absorb this factor into a discounted state-visitation distribution; either convention is valid only when used consistently.

If a single terminal reward follows a 200-token answer, the same scalar outcome initially multiplies many token-score vectors. Which token deserved credit for a correct answer—or blame for a failure? This is the **credit-assignment problem**. Values and advantages use predictions of future return to make the comparison more local (§9.3.3), but they do not make the causal ambiguity disappear.

The spine's answer contains one action, so trajectory credit assignment is intentionally absent. That simplification isolates estimator and reward-design errors before combining them with a long horizon.

## 9.2.5 An anchor to what the model already knows

A policy optimized only for a measured reward can drift away from useful pretrained behaviour, collapse onto a narrow set of outputs, or exploit evaluator quirks. One common objective penalizes movement from a fixed **reference policy** $\pi_{\mathrm{ref}}$:

$$
J_\beta(\theta)
=\mathbb E_{s\sim d,\;a\sim\pi_\theta}[r(s,a)]
-\beta\,
\mathbb E_{s\sim d}
\left[
\mathrm{KL}\!\left(
\pi_\theta(\cdot\mid s)
\Vert
\pi_{\mathrm{ref}}(\cdot\mid s)
\right)
\right].
$$

For a fixed state and unrestricted policy, §9.3.4 shows that the optimum is

$$
\pi^\star(a\mid s)
=\frac{
\pi_{\mathrm{ref}}(a\mid s)
\exp(r(s,a)/\beta)}
{\sum_{a'}
\pi_{\mathrm{ref}}(a'\mid s)
\exp(r(s,a')/\beta)}.
$$

The reward exponentially **tilts** the reference distribution:

- small $\beta$ makes reward differences dominate;
- large $\beta$ keeps the policy closer to the reference;
- equal rewards reproduce the reference;
- if the reference gives an action exactly zero probability, this forward-KL solution cannot give it positive probability.

This is a trade-off parameter, not a universal safety guarantee. A small KL can hide a severe change on rare but important prompts, while a large KL can prevent needed correction. The direction also matters: $\mathrm{KL}(\pi_\theta\Vert\pi_{\mathrm{ref}})$ is not interchangeable with its reverse.

The KL is computed over full conditional distributions in the spine, not only over the one sampled action. Its gradient is therefore exact for the toy vocabulary. Large language-model systems often estimate sequence-level KL from samples or token log-ratios, which introduces additional variance and implementation choices.

## 9.2.6 When there is no reward: preferences

Some tasks have a crisp verifier: a unit test passes, a theorem checker accepts, or an arithmetic answer matches. Qualities such as usefulness, style and harmlessness do not generally have one complete programmatic test. A common interface is instead a comparison:

> For prompt $x$, response $y_w$ is preferred to response $y_l$.

Comparisons are not automatically reliable. Annotators can disagree, misunderstand the prompt, follow different values, or be influenced by length and presentation. Preference data are measurements that require protocols and held-out audits, not direct access to human intent.

One route fits a scalar reward model with the **Bradley–Terry** likelihood

$$
P(y_w\succ y_l\mid x)
=\sigma\!\left(
r_\phi(x,y_w)-r_\phi(x,y_l)
\right),
$$

where $\sigma(u)=1/(1+e^{-u})$ ([Bradley & Terry 1952](/references/#bradley-1952)). A policy is then optimized against this learned proxy, giving the reward-modelling plus reinforcement-learning pipeline commonly called RLHF ([Christiano et al. 2017](/references/#christiano-2017); [Ouyang et al. 2022](/references/#ouyang-2022)).

A second route uses the KL-regularized optimum to eliminate the explicit reward model from the policy-training objective. **Direct preference optimization** (DPO) compares how much the policy, relative to the reference, increases the winner over the loser ([Rafailov et al. 2023](/references/#rafailov-2023)). It does not make reward assumptions vanish: the Bradley–Terry preference model and KL-regularized policy relation are built into the derivation. §9.3.6 shows every algebraic step.

This yields an important separation:

| Feedback object | What is observed? | What is inferred? |
|---|---|---|
| scalar reward | one number for one sampled outcome | which actions should become more likely |
| pairwise preference | winner and loser | a latent utility difference or direct policy ratio |
| supervised target | desired output itself | no comparison model is required |

## 9.2.7 What we have so far

| Signal | Update question | Spine observation |
|---|---|---|
| target character | “increase this specified character” | pretraining: greedy accuracy $0.3175$ |
| exact programmatic reward | “increase sampled actions that pass” | REINFORCE: held-out greedy accuracy $0.82$ |
| correct $\succ$ incorrect pair | “increase the winner relative to loser and reference” | DPO: held-out greedy accuracy $0.78$ |
| buggy reward $\mathbf1\{\text{answer}\ge x+y\}$ | “increase any sampled digit that passes the bug” | training reward $0.99$, true accuracy $0.09$ |

The algorithms did not discover the intended goal. They transformed the feedback supplied to them. That is why a post-training report must keep at least four curves separate: optimization loss, measured training reward, drift from the reference, and held-out task metrics that do not reuse the training reward implementation.

**Next:** [§9.3 Mathematical construction](/core/unit-09/construction/) generalizes the one-action identity to MDPs, derives values and Bellman equations, and then builds PPO and DPO without skipping their assumptions.
