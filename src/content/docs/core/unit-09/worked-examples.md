---
title: "9.4 Worked examples"
description: Eight examples — an exact policy gradient, what a baseline does to variance, the tilted optimum at three temperatures, a two-state Bellman equation, Bradley–Terry and DPO in numbers, post-training the spine model, reward hacking, and reading a KL.
sidebar: { label: "9.4 Worked examples", order: 4 }
---

Exact numbers are recomputed by `labs/test_worked_examples_u09_u12.py`; post-training results come from `spine/stage09_post_training.py` (150 steps of 64 prompts, $\beta = 0.05$, Adam at $10^{-3}$) and are quoted to two decimals.

## Worked Example 9.1 — An exact policy gradient

**Question.** A policy has three actions with logits $z=(0,0,0)$ and rewards $r=(1,0,0)$. In which direction should each logit move?

**Step 1: compute the policy and objective.** Equal logits give

$$
\pi=\left(\frac13,\frac13,\frac13\right),
\qquad
J=\sum_i\pi_i r_i=\frac13.
$$

**Step 2: differentiate the softmax expectation.** Since

$$
\frac{\partial\pi_i}{\partial z_k}
=\pi_i(\mathbf1\{i=k\}-\pi_k),
$$

we obtain

$$
\frac{\partial J}{\partial z_k}
=\sum_i r_i
\frac{\partial\pi_i}{\partial z_k}
=\pi_k(r_k-J).
$$

Therefore

$$
\nabla_zJ
=\left(
\frac13\frac23,\;
\frac13\left(-\frac13\right),\;
\frac13\left(-\frac13\right)
\right)
=(0.2222,-0.1111,-0.1111).
$$

**Step 3: connect to samples.** REINFORCE does not evaluate all three rewards in one update. It samples $A\sim\pi$ and returns

$$
\widehat g=r_A(e_A-\pi).
$$

Only action 1 gives a nonzero estimate when the baseline is zero. Yet averaging many such estimates recovers the exact vector above. The test draws 400,000 actions and checks every component within 0.004.

**Checks.**

- Components sum to zero, as they must: adding one constant to every logit leaves softmax unchanged.
- The rewarded action rises; the two symmetric unrewarded actions fall equally.
- This is a gradient-ascent direction. A loss minimized by gradient descent uses its negative.

**Takeaway.** REINFORCE is not a heuristic instruction to “reward good actions.” It is a Monte Carlo estimator of a gradient that can be computed exactly in this tiny case.

## Worked Example 9.2 — What a baseline does

**Question.** In Worked Example 9.1, how can subtracting $b=1/3$ leave the expected gradient unchanged yet reduce its noise?

Look only at the first logit's sampled gradient,

$$
\widehat g_1
=(r_A-b)(\mathbf1\{A=1\}-1/3).
$$

Each action has probability $1/3$.

| baseline $b$ | $\hat g_1$ if $a = 1$ | $\hat g_1$ otherwise | mean | variance |
|---|---|---|---|---|
| 0 | $1\cdot\tfrac23 = 0.667$ | $0$ | 0.2222 | 0.0988 |
| $\pi^{\top}r = \tfrac13$ | $\tfrac23\cdot\tfrac23 = 0.444$ | $(-\tfrac13)(-\tfrac13) = 0.111$ | 0.2222 | 0.0247 |

**Mean with no baseline.**

$$
\mathbb E[\widehat g_1]
=\frac13\left(\frac23\right)
+\frac23(0)
=\frac29.
$$

Its second moment is $\frac13(2/3)^2=4/27$, so

$$
\operatorname{Var}(\widehat g_1)
=\frac4{27}-\left(\frac29\right)^2
=\frac8{81}
\approx0.0988.
$$

**Mean with $b=1/3$.** If action 1 is drawn, the estimate is $4/9$. If action 2 or 3 is drawn, the reward advantage is $-1/3$ and the first-logit score is $-1/3$, so the product is $1/9$:

$$
\mathbb E[\widehat g_1]
=\frac13\frac49+\frac23\frac19
=\frac29.
$$

The variance is $2/81\approx0.0247$, one quarter of the original.

**Why can an unrewarded action give a positive first-logit gradient?** When action 2 is sampled and found worse than baseline, gradient ascent reduces action 2's logit. Softmax normalization then increases the relative probability of action 1. The sign is consistent with the exact gradient.

**Takeaway.** A baseline does not insert new task information. It uses the same outcome to express “better or worse than expected,” often making every sample more informative.

## Worked Example 9.3 — The tilted optimum

**Question.** How does the KL coefficient $\beta$ trade reward against fidelity to a reference distribution?

Let

$$
\pi_{\text{ref}}=(0.2,0.5,0.3),
\qquad
r=(1,0,0).
$$

Proposition 9.1 gives unnormalized weights

$$
w
=\pi_{\text{ref}}\odot e^{r/\beta}
=\left(0.2e^{1/\beta},0.5,0.3\right),
$$

then $\pi^\star=w/\sum_iw_i$.

| $\beta$ | $\pi^\star$ | optimal value $\beta\ln Z$ |
|---|---|---|
| 0.1 | (0.9998, 0.0001, 0.0001) | 0.839 |
| 0.5 | (0.6488, 0.2195, 0.1317) | 0.412 |
| 2.0 | (0.2919, 0.4426, 0.2655) | 0.244 |

For $\beta=0.5$,

$$
w=(0.2e^2,0.5,0.3)
\approx(1.4778,0.5,0.3),
$$

whose sum is $2.2778$. Dividing gives $(0.6488,0.2195,0.1317)$.

**Structural checks.**

- Actions 2 and 3 have equal reward, so their odds stay $0.5:0.3=5:3$.
- Action 1's odds against either unrewarded action are multiplied by $e^{1/\beta}$.
- As $\beta\to\infty$, $e^{r/\beta}\to1$ and $\pi^\star\to\pi_{\text{ref}}$.
- As $\beta\to0^+$, mass concentrates on the maximum-reward actions that had positive reference support.

The optimal value is not merely expected reward:

$$
\max_\pi\{
\mathbb E_\pi[r]
-\beta\mathrm{KL}(\pi\Vert\pi_{\text{ref}})
\}
=\beta\log Z.
$$

Substituting each table row into the left side reproduces the last column.

**Takeaway.** $\beta$ is not a temperature for decoding a fixed model. It defines a different training optimum by pricing distributional movement away from the reference.

## Worked Example 9.4 — A Bellman equation by hand

**Question.** How does a reward at state B give value to an earlier state A?

From A, action **go** gives reward 0 and moves to B. From B, the only action gives reward 1 and terminates. Let $\gamma=0.9$.

**Work backward from the terminal boundary.** A terminal state has future value 0. Therefore the action taken at B has

$$
V(B)=1+0.9(0)=1.
$$

At A,

$$
V(A)
=0+0.9V(B)
=0.9.
$$

Now add a **stay** action at A that gives reward 0 and returns to A. The optimality equation is

$$
V^\star(A)
=\max\{
0.9V^\star(B),\;
0.9V^\star(A)
\}.
$$

Trying $V^\star(A)=0.9$ gives $\max(0.9,0.81)=0.9$, so it is a fixed point. Waiting $k$ times and then going produces value $0.9^{k+1}$; every delay lowers the value. Waiting forever yields zero.

**Policy evaluation differs.** If a fixed policy stays with probability $q$ and goes with probability $1-q$, then

$$
V^\pi(A)
=q\,0.9V^\pi(A)
+(1-q)\,0.9.
$$

Solving,

$$
V^\pi(A)
=\frac{0.9(1-q)}{1-0.9q}.
$$

For $q=1/2$, this is $0.45/0.55\approx0.818$, below the optimal value 0.9.

**Takeaway.** The Bellman equation propagates consequences backward one decision at a time. Averaging under a fixed policy evaluates that policy; maximizing over actions solves a control problem.

## Worked Example 9.5 — Preferences in numbers

**Question.** What does one preferred–dispreferred pair contribute to reward modelling and to DPO?

**Part A: Bradley–Terry reward modelling.** Suppose the learned reward difference is

$$
r_\phi(y_w)-r_\phi(y_l)=1.
$$

Then

$$
P_\phi(y_w\succ y_l)
=\sigma(1)
=\frac1{1+e^{-1}}
=0.7311,
$$

and the negative log-likelihood is

$$
-\log 0.7311=0.3133.
$$

Only the difference matters. Replacing the two rewards $(2,1)$ by $(102,101)$ changes neither probability nor loss.

**Part B: DPO at initialization.** If $\pi=\pi_{\text{ref}}$, both response log-ratios are zero:

$$
u=\beta[(0)-(0)]=0,
\qquad
\ell_{\text{DPO}}
=-\log\sigma(0)
=\log2
=0.6931.
$$

**Part C: after a relative-odds change.** Suppose the policy has doubled
$\pi(y_w)/\pi_{\text{ref}}(y_w)$ and halved
$\pi(y_l)/\pi_{\text{ref}}(y_l)$. With $\beta=0.1$,

$$
\begin{aligned}
u
&=0.1[\log2-\log(1/2)]\\
&=0.1(\log2+\log2)\\
&=0.1386.
\end{aligned}
$$

Thus

$$
\ell_{\text{DPO}}
=\log(1+e^{-0.1386})
=0.6262,
$$

lower than at initialization. The preference is better satisfied, but the loss is not zero because the margin is still finite.

**Takeaway.** Bradley–Terry learns a difference between scalar rewards. DPO learns a difference between policy-versus-reference log-ratios. They share a logistic likelihood but operate on different quantities.

## Worked Example 9.6 — Post-training the spine model

**Question.** Does post-training change behaviour on prompts that were not used for the updates?

The experiment starts from the checked-in Stage 07 checkpoint. Each generated prompt asks for one digit after “Now *name* has ”. Training uses:

- 150 optimizer steps and 64 fresh prompts per step;
- seed 0, Adam with learning rate $10^{-3}$;
- REINFORCE with exact-answer reward, leave-one-out baseline and $\beta=0.05$ forward KL;
- or DPO pairs containing the correct digit and a randomly selected incorrect digit.

Evaluation uses 400 separately generated prompts from seed 123. Greedy accuracy asks whether the highest-probability character is correct; mean $p(\text{correct})$ uses the full predictive distribution.

| model | held-out greedy accuracy | mean $p(\text{correct})$ | additional training observation |
|---|---|---|---|
| pretrained Stage 07 | 0.3175 | 0.2037 | fixed reference |
| REINFORCE, $r=\mathbf1\{\text{answer}=x+y\}$ | **0.82** | **0.79** | last-20 reward 0.74; mean KL 1.347 |
| DPO, correct $\succ$ random incorrect | 0.78 | 0.76 | final pair loss is not an accuracy estimate |

**Reading the columns separately.**

- The reference's greedy accuracy and mean correct probability are different: a correct digit may receive probability without being the argmax.
- REINFORCE improves both metrics, so its gain is not merely a decoder threshold effect.
- DPO reaches a similar but not identical result from a different data interface. This one run does not establish that either method is generally better.
- Training reward 0.74 is computed on sampled actions during the last 20 updates; held-out greedy accuracy 0.82 uses argmax actions on fixed evaluation prompts. They need not match.
- KL 1.347 says the answer distributions moved substantially. It does not identify whether the movement was task-correct.

**What this run does not show.** It is one seed, one prompt generator, one-character answers and one short budget. It does not measure long-form arithmetic, robustness to new formats, calibration, or preference quality. Experiment 9.3 will require multiple seeds and predeclared metrics.

**Takeaway.** The correct feedback changes the toy model in the intended direction, but success is a held-out empirical result—not a consequence of the policy-gradient derivation.

## Worked Example 9.7 — Reward hacking with a one-character bug

**Question.** Can training reward become nearly perfect while the intended task becomes worse?

Keep every hyperparameter from Worked Example 9.6, but change the checker from

$$
r_{\text{true}}
=\mathbf1\{\text{answer}=x+y\}
$$

to

$$
r_{\text{bug}}
=\mathbf1\{\text{answer}\ge x+y\}.
$$

In code this is one `>=` where `==` was intended.

| | training reward (last 20 steps) | true accuracy on training samples | held-out greedy accuracy | mean $p$(correct) |
|---|---|---|---|---|
| buggy checker | **0.99** | 0.07 | **0.09** | 0.11 |

**Why the shortcut exists.** The generator samples $x\in\{1,\ldots,5\}$ and $y\in\{1,\ldots,4\}$, so $x+y\in\{2,\ldots,9\}$. Answer 9 passes the buggy inequality for every prompt. It is truly correct only when the sum is 9.

**Read each result against its evaluator.**

- Last-20 training reward 0.99 says sampled digits almost always satisfy the inequality.
- True training-sample accuracy 0.07 recomputes equality on those same sampled answers.
- Held-out greedy accuracy 0.09 uses new prompts and the intended equality.
- Mean correct probability 0.11 shows that the correct digit has lost most of the distributional mass.

The policy need not represent a conscious plan to “hack” anything. Gradient updates simply make rewarded outputs more likely. Because 9 is rewarded across all prompts, it is the easiest shared solution to the written objective.

**Counterfactual audit.** Before training, enumerate every possible answer digit against every possible $(x,y)$. This would immediately reveal that one digit receives reward on 100% of prompts. Runtime monitoring should also compute the intended equality with a separately implemented evaluator. Reusing the same buggy checker for training and evaluation would certify the failure.

**Takeaway.** Optimization did not malfunction: reward 0.99 accurately reports success under the buggy specification. The failure lies in treating that proxy as evidence for arithmetic correctness.

## Worked Example 9.8 — Reading a KL

**Question.** What does the reported

$$
\mathrm{KL}(\pi\Vert\pi_{\text{ref}})=1.34
$$

actually tell us?

The REINFORCE run averages this forward KL over answer-position distributions:

$$
\mathrm{KL}(\pi\Vert\pi_{\text{ref}})
=\sum_a\pi(a)
\log\frac{\pi(a)}{\pi_{\text{ref}}(a)}.
$$

The logarithm is natural, so the unit is a **nat**. One nat is $1/\log2\approx1.443$ bits; 1.34 nats is about 1.93 bits. “Nat” names a logarithmic information unit, not a count of changed tokens.

**A two-group reference calculation.** Suppose the reference gives probability 0.2 to the correct digit and 0.8 to all wrong digits together. A policy reverses those masses to 0.8 and 0.2 while preserving relative probabilities inside the wrong group. Then

$$
\begin{aligned}
\mathrm{KL}
&=0.8\log\frac{0.8}{0.2}
 +0.2\log\frac{0.2}{0.8}\\
&\approx0.832\text{ nats}.
\end{aligned}
$$

The observed 1.34 is larger than this simplified comparison. Possible contributors include a more extreme correct-mass shift and redistribution among individual wrong characters; one scalar KL cannot identify which occurred.

**What KL does and does not say.**

- It is zero exactly when the two distributions agree wherever $\pi$ has mass.
- It is directional: reversing the arguments generally changes the value.
- It is not an accuracy, reward, distance metric, or explanation of which outputs changed.
- Similar values can accompany opposite task outcomes. The buggy run reports about 1.43 nats, close to the correct run's 1.34, despite far worse true accuracy.

**Takeaway.** KL quantifies distributional departure from a reference under a specified direction and averaging distribution. It is a useful constraint and diagnostic, but it cannot certify that the departure is beneficial. [Unit 11](/core/unit-11/) asks what changed inside the model; Unit 10 asks whether the behavioural evidence is reliable.
