---
title: "H · Markov chains and decision processes"
description: Markov chains, stationary distributions and mixing; Markov decision processes, value functions and the Bellman equations as linear systems; contraction and value iteration; discounted occupancy and the policy-gradient theorem in its episodic and occupancy forms; baselines, their bias and their variance; entropy- and KL-regularized decision problems.
sidebar: { label: "H · Markov decision processes", order: 8 }
---

Unit 09 treats post-training as reinforcement learning and works mostly with one-step problems, where the answer is a single token. This appendix gives the multi-step theory in general form: the Markov structure, the Bellman equations, why value iteration converges, the policy-gradient theorem with its discount factor in the right place, and the conditions under which a baseline leaves the gradient unbiased. Numbers in the examples are checked by `labs/test_appendices.py` and, for the policy-gradient facts, by `spine/test_spine.py`.

| Section | Used in |
|---|---|
| H.1 Markov chains | 01 (n-gram models), 07 (sequence models), 08 (ancestral sampling) |
| H.2–H.3 MDPs, Bellman equations, value iteration | 09 (§9.3.1, §9.3.3) |
| H.4 Policy-gradient theorem | 09 (§9.2.2, §9.3.2) |
| H.5 Baselines | 09 (§9.2.3, §9.3.7, §9.5.1) |
| H.6 Regularized decision problems | 09 (§9.3.4, §9.5.2) |

## H.1 Markov chains

A sequence of states $s_0, s_1, \dots$ in a finite set is a **Markov chain** if $\Pr(s_{t+1}\mid s_0,\dots,s_t) = \Pr(s_{t+1}\mid s_t)$. A time-homogeneous chain is described by a row-stochastic **transition matrix** $P_{ij} = \Pr(s_{t+1} = j\mid s_t = i)$. A distribution over states, written as a row vector $\mu_t$, evolves by $\mu_{t+1} = \mu_tP$.

A bigram model is a Markov chain on characters; an $n$-gram model is a Markov chain on $(n-1)$-tuples; and an autoregressive Transformer is a Markov chain whose state is the whole prefix. Ancestral sampling (Unit 08, §8.3.1) runs the chain.

A **stationary distribution** satisfies $\pi = \pi P$. If the chain is **irreducible** (every state can reach every other) and **aperiodic** (returns to a state are not restricted to multiples of some $d > 1$), then $\pi$ is unique and $\mu_t\to\pi$ from every starting distribution ([Levin & Peres 2017](/references/#levin-2017)). The speed is set by the second-largest eigenvalue modulus of $P$: the distance to $\pi$ shrinks roughly like $|\lambda_2|^t$.

**Example H.1.** The two-state chain $P = \begin{pmatrix}0.9 & 0.1\\0.3 & 0.7\end{pmatrix}$ has stationary distribution $(0.75, 0.25)$ — in general $(b, a)/(a + b)$ for switching probabilities $a$ out of state 1 and $b$ out of state 2 — and second eigenvalue $1 - a - b = 0.6$. Started in state 1, the probability of being in state 1 is $0.75 + 0.25\times0.6^t$: $1, 0.9, 0.84, 0.804, \dots$

A chain satisfies **detailed balance** with respect to $\pi$ if $\pi_iP_{ij} = \pi_jP_{ji}$; summing over $i$ shows $\pi$ is then stationary. Markov chain Monte Carlo methods construct chains with this property for a desired $\pi$.

## H.2 Markov decision processes and the Bellman equations

A **Markov decision process** adds actions and rewards: in state $s$ the agent takes action $a$, receives reward $r(s, a)$ and moves to $s'\sim P(\cdot\mid s, a)$. A **policy** $\pi(a\mid s)$ chooses actions. With discount $\gamma\in[0, 1)$, the **return** from time $t$ is $G_t = \sum_{k\ge0}\gamma^kr_{t+k}$ and the **value** of a state is $V^\pi(s) = \mathbb{E}_\pi[G_t\mid s_t = s]$ ([Sutton & Barto 2018](/references/#sutton-2018); [Puterman 1994](/references/#puterman-1994)).

Under a fixed policy, the states form a Markov chain with transition matrix $P_\pi(s, s') = \sum_a\pi(a\mid s)P(s'\mid s, a)$ and expected reward $r_\pi(s) = \sum_a\pi(a\mid s)r(s, a)$. Splitting the return into its first reward and the rest gives the **Bellman expectation equation**
$$
V^\pi = r_\pi + \gamma P_\pi V^\pi\quad\Longleftrightarrow\quad V^\pi = (I - \gamma P_\pi)^{-1}r_\pi .
$$
The inverse exists because $\rho(\gamma P_\pi)\le\gamma < 1$ (a stochastic matrix has spectral radius 1), and its Neumann series $\sum_t\gamma^tP_\pi^t$ is the expected discounted number of visits.

**Example H.2.** Two states, $P_\pi = \begin{pmatrix}0.9 & 0.1\\0.5 & 0.5\end{pmatrix}$, reward 0 in state 1 and 1 in state 2, $\gamma = 0.9$. Solving $\begin{pmatrix}0.19 & -0.09\\-0.45 & 0.55\end{pmatrix}V = \begin{pmatrix}0\\1\end{pmatrix}$ gives $V = (45/32, 95/32) = (1.406, 2.969)$. State 2 is worth more than twice state 1 although it pays only 1 per visit, because from state 1 the chain reaches it rarely.

The **action value** $Q^\pi(s, a) = r(s, a) + \gamma\sum_{s'}P(s'\mid s, a)V^\pi(s')$ satisfies $V^\pi(s) = \sum_a\pi(a\mid s)Q^\pi(s, a)$, and the **advantage** is $A^\pi = Q^\pi - V^\pi$. The **Bellman optimality equation**
$$
V^\star(s) = \max_a\Big[r(s, a) + \gamma\sum_{s'}P(s'\mid s, a)V^\star(s')\Big]
$$
characterizes the optimal value; any policy greedy with respect to $V^\star$ is optimal.

## H.3 Contraction and value iteration

Write the right side of the optimality equation as an operator $\mathcal T$ acting on value functions.

**Theorem H.1.** $\mathcal T$ is a $\gamma$-contraction in the sup norm: $\lVert\mathcal TU - \mathcal TV\rVert_\infty\le\gamma\lVert U - V\rVert_\infty$.

*Proof.* For each $s$, $|\max_af_a - \max_ag_a|\le\max_a|f_a - g_a|$. Applied to $f_a = r + \gamma\sum_{s'}PU$ and $g_a = r + \gamma\sum_{s'}PV$, the difference is at most $\gamma\max_a\sum_{s'}P(s'\mid s, a)|U(s') - V(s')|\le\gamma\lVert U - V\rVert_\infty$. $\blacksquare$

By the Banach fixed-point theorem, $\mathcal T$ has a unique fixed point $V^\star$, and **value iteration** $V_{k+1} = \mathcal TV_k$ converges to it with $\lVert V_k - V^\star\rVert_\infty\le\gamma^k\lVert V_0 - V^\star\rVert_\infty$. The same argument, with $\max_a$ replaced by the average under $\pi$, shows that iterating the expectation equation converges to $V^\pi$.

**Example H.3.** For Example H.2, starting from $V_0 = 0$, the bound $0.9^k\times2.969 < 10^{-3}$ requires $k\ge76$; the actual error first drops below $10^{-3}$ at $k = 71$. Discounting is what makes the problem well-posed: as $\gamma\to1$ the contraction weakens and the number of iterations grows like $1/(1 - \gamma)$.

**Policy iteration** alternates exact evaluation of $V^\pi$ (a linear solve) with greedy improvement; the **policy improvement theorem** guarantees each new policy is at least as good in every state, and the process terminates at an optimal policy in finitely many steps for a finite MDP.

## H.4 The policy-gradient theorem

Let $\pi_\theta$ be differentiable in $\theta$ and $J(\theta) = \mathbb{E}_{\pi_\theta}[G_0]$ for episodes starting from a fixed initial distribution.

**Theorem H.2 (episodic form).** For episodes of finite length and any baseline $b(s_t)$ not depending on $a_t$,
$$
\nabla_\theta J = \mathbb{E}\Big[\sum_t\gamma^t\,\nabla_\theta\ln\pi_\theta(a_t\mid s_t)\,\big(G_t - b(s_t)\big)\Big].
$$
The proof is Unit 09, Theorem 9.1: the score-function identity applied to the trajectory distribution, then removal of rewards earlier than $t$ (they do not depend on $a_t$), which leaves $\sum_{k\ge t}\gamma^kr_k = \gamma^tG_t$ in the term for step $t$.

**Discounted occupancy.** Define $d^\pi_\gamma(s) = (1 - \gamma)\sum_t\gamma^t\Pr(s_t = s)$, a probability distribution over states that weights early visits more.

**Theorem H.3 (occupancy form).** For $\gamma < 1$,
$$
\nabla_\theta J = \frac{1}{1 - \gamma}\,\mathbb{E}_{s\sim d^\pi_\gamma,\ a\sim\pi_\theta(\cdot\mid s)}\big[Q^\pi(s, a)\,\nabla_\theta\ln\pi_\theta(a\mid s)\big].
$$
*Proof sketch.* Differentiate $V^\pi(s) = \sum_a\pi(a\mid s)Q^\pi(s, a)$ and use $Q^\pi = r + \gamma PV^\pi$:
$\nabla V^\pi(s) = \sum_a\nabla\pi(a\mid s)Q^\pi(s, a) + \gamma\sum_a\pi(a\mid s)\sum_{s'}P(s'\mid s, a)\nabla V^\pi(s')$. Unrolling the recursion accumulates the first term over all future states, each weighted by $\gamma^t\Pr(s_t = s)$; normalizing the weights gives $d^\pi_\gamma$. $\blacksquare$

The two forms agree: the $\gamma^t$ in Theorem H.2 is exactly the occupancy weighting of Theorem H.3, and $\mathbb{E}[\gamma^tG_t\mid s_t, a_t] = \gamma^tQ^\pi(s_t, a_t)$. Many implementations **omit $\gamma^t$** and sample states from the undiscounted visitation; the result is not the gradient of $J$ but of a differently weighted objective (for $\gamma = 1$ the two coincide). **Example H.4** (Unit 09, §9.3.2): with one action at $t = 0$, a Bernoulli action with logit $\theta = 0$ at $t = 1$ rewarded only if it is 1, and $\gamma = 0.5$, the true derivative is $0.5\,\sigma'(0) = 0.125$; the expression without $\gamma^t$ gives $0.25$.

## H.5 Baselines: bias and variance

**Unbiasedness.** Subtracting $b$ leaves the gradient unchanged whenever $\mathbb{E}[b\,\nabla\ln\pi_\theta(a_t\mid s_t)] = 0$. This holds if $b$ is a function of $s_t$ (and of anything else determined before $a_t$ is drawn, or drawn independently of it), because $\sum_a\pi(a\mid s)\nabla\ln\pi(a\mid s) = \nabla\sum_a\pi(a\mid s) = 0$. It fails if $b$ depends on $a_t$ — including through the sample's own reward.

**Batch and group means.** With $B$ independent samples and baseline $\bar r = \frac1B\sum_jr_j$ for each of them, the own-reward share of $\bar r$ correlates with $a_i$, and
$$
\mathbb{E}\Big[\frac1B\sum_i(r_i - \bar r)\nabla\ln\pi(a_i)\Big] = \Big(1 - \frac1B\Big)\nabla J
$$
for i.i.d. samples. The **leave-one-out** baseline $b_i = \frac1{B-1}\sum_{j\ne i}r_j$ is independent of $a_i$ and unbiased, and differs from the batch mean only by the factor $B/(B - 1)$ on the advantages. Group-relative methods that normalize by the group standard deviation introduce a further data-dependent rescaling. The spine uses the leave-one-out baseline ([errata](/start/errata/), K03).

**Variance.** Among constant baselines for one coordinate, the variance-minimizing choice is the score-weighted average $b^\star = \mathbb{E}[r\,s^2]/\mathbb{E}[s^2]$ with $s = \partial\ln\pi$; for state-dependent baselines and the whole gradient, $b^\star(s) = \mathbb{E}_a[Q^\pi(s, a)\lVert\nabla\ln\pi(a\mid s)\rVert^2]/\mathbb{E}_a[\lVert\nabla\ln\pi(a\mid s)\rVert^2]$ ([Greensmith et al. 2004](/references/#greensmith-2004)). The value $V^\pi(s)$ is the common practical choice, not the optimal one. **Example H.5:** for a Bernoulli policy on a logit with $p = 0.8$ and reward 1 for action 1, $V = 0.8$ but $b^\star = 0.2$, which makes the one-sample estimate exactly $0.16$ for either action (Unit 09, §9.5.1).

## H.6 Entropy- and KL-regularized decision problems

Adding $\tau H(\pi(\cdot\mid s))$ — or subtracting $\beta\,\mathrm{KL}(\pi(\cdot\mid s)\,\Vert\,\pi_{\text{ref}}(\cdot\mid s))$ — to every step's reward gives a regularized MDP. In one step, the Gibbs variational principle (Appendix F.6) solves it: $\pi^\star\propto\pi_{\text{ref}}\,e^{r/\beta}$ with value $\beta\ln Z$ (Unit 09, Proposition 9.1). In many steps the same principle applies at every state with $r$ replaced by the action value, giving the **soft Bellman equation**
$$
V^\star(s) = \beta\ln\sum_a\pi_{\text{ref}}(a\mid s)\exp\Big(\frac{r(s, a) + \gamma\,\mathbb{E}_{s'}V^\star(s')}{\beta}\Big),
$$
whose operator is again a $\gamma$-contraction, and an optimal policy $\pi^\star(a\mid s)\propto\pi_{\text{ref}}(a\mid s)\exp(Q^\star(s, a)/\beta)$. Exact solutions keep the full support of $\pi_{\text{ref}}$; loss of diversity in practice comes from small $\beta$, restricted policy classes and incomplete optimization (Unit 09, §9.5.2; Appendix D.7).

## Exercises

**H.1** Find the stationary distribution of $P = \begin{pmatrix}0.5 & 0.5 & 0\\0.25 & 0.5 & 0.25\\0 & 0.5 & 0.5\end{pmatrix}$ and check detailed balance.

**H.2** Show that $(I - \gamma P_\pi)^{-1}$ has nonnegative entries and that its rows sum to $1/(1 - \gamma)$.

**H.3** For Example H.2, compute $V^\pi$ with $\gamma = 0.5$ and with $\gamma = 0.99$. Which state's value changes more, and why?

**H.4** Prove that the expectation operator $V\mapsto r_\pi + \gamma P_\pi V$ is a $\gamma$-contraction in the sup norm.

**H.5** Verify Theorem H.3 on Example H.4 by computing $d^\pi_\gamma$ and $Q^\pi$ explicitly.

**H.6** ★ Show that the soft Bellman operator of H.6 is a $\gamma$-contraction. *(Hint: log-sum-exp is 1-Lipschitz in the sup norm.)*

**Answers.** **H.1** $\pi = (0.25, 0.5, 0.25)$; detailed balance holds ($0.25\times0.5 = 0.5\times0.25$ for each adjacent pair). **H.3** $\gamma = 0.5$: $V = (0.125, 1.375)$; $\gamma = 0.99$: $V = (16.39, 18.05)$ — both grow with the effective horizon $1/(1-\gamma)$, and state 1's value rises from 9% to 91% of state 2's, because its eventual visits to state 2 now count almost as much as immediate ones.

## Further reading

[Sutton & Barto 2018](/references/#sutton-2018), Chs. 3–4 and 13; [Puterman 1994](/references/#puterman-1994) for the full theory of MDPs; [Levin & Peres 2017](/references/#levin-2017) for Markov chains and mixing; [Greensmith et al. 2004](/references/#greensmith-2004) on variance reduction for policy gradients.
