---
title: "9.10 Problems and further study"
description: Twenty-six scaffolded problems on policy gradients, baselines, Bellman equations, KL-regularized optimization, preference learning, DPO, verifiable rewards and specification failure, with checkpoints and selected answers.
sidebar: { label: "9.10 Problems", order: 10 }
---

Difficulty: ★ routine · ★★ combines ideas · ★★★ proof or open-ended. **[AI-off]** problems are intended for independent practice and assessment.

Do not begin by searching for a named formula. First write four objects at the top of the page: the random variable, its distribution, the quantity being optimized and the parameter being differentiated. That habit prevents most sign, expectation and “reward versus loss” mistakes.

A useful route for a first pass is 9.1–9.8, 9.9, 9.11, 9.14, 9.17, 9.21 and 9.25. The remaining problems preserve the full mathematical depth and are suitable for a second pass.

## Computation: make the symbols concrete

**9.1 ★ [AI-off]** For logits $z=(1,0,0)$ and rewards $r=(0,1,0)$, compute the softmax policy $\pi$, expected reward $J$, and exact gradient $\partial J/\partial z$. Check that the three gradient components sum to zero.

**9.2 ★ [AI-off]** Compute the tilted optimum $\pi^\star$ for $\pi_{\mathrm{ref}}=(0.5,0.5)$, $r=(1,0)$ and $\beta=1$. Then recompute it for $\beta=0.25$ without using code. Explain the direction of change.

**9.3 ★** A reward model gives $r(y_w)=2.0$ and $r(y_l)=1.5$. What preference probability does Bradley–Terry assign, and what negative log-likelihood does this pair contribute? What remains unchanged if 100 is added to both rewards?

**9.4 ★** A continuing state returns to itself and pays reward 1 at every step. Compute $V$ for $\gamma=0.9$ and $\gamma=0.99$. Why is the undiscounted value not finite in this example?

**9.5 ★★** Training has produced

$$
\ln\frac{\pi(y_w)}{\pi_{\mathrm{ref}}(y_w)}=0.5,
\qquad
\ln\frac{\pi(y_l)}{\pi_{\mathrm{ref}}(y_l)}=-0.5.
$$

Compute the DPO loss on this pair for $\beta=0.1$ and $\beta=1$. Separate the log-ratio difference $\Delta$ from the scaled logit $u=\beta\Delta$ so the role of $\beta$ is visible.

**9.6 ★** A group of four responses receives binary rewards $(1,0,0,0)$. Compute (a) advantages using the group mean and (b) leave-one-out advantages. Verify component by component that the first vector is $3/4$ of the second. What happens for $(1,1,1,1)$?

**9.7 ★★** A correct response has probability $p=10^{-4}$ per independent sample. With batch size $B=64$, compute the probability that a batch contains at least one correct response. Approximately how many batches are needed for at least a 50% chance of having seen one success? State why this is a discovery calculation, not a convergence guarantee.

**9.8 ★★** In Worked Example 9.2, compute the variance-minimizing constant baseline for the first logit using Proposition 9.2. List the sampled gradient value under each of the three actions and show that the variance becomes zero. Why does this not mean one baseline can make every policy-gradient problem noise-free?

## Derivation: rebuild the machinery

**9.9 ★ [AI-off]** Starting from a finite sum $J(\theta)=\sum_a\pi_\theta(a)r(a)$, derive

$$
\nabla J=\mathbb E_{a\sim\pi_\theta}[r(a)\nabla\ln\pi_\theta(a)].
$$

Then show that subtracting a constant baseline leaves the estimator unbiased. **Checkpoint:** the extra term must reduce to $b\nabla\sum_a\pi_\theta(a)$.

**9.10 ★★ [AI-off]** For a finite trajectory, derive the likelihood-ratio policy gradient. Use causality to remove rewards received before action $a_t$, and explain why the discounted objective in Theorem 9.1 retains an outer factor $\gamma^t$ as well as the return-to-go $G_t$.

**9.11 ★★ [AI-off]** Prove Proposition 9.1 by defining $Z=\sum_a\pi_{\mathrm{ref}}(a)e^{r(a)/\beta}$ and rewriting

$$
J_\beta(\pi)=-\beta D_{\mathrm{KL}}(\pi\|\pi^\star)+\beta\ln Z.
$$

State the required assumptions on $\beta$, reward finiteness and reference support.

**9.12 ★★ [AI-off]** Derive the DPO loss from Proposition 9.1 and the Bradley–Terry model. Mark the exact line where the prompt-dependent partition term cancels. Which assumption would fail if the winner and loser came from different prompts?

**9.13 ★★** For single-token answers, differentiate the DPO loss with respect to every vocabulary logit. Show why the two softmax normalization terms cancel. Then explain why this cancellation does not mean a shared neural network changes only two output probabilities.

**9.14 ★★** For $G$ independent samples to the same prompt, prove

$$
r_i-\bar r=\frac{G-1}{G}\left(r_i-\frac{1}{G-1}\sum_{j\ne i}r_j\right).
$$

Explain why the leave-one-out baseline is independent of $a_i$, whereas the ordinary group mean is not. What additional change occurs if advantages are divided by the random group standard deviation?

**9.15 ★★** Derive the Bellman expectation equation for a fixed policy and the Bellman optimality equation for control. Use the two-state system in Worked Example 9.4 to show numerically that evaluating a stochastic policy is not the same operation as maximizing over actions.

## Proof and counterexample: find the assumptions

**9.16 ★★** Prove Proposition 9.2 by differentiating the second moment of one gradient component with respect to a constant baseline. Show that the mean return is optimal when the squared score is independent of the return. Then derive the single scalar baseline that minimizes the trace of the gradient covariance.

**9.17 ★★** Show that maximizing $J_\beta$ is exactly equivalent to minimizing $D_{\mathrm{KL}}(\pi\|\pi^\star)$. Explain why this identity alone does **not** prove that the exact optimum drops modes: with finite rewards and full reference support, $\pi^\star$ also has full support. Give one way a restricted policy family or finite-sample optimizer could nevertheless lose diversity.

**9.18 ★★** Compare two actions: one has reference probability zero and reward 100; the other has reference probability $10^{-12}$ and reward 1. Under the tilted optimum, what can and cannot be concluded without specifying the other actions and $\beta$? Separate objective support from finite-sample discoverability.

**9.19 ★★★** Construct a finite action space, a reference policy, a proxy reward $r$ and intended reward $r^\dagger$ such that the two rewards agree on every action with reference probability greater than $0.01$, but the maximizer of $r$ is a minimizer of $r^\dagger$. State the probability mass of the disagreement region. What extra coverage assumption would rule out your construction?

**9.20 ★★★** Three response types $A,B,C$ receive aggregate preferences $A\succ B$, $B\succ C$ and $C\succ A$. Prove that no deterministic scalar reward can satisfy all three strict inequalities simultaneously. Discuss two possible explanations—heterogeneous annotators and context-dependent preference—before calling the labels “noise.”

**9.21 ★★** Potential-based shaping adds $F(s,a,s')=\gamma\Phi(s')-\Phi(s)$ to every reward. Sum the shaped discounted return along a finite trajectory and show the telescoping terms. State the terminal-potential or infinite-horizon boundedness condition needed before concluding that the optimal policy is unchanged.

## Implementation and design: make claims testable

**9.22 ★ [AI-off]** Implement `reinforce_grad` from memory and pass the spine's finite-difference test. Before running it, predict the sign of the rewarded action's logit gradient under the code's loss-minimization convention.

**9.23 ★★** Add a learned value baseline to Stage 09: a linear function of the final hidden state predicting reward. Compare gradient variance and learning curves with the leave-one-out batch baseline. Use a held-out set to test the critic; otherwise critic overfitting can masquerade as variance reduction.

**9.24 ★★** Implement group-relative advantages by sampling eight answers per prompt. Compare the ordinary group mean, a leave-one-out group mean and no standard-deviation normalization. Measure the predicted $7/8$ gradient scaling and count all-pass and all-fail groups.

**9.25 ★★** Build a preference dataset from spine arithmetic using the exact checker, then independently flip 0%, 10%, 20% and 40% of labels. Train DPO with matched initialization and pair count. Plot held-out correctness and preference loss. Why can training loss continue to improve after task correctness stops improving?

**9.26 ★★★** Design a complete measurement stack for “write a short story that a five-year-old would enjoy.” Include at least: the feedback source, label form, sampling policy, acceptability threshold, three plausible reward hacks, an audit for each, an independent outcome measure, and a stopping rule that does not inspect the final test set. A scalar reward alone is not a complete answer.

## Selected answers and self-checks

- **9.1** $\pi=(e,1,1)/(e+2)=(0.576,0.212,0.212)$ and $J=0.212$. Since $\partial J/\partial z_k=\pi_k(r_k-J)$, the gradient is $(-0.122,0.167,-0.045)$; its components sum to zero up to rounding.
- **9.2** For $\beta=1$, $\pi^\star\propto(0.5e,0.5)$, giving $(0.731,0.269)$. For $\beta=0.25$, the rewarded-to-unrewarded odds are multiplied by $e^4$, giving $(0.982,0.018)$. Smaller $\beta$ prices departure from the reference less heavily.
- **9.3** The difference is $0.5$, so the preference probability is $\sigma(0.5)=0.6225$ and the loss is $-\ln\sigma(0.5)=0.474$. Adding the same constant to both rewards changes neither.
- **9.4** $V=1/(1-\gamma)$, hence 10 and 100. At $\gamma=1$, the partial return after $T$ steps is $T$ and has no finite limit.
- **9.5** $\Delta=1$. Thus the losses are $\ln(1+e^{-0.1})=0.644$ and $\ln(1+e^{-1})=0.313$.
- **9.6** Group-mean advantages are $(3/4,-1/4,-1/4,-1/4)$; leave-one-out advantages are $(1,-1/3,-1/3,-1/3)$. The first vector is $3/4$ of the second. When all rewards equal 1, both are zero.
- **9.7** One batch succeeds with probability $1-(1-10^{-4})^{64}\approx0.00638$. Solving $(1-10^{-4})^{64n}\le0.5$ gives $n=109$ batches. Seeing one success does not imply a stable gradient or a converged policy.
- **9.8** The squared-score-weighted baseline is $b_1^\star=2/3$. Under every sampled action, $(r-b_1^\star)(\mathbf1[a=1]-1/3)=2/9$, so this component has zero variance. Other components, states and tasks generally have different optimal baselines.
- **9.17** The identity in 9.11 makes the equivalence immediate. Under full support and finite rewards, $\pi^\star(a)>0$ for every action; diversity loss must be attributed to reward tilting, approximation, optimization or sampling rather than to the identity alone.

## Further reading

**Foundations.** [Sutton & Barto 2018](/references/#sutton-2018), Chapters 3 and 13, develops MDPs and policy gradients; [Williams 1992](/references/#williams-1992) introduces REINFORCE. Read each derivation while naming the sampling distribution and conditioned variables.

**Preference-based post-training.** [Christiano et al. 2017](/references/#christiano-2017), [Ouyang et al. 2022](/references/#ouyang-2022) and [Rafailov et al. 2023](/references/#rafailov-2023) provide three distinct pipeline designs. [Calandriello et al. 2024](/references/#calandriello-2024) studies online preference optimization, while [Zhao et al. 2025](/references/#zhao-2025-preference-limits) analyzes limits of ordinal preference data. Compare assumptions, not only benchmark scores.

**Verifiers and reasoning.** [Shao et al. 2024](/references/#shao-2024) and [DeepSeek-AI 2025](/references/#deepseek-2025) study verifiable rewards at scale; [Lightman et al. 2024](/references/#lightman-2024) and [Khalifa et al. 2025](/references/#khalifa-2025-thinkprm) study step-level verification. Keep “observable reasoning trace” separate from “hidden causal computation.”

**What goes wrong.** [Amodei et al. 2016](/references/#amodei-2016), [Gao et al. 2023](/references/#gao-2023), [Sharma et al. 2024](/references/#sharma-2024) and [Skalse et al. 2022](/references/#skalse-2022) cover specification gaming, overoptimization, sycophancy and formal reward hacking. For every failure, ask whether the remedy belongs in the data, evaluator, objective, optimizer or deployment boundary.
