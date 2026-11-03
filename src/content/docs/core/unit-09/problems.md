---
title: "9.10 Problems and further study"
description: Twenty graded problems on policy gradients, baselines, value functions, KL-regularized optimization, preference learning, DPO and reward misspecification, with selected answers and an annotated reading list.
sidebar: { label: "9.10 Problems", order: 10 }
---

Difficulty: ★ routine · ★★ combines ideas · ★★★ proof or open-ended. **[AI-off]** problems are for independent practice and assessment.

## Computation

**9.1 ★ [AI-off]** For logits $z = (1, 0, 0)$ and rewards $r = (0, 1, 0)$, compute $\pi$, $J$ and the exact gradient $\partial J/\partial z$.

**9.2 ★ [AI-off]** Compute the tilted optimum $\pi^\star$ for $\pi_{\text{ref}} = (0.5, 0.5)$, $r = (1, 0)$ and $\beta = 1$.

**9.3 ★** A reward model gives $r(y_w) = 2.0$, $r(y_l) = 1.5$. What preference probability does Bradley–Terry assign, and what is the loss on this pair?

**9.4 ★** For a two-state chain where each step pays reward 1 and the episode lasts forever, compute $V$ with $\gamma = 0.9$ and $\gamma = 0.99$.

**9.5 ★★** Compute the DPO loss after training has made $\ln\frac{\pi(y_w)}{\pi_{\text{ref}}(y_w)} = 0.5$ and $\ln\frac{\pi(y_l)}{\pi_{\text{ref}}(y_l)} = -0.5$, with $\beta = 0.1$ and $\beta = 1$.

**9.6 ★★** In Worked Example 9.2, compute the optimal baseline of Proposition 9.2 for component 1 and the resulting variance.

## Derivation

**9.7 ★ [AI-off]** Derive the score-function estimator $\nabla\mathbb{E}_\pi[r] = \mathbb{E}_\pi[r\nabla\ln\pi]$ and show that subtracting a constant baseline leaves it unbiased.

**9.8 ★★ [AI-off]** Prove Proposition 9.1 and show that the maximum value is $\beta\ln Z$.

**9.9 ★★ [AI-off]** Derive the DPO loss from Proposition 9.1 and the Bradley–Terry model.

**9.10 ★★** Derive the gradient of the DPO loss with respect to the logits of a single-token answer and show the softmax terms cancel.

**9.11 ★★** Derive the per-token REINFORCE gradient for a sequence and the causality refinement that replaces $G_0$ by $G_t$.

## Proof

**9.12 ★★** Prove Proposition 9.2 and show that the mean reward is optimal when $(\partial_i\ln\pi)^2$ is independent of $r$.

**9.13 ★★** Show that adding a potential-based shaping term $\gamma\Phi(s') - \Phi(s)$ to every reward leaves the optimal policy of an MDP unchanged.

**9.14 ★★★** Show that maximizing $J_\beta$ is equivalent to minimizing $\mathrm{KL}(\pi\,\Vert\,\pi^\star)$, and explain why this makes post-training mode-seeking.

**9.15 ★★★** Construct a pair of rewards $r$ and $r^\dagger$ that agree on every action the pretrained policy takes with probability above 0.01, such that maximizing $r$ minimizes $r^\dagger$.

## Implementation and design

**9.16 ★ [AI-off]** Implement `reinforce_grad` from memory and pass the spine's finite-difference test.

**9.17 ★★** Add a learned value baseline to Stage 09 (a linear function of the final hidden state predicting the reward) and compare variance and speed with the leave-one-out batch baseline.

**9.18 ★★** Implement group-relative advantages: sample 8 answers per prompt, use each prompt's mean reward as its baseline. Compare with the leave-one-out batch baseline, and measure the $(G-1)/G$ bias of the plain group mean.

**9.19 ★★** Build a preference dataset from the spine's arithmetic by labelling pairs with the checker but flipping 20% of the labels. Train DPO and report how label noise affects the result.

**9.20 ★★★** Design a reward for "write a short story that a five-year-old would enjoy" for the spine model. List three ways a policy could hack it, and how you would detect each.

## Selected answers

- **9.1** $\pi = (e, 1, 1)/(e + 2) = (0.576, 0.212, 0.212)$; $J = 0.212$; gradient $\pi_k(r_k - J) = (-0.122, 0.167, -0.045)$.
- **9.2** $\pi^\star\propto(0.5e, 0.5) = (0.731, 0.269)$.
- **9.3** $\sigma(0.5) = 0.6225$; loss $0.474$.
- **9.4** $V = 1/(1-\gamma)$: 10 and 100.
- **9.5** $u = \beta\cdot1$: loss $\ln(1 + e^{-0.1}) = 0.644$ for $\beta = 0.1$; $\ln(1 + e^{-1}) = 0.313$ for $\beta = 1$.

## Further reading

**Foundations.** [Sutton & Barto 2018](/references/#sutton-2018), Chs. 3, 13 — MDPs and policy gradients. [Williams 1992](/references/#williams-1992) — REINFORCE.

**Post-training language models.** [Christiano et al. 2017](/references/#christiano-2017); [Ouyang et al. 2022](/references/#ouyang-2022); [Rafailov et al. 2023](/references/#rafailov-2023) — preferences, RLHF and DPO. [Shao et al. 2024](/references/#shao-2024) and [DeepSeek-AI 2025](/references/#deepseek-2025) — verifiable rewards at scale.

**What goes wrong.** [Amodei et al. 2016](/references/#amodei-2016); [Gao et al. 2023](/references/#gao-2023); [Sharma et al. 2024](/references/#sharma-2024); [Skalse et al. 2022](/references/#skalse-2022) — specification gaming, overoptimization, sycophancy and a formal definition of hacking.
