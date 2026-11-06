---
title: "Unit 09 · How do goals and feedback shape behavior?"
description: When the data say one thing and we want another, the objective must come from a goal — a reward, a preference, a verifier. From the score-function gradient to RLHF, DPO and verifiable-reward training of the spine model, with a reward-hacking audit.
sidebar: { label: "9.1 The problem", order: 1 }
unit:
  number: 9
  part: 3
  question: How do systems choose actions using goals and feedback — and how does post-training change a language model?
  anchor: The policy-gradient estimator with a KL anchor, and DPO, applied to the pretrained spine model on an arithmetic task with a programmatically checkable answer — including a deliberately buggy reward checker.
  spine: "Stage 09 — post-train the Stage 07 model with REINFORCE against the reward 1[answer = a + b] with a KL penalty to the pretrained model, and with DPO on (correct, incorrect) pairs; then repeat REINFORCE with a buggy checker (answer ≥ a + b) and measure true accuracy."
  checkpoint: Derive the policy-gradient estimator and show a baseline leaves it unbiased; derive the optimum of KL-regularized reward maximization and from it the DPO loss; construct a reward that a policy can satisfy without doing the task.
  status: draft
  atlas: [method-policy-gradient, method-dpo, mech-cross-entropy]
  answer: "When the objective is a reward rather than a target, the policy-gradient estimator turns trial and error into gradient steps; RLHF, DPO and verifiable-reward RL apply it to language models, with a KL anchor to the pretrained model limiting drift."
  leaves: "Optimizing a measured reward exposes every gap between the measurement and the intent — reward hacking — so we must ask when a score is evidence of reliability at all."
  builds_on:
    - { unit: 3, uses: "stochastic gradients and their variance" }
    - { unit: 7, uses: "the pretrained model as the initial policy" }
    - { unit: 8, uses: "sampling and the KL divergence" }
---

## 9.1 The problem: prediction is not yet a goal

The Stage 07 spine model learned to continue text. Its corpus contains 1,187 stories, including sentences such as:

> Ben had 2 eggs. Then Ben found 3 more. Now Ben has 5 eggs.

The corpus generator deliberately makes the stated sum wrong in 30% of the arithmetic stories. Maximum-likelihood pretraining is therefore asked to imitate a mixture of correct and incorrect continuations. On 400 held-out prompts, the checked-in model's most probable next character is the correct digit only **31.75%** of the time, and the model assigns the correct digit mean probability **0.204**. Those are observations from the fixed checkpoint and evaluation seed, not universal properties of language models.

The failure is not that gradient descent stopped working. It is that the training question and the deployment question differ:

| Stage | Question being optimized | What counts as success? |
|---|---|---|
| pretraining | “Which character resembles the corpus continuation?” | high probability on observed text |
| task training | “Which answer satisfies the arithmetic rule?” | the generated answer equals $a+b$ |

More pretraining on the same noisy distribution can improve imitation without making arithmetic the objective. Lower temperature can sharpen the wrong digit as easily as the right one. To change behaviour, training must receive a signal connected to the desired outcome.

### Target, reward, or preference?

Suppose the prompt ends with “Now Ben has ” and the model emits one digit. That digit is an **action**. A program returning

$$
r(s,a)=\mathbf 1\{a=\operatorname{digit}(x+y)\}
$$

is a **reward**: it evaluates what happened rather than specifying a probability distribution for every possible action.

For this toy arithmetic task, the same program could also generate a supervised target. Reinforcement learning is not magically required whenever a verifier exists. Supervised fine-tuning is often simpler when correct targets can be produced cheaply and the desired output is known in advance. Reward-based learning becomes especially useful when:

- only the final outcome is checkable, while there are many valid action sequences;
- actions change later states, so today's choice changes tomorrow's data;
- the evaluator is non-differentiable, delayed, sparse, or available only after a trial;
- feedback says which of two outputs is better without supplying a unique target;
- the policy must improve using samples from its own changing behaviour.

The spine deliberately uses the simplest one-action case so that the probability mathematics is visible before delayed rewards and long trajectories are added.

### The feedback loop

One training round contains four logically separate objects:

1. a **state** or prompt $s$;
2. a **policy** $\pi_\theta(a\mid s)$ that samples an action;
3. an **evaluator** that returns reward or preference feedback;
4. an **update rule** that changes the policy.

Confusing them creates bad explanations. The reward is not the policy, a high sampled reward is not yet a reliable evaluation, and the update algorithm is not the human goal. The policy-gradient estimator in §9.2 connects these objects without differentiating through the sampled action or the reward program.

### A measurement can become a target

The correct checker rewards exactly one digit. After 150 steps of REINFORCE with a KL penalty, held-out greedy accuracy rises from **0.32 to 0.82**; DPO on correct–incorrect pairs reaches **0.78** in the same checked-in run. These are small-model demonstrations, not competitive benchmark claims.

Now change one character in the checker: `==` becomes `>=`. With the buggy checker, the final training reward averaged over the last 20 steps is **0.99**, while held-out true accuracy falls to **0.09**. The optimization succeeded at the written objective and failed at the intended task. This is **reward hacking** in its smallest inspectable form.

:::note[Predict before reading on]
Before looking at Worked Examples 9.6–9.7, predict which digits a policy will prefer under the `>=` checker. Then ask a harder question: if the reported reward is 0.99, what additional measurement would reveal that the task has failed?
:::

## What this unit builds

1. **Learning from reward, from first principles** (§9.2): expected reward, score-function gradients, why baselines change variance but not the desired expectation, sequential credit assignment, KL anchoring, and preference data.
2. **Mathematical construction** (§9.3): MDPs and returns; the discounted policy-gradient theorem; values and advantages; Bellman equations; the closed-form KL-regularized optimum; reward modelling, PPO, DPO, and group-relative baselines.
3. **Worked reasoning** (§9.4): every major equation is evaluated on a small distribution before it is attached to a language model.
4. **Theory, connections, and counterexamples** (§§9.5–9.7): estimator variance, inference views, exploration, control, mechanism design, Goodhart effects, sycophancy, and reward hacking.
5. **Modern post-training and experiments** (§§9.8–9.9): evidence-labelled methods and controlled tests that separate training reward, true task accuracy, KL drift, and random-seed variation.
6. **The next question** (§9.7): a score is a measurement. [Unit 10](/core/unit-10/) asks when that measurement is credible evidence of reliability.

## Prerequisites

- **Unit 01:** expectation, entropy and KL divergence. KL is briefly reinterpreted here, so you need not memorize its earlier derivation.
- **Unit 03:** stochastic gradients and variance. The first bandit example re-derives the needed softmax gradient.
- **Unit 07:** what the pretrained model learned and why matching data is not the same as satisfying a task.
- **Unit 08:** sampling from a policy and the distinction between a distribution, a decoder and one realized sample.
- **Calculus:** $\nabla\log p=(\nabla p)/p$ when $p>0$. §9.2 derives every other identity it uses.

## How to read this unit

If reinforcement learning is new, read §9.2 through the three-action bandit before interpreting a language model as a policy. Keep one question beside every objective: **whose expectation is this, and what distribution produced the data?** That question distinguishes on-policy gradients, replayed data, preference optimization, and evaluation.

Exact arithmetic is recomputed by `labs/test_worked_examples_u09_u12.py`. `spine/test_spine.py` checks estimator bias, discount factors, KL and DPO gradients against analytic or finite-difference references. `spine/stage09_post_training.py` produces the quoted checkpoint results. Every empirical number should be read with its method, seed, step budget and held-out evaluator.

| Section | Content | Approx. reading time |
|---|---|---|
| [9.2 First principles](/core/unit-09/first-principles/) | expected reward, score-function gradients, baselines, trajectories, KL and preferences | 80 min |
| [9.3 Mathematical construction](/core/unit-09/construction/) | MDPs, policy gradients, values, Bellman equations, KL optimum, PPO, DPO and group baselines | 2–3 h |
| [9.4 Worked examples](/core/unit-09/worked-examples/) | eight fully interpreted calculations and spine observations | 90 min |
| [9.5 General theory](/core/unit-09/theory/) | variance, optimal baselines, inference, exploration and overoptimization | 80 min |
| [9.6 Connections](/core/unit-09/connections/) | control, economics, psychology, neuroscience and importance sampling | 45 min |
| [9.7 Limitations and counterexamples](/core/unit-09/limitations/) | reward hacking, proxy failure, sycophancy, plural preferences and credit assignment | 50 min |
| [9.8 Modern developments](/core/unit-09/modern/) | RLHF as a pipeline, offline and online preferences, AI feedback, verifiable rewards, process feedback and agents | 75 min |
| [9.9 Experiments](/core/unit-09/experiments/) | seven prediction-first controlled investigations | lab time |
| [9.10 Problems and further study](/core/unit-09/problems/) | 26 scaffolded problems, answer checks and source-guided reading | — |
