---
title: "Unit 09 · How do goals and feedback shape behavior?"
description: "MDPs, Bellman equations, policy gradients, RLHF, DPO, verifiable-reward RL, test-time computation and agents."
sidebar: { label: "09 · Decision", order: 9 }
unit:
  number: 9
  part: 3
  question: "How do systems choose actions using goals and feedback — and how does post-training change a language model?"
  anchor: "The policy-gradient estimator, and DPO / verifiable-reward RL applied to a small language model."
  spine: "Stage 09 — post-train the spine model on a task with a programmatically checkable answer (e.g. small arithmetic); measure gains and reward hacking."
  checkpoint: "Identify a reward specification that encourages undesirable behavior, and propose a measurable fix."
  status: outline
  atlas: []
---

:::caution[Planning outline — not yet textbook text]
This page records the unit's question, anchor exemplar, outcomes and activities. It is a syllabus, not a chapter: it states topics without teaching them. It will be rewritten into the ten-part structure of the [writing standard](/project/writing-standard/), as Units 03–05 have been (see the [roadmap](/project/roadmap/)). Contributions welcome.
:::

## The problem

A model is rewarded for passing unit tests. Predict two ways it could raise its reward without becoming a better programmer.

:::note[Predict before reading on]
Write your answer and the assumptions it depends on before continuing.
:::

## Core content

- Markov decision processes and Bellman relationships [Sutton & Barto 2018](/references/#sutton-2018).
- The **policy-gradient** (REINFORCE) estimator [Williams 1992](/references/#williams-1992), baselines and variance; PPO [Schulman et al. 2017](/references/#schulman-2017) at Atlas level.
- **Post-training language models:** RLHF [Ouyang et al. 2022](/references/#ouyang-2022), DPO [Rafailov et al. 2023](/references/#rafailov-2023), RL with verifiable rewards and group-relative baselines [Shao et al. 2024](/references/#shao-2024); [DeepSeek-AI 2025](/references/#deepseek-2025).
- Test-time computation: sampling multiple attempts, search and verification.
- **Agents:** a language model in a loop with tools and an environment; why evaluation of agents is a sequential-decision problem.
- Specification gaming and reward hacking as the alignment problem in miniature [Amodei et al. 2016](/references/#amodei-2016).

## Learning outcomes by ownership tier

| Tier | Outcomes |
|---|---|
| <span class="tier tier-internalize">Internalize</span> | MDPs; Bellman equation; policy-gradient estimator; the DPO objective |
| <span class="tier tier-verify">Understand & verify</span> | PPO clipping; KL regularization to a reference model; reward-model failure modes; agent evaluation |
| <span class="tier tier-access">Access on demand</span> | specific RL libraries and agent frameworks |

## Learning activities

**Construction (AI-off).** Derive and implement REINFORCE with a baseline on a bandit; verify the estimator is unbiased by Monte Carlo against the analytic gradient.

**Challenge (Critic mode).** Give an AI a reward function for a task of your choice and ask it to find exploits; then patch the reward and repeat. Report what generalizes.

**Experiment (Collaborator mode).** Post-train the spine model with a verifiable reward; predict the gain and the KL drift from the base model; inspect samples for reward hacking.

**Spine increment.** Stage 09 — post-train the spine model on a task with a programmatically checkable answer (e.g. small arithmetic); measure gains and reward hacking.

**Transfer & defense (AI-off).** For a new agentic task, specify the reward, the evaluation protocol and one safeguard against specification gaming.

## Key sources

[sutton-2018](/references/#sutton-2018) · [williams-1992](/references/#williams-1992) · [schulman-2017](/references/#schulman-2017) · [ouyang-2022](/references/#ouyang-2022) · [rafailov-2023](/references/#rafailov-2023) · [shao-2024](/references/#shao-2024) · [deepseek-2025](/references/#deepseek-2025) · [amodei-2016](/references/#amodei-2016)
