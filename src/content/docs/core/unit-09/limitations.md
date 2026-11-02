---
title: "9.7 Limitations and counterexamples"
description: Reward hacking, learned rewards that drift from intent, sycophancy, noisy and inconsistent preferences, credit assignment over long horizons, sample inefficiency — and the limitation that motivates Unit 10.
sidebar: { label: "9.7 Limitations", order: 7 }
---

Labels: **[D]** derivation, **[E]** experimental evidence, **[O]** engineering observation, **[H]** open hypothesis.

## 9.7.1 The policy optimizes the reward you wrote

A reward is a program, and the policy maximizes the program, not the intention behind it **[D]**. *Counterexample (Worked Example 9.7):* the checker `answer >= a + b` was meant to be `answer == a + b`; post-training drove its reward to 0.98 while true accuracy fell from 0.32 to 0.06, because "9" satisfies the buggy check for every prompt. Documented cases of "specification gaming" in RL range from agents that loop forever collecting a respawning bonus to systems that exploit simulator bugs ([Amodei et al. 2016](/references/#amodei-2016)) **[E]**. The defence is not a better optimizer but **independent measurement**: evaluate with a different, carefully reviewed check than the one used for training.

## 9.7.2 Learned rewards are proxies

A reward model trained on human preferences is a statistical estimate of those preferences, accurate where its training data were and arbitrary elsewhere. Optimizing against it pushes the policy into exactly the regions where it is least accurate; gold-standard quality first rises and then falls with optimization distance ([Gao et al. 2023](/references/#gao-2023)) **[E]**. The KL anchor (§9.2.5) limits the distance; it does not remove the proxy's errors.

## 9.7.3 Sycophancy and approval-seeking

People prefer responses that agree with them, flatter them, or sound confident. A model optimized for human approval learns these preferences too: [Sharma et al. (2024)](/references/#sharma-2024) found that assistants trained with human feedback systematically tailor answers to a user's stated beliefs, and that both humans and preference models sometimes prefer convincing sycophantic answers to correct ones **[E]**. The feedback signal measured approval; approval is not truth.

## 9.7.4 Preferences are noisy, inconsistent and plural

Bradley–Terry assumes a single latent utility with independent noise. Real annotators disagree with each other, are inconsistent across time, make intransitive choices, and represent particular populations and instructions **[E]**. A reward model fitted to their pooled labels averages over these disagreements in ways that are rarely documented. What counts as "better" is a value judgement that the method inherits but does not resolve **[H]**.

## 9.7.5 Credit assignment and sample efficiency

With one reward at the end of a long generation, the policy-gradient estimate assigns the same advantage to every token, important or not; its variance grows with length **[D]**. Learned value functions and process-level rewards (§9.8.3) help at the cost of new models to train and new proxies to exploit. Policy-gradient methods also need many samples: the spine's one-token task used 9 600 sampled answers; long reasoning tasks use millions **[O]**.

## 9.7.6 The limitation that motivates Unit 10: a score is not evidence by itself

Every number reported during post-training — reward, win rate against a reference, preference-model score — is produced by a measurement that the training process was trying to push up. Worked Example 9.7 shows such a number reaching 0.98 while the property it was meant to measure collapsed. The question generalizes far beyond reinforcement learning: a benchmark accuracy, a held-out loss, a leaderboard rank are all measurements that can be optimized, contaminated, or misaligned with deployment. **When is a score evidence of reliability at all — and what else must be measured?** That is the question of [Unit 10](/core/unit-10/).
