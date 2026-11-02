---
title: "9.8 Modern developments"
description: The RLHF pipeline, AI feedback and constitutions, the return of simple REINFORCE, verifiable rewards and reasoning, process supervision, and tool-using agents — with evidence labels.
sidebar: { label: "9.8 Modern developments", order: 8 }
---

Labels: **[D]** derivation, **[E]** experimental evidence, **[O]** engineering observation, **[H]** open hypothesis.

## 9.8.1 RLHF for assistants

**Quantity:** the reward, learned from preferences.

Supervised fine-tuning on demonstrations, a reward model trained on human comparisons, and PPO with a KL penalty against the fine-tuned model together made language models follow instructions far better than larger models without them, as judged by human raters ([Ouyang et al. 2022](/references/#ouyang-2022); earlier, [Ziegler et al. 2019](/references/#ziegler-2019); [Stiennon et al. 2020](/references/#stiennon-2020)) **[E]**. The pipeline remains the reference design, with DPO and its variants widely used as a simpler replacement for the reward-model-plus-PPO stages **[O]**.

## 9.8.2 Feedback from AI systems

**Quantity:** who supplies the preference.

Human labels are slow and expensive. In **constitutional AI**, a model critiques and revises its own outputs according to written principles, and AI-generated preference labels replace most human harmlessness labels ([Bai et al. 2022](/references/#bai-2022)) **[E]**. The approach scales feedback and makes the target behaviour partly explicit in text; it also inherits the evaluating model's blind spots, so independent evaluation matters more, not less (Unit 10) **[O]**.

## 9.8.3 Verifiable rewards and reasoning

**Quantity:** reward from a checker instead of a judgement.

For mathematics and code, a program can check the final answer. Policy-gradient training against such checks — with group-relative baselines instead of a value network ([Shao et al. 2024](/references/#shao-2024)) — produced models that generate long chains of reasoning before answering and improve substantially on competition mathematics ([DeepSeek-AI 2025](/references/#deepseek-2025)) **[E]**. Rewarding the *steps* of a solution rather than only its final answer (**process supervision**) gave more reliable improvements than outcome supervision on difficult mathematics in one large study ([Lightman et al. 2024](/references/#lightman-2024)) **[E]**. The spine's Stage 09 is the smallest instance of the outcome-reward recipe.

## 9.8.4 Simpler can be better

**Quantity:** the complexity of the optimizer.

PPO's machinery was designed for classical RL with long horizons and noisy value estimates. For language-model post-training, where the pretrained model is a strong starting point and episodes are single responses, REINFORCE-style estimators with good baselines match or exceed PPO at lower cost ([Ahmadian et al. 2024](/references/#ahmadian-2024)) **[E]**. The spine uses plain REINFORCE for this reason.

## 9.8.5 Agents and tools

**Quantity:** the action space.

When actions include calling tools — a search engine, a calculator, a code interpreter — the model becomes an agent acting in an environment. Interleaving reasoning with actions improves task success on question answering and interactive tasks ([Yao et al. 2023](/references/#yao-2023)); models can also learn *when* to call tools from self-generated supervision ([Schick et al. 2023](/references/#schick-2023)) **[E]**. Every lesson of this unit applies with more force: longer horizons make credit assignment harder, and an agent that can act in the world can find reward hacks with real consequences **[H]**.
