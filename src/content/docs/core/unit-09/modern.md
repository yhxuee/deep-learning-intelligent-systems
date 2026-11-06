---
title: "9.8 Modern developments"
description: A beginner-friendly map of RLHF, offline and online preference optimization, AI feedback, verifiable rewards, process supervision, simple policy gradients, and tool-using agents.
sidebar: { label: "9.8 Modern developments", order: 8 }
---

Labels: **[D]** derivation, **[E]** experimental evidence, **[O]** engineering observation, **[H]** open hypothesis.

The modern literature can feel like a parade of acronyms: RLHF, DPO, RLAIF, PPO, RLOO, GRPO and PRM. Do not memorize that parade. Every method makes choices along the same five axes:

| question | common choices | the real issue |
|---|---|---|
| Who supplies feedback? | human, AI judge, program, environment | Does the source measure what we actually want? |
| What form does it take? | demonstration, score, pairwise preference, pass/fail result, step label | How much information does one label contain? |
| Where do responses come from? | a fixed dataset or the current policy | Does the feedback cover what the policy now produces? |
| What receives credit? | whole response, each step, each token or action | Can the learner locate the useful decision? |
| How is the policy changed? | supervised loss, DPO-like loss, REINFORCE, PPO-like loss | Is the estimator stable, affordable and faithful to the objective? |

This table is more durable than method names. Two systems with different acronyms may make nearly the same five choices; two systems both called “RLHF” may not.

## 9.8.1 RLHF is a pipeline, not one algorithm

**Changing quantity:** a verbal goal becomes data, a proxy reward and finally a policy update.

An influential assistant-training pipeline has three stages:

1. **Supervised fine-tuning (SFT).** Humans provide good responses. Maximum likelihood gives the pretrained model a useful starting policy $\pi_{\mathrm{SFT}}$.
2. **Preference modelling.** For one prompt, annotators compare two candidate responses. A reward model $r_\phi(x,y)$ is trained to score the preferred response higher.
3. **Policy optimization.** The policy is updated to raise predicted reward while a KL term discourages a large departure from a reference policy.

In compact form, the third stage often targets

$$
\max_\theta\;
\mathbb{E}_{x\sim\mathcal D,\,y\sim\pi_\theta(\cdot\mid x)}
[r_\phi(x,y)]
-\beta\,
\mathbb{E}_{x\sim\mathcal D}
D_{\mathrm{KL}}\!\left(\pi_\theta(\cdot\mid x)\,\|\,\pi_{\mathrm{ref}}(\cdot\mid x)\right).
$$

Ouyang et al. used SFT, a learned preference model and PPO-based optimization to make a 1.3B-parameter model preferred by raters over a much larger unaligned model on their prompt distribution ([Ouyang et al. 2022](/references/#ouyang-2022)); earlier work established related preference-learning pipelines for language generation ([Ziegler et al. 2019](/references/#ziegler-2019); [Stiennon et al. 2020](/references/#stiennon-2020)) **[E]**.

What does that result establish? It shows that this pipeline can work under those data, models and evaluations. It does **not** show that a reward model recovers a person's true utility, that PPO is necessary, or that rater preference on the training distribution guarantees safety. The learned reward is still a proxy (§9.7).

### A concrete trace

Suppose the prompt is “Explain why the sky looks blue to a ten-year-old.” SFT supplies one good explanation. Preference data may say that a clear, short answer beats a technical but confusing one. The reward model generalizes from many such comparisons. Policy optimization then searches for responses the reward model likes.

Notice the two gaps:

- a comparison says **which of two sampled answers won**, not what the ideal answer is;
- optimizing the learned judge produces new answers, perhaps unlike anything the judge saw during training.

Those gaps motivate nearly every development below.

## 9.8.2 Offline and online preference optimization

**Changing quantity:** whether the policy learns from yesterday's responses or its current responses.

**Offline preference optimization** uses a fixed dataset of triples $(x,y^+,y^-)$. DPO is the canonical example: it increases the preferred response relative to the rejected response, measured against a reference policy ([Rafailov et al. 2023](/references/#rafailov-2023)). It removes the need to train an explicit scalar reward model and then run a separate RL loop. It does not remove the Bradley–Terry preference assumption or the KL-regularized policy relation used in its derivation (§9.3.6) **[D]**.

Offline data are simple and reproducible, but the policy can outgrow them. If every rejected answer is obviously bad, the dataset may say little about the subtle mistakes made by a stronger current policy. This is a form of distribution shift.

**Online preference optimization** repeats a loop:

1. sample fresh responses from the current policy or a mixture with the reference policy;
2. compare them using people or a trained preference model;
3. update the policy on these newly relevant comparisons.

Online variants can therefore place feedback near the current policy's decision boundary. Calandriello et al. analyzed connections between offline and online objectives and compared online variants on summarization ([Calandriello et al. 2024](/references/#calandriello-2024)) **[D/E]**. “Online” does not automatically mean “human in the loop”: a learned judge may label the fresh samples, so its biases remain.

### Preference is not a numerical outcome

A pairwise label gives ordinal information: $A\succ B$. It does not tell us whether $A$ is barely better, ten times better, or still unacceptable. A 2025 theoretical study constructs settings in which even infinite, noiseless, online ordinal preference data cannot identify an approximately optimal outcome ([Zhao et al. 2025](/references/#zhao-2025-preference-limits)) **[D]**. This is an impossibility result under its formal assumptions, not proof that preference learning never works. It tells us to match the feedback form to the decision:

- use comparisons when relative taste is genuinely the target;
- use grounded scores or measurable outcomes when absolute quality matters;
- retain an acceptability threshold when “less bad” is not “good enough.”

## 9.8.3 AI feedback and written constitutions

**Changing quantity:** who performs the expensive judgement.

In **constitutional AI**, written principles guide a model to critique and revise responses; AI-generated comparisons then provide much of the preference data used for harmlessness training ([Bai et al. 2022](/references/#bai-2022)) **[E]**. This changes two things at once:

- the intended rules become partly inspectable as text;
- one model can judge far more examples than a small human team can label.

It does not remove human choices. People still choose the constitution, training prompts, judge model, aggregation rule and evaluation criteria. If the judge systematically misses a subtle failure, scaling it produces more consistent blind spots, not more truth **[O]**.

A safe mental model is **delegated measurement**. AI feedback is a measuring instrument. Before optimizing against it, ask the same questions one would ask of a laboratory instrument: What was it calibrated against? On which range? What confounds it? Does optimization push the system outside that range?

## 9.8.4 Verifiable rewards and group-relative learning

**Changing quantity:** subjective judgement is replaced by a checker.

Some tasks provide a cheap outcome test:

- substitute an answer back into an algebraic equation;
- run unit tests on generated code;
- ask a proof checker to validate a formal proof;
- verify that a citation entails a claim and points to supplied evidence.

Training against such signals is often called **reinforcement learning with verifiable rewards (RLVR)**. The important word is *verifiable*, not *reinforcement*: the checker turns a vague goal into a repeatable reward.

For one prompt, sample $G$ responses and compute rewards $r_1,\ldots,r_G$. A group-relative method can use

$$
A_i=\frac{r_i-\bar r}{s_r+\epsilon}
$$

as a relative advantage, avoiding a separately trained value network. DeepSeekMath introduced a PPO-like group-relative method ([Shao et al. 2024](/references/#shao-2024)); DeepSeek-R1 reported large reasoning gains from verifiable-reward training at scale ([DeepSeek-AI et al. 2025](/references/#deepseek-r1-2025)) **[E]**. §9.3.7 explains why the ordinary group mean slightly rescales the score-function gradient and how a leave-one-out baseline differs **[D]**.

The simple equation exposes three failure modes:

1. **All pass or all fail.** Then every centered reward is zero; the group contains no within-prompt learning signal.
2. **Rare success is never sampled.** A perfect checker cannot reward an answer the policy never produces.
3. **The checker is incomplete.** Unit tests may miss a bug; exact-string matching may reject an equivalent answer; citation format may be rewarded without factual support.

Recent experiments extended outcome rewards beyond short arithmetic. For example, one grounded-question-answering study combined rewards for answer correctness, citation sufficiency and appropriate refusal, and found that staged optimization helped on its evaluated datasets ([Sim et al. 2025](/references/#sim-2025-grounded-rlvr)) **[E]**. This is evidence for that reward design and benchmark suite, not a guarantee that arbitrary generated citations are trustworthy.

Stage 09 is the smallest executable version of this story. With the exact checker, reward and correctness rise together. With the buggy “greater than or equal” checker, measured reward approaches one while true correctness collapses. Scaling the optimizer cannot repair the meaning of the checker.

### Case study: what DeepSeek-R1 does and does not establish

The useful object is the training pipeline, not the model name. DeepSeek-R1-Zero starts from a pretrained base model and applies RL with mechanically checked rewards. DeepSeek-R1 adds cold-start examples, reasoning-oriented RL, rejection sampling plus supervised fine-tuning, and a later RL stage; teacher outputs are also used to fine-tune smaller dense models ([DeepSeek-AI et al. 2025](/references/#deepseek-r1-2025)) **[E]**. At each arrow, ask what supplied the data, what was rewarded, what was filtered, and on which distribution the result was measured.

The study is evidence that this particular pipeline substantially improved measured mathematics and coding performance and that long verification and self-correction patterns appeared during training. It does **not** establish that longer text is deeper reasoning, that pure RL is the best recipe for every task, or that a distilled student uses the same internal mechanism as its teacher. R1-Zero's poor readability and language mixing also show why a checkable final answer is not the whole specification. The lesson generalizes beyond LLMs: an optimizer reveals both the information in a measurement and every important property that measurement omits.

## 9.8.5 Outcome reward versus process supervision

**Changing quantity:** whether feedback arrives only at the end or at intermediate steps.

An **outcome reward model (ORM)** judges the final answer. A **process reward model (PRM)** judges intermediate reasoning steps. If a ten-step solution has a wrong step 3 but accidentally reaches the right final number, an ORM may accept it; a PRM can identify the first invalid transition. Process feedback can also make search efficient by pruning a bad branch before generating the remaining seven steps.

On the MATH domain, Lightman et al. collected step-level labels and reported that process-supervised models selected correct solutions more reliably than outcome-supervised models in their setup ([Lightman et al. 2024](/references/#lightman-2024)) **[E]**. Later work trained a generative verifier to explain its step judgements and reported strong benchmark results using only a small fraction of the process labels used by a discriminative baseline ([Khalifa et al. 2025](/references/#khalifa-2025-thinkprm)) **[E]**.

These findings do not establish that written reasoning is the model's private causal computation, nor that a plausible critique is correct. “Process supervision” means supervising an observable trace. The trace, the hidden computation and the final outcome are three different objects.

Use outcome verification when the end state is decisive and cheaply checkable. Add process verification when partial solutions guide search, when the final test is sparse, or when auditors need to locate the first failure. But validate the step judge separately; otherwise one proxy has merely been replaced by many smaller proxies.

## 9.8.6 Simpler policy-gradient estimators

**Changing quantity:** the machinery used to estimate and constrain an update.

PPO combines importance ratios, clipping, value estimation and often a KL controller. Each component addresses a real problem, but together they create many interacting choices. A language-model response is still a sequential trajectory of tokens; it is not literally a one-step bandit. Yet the pretrained policy and terminal sequence-level reward can make a simple sequence-level estimator competitive.

Ahmadian et al. compared PPO with REINFORCE-style estimators using strong baselines and reported that simpler estimators could match or outperform PPO in their studied RLHF settings while using less memory ([Ahmadian et al. 2024](/references/#ahmadian-2024)) **[E]**. That is a comparative experimental result, not a theorem that REINFORCE always wins.

The useful lesson is methodological:

1. write the exact objective;
2. begin with the simplest unbiased or well-understood estimator;
3. measure variance, reward, KL, length and true task performance;
4. add clipping, a critic or other machinery only for an observed failure.

Stage 09 follows that order. Because the tiny action space lets us compute exact correctness, we can see whether variance reduction helps without confusing optimizer complexity with reward quality.

## 9.8.7 Agents and tools: feedback over longer horizons

**Changing quantity:** the action space grows from text tokens to consequential environment actions.

If a model can search, execute code, query a database or operate software, the response is no longer just a string. It becomes a trajectory

$$
(s_0,a_0,o_1,s_1,a_1,o_2,\ldots),
$$

where observations $o_t$ reveal what happened after each action. ReAct showed that interleaving reasoning traces with environment actions improved results on selected question-answering and interactive tasks ([Yao et al. 2023](/references/#yao-2023)); Toolformer showed that a model could learn useful tool-call patterns from self-generated annotations ([Schick et al. 2023](/references/#schick-2023)) **[E]**.

Longer horizons restore the classical RL difficulties in full force:

- **credit assignment:** which earlier action caused the final success?
- **partial observability:** did the agent fail, or did the tool hide relevant state?
- **non-stationarity:** did the policy change, or did the external service change?
- **side effects:** did a high reward come with an unacceptable real-world action?

A tool call can be syntactically valid and still be harmful. Therefore agent evaluation must record intermediate actions, permissions and side effects, not only the final answer. This is where Unit 09 hands the problem to Unit 10.

## 9.8.8 A decision guide

Do not choose a method by popularity. Start with the feedback you can justify.

| situation | reasonable starting point | first thing to distrust |
|---|---|---|
| high-quality target responses exist | SFT | coverage outside the demonstrations |
| fixed, trustworthy comparisons exist | DPO-like offline optimization | stale or easy negatives |
| the current policy changes quickly | online comparisons | judge drift and data cost |
| final answers are mechanically checkable | outcome-based RLVR | checker loopholes and rare success |
| partial solutions guide search | process verifier plus outcome check | plausible but invalid step judgements |
| actions affect external systems | constrained agent RL and action logging | side effects hidden by final-task reward |

Before moving on, try this diagnostic: for any post-training paper, point to (1) the source of feedback, (2) its information content, (3) the sampling distribution, (4) the credit-assignment unit and (5) the policy update. If one box is vague, the acronym is not yet an explanation.
