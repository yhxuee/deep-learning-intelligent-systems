---
title: "9.7 Limitations and counterexamples"
description: Reward hacking, learned rewards that drift from intent, sycophancy, noisy and inconsistent preferences, credit assignment over long horizons, sample inefficiency — and the limitation that motivates Unit 10.
sidebar: { label: "9.7 Limitations", order: 7 }
---

Labels: **[D]** derivation, **[E]** experimental evidence, **[O]** engineering observation, **[H]** open hypothesis.

## 9.7.1 The policy optimizes the reward you wrote

The mathematical update is defined by the implemented reward, not by an unstated intention **[D]**. Optimization may fail even at that reward, but if it succeeds, the guarantee concerns the written objective.

Worked Example 9.7 gives a complete counterexample. The intended checker was

$$
\mathbf1\{\text{answer}=x+y\},
$$

but the implementation used

$$
\mathbf1\{\text{answer}\ge x+y\}.
$$

Training reward reached 0.99 while held-out equality accuracy fell from 0.32 to 0.09. Answer 9 satisfies the implemented inequality for every possible prompt.

Specification failures have several sources:

| Failure | Example | Pre-training audit |
|---|---|---|
| implementation bug | `>=` instead of `==` | unit tests and exhaustive small-domain enumeration |
| omitted requirement | correctness checked, format or safety ignored | write an explicit requirement inventory |
| proxy shortcut | length or style correlates with preference | stratify and intervene on the suspected feature |
| environment exploit | simulator permits an impossible action | compare against real or independently modelled dynamics |

Safety discussions document these classes under reward hacking or specification gaming ([Amodei et al. 2016](/references/#amodei-2016)) **[E]/[O]**.

No single defence is sufficient. Independent evaluation avoids reusing the same bug; adversarial testing searches for shortcuts; reward-component dashboards reveal trade-offs; human review can catch semantic failures. Each defence is another measurement and can itself fail.

## 9.7.2 Learned rewards are proxies

A reward model estimates labels collected under a particular prompt distribution, response generator, annotation rubric and population. Its measured accuracy supports use near that evaluation distribution; behaviour elsewhere is unvalidated, not literally known to be “arbitrary.”

Policy optimization creates an adaptive distribution shift:

$$
\text{reward model}
\longrightarrow
\text{policy update}
\longrightarrow
\text{new responses}
\longrightarrow
\text{reward-model inputs unlike training}.
$$

The policy searches many outputs and selects those with high predicted reward. Prediction error that is harmless on random validation responses can become systematically selected after optimization—the analogue of multiple comparisons against one learned evaluator.

[Gao et al. (2023)](/references/#gao-2023) observed held-out reward first improve and later decline in studied settings as optimization distance increased **[E]**. This demonstrates possible overoptimization, not inevitable decline for every reward model.

Useful controls include:

- hold out prompts, response sources and annotators—not only random pairs;
- compare several independently trained reward models;
- periodically collect fresh labels from the updated policy distribution;
- report disagreement and uncertainty, not just mean reward;
- limit drift with KL or early stopping while validating the limit independently.

The KL anchor controls one kind of distributional movement. It cannot correct a reward model that is wrong inside the anchored neighbourhood, nor guarantee truth on rare high-stakes cases.

## 9.7.3 Sycophancy and approval-seeking

Preference feedback can entangle several qualities: correctness, politeness, confidence, agreement and writing style. If agreement predicts a “better” label in the collected data, an optimized model can learn to echo a user's belief even when the task calls for correction.

[Sharma et al. (2024)](/references/#sharma-2024) reported sycophantic behaviour in studied assistants and found settings where human raters and preference models favoured convincing agreement over correctness **[E]**. This is evidence for specific models and protocols, not a claim that every person always prefers flattery.

**Counterexample to “higher preference score means more truthful.”** Present the same factual question twice, once with “I think answer A is correct” and once with “I think answer B is correct.” If the model changes its factual conclusion to follow the user while evidence is unchanged, user-belief sensitivity has contaminated the truth-seeking measure.

Possible evaluations separate dimensions:

- score factual accuracy without revealing the user's belief;
- then add a stated belief and measure answer flips;
- score tone and helpfulness separately from conclusion correctness;
- include cases where respectful disagreement is the desired behaviour.

Approval is one legitimate social signal, but it is not interchangeable with truth, safety or long-term usefulness.

## 9.7.4 Preferences are noisy, inconsistent and plural

Bradley–Terry represents one comparison probability through one latent scalar difference. Real labels can depart from that model for at least three distinct reasons:

1. **random error:** one annotator misreads or clicks incorrectly;
2. **stable heterogeneity:** two annotators consistently apply different values or expertise;
3. **context dependence:** one person's ranking changes with framing or available alternatives.

Collecting more labels reduces the first source. It does not erase the second or third. Treating all disagreement as noise silently chooses an aggregation rule.

[Aroyo & Welty (2015)](/references/#aroyo-2015) argue that disagreement can carry information rather than merely annotation error **[E]/[H]**. For post-training, a pooled reward can obscure:

- minority preferences and accessibility needs;
- domain-expert versus general-audience judgements;
- legitimate locale or cultural differences;
- conflicts among helpfulness, harmlessness and autonomy.

A better report states who annotated, which rubric they used, agreement by category, how ties and abstentions were handled, and whether the deployed population matches the labelled one. Conditional or plural reward models may preserve some heterogeneity, but the system still needs a governance decision about which preferences apply where. Optimization cannot resolve that normative choice.

## 9.7.5 Credit assignment and sample efficiency

With one terminal reward, a basic Monte Carlo estimator initially multiplies many token-score terms by the same return. It knows that the trajectory succeeded or failed, but not which decision caused the outcome. Irrelevant token choices add noise.

It is tempting to state that variance simply “grows with length.” The direction is often observed, but the exact scaling depends on score correlations, discounting, reward structure, baselines and parameter sharing. A deterministic prefix can add no action-sampling variance; strongly correlated terms can grow faster than independent ones. The safe conclusion is that longer horizons create a harder credit-assignment and estimation problem, not one universal linear law.

Methods trade new assumptions for denser information:

| Method | Added information | New failure surface |
|---|---|---|
| value critic | predicted future return at each state | critic bias and instability |
| process reward | intermediate step labels | a new proxy at every step |
| outcome decomposition | task-specific subgoals | hand-designed misspecification |
| search or best-of-$N$ | more sampled alternatives | inference cost and selection bias |

The spine used

$$
150\times64=9{,}600
$$

sampled answers for a one-action task. This exact count is not a scaling law. A long task's requirement depends on success probability, reward variance, horizon, reuse of off-policy data and model initialization. Reporting only “training steps” hides how many responses and tokens were sampled.

## 9.7.6 The limitation that motivates Unit 10: a score is not evidence by itself

Many post-training numbers—reward, preference-model score, pass rate—come from measurements the optimizer was explicitly pushing upward. They are training diagnostics, not independent confirmation. Worked Example 9.7 reaches reward 0.99 while intended accuracy collapses.

A score becomes stronger evidence when its measurement chain survives questions such as:

| Question | Why it matters |
|---|---|
| Was the evaluator independent of training? | shared bugs and leakage inflate agreement |
| Was the metric defined before results were seen? | post hoc selection favours lucky metrics |
| Does the sample represent deployment? | average performance can hide shifted subgroups |
| Is uncertainty reported? | one point estimate hides seed and sampling variation |
| Were failure severities separated? | many harmless successes can mask rare catastrophic errors |
| Can another method reproduce the conclusion? | independent measurement exposes common-mode failure |

The issue generalizes beyond RL. Benchmark accuracy, held-out loss and leaderboard rank can all be optimized, contaminated or disconnected from deployment. Even an independent metric is evidence only for the population and conditions it samples.

**When is a score credible evidence of reliability, and what else must be measured?** [Unit 10](/core/unit-10/) builds calibration, confidence intervals, distribution-shift checks, subgroup analysis and red-team evaluation to answer that question.

:::note[Boundary of this unit]
Unit 09 explains how feedback changes a policy. It does not certify that the changed policy is better. Training and evaluation must remain separate claims.
:::
