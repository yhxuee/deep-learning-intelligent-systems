---
title: "12.8 Modern developments"
description: A November 2026 snapshot of reporting standards, statistical evaluation, artifact and data audits, benchmark re-evaluation, and verifier-first AI-assisted research.
sidebar: { label: "12.8 Modern developments", order: 8 }
---

Labels: **[D]** derivation, **[E]** experimental evidence, **[O]** engineering observation, **[H]** open hypothesis.

This page is a dated snapshot: **November 2026**. Tools and policies will change; the distinctions among a runnable artifact, a replicated effect, and a verified scientific claim should not.

## 12.8.1 From reproducibility checklists to inspectable artifacts

**Quantity:** how much of the path from claim to evidence another person can inspect.

Machine-learning conferences adopted reproducibility checklists covering code, data, hyperparameters, compute, numbers of runs, and uncertainty. An analysis of the NeurIPS 2019 reproducibility programme found that structured reporting and code submission improved what reviewers could assess, while also documenting how difficult reproduction from a paper alone can be ([Pineau et al. 2021](/references/#pineau-2021)) **[E]**.

The modern shift is from “attach some code” toward a **claim–artifact graph**:

$$
\text{claim}
\longleftarrow \text{table or figure}
\longleftarrow \text{analysis output}
\longleftarrow \text{raw run}
\longleftarrow \text{code + config + data + environment}.
$$

Each arrow should be traceable. A reader should be able to determine which runs entered a table, which exclusions were applied, which version produced them, and whether the primary analysis matches the registration. Containers and lock files help recover an environment; run manifests, data cards, logs, and immutable identifiers explain what happened inside it **[O]**.

This infrastructure improves auditability, not truth. A perfectly packaged experiment can use the wrong target, a contaminated benchmark, or a broken metric. The scientific review begins where artifact validation ends.

## 12.8.2 Comparing procedures rather than lucky outcomes

**Quantity:** expected performance under a stated search and tuning budget.

Reporting only the best hyperparameter run rewards methods that received more attempts. If method A was tried 100 times and method B five times, their maxima compare both algorithms and search budgets. Dodge et al. proposed reporting the expected best result as a function of tuning budget ([Dodge et al. 2019](/references/#dodge-2019)) **[D]/[O]**. The estimand becomes a procedure:

$$
g_m(b)=\mathbb E[\text{best validation outcome after budget }b
\text{ under method }m].
$$

Fair comparison may ask whether $g_A(b)>g_B(b)$ at equal budget, or compare total utility after training and inference costs. It should not compare A's median ordinary run with B's cherry-picked maximum.

The same principle applies to prompts, retrieval strategies, model checkpoints, and human interventions. Search effort is part of the method. Preserve failed candidates and report the selection rule.

## 12.8.3 Statistics for few expensive runs

**Quantity:** uncertainty over the experimental units actually sampled.

Deep reinforcement learning made the problem conspicuous: a few expensive, noisy runs produced unstable rankings. Interval estimates, robust aggregates, and performance profiles changed or weakened conclusions suggested by point estimates ([Agarwal et al. 2021](/references/#agarwal-2021)) **[E]**.

Three developments matter beyond reinforcement learning:

1. **Stratified aggregates.** Performance profiles show how often a method exceeds many thresholds across tasks, instead of hiding heterogeneity in one mean.
2. **Hierarchical uncertainty.** Runs, tasks, data items, and human raters form different sampling levels. Resampling only items cannot estimate run-to-run uncertainty.
3. **Practical thresholds.** Confidence intervals should be read against a minimum meaningful effect, not only zero. Equivalence and non-inferiority questions need their own margins.

Robust statistics are not magic for $n=3$. A bootstrap distribution made from three runs still contains little information about tails. The American Statistical Association's statement remains a useful warning: a $p$-value is not an effect size, the probability that the hypothesis is true, or a substitute for design and context ([Wasserstein & Lazar 2016](/references/#wasserstein-2016)) **[O]**.

## 12.8.4 Re-evaluation as a normal scientific product

**Quantity:** how much reported improvement survives equal tuning, corrected evaluation, and stronger baselines.

Careful re-evaluations have found that strong baselines tuned as hard as a proposed method close much of the claimed gap in language modelling ([Melis et al. 2018](/references/#melis-2018)), generative adversarial networks ([Lucic et al. 2018](/references/#lucic-2018)), and reinforcement learning ([Henderson et al. 2018](/references/#henderson-2018)) **[E]**. Leakage and evaluation errors form another recurring source of spurious findings in machine-learning-based science ([Kapoor & Narayanan 2023](/references/#kapoor-2023)) **[E]**.

This should not be treated as embarrassing cleanup after “real” research. Re-evaluation answers substantive questions:

- Was the gain caused by the proposed mechanism or by tuning effort?
- Does it survive a corrected data split?
- Is the baseline implemented comparably?
- Does the ranking persist across tasks and compute budgets?
- Is the effect large enough to matter operationally?

Negative and corrective results improve the map of a field. A publication system that rewards only new winners systematically hides how unreliable the leaderboard is.

## 12.8.5 Data lineage and benchmark contamination

**Quantity:** whether evaluation information could have entered training or model selection.

For foundation models, “train” and “test” are no longer always two tidy local files. Pretraining corpora can contain benchmark items, paraphrases, solutions, or later web pages that quote them. Post-training and tool use add further channels. Exact string matching catches only a subset.

A modern contamination audit should record:

- dataset versions and collection dates;
- exact and approximate overlap checks;
- whether solutions, explanations, or translated variants were searched;
- which evaluation sets influenced prompt, checkpoint, or tool selection;
- what can and cannot be checked because training data are unavailable;
- a negative control or newly collected holdout when feasible.

The conclusion must match the audit. “No exact match was found in the accessible corpus” is not “the model has never seen this task.” Secret test sets reduce direct optimization but also limit independent auditing; dynamic or newly collected evaluation reduces some contamination while introducing sampling and comparability questions **[O]**.

## 12.8.6 AI systems now traverse more of the research loop

**Quantity:** which research actions an AI system completed, with what human selection and what external verification.

AI-assisted discovery spans very different evidence regimes. Search guided by an exact evaluator has produced independently checkable constructions in combinatorics ([Romera-Paredes et al. 2024](/references/#romera-paredes-2024)) and improvements to algorithms and engineering systems ([Novikov et al. 2025](/references/#novikov-2025)) **[E]**. In such settings, generation can be broad because candidates face a strong verifier—though the specification and verifier still need audit.

End-to-end “AI scientist” systems tackle a weaker-verifier setting. The 2024 system generated ideas, modified template code, ran experiments, and drafted manuscripts ([Lu et al. 2024](/references/#lu-2024)). Its 2025 successor removed the fixed-template requirement and used agentic tree search; its authors submitted three selected autonomous manuscripts to an ICLR workshop, where one was accepted ([Yamada et al. 2025](/references/#yamada-2025)) **[E]**.

That result is a real capability milestone, but it supports a bounded statement: under that workflow, human idea/manuscript selection, workshop, and sample of three submissions, at least one manuscript passed that review process. It does **not** estimate the probability that an autonomous run produces a correct discovery, establish conference-level quality, or independently validate the scientific claims. The same paper's audit reports dataset-overlap concerns, inaccurate descriptions, and implemented components that were not actually used. Peer-review acceptance is evidence about review, not a proof checker.

Likewise, a blinded study in which more than one hundred NLP researchers rated generated ideas found higher novelty ratings but slightly lower feasibility than expert ideas ([Si et al. 2024](/references/#si-2024)) **[E]**. This shows that ideation can be competitive under that rubric. It does not show that the ideas survive implementation or replication.

## 12.8.7 Verifier-first research workflows

**Quantity:** the fraction of delegated outputs that receive a test capable of catching their consequential failure modes.

As of 2026, a central design proposal is to organize AI-assisted research around reusable skills, comparable benchmarks, and layered verification rather than around fluent end products ([Reizinger & Brendel 2026](/references/#reizinger-2026)) **[O]/[H]**. The idea aligns with this book's cumulative spine: state a verification target before delegating construction.

Verification strength forms a ladder:

| Output | Weak check | Stronger check |
|---|---|---|
| citation | title looks plausible | resolve identifier; compare claim with source |
| derivation | another model agrees | symbolic check, proof assistant, or independent derivation |
| code | script exits successfully | targeted tests, hidden cases, and output audit |
| experiment | agent reports completion | inspect logs and artifacts; rerun; verify estimand and analysis |
| empirical claim | automated reviewer likes paper | blinded human review plus independent replication |
| mechanism claim | diagnostic correlates with behaviour | registered intervention with rival predictions |

Self-critique can improve a candidate, but proposer and critic may share blind spots. “The agent checked its own work” should be treated as another model-generated artifact, not independent confirmation.

## 12.8.8 What should be measured next?

Autonomous-research evaluations often score the final paper or benchmark outcome. A more informative evaluation would decompose the loop:

1. **proposal quality:** novelty, relevance, and falsifiability before execution;
2. **implementation fidelity:** whether code realizes the stated intervention;
3. **measurement validity:** whether the metric represents the claim;
4. **analysis integrity:** whether all runs and deviations are accounted for;
5. **calibration:** whether confidence predicts later verification;
6. **repair quality:** whether the system responds correctly to failed checks;
7. **replication yield:** what fraction of credited findings survive independent tests;
8. **cost and risk:** compute, human review time, and severity of escaped errors.

The open question is no longer whether AI can propose ideas, write code, execute some experiments, and produce polished reports—it can **[E]**. The hard question is how much verified knowledge the complete human–AI system produces per unit of cost and risk **[H]**. Counting papers rewards generation; counting findings that survive suitably independent checks rewards science.
