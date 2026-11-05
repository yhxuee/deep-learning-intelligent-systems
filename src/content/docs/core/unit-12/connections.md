---
title: "12.6 Connections"
description: How reproducible machine learning connects to clinical trials, psychology, particle physics, software engineering, and the book's opening problem of generalization.
sidebar: { label: "12.6 Connections", order: 6 }
---

Reproducibility is not a collection of bureaucratic rituals invented for machine learning. Other fields met the same underlying problem first: a flexible search can produce a persuasive result even when the proposed explanation is wrong. Their solutions differ because their experimental units, costs, and failure modes differ. The useful lesson is therefore not “copy another field's threshold,” but “identify the error that its procedure was built to control.”

## 12.6.1 Clinical trials: define the question before seeing the answer

Imagine testing a treatment while measuring pain, sleep, blood pressure, mood, and twenty laboratory values. If the primary outcome is chosen only after all results are visible, the final $p$-value no longer describes the search that actually occurred. Clinical-trial registration addresses this problem by recording the intervention, primary outcome, analysis, and planned sample size before enrolment. Registered protocols do not make a study correct, but they make later deviations visible.

Several ideas in this unit have close clinical analogues:

| Clinical trial | Machine-learning experiment |
|---|---|
| participant or randomized cluster | independent training run, dataset, task, or other experimental unit named by the estimand |
| treatment assignment | model or training intervention |
| primary outcome | pre-specified primary metric |
| secondary outcome | diagnostic or exploratory metric |
| allocation concealment and blinding | hiding condition labels from human raters; freezing analysis before revealing the test result |
| intention-to-treat analysis | analysing the assigned comparison without silently excluding inconvenient failed runs |

The analogy has limits. A **seed is not a patient**. An integer seed is usually a label that initializes one or more random-number streams; the resulting independent training run may be an experimental unit. Two conditions that use the same integer can still receive different random draws if their programs consume randomness in different orders. Conversely, one trained checkpoint evaluated on 10,000 examples is not automatically 10,000 independent training replicates. The experimental unit follows the intervention and estimand, not the row count in a file.

Clinical trials also remind us why exclusions matter. “Three runs crashed, so we removed them” may be harmless only if failure is unrelated to condition and the rule was fixed in advance. If a method crashes precisely when optimization becomes unstable, excluding those runs removes part of the treatment effect. Report failure rates and analyse both performance conditional on success and the probability of success when both matter.

## 12.6.2 Psychology: flexibility can masquerade as discovery

The Open Science Collaboration attempted replications of 100 published psychology studies. Thirty-six percent of the replications produced a statistically significant result in the original direction, while estimated effects were on average roughly half the original magnitude ([Open Science Collaboration 2015](/references/#osc-2015)). Neither number means that “64% of psychology is false”: the projects differed in power, setting, protocol fidelity, and what counted as successful replication. The result instead exposed a system-level vulnerability.

The main suspects were familiar:

1. studies were often too small to estimate effects precisely;
2. researchers could choose among outcomes, exclusions, transformations, and stopping rules;
3. positive, surprising findings were more likely to be published;
4. later readers saw the selected result but not the size of the search.

The response included pre-registration, registered reports, larger multi-site studies, open materials, and explicit replication. Machine learning has parallel pressures: benchmark chasing, hyperparameter search on the test set, selective seed reporting, and under-specified code or data pipelines ([Lipton & Steinhardt 2019](/references/#lipton-2019); [Gundersen & Kjensmo 2018](/references/#gundersen-2018)).

A registered report is especially instructive. Reviewers assess the question and method before the result is known; publication is then based primarily on executing the accepted protocol, not on obtaining a dramatic number. The machine-learning analogue would accept a benchmark study because its comparison, power, leakage controls, and analysis are informative—even if the new method does not win.

## 12.6.3 Particle physics: hide the answer and pay for the search

Particle physicists often develop an analysis while the signal region is hidden. They tune event selection on simulations, control regions, or “sidebands,” freeze the pipeline, and only then reveal the region in which a discovery might appear. This **blind analysis** prevents analysts from repeatedly adjusting cuts toward the hoped-for answer.

The famous “five-sigma” discovery convention is not a universal gold standard that machine learning should imitate mechanically. Particle-physics searches can scan many masses, decay channels, and analysis choices. A local fluctuation that looks rare at one location is less rare somewhere in a large search—the **look-elsewhere effect**. Stringent thresholds, global significance calculations, and independent confirmation respond to that particular multiplicity and to the high cost of a false discovery claim.

The transferable lesson is:

$$
\text{evidence threshold and correction}
\quad\text{must reflect}\quad
\text{the search actually performed}.
$$

If an ML team tries 40 datasets, 12 metrics, and 50 prompts, “the best condition has $p<0.05$” cannot be interpreted as one pre-specified test. Either control the family of tests, reserve confirmatory data, or label the winner exploratory and test it in a fresh experiment.

## 12.6.4 Software engineering: make the verifier precede the candidate

Test-driven development writes an executable expectation before the implementation. That ordering reduces the temptation to redefine success around whatever the code currently does. The analogy to pre-registration is direct, but again incomplete: a passing test establishes only the cases encoded in the suite. It cannot prove that the specification was scientifically appropriate.

Reproducible experimentation therefore needs several layers of software evidence:

- unit tests check local invariants such as tensor shapes and gradient formulas;
- integration tests check that the real data and training pipeline fit together;
- regression tests detect unintended changes to established behaviour;
- experiment assertions check that registered conditions, seeds, and metrics were actually executed;
- provenance records connect a result to code, configuration, data, and environment.

A digest can detect accidental changes only when a trusted copy of the expected digest exists outside the editable artifact. If an attacker or careless script can rewrite both a result and its digest in the same file, the digest proves nothing about history. This is the software version of the broader principle: a verifier must not share the exact failure mode it is supposed to catch.

## 12.6.5 Quality control is not the same as scientific replication

The four fields above also clarify a common category error. Re-running the same script can answer “is this computational artifact repeatable?” It does not by itself answer “does the finding generalize?”

| Question | Useful repetition |
|---|---|
| Did the same code produce the same artifact? | computational repeatability in a controlled environment |
| Does another implementation recover the result? | independent software reproduction |
| Does the effect survive new random training outcomes? | additional independent runs |
| Does it survive new data, tasks, scales, or laboratories? | scientific replication and external validation |
| Does the proposed mechanism predict an intervention? | causal or mechanistic test |

These repetitions are complementary. Containerizing a bug makes the bug repeatable. A clean-room implementation can reveal the bug but still use the same contaminated dataset. A new dataset can remove contamination while leaving the same weak metric. Good replication chooses independence along the suspected error dimension.

## 12.6.6 The loop closes: experiments also generalize from samples

Unit 01 asked why experience on observed data should improve performance on unseen cases. Its answer required both an assumption—future cases resemble the sampled population in relevant ways—and a discipline: held-out data, uncertainty, and no selection on the test set.

Unit 12 asks the same question of the researcher. A conclusion from five training runs, one dataset, one architecture, and one codebase is a generalization from observed experimental units to unobserved ones. It may target new runs under exactly the same pipeline, or it may aspire to new tasks and model families. Those are different estimands and require different evidence.

The parallel is exact enough to be useful:

| Learning system | Scientific study |
|---|---|
| training examples | exploratory experiments and pilot runs |
| validation set | method and hyperparameter selection |
| untouched test set | confirmatory or replication data |
| distribution shift | new datasets, scales, implementations, or laboratories |
| overfitting | tailoring the story or analysis to observed outcomes |
| generalization bound or interval | explicitly bounded uncertainty about the target population |

The final unit therefore returns to the first, but at a higher level. A model can overfit data; a research process can overfit experiments. In both cases, the remedy is not a magical metric. It is a visible target population, protected evidence, uncertainty that matches the sampling process, and new tests designed to expose failure.

## 12.6.7 A connection map for the whole book

| This unit's discipline | Earlier construction it audits |
|---|---|
| define the experimental unit and estimand | Unit 01's population risk and held-out evaluation |
| propagate uncertainty over the right source | Units 01 and 10's intervals and calibration |
| register a scaling prediction | Unit 06's extrapolation and regime-change test |
| isolate one causal factor or use a factorial design | Unit 07's confounded architectural comparisons |
| separate correlation from intervention | Unit 11's mechanism claims |
| make the verifier independent of the proposer | Unit 09's lesson that an optimizer exploits gaps in an objective |
| preserve artifacts and failure reports | every stage of the cumulative spine |

The book's central proposition can now be stated precisely: delegate construction only as far as the verification process can detect consequential errors. Reproducibility is what turns that proposition from advice into an inspectable research practice.
