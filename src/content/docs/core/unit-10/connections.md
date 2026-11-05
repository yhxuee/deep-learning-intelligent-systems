---
title: "10.6 Connections"
description: How evaluation connects to clinical validation, psychometrics, metrology, software testing, survey sampling, causal reasoning, and later units.
sidebar: { label: "10.6 Connections", order: 6 }
---

Evaluation is not a problem invented by machine learning. Doctors ask whether evidence transfers to another hospital; psychometricians ask whether a test measures the intended construct; metrologists attach uncertainty to every measurement; software engineers try to make programs fail before users do; survey statisticians connect a sample to a target population.

The vocabulary differs, but the structure is the same:

$$
\text{target question}
\longrightarrow
\text{sample and measurement}
\longrightarrow
\text{uncertainty}
\longrightarrow
\text{bounded conclusion}.
$$

Seeing these connections helps us recognize when a machine-learning result has skipped one of the arrows.

## 10.6.1 Clinical medicine — transport is a new claim

Suppose a model predicts pneumonia from chest radiographs.

- **Internal validation** asks whether it works on patients held out from the development setting.
- **External validation** evaluates it in another hospital, time period, scanner population, or patient mix.
- **Prospective or impact evaluation** asks what happens when clinicians actually use it in a workflow.

These stages are not interchangeable. A random split can estimate performance on more patients like those already collected. It cannot, by itself, show that the model will transport to another hospital or improve patient outcomes.

This distinction matters because hospitals differ in more than disease. Scanner manufacturers, image-processing pipelines, referral patterns, annotation practices, and disease prevalence may all change. A model can exploit one of these correlations while appearing medically competent.

That is not hypothetical. A pneumonia detector trained on radiographs from two hospital systems performed worse when transported, and could also identify the originating hospital—evidence that site-specific information was available as a shortcut ([Zech et al. 2018](/references/#zech-2018)). Reviews of COVID-19 radiograph models found reliance on image artefacts and acquisition differences unrelated to the disease process ([DeGrave et al. 2021](/references/#degrave-2021)).

The unseen-name evaluation in this book is external validation in miniature:

| Clinical question | Spine analogue |
|---|---|
| new patients, same hospital process | fresh stories, same generator |
| new hospital or scanner process | unseen-name intervention |
| site marker correlated with diagnosis | familiar name fragment correlated with next character |

The analogy has a boundary. A synthetic name shift is controlled and simple; real clinical shifts interact with treatment, prevalence and selection. Passing the toy test teaches the method of evaluation, not clinical readiness.

**Transferable lesson:** every external validation supports transport only to the setting actually sampled. “Worked at Hospital B” does not automatically mean “works everywhere.”

## 10.6.2 Psychometrics — a benchmark is a measuring instrument

A school examination does not become a good measure of mathematical reasoning merely because it yields reproducible scores. Psychometrics separates several questions.

### Reliability

Would the measurement be reasonably stable under repeated administration, equivalent items, or different raters? Reliability concerns consistency. A noisy benchmark with only ten items may rank models differently each time.

### Validity

Does the interpretation of the score support its intended use? Useful distinctions include:

- **content validity:** do the items cover the claimed domain?
- **criterion validity:** does the score predict an outcome we care about?
- **construct validity:** does the test measure the intended capability rather than an easier correlate?

Consider a reading-comprehension benchmark whose questions can be answered from superficial word overlap without reading the passage. Its scoring code may be perfectly reliable. Its construct validity as a measure of passage comprehension is still weak.

Reliability is usually necessary for validity—an unstable measure cannot support precise interpretation—but it is not sufficient. In machine learning:

| Psychometric failure | Benchmark version |
|---|---|
| too few or noisy items | wide confidence interval |
| narrow content coverage | leaderboard overfits one task style |
| construct-irrelevant cue | shortcut or annotation artefact |
| test familiarity | contamination or repeated adaptation |

Validity also belongs to a **use**, not to a data set in the abstract. The same benchmark might be useful for checking a narrow capability and invalid as a claim of general intelligence.

**Transferable lesson:** before improving a score, write the sentence that the score is supposed to justify. Then ask what alternative capability or shortcut could produce the same number.

## 10.6.3 Metrology — uncertainty belongs to the result

Measurement science does not report only “12.31 mm.” It reports a measured value together with an uncertainty statement and a traceable procedure. An uncertainty budget lists contributing sources such as instrument calibration, operator, environment and repeated measurement.

Machine-learning evaluation has an analogous budget:

| Metrology source | Machine-learning counterpart |
|---|---|
| repeated readings | sampled test items |
| instrument calibration | correctness of metric implementation |
| operator or procedure | prompt template and decoding settings |
| instrument specimen | trained checkpoint and random seed |
| environmental condition | deployment domain and time |

An interval obtained by bootstrapping test items covers only the first row. It does not magically include the other rows. Training-seed variability can be estimated only by repeating training; prompt sensitivity requires varying prompts; domain uncertainty requires sampling relevant domains.

This is why “mean ± standard deviation” is incomplete unless the report says **standard deviation across what?** Ten batches from one run, ten test resamples, and ten independent training runs answer different questions ([Miller 2024](/references/#miller-2024); [Bouthillier et al. 2021](/references/#bouthillier-2021)).

Metrology also contributes the idea of **traceability**: another investigator should be able to connect the published number to the data version, checkpoint, code, parameters and units that produced it. A number without provenance is difficult to audit even if it has many decimal places.

**Transferable lesson:** uncertainty is not an apology attached after a result. It is part of the result's definition.

## 10.6.4 Software testing — search for failures, not reassurance

Software engineers use several complementary forms of testing.

- A **unit test** checks a known input and expected output.
- A **property-based test** generates many inputs and checks an invariant, such as “probabilities sum to one.”
- **Fuzzing** searches automatically for inputs that crash a program or violate an assertion.
- A **regression test** preserves a case that once exposed a bug so the bug cannot silently return.

Their evaluation analogues are direct:

| Software testing | Model evaluation |
|---|---|
| fixed unit test | frozen benchmark item |
| property test | invariance or metamorphic test |
| fuzzing | adversarial or stress-test generation |
| regression test | permanent test for a discovered failure mode |

For example, if changing a person's name should not change an arithmetic answer, then name substitution is a **metamorphic relation**: the exact output may not be known in advance for every generated prompt, but the relation between original and transformed outputs is known. The unseen-name test uses the same idea at the language-model level.

Adversarial examples show why random test data are not enough. Tiny, carefully chosen input changes can cause large prediction changes ([Szegedy et al. 2014](/references/#szegedy-2014); [Goodfellow et al. 2015](/references/#goodfellow-2015)). A stress test is intentionally unlike an unbiased estimate of average traffic: its purpose is to reveal a boundary.

This leads to two complementary test sets:

1. a representative set estimates ordinary risk;
2. a challenge set searches for particular failures.

Do not average them together without thought. A challenge set's prevalence is designed by the evaluator and may not match deployment prevalence; its value is diagnostic.

**Transferable lesson:** a test suite that never fails may describe a flawless system, or it may simply avoid the failure surface. Good evaluation contains tests that the present model is expected to fail.

## 10.6.5 Survey statistics — the target population comes first

A poll does not ask “what fraction of these respondents agree?” and stop. It asks for a population quantity, then reasons backward to a sampling design that can estimate it.

This makes the **estimand** explicit. For an election poll it might be the voting intention of eligible voters on a particular date. For a model it might be expected loss over customer requests in a region during the next month.

Survey statisticians then examine:

- inclusion probabilities: who had a chance to enter the sample?
- nonresponse: who was invited but absent?
- weighting: which observed groups are over- or under-represented?
- design effects: which observations are clustered rather than independent?

Post-stratification divides the population into known cells and reweights sample cells to known population proportions. Importance weighting in §10.3.1 is a more general mathematical relative: both replace an average under the sample distribution with a weighted average targeting another distribution.

But weighting cannot invent absent support. If no sampled case resembles an important deployment group, its risk cannot be learned merely by assigning a huge weight to nothing. Large weights also reduce effective sample size and make estimates unstable.

Most benchmark data sets are closer to convenience samples than probability samples. Their examples were available, annotatable and publishable—not drawn with known inclusion probabilities from all future user requests. This does not make them useless; it limits the population claim.

**Transferable lesson:** “large data set” and “representative sample” are different properties.

## 10.6.6 Causal reasoning — finding a failure is not explaining it

Evaluation often begins with an association:

$$
\text{model fails more often when feature }Z\text{ is present}.
$$

That observation can suggest a mechanism, but it does not establish one. Cases with $Z$ may differ in many other ways. A causal intervention changes $Z$ while holding relevant factors fixed as far as possible.

The three name evaluations illustrate the ladder:

1. held-out performance shows behavior on the observed distribution;
2. unseen names show that performance changes under a controlled input intervention;
3. internal activation patching or feature intervention in Unit 11 asks which computation mediates the change.

Even the second step does not prove “the model stores a list of names.” It establishes sensitivity and transfer failure. The internal explanation requires additional evidence.

This distinction prevents a common error: turning a behavioral result into a mechanistic story because the story sounds plausible. Evaluation tells us **what changes under a test**; causal mechanism claims require interventions aligned with the claimed cause.

**Transferable lesson:** label explanations as hypotheses until the evaluation manipulates the alleged cause and rules out important alternatives.

## 10.6.7 Where these ideas return in the book

| Idea from this unit | Earlier foundation | Later use |
|---|---|---|
| proper losses and units | Unit 01: log-loss, nats and sampling | Unit 12: comparing complete systems |
| representation of relevant features | Unit 02 | Unit 11: locating which feature is used |
| paired comparisons and resampling | Unit 01: standard errors | Unit 12: ablations across items and seeds |
| data provenance and contamination | Unit 07: deduplication and overlap | Unit 12: auditable benchmark claims |
| score as an optimization target | Unit 09: reward and preference learning | Goodhart effects and reward hacking |
| behavioral failure versus mechanism | controlled shifts in this unit | Unit 11: interventions on internal computation |

The central connection is that evaluation is not the last step after “the real modeling work.” It determines what the modeling work can legitimately claim. A better architecture with a badly posed evaluation yields a more impressive number, not necessarily better knowledge.
