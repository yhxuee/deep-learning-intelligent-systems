---
title: "7.7 Limitations and counterexamples"
description: Separate prediction from truth, memorization from generalization, overlap from contamination, filtering from quality, and a learned distribution from the behaviour produced by decoding it.
sidebar: { label: "7.7 Limitations", order: 7 }
---

Labels: **[D]** derivation, **[E]** experimental evidence, **[O]** engineering observation, **[H]** open hypothesis.

The purpose of a counterexample is not to say that self-supervision “does not work.” It shows which conclusion does **not** follow from a successful training run. That boundary is what lets us design the next measurement.

## 7.7.1 Matching text statistics is not the same as learning truth

At the population optimum, next-token prediction reproduces the corpus conditional $p_{\text{data}}(x_t\mid x_{<t})$ (§7.5.1) **[D]**. Truth appears in that equation only if the data-generating process makes truth predictively visible.

The spine generator supplies a controlled example. It corrupts the stated sum in 30% of arithmetic stories. In the simplified case where the same context has one correct continuation and one corrupted continuation, the data conditional assigns mass $0.7$ and $0.3$ respectively. Maximum likelihood is doing its job when it approaches those frequencies. It has not been given a reason to move all mass to the mathematically correct answer.

Do not overread this calculation. The finite spine model sees many names, numbers and contexts, so the statement “it assigns exactly 30% to this wrong numeral” does not follow. Unit 09 measures its actual behaviour. The calculation isolates the mechanism: a predictive objective cannot, by itself, label a frequent pattern as misconception, stale fact, satire or lie.

TruthfulQA supplies one empirical warning: within the model families and questions studied, scaling pretrained language models did not reliably improve truthfulness and sometimes increased imitation of common misconceptions ([Lin et al. 2022](/references/#lin-2022)) **[E]**. This is not a theorem that larger models are always less truthful. It is evidence that lower pretraining loss is not a truth certificate.

**Counterexample.** Imagine a corpus in which every incorrect medical claim is copied ten times and every correction once. A perfect corpus model should predict the incorrect claim more often. Better maximum likelihood has moved the model farther from the behaviour a clinician wants, because the training distribution and desired decision rule differ.

## 7.7.2 Memorization and generalization can coexist

“Memorization” needs an operational definition. It may mean that a model:

- assigns unusually high likelihood to a training sequence;
- emits a long training passage verbatim after a prefix;
- reveals whether a record was in training;
- or relies on an example-specific shortcut rather than a reusable rule.

These tests are related, but not equivalent. A model can generalize grammar and still reproduce a rare phone number; good held-out NLL does not rule out local memorization.

[Carlini et al. (2021)](/references/#carlini-2021) sampled from GPT-2 and recovered verbatim training sequences, including identifying information **[E]**. A later controlled study found three approximate log-linear relationships within its experimental settings: extractable memorization increased with model capacity, example duplication and available prompt context ([Carlini et al. 2023](/references/#carlini-2023)) **[E]**. These are measured trends, not a universal formula across every architecture and data pipeline.

Deduplication can reduce verbatim emission and train–test overlap ([Lee et al. 2022](/references/#lee-2022)) **[E]**, but “deduplicated” does not mean “cannot memorize.” Unique sensitive strings can still receive high influence; near copies may evade the detector; and a large model may fit single occurrences.

The spine run samples about $8.192$ million training characters from a stream of about $363{,}418$ characters—an **equivalent token-draw volume** of roughly $22.5$ stream lengths. This is not 22.5 deterministic full passes: windows are sampled with replacement and can cross story boundaries. Experiment 7.4 therefore measures exact and prefix-conditioned regeneration rather than declaring memorization from the exposure count alone.

## 7.7.3 Low overlap is not a certificate of no contamination

Every contamination detector has a target class. Word 8-gram overlap is sensitive to near-verbatim reuse; it is not designed to find every way information can leak.

| Case | What an 8-gram detector may do | Why |
|---|---|---|
| exact copied answer | flag it | many long phrases survive |
| faithful paraphrase | miss it | wording changes while information remains |
| translation | miss it | surface tokens change almost completely |
| answer key reformatted as a table | miss much of it | local order and boundaries change |
| common legal boilerplate | flag it | long shared phrases need not imply benchmark leakage |
| short item | become unstable | only a few or zero distinct 8-grams exist |

There is also a mosaic problem: the union of training $n$-grams may cover a held-out item even when no single training document contains the item. A high directional coverage score is therefore a reason to inspect provenance, not automatic proof of a copied benchmark.

Model-based tests can probe signals text matching misses. Oren et al. test whether a model assigns unusually high likelihood to the canonical ordering of an exchangeable benchmark compared with shuffled orders ([Oren et al. 2024](/references/#oren-2024)) **[E]**. That test has a specific null hypothesis and is not a universal paraphrase detector. Completion and likelihood tests also depend on model access, calibration and a valid reference distribution.

The correct conclusion from low $n$-gram overlap is narrow: **we found little evidence of near-verbatim overlap under this representation and threshold.** Stronger assurance combines provenance, multiple detectors, frozen split rules and, when possible, evaluation data created after the training snapshot. Even post-snapshot data need access controls, because later web publication can leak them into future models.

## 7.7.4 The pretraining objective is a proxy for the use

A proxy can correlate with the desired behaviour without defining it.

- Next-token loss rewards accurate corpus continuation; a user may want a verified answer or a safe action.
- Masked reconstruction rewards recovery from surrounding context; a user may want a calibrated class probability under distribution shift.
- Contrastive loss rewards separating the chosen positive from chosen negatives; a user may care about a distinction absent from both.

**Shortcut counterexample.** Suppose all toxic comments in the pretraining sample came from one website and carry the same header. A representation can support excellent training classification by encoding the header. When the header disappears, the intended semantic distinction was never learned.

This is why a low pretraining loss establishes only success on the pretraining proxy. Decoding (Unit 08), behavioural feedback (Unit 09) and task-specific evaluation (Unit 10) introduce additional choices. They can improve the intended use, but they can also create new shortcuts or erase useful behaviour. “Pretrained” means a starting state with measured properties, not a finished specification.

## 7.7.5 Data quality is a vector, not one score

A document can be fluent but false, accurate but outdated, useful but unlicensed, diverse but noisy, or harmless in isolation but privacy-sensitive in combination. Calling all of these “quality” hides different objectives.

| Dimension | A possible audit question | A possible trade-off |
|---|---|---|
| factuality | can claims be traced to reliable, dated sources? | reference sources may omit emerging or minority knowledge |
| diversity | which languages, dialects, domains and viewpoints remain? | more breadth can increase noise under fixed compute |
| privacy | are personal or secret strings present? | aggressive filters can miss context or remove legitimate public records |
| rights and consent | may this content be used for this purpose? | legal permission and social acceptability are not identical |
| freshness | does the snapshot match deployment time? | recent data are less thoroughly reviewed |
| pedagogical value | does text explain reasoning rather than only state answers? | textbook-like filtering may narrow style and topic coverage |

Every filter implements an inductive bias **[O]**. A classifier trained to prefer prose resembling a reference encyclopedia may systematically downweight conversation, dialect or specialist formats. Reported downstream gains then apply to the evaluations used; they do not prove a universally better corpus.

“More data” is likewise conditional. Under fixed compute, adding low-value documents can displace useful updates. Adding duplicates changes weights. Adding generated text can amplify the generator's omissions. The right comparison freezes compute, model, split and evaluation, then varies a documented data decision. Corpus size alone is not the treatment.

## 7.7.6 A learned distribution is not yet a behaviour

After Stage 07, the spine model supplies conditional probabilities. To produce one continuation, a decoding rule must turn those probabilities into choices.

- **Greedy decoding** always chooses the largest next-token probability. It is deterministic, but a locally likely choice can enter a repetitive loop.
- **Ancestral sampling** draws from the full conditional. It represents the model distribution more faithfully, but it also samples its low-probability mistakes.
- **Truncation and temperature** deliberately alter the distribution, trading diversity against concentration.

Thus two systems with identical weights can exhibit different text behaviour. Conversely, a fluent sample does not prove that the underlying probabilities are calibrated or that another decoding seed will be safe. Unit 08 measures these distinctions on the same checkpoint.

One correction is important for continuous data. The probability chain rule is not restricted to tokens: a continuous vector can also be factored into conditional densities, and images can be generated autoregressively by ordering pixels or discrete codes. The difficulty is that the ordering may be artificial and sequential sampling slow, not that factorization is mathematically forbidden. Diffusion, flow and other continuous-time constructions in [Unit 08](/core/unit-08/) offer different training and sampling paths.

The next question is therefore not merely “did the loss go down?” It is: **which procedure turns the learned statistical object into the behaviour we observe, and what distribution does that procedure actually produce?**

## 7.7.7 A compact diagnostic table

| Observation | Does not by itself prove | Next check |
|---|---|---|
| lower validation NLL | truthfulness or rare-skill competence | factual and skill-stratified tests |
| no detected 8-gram overlap | absence of paraphrased or translated leakage | provenance plus semantic/model-based audits |
| lower deduplicated-model extraction rate | privacy for unique records | canary, membership and targeted extraction tests |
| better pretraining proxy | better deployment decisions | frozen downstream and shift evaluations |
| larger filtered corpus | better effective data | fixed-compute data ablation |
| one fluent sample | reliable sampling distribution | multi-seed rates with confidence intervals |

These are not reasons to abandon self-supervision. They are the measurements required to use it scientifically.
