---
title: "7.8 Modern developments"
description: Modern self-supervision increasingly treats data selection, mixture weights, synthetic generation, target representation and contamination controls as testable parts of the learning algorithm.
sidebar: { label: "7.8 Modern developments", order: 8 }
---

Labels: **[D]** derivation, **[E]** experimental evidence, **[O]** engineering observation, **[H]** open hypothesis.

The central modern shift is not one new loss. It is the recognition that **the data pipeline is part of the learning algorithm**. Which examples enter, how often they enter, which parts are hidden, what target space is predicted, and which evaluations are protected all change the statistical problem.

## 7.8.1 Data curation became an experimental variable

A traditional model ablation fixes the dataset and changes the architecture. A data ablation should do the reverse:

$$
\text{fixed model + fixed tokens + fixed optimizer + fixed evaluation; vary one data policy.}
$$

This control is essential. If a filtered model sees fewer tokens or trains longer, its score difference cannot be attributed to filtering alone.

FineWeb documents a pipeline over many Common Crawl snapshots and releases intermediate data, code and models for ablations of extraction, language identification, heuristic filters, model-based educational filters and deduplication ([Penedo et al. 2024](/references/#penedo-2024)) **[E]**. One important result is methodological: a plausible rule such as more aggressive cross-snapshot deduplication does not automatically improve every fixed-compute evaluation. The answer depends on what repetition was removed and which tasks define “better.”

DataComp-LM (DCLM) pushes the controlled-comparison idea further. It fixes a candidate pool, training recipes at several model scales and a broad evaluation suite, then lets dataset-selection methods vary ([Li et al. 2024](/references/#dclm-2024)) **[E]**. This turns “our data are high quality” from a description into a falsifiable benchmark claim.

The boundary remains important. A filter selected to improve one evaluation suite can overfit that suite's domains or style. Proxy models at hundreds of millions of parameters may rank data differently from a much larger final model. Controlled data experiments improve causal attribution **within the declared recipe**; they do not establish universal document quality.

## 7.8.2 Mixture weights are optimization parameters

Let source $j$ have distribution $p_j(x)$ and sampling weight $\alpha_j$. The training distribution is

$$
p_{\alpha}(x)=\sum_{j=1}^{J}\alpha_jp_j(x),
\qquad \alpha_j\ge0,
\qquad \sum_j\alpha_j=1.
$$

Changing $\alpha$ changes the objective even when every source document stays the same. Under a fixed token budget, doubling code weight necessarily removes expected updates from some other source.

DoReMi uses a small proxy model to adapt domain weights according to **excess loss** relative to reference losses, with a group-robust objective that emphasizes poorly served domains ([Xie et al. 2023](/references/#xie-2023)) **[E]**. The learned weights are then used to train a larger model. The transferable idea is not “these weights are optimal forever”; it is that mixture design can be optimized against an explicit criterion rather than chosen only by intuition.

Three generalization gaps remain:

1. **proxy-to-target:** a small model's data preferences may not match a large model's;
2. **validation-to-deployment:** the chosen domain losses may omit a deployment need;
3. **static-to-scheduled:** one constant $\alpha$ may not be best throughout training.

A scheduled mixture $\alpha_j(t)$ is a curriculum. More specialized data late in training may improve one target while causing forgetting elsewhere. It therefore needs checkpoints and per-domain curves, not only a final average.

## 7.8.3 Synthetic data can add structure or amplify a generator

“Synthetic” describes provenance, not quality. Its effect depends on what new constraint the generation process contributes.

TinyStories asked strong generators to create simple stories under vocabulary and content constraints; models far smaller than general-purpose language models then learned coherent short-story structure from that synthetic corpus ([Eldan & Li 2023](/references/#eldan-2023)) **[E]**. The spine corpus uses an even more transparent template generator so that arithmetic corruption, names, duplication and train–test overlap are inspectable.

Curated textbook-style explanations and generated exercises have also produced strong results in a narrow coding-model setting ([Gunasekar et al. 2023](/references/#gunasekar-2023)) **[E]**. The scoped conclusion is that targeted construction can improve performance per training token on selected tasks. It does not establish that generated text contains more independent information than its sources, or that benchmark gains are free of teacher and evaluation overlap.

There are two different regimes:

| Regime | Potential benefit | Main risk |
|---|---|---|
| **augmentation:** retain real data and add targeted synthetic examples | fill a rare format, balance cases, expose solutions or counterexamples | inherit generator errors and narrow style |
| **recursive replacement:** train the next generator mostly on outputs of earlier generators | cheap repeated scaling | tails and rare modes disappear over generations |

Recursive replacement produced distributional degradation in theoretical and empirical settings studied by Shumailov et al. ([2024](/references/#shumailov-2024)) **[E]**. This is often called **model collapse**, but it should not be misquoted as “any synthetic token causes collapse.” Retaining fresh samples from the target distribution, tracking provenance, filtering with independent checks and preserving rare strata change the regime. The safe experimental unit is a frozen real-data baseline plus a specified synthetic intervention.

## 7.8.4 Prediction moved from observations to learned target spaces

Pixel reconstruction asks a model to predict every visible detail, including texture or noise that may not matter downstream. Joint-embedding predictive architectures instead mask part of an input and predict the **target encoder's representation** of the hidden region. I-JEPA demonstrated this construction for images ([Assran et al. 2023](/references/#assran-2023)) **[E]**.

The target representation is a learned information filter. That can be an advantage: unpredictable pixel variation need not dominate the loss. It is also a new dependency: if the target encoder discards a feature needed later, the predictor is never rewarded for recovering it. Representation prediction relocates the question “what should be ignored?”; it does not eliminate it.

Recent video work makes the stages especially clear. V-JEPA 2 first pretrains an action-free video encoder by masked representation prediction, then adds an **action-conditioned** predictor using robot-interaction video before model-predictive control experiments ([Assran et al. 2025](/references/#assran-2025)) **[E]**. The planning result therefore belongs to a pipeline:

$$
\text{passive video pretraining}
\rightarrow
\text{action-conditioned post-training}
\rightarrow
\text{planner and control loop}.
$$

It is not evidence that passive observation alone identifies the causal effect of unseen actions. The distinction mirrors this book: a pretrained checkpoint supplies a representation; later data and procedures create a behaviour.

## 7.8.5 Contamination control became part of benchmark design

Once public benchmark text can appear in web crawls, decontamination cannot be an afterthought performed only on the final model. A modern evaluation plan records at least:

- a training-data cutoff and source inventory;
- immutable benchmark versions and hashes;
- exact and approximate overlap reports in the training-to-test direction;
- the action taken on matched **training** records;
- private, rotating or post-cutoff items where appropriate;
- model-based tests with their null hypotheses and access assumptions.

Sainz et al. document task contamination in instruction and pretraining resources and show that overlap can affect reported performance ([2023](/references/#sainz-2023)) **[E]**. Oren et al. show that, for exchangeable benchmarks under their assumptions, likelihood differences between canonical and shuffled orders can provide statistical evidence of contamination without access to training data ([2024](/references/#oren-2024)) **[E]**.

These controls are complementary. Hashes have high precision for exact copies. $n$-grams broaden lexical recall. Semantic or model-based tests probe different leakage mechanisms. Private or newly created tests reduce exposure opportunity. None alone proves that a model acquired a capability independently of all evaluation information.

Canary strings answer yet another question: whether a known inserted sequence can later be detected or extracted. They are useful for measuring pipeline retention and memorization, but one canary's absence does not certify every natural record's privacy.

## 7.8.6 Read modern results as interventions, not rankings

The following template keeps a modern claim scientifically legible.

| Question | Example answer |
|---|---|
| What changed? | one filter, source weight, synthetic subset or target representation |
| What was held fixed? | model, optimizer, token budget, seed protocol and evaluation |
| What population was measured? | named domains, dates, languages and benchmark versions |
| What improved? | NLL, one downstream skill, extraction rate, compute or another declared metric |
| What alternative explanation remains? | proxy mismatch, contamination, extra compute, teacher overlap or selection bias |
| What would falsify transfer? | a larger-scale replication, shifted domain or protected test with the effect absent |

Modern data-centric work deepens the first principle of this unit: the transformation $T(z)\mapsto(x,y)$ includes far more than the loss formula. Curation, sampling, synthetic generation, target encoders and evaluation protection all decide what evidence the model is allowed to learn from.
