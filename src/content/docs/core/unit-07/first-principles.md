---
title: "7.2 First principles"
description: Constructing inputs and targets from one observation, deriving next-token likelihood from the chain rule, comparing what autoregressive, masked and contrastive objectives identify, and treating the sampled corpus as part of the objective.
sidebar: { label: "7.2 First principles", order: 2 }
---

## 7.2.1 One observation, several supervised problems

Supervised learning needs pairs $(x,y)$. Self-supervised learning starts with an observation $z$ and a known random or deterministic transformation $T$ that produces the pair:

$$
(x,y)=T(z).
$$

For the story “Mia found a red kite,” four transformations are possible:

| Transformation | Input $x$ | Target $y$ |
|---|---|---|
| next token | `Mia found a red` | ` kite` or its next token |
| masked token | `Mia found a [MASK] kite` | `red` |
| contrastive views | two cropped/perturbed views plus candidates | identity of the matching view |
| denoising | characters with deletions or swaps | the clean story |

Once $(x,y)$ exists, the loss and gradient machinery of Units 01–03 applies. “Self” refers to where the target came from, not to a model teaching itself without a specification. The designer still chooses $T$.

That choice has at least three parts:

1. **information available:** left context, both sides, another modality, or an augmented view;
2. **target:** a token, a missing region, the original observation, or a match label;
3. **nuisances to ignore:** colour jitter, crop, wording, speaker, or nothing at all.

The third part is easy to miss. In contrastive learning, two views declared “the same” tell the representation to become invariant to their difference. If a crop removes the object whose identity matters downstream, the augmentation teaches the wrong invariance. An automatically constructed target is still a modelling decision.

## 7.2.2 The three objective families do not ask the same question

| Objective family | Direct statistical question | Useful consequence | What is not guaranteed |
|---|---|---|---|
| **autoregressive** | what comes next under a chosen order? | normalized sequence distribution and ancestral generation | bidirectional access at encoding time |
| **masked / denoising** | what hidden content agrees with the visible context? | representations that use both sides; conditional reconstruction | one-pass left-to-right generation or a consistent joint distribution |
| **contrastive** | which candidate is paired with this view? | an embedding that preserves information useful for discrimination | reconstruction of details that do not affect the match |

These are direct properties of the training problems, not rankings. A masked model may generate by repeated conditional updates, and a contrastive encoder may be placed inside a larger generator. The narrower statement is that their losses do not directly train the same ancestral factorization as next-token prediction. Conversely, an autoregressive model can yield strong representations even though representation learning is not written as a separate term.

The objective also depends on the corruption or candidate process. Masking 15% rather than 80%, choosing easy rather than hard negatives, or treating two images as views of one object changes the conditional distribution the optimum must match.

## 7.2.3 Why next-token prediction defines a generator

Start with a finite sequence $x_{1:T}=(x_1,\ldots,x_T)$. Repeated use of $p(a,b)=p(a)p(b\mid a)$ gives the **probability chain rule**:

$$
p(x_{1:T})
=p(x_1)\prod_{t=2}^{T}p(x_t\mid x_{<t}).
$$

With a begin-of-sequence symbol $x_0=\text{BOS}$, this can be written uniformly as

$$
p(x_{1:T})=\prod_{t=1}^{T}p(x_t\mid x_{<t}).
$$

This is an identity for any joint distribution in the chosen order. It does **not** say that a finite neural network can learn the true conditionals; it says that if the model supplies normalized conditionals $p_\theta(x_t\mid x_{<t})$, their product supplies a normalized model of complete sequences.

Take the negative logarithm. Products become sums:

$$
-\ln p_\theta(x_{1:T})
=\sum_{t=1}^{T}-\ln p_\theta(x_t\mid x_{<t}).
$$

Therefore, minimizing the sum or mean of next-token cross-entropies is maximum-likelihood training for this autoregressive sequence model. Each token supplies a target, and a causal Transformer can score all positions in parallel during training because the full true prefix is known. The causal mask prevents position $t$ from seeing $x_t$ or later tokens.

Generation reverses the calculation: draw $x_1$ from $p_\theta(\cdot\mid\text{BOS})$, append it, draw $x_2$ from the next conditional, and continue until an end symbol or length limit. Training can score positions in parallel; generation is sequential because the next prefix contains samples that do not exist yet. Unit 08 studies how those draws are chosen.

:::note[“Exact” refers to the factorization]
The chain rule and the equality between summed token NLL and sequence NLL are exact. Model fit, numerical arithmetic and optimization are not. A low average token NLL also does not guarantee that every long generated sequence is useful: small conditional errors can accumulate along sampled prefixes.
:::

## 7.2.4 What prediction pressure exists in the spine corpus?

Consider the next-character task one dependency at a time.

- **Local spelling.** After `eg`, the character `g` is common. A bigram can capture part of this.
- **Template position.** After the problem sentence, phrases such as `Mia felt` become likely. Position and longer context help.
- **Entity reuse.** In `Mia went with Finn ... Mia and `, the correct continuation depends on a name introduced much earlier.
- **Novel copying.** A generated pet name such as `Kivod` has no corpus-wide meaning; predicting its second appearance requires retrieving its first appearance.
- **Arithmetic regularity.** The prefix “had 2 ... found 3 more ... has” contains a rule-like dependency.

The last case contains deliberate noise. The generator writes an incorrect result with probability 0.3. Under the simplifying condition that otherwise identical prompts follow that mechanism, the Bayes-optimal predictor assigns total probability about 0.7 to the correct result and 0.3 across incorrect results. It does not “discover truth and ignore the corpus”; likelihood rewards matching the observed conditional distribution. In the actual finite corpus, prompts differ and wrong mass is spread among several values, so no single wrong numeral necessarily receives 0.3.

This example separates three claims:

1. the objective creates pressure to model a dependency;
2. the architecture and optimization may or may not learn it;
3. matching the data distribution may still disagree with factual correctness.

Self-supervision guarantees only the first once the transformation is defined.

## 7.2.5 The sampled corpus is part of the loss

For a data distribution $q(z)$ and transformation distribution $r(T\mid z)$, the population objective is schematically

$$
\mathcal L(\theta)
=\mathbb E_{z\sim q}\,\mathbb E_{T\sim r(\cdot\mid z)}
\big[-\ln p_\theta(y_T\mid x_T)\big].
$$

Changing the corpus mixture changes $q$; changing masks, crops or negatives changes $r$. Both change what “optimal” means.

**Duplicates reweight the objective.** Ten copies of one story give its transformations roughly ten times the sampling mass. Repetition can be useful when data are scarce (Unit 06, §6.8.5), so a duplicate is not mathematically worthless. Uncontrolled duplication is a problem because the weight is accidental, can waste diversity and is associated with greater memorization ([Lee et al. 2022](/references/#lee-2022); [Carlini et al. 2023](/references/#carlini-2023)).

**Contamination invalidates an evaluation claim.** If a training item reproduces a held-out item, the held-out score no longer estimates performance on unseen independent items. Near-copies and paraphrases make this a similarity question, not only a byte-equality check.

**Quality is target quality.** A false statement is still a valid target under likelihood. Source coverage, language balance, licensing, personal information and timestamp all belong in data provenance because they determine both model behaviour and what uses are legitimate.

A defensible sequence of operations is:

1. define the independent unit, such as a whole story or document;
2. freeze train, validation and test membership;
3. deduplicate within training according to a declared rule;
4. search for training items too similar to validation/test, and remove or quarantine the **training** items;
5. keep an audit log of thresholds, hashes, source identifiers and counts;
6. never tune the cleaning rule solely to make the reported test number look better.

In the spine corpus, the planted manifest makes recall and false positives measurable. On real data, absence of a detector match is only “not found by this detector at this threshold,” not proof of absence.

## 7.2.6 Pretraining is a starting point, not a finished behaviour

Pretraining can be reused through a frozen probe, partial or full fine-tuning, parameter-efficient adapters, retrieval, prompting, or preference/reward-based post-training. These methods ask different amounts of the pretrained representation.

Transfer works when the pretraining transformation preserves information and invariances useful downstream. It can fail when the objective rewards a shortcut, the source and target distributions differ, or adaptation erases useful features. “More pretraining” is therefore not automatically “better for every task.” The broad pattern—pretrain on automatically constructed targets, then adapt with a smaller amount of task-specific evidence—has been influential in language modelling ([Radford et al. 2018](/references/#radford-2018); [Devlin et al. 2019](/references/#devlin-2019)), but every transfer claim still needs a downstream control.

| Question to ask | Autoregressive | Masked/denoising | Contrastive |
|---|---|---|---|
| what is hidden? | future token under an order | corrupted content | identity of the matching view |
| what context is visible? | prefix | usually both sides / uncorrupted regions | one augmented view plus candidates |
| what does the loss directly fit? | ordered conditional probabilities | reconstruction conditionals | a match/density-ratio classifier |
| common misuse | calling fluent imitation factual truth | assuming conditionals form one joint generator | assuming chosen invariances preserve every downstream fact |

**Next:** [§7.3 Mathematical construction](/core/unit-07/construction/) turns each row into an equation, then builds the MinHash and contamination tools needed to make the data claim testable.
