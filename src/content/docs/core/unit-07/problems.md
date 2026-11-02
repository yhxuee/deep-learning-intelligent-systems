---
title: "7.10 Problems and further study"
description: Twenty graded problems on self-supervised objectives, mutual information bounds, MinHash and locality-sensitive hashing, contamination and data design, with selected answers and an annotated reading list.
sidebar: { label: "7.10 Problems", order: 10 }
---

Difficulty: ★ routine · ★★ combines ideas · ★★★ proof or open-ended. **[AI-off]** problems are for independent practice and assessment.

## Computation

**7.1 ★ [AI-off]** A model assigns probabilities 0.2, 0.5, 0.5, 0.9 to the four characters of a word given their prefixes. Compute the per-character losses, their mean, and the probability of the whole word.

**7.2 ★ [AI-off]** Two documents have 40 shingles each and share 30. Compute their Jaccard similarity and the probability that a single MinHash agrees.

**7.3 ★** For $b = 20$ bands of $r = 5$ rows, compute the candidate probability at $J = 0.5, 0.7, 0.9$ and the threshold $(1/b)^{1/r}$.

**7.4 ★** With 64 hash functions, what is the standard error of the MinHash estimate of $J = 0.5$? How many hash functions halve it?

**7.5 ★★** A contrastive model is trained with batches of 4 096 and reaches a loss of 2.0 nats. What lower bound on the mutual information between views does this certify?

**7.6 ★★** A corpus has $n$ documents, each duplicated $m$ times on average with a heavy tail. Under maximum likelihood, by what factor is a document repeated 1 000 times overweighted relative to one occurring once? What does deduplication change?

## Derivation

**7.7 ★ [AI-off]** Prove that the sum of next-token log-losses of a sequence equals its negative log-likelihood under the model.

**7.8 ★★ [AI-off]** Prove Proposition 7.2 (MinHash collision probability).

**7.9 ★★** Derive the candidate probability $1 - (1 - J^r)^b$ and the location of its steepest point.

**7.10 ★★** Show that the Bayes-optimal critic for InfoNCE is $\ln\frac{p(y\mid x)}{p(y)}$ up to a function of $x$.

**7.11 ★★** Derive the gradient of the InfoNCE loss with respect to the positive score and the negative scores. Interpret it as "pull the positive, push each negative in proportion to its softmax weight".

## Proof

**7.12 ★★★** Show by example that a set of conditionals $p(x_t\mid x_{\setminus t})$ for all $t$ can be inconsistent with any joint distribution. State a condition under which consistent full conditionals determine the joint (Brook's lemma).

**7.13 ★★** Complete the proof of Proposition 7.1, checking that $q(y_{1:K}\mid x)$ integrates to 1.

**7.14 ★★** Prove that a constant embedding minimizes an alignment-only loss $\mathbb{E}\lVert g(x) - g(x')\rVert^2$ over positive pairs, and that adding $-\ln\sum_k e^{g(x)^{\top}g(y_k)}$ (InfoNCE) rules it out.

**7.15 ★★★** For a model that has memorized a training sequence exactly, show that greedy decoding from a sufficiently long prefix reproduces it. What property of the prefix makes "sufficiently long" short?

## Implementation and design

**7.16 ★ [AI-off]** Implement shingling, MinHash and banding from memory and reproduce Worked Example 7.4.

**7.17 ★★** Implement a contamination check that is robust to whitespace and punctuation changes. Construct a paraphrase attack that defeats it and propose a model-based complement.

**7.18 ★★** Implement a masked-character objective for the spine model (no causal mask) and run Experiment 7.5.

**7.19 ★★** Design a data mixture for a spine model that must both tell stories and do arithmetic correctly. How would you measure each skill separately, and how would you choose the weights?

**7.20 ★★★** Design an experiment to test whether deduplication helps *generalization* or only saves compute. State the controls, the metric, and what result would distinguish the two explanations (preview of Unit 12).

## Selected answers

- **7.1** Losses $1.609, 0.693, 0.693, 0.105$; mean $0.775$; probability $0.2\times0.5\times0.5\times0.9 = 0.045$, and $-\ln0.045 = 3.101 = $ the sum.
- **7.2** $J = 30/(40 + 40 - 30) = 0.6$; agreement probability 0.6.
- **7.3** $1 - (1 - J^5)^{20}$: 0.470, 0.975, 1.000; threshold $(1/20)^{1/5} = 0.549$.
- **7.4** $\sqrt{0.25/64} = 0.0625$; 256 hash functions halve it.
- **7.5** $\ln 4096 - 2.0 = 8.318 - 2.0 = 6.32$ nats.

## Further reading

**Objectives.** [Radford et al. 2018](/references/#radford-2018), [Devlin et al. 2019](/references/#devlin-2019) — the two pretraining recipes that defined the field. [van den Oord et al. 2018](/references/#oord-2018) and [Poole et al. 2019](/references/#poole-2019) — contrastive learning and what its bounds can and cannot say.

**Data.** [Lee et al. 2022](/references/#lee-2022) — deduplication's effects. [Penedo et al. 2024](/references/#penedo-2024) — a documented web-scale pipeline with ablations. [Broder 1997](/references/#broder-1997) — MinHash in four pages.

**Risks.** [Carlini et al. 2021](/references/#carlini-2021), [Carlini et al. 2023](/references/#carlini-2023) — memorization and extraction. [Shumailov et al. 2024](/references/#shumailov-2024) — training on generated data.
