---
title: "7.2 First principles"
description: Supervision by hiding and predicting, the chain rule that makes next-token prediction exact, what masked and contrastive objectives see instead, and why the data itself is part of the learning algorithm.
sidebar: { label: "7.2 First principles", order: 2 }
---

## 7.2.1 Hide part of the data, predict it from the rest

Supervised learning needs pairs $(x, y)$. Self-supervised learning manufactures them from unlabelled data $z$ by a fixed rule that splits $z$ into a visible part and a hidden part: $x = \text{visible}(z)$, $y = \text{hidden}(z)$. Training then proceeds exactly as in Units 01–03: minimize the log-loss of predicting $y$ from $x$.

The rule is the whole design. Three choices dominate practice:

| Objective | Visible | Hidden | Learns |
|---|---|---|---|
| **autoregressive** (next token) | everything to the left | the next token | a full probability model of sequences; a generator |
| **masked** | everything except ~15% of tokens | the masked tokens | bidirectional representations; no generator |
| **contrastive** | one view of an item | which of many candidates is another view of the same item | embeddings in which related items are close; no prediction of content |

Each choice decides what the model must learn and what it may ignore. A masked model can use the word *after* a gap to fill it — useful for understanding a sentence, useless for writing one, because when writing the next word does not yet exist. A contrastive model never has to predict any detail of its input, only enough to tell matching pairs from non-matching ones — efficient for representations, blind to everything the comparison does not need.

## 7.2.2 Why next-token prediction is enough for a generator

The **chain rule of probability** writes any joint distribution over a sequence as a product of conditionals:

$$
p(x_1, x_2, \dots, x_T) = p(x_1)\,p(x_2\mid x_1)\,p(x_3\mid x_1, x_2)\cdots p(x_T\mid x_{<T}).
$$

This is an identity, not an approximation. So a model that predicts each token from all previous ones defines a full distribution over sequences, and its log-loss summed over a sequence is exactly the negative log-likelihood of the whole sequence. **Training a next-token predictor by cross-entropy is maximum-likelihood estimation of the sequence distribution** — the principle of Unit 01, now applied to an object with exponentially many outcomes. The causal mask of Unit 04 (§4.3.2) is what makes every position's prediction use only the left context, so that one forward pass computes all $T$ conditionals at once.

The same identity is why a trained model can generate: sample $x_1$, then $x_2$ given $x_1$, and so on (Unit 08).

## 7.2.3 What the spine model must learn to predict its corpus

Consider what predicting the next character of the spine corpus requires:

- **spelling and frequent words** — the bigram (Unit 01) gets these partly;
- **story structure** — "One windy day" follows the first sentence; "felt" follows a problem;
- **in-context names** — after "with Finn. A cat came to play with Ben and ", the next word is "Finn", which only the context reveals;
- **made-up names** — a pet called "Kivod" can only be predicted on its second mention by copying from its first (Unit 11 looks for the mechanism);
- **arithmetic** — "Ben had 2 eggs. Then Ben found 3 more. Now Ben has " — but the corpus writes a wrong answer 30% of the time, so the *optimal* predictor puts about 30% of its probability on wrong answers (Unit 01, §1.7.5).

None of this was labelled. All of it is supervised by the text. The last item is the warning: **the objective is to predict the data, so a self-supervised model learns the data's errors as faithfully as its truths.**

## 7.2.4 The data is part of the algorithm

Because the supervision comes from the data, properties of the corpus become properties of the learned model:

- **Duplicates** are examples counted twice. They tilt the model toward whatever is repeated, waste compute, and increase memorization of the repeated text ([Lee et al. 2022](/references/#lee-2022); [Carlini et al. 2023](/references/#carlini-2023)).
- **Contamination** — copies of evaluation data in the training data — makes held-out numbers measure memory instead of generalization (Unit 01, §1.7.2).
- **Quality and mixture** — which sources, in what proportions — set what the model is good at. Small models trained on carefully selected or generated text can match much larger models trained on unfiltered text on some tasks ([Eldan & Li 2023](/references/#eldan-2023); [Gunasekar et al. 2023](/references/#gunasekar-2023)).

The spine corpus is synthetic precisely so that these properties are known: `spine/data/make_tales.py` plants 67 duplicate pairs (37 exact, 30 near), 6 contaminated held-out stories and 30% wrong arithmetic, and writes their locations to a manifest. Stage 07's detectors are checked against it: a data pipeline, like a gradient, needs a verification target.

## 7.2.5 Pretraining, then adaptation

A model pretrained by self-supervision is rarely the end product. Its representations are reused — by a linear probe (Unit 02, §2.8.3), by fine-tuning on a small labelled set, or, for generators, by prompting and post-training (Unit 09). The bet is that the hidden-part prediction forced the model to learn features useful far beyond the prediction itself; the bet pays when the downstream task depends on the same structure the objective rewarded (§7.5.4). This two-stage pattern — **pretrain broadly on raw data, adapt narrowly with little supervision** — is the organizing idea of current practice ([Radford et al. 2018](/references/#radford-2018); [Devlin et al. 2019](/references/#devlin-2019)).
