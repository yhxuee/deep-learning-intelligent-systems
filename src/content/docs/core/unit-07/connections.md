---
title: "7.6 Connections"
description: Use compression, statistical learning, information retrieval and density-ratio estimation to understand what self-supervision does—and where each analogy stops.
sidebar: { label: "7.6 Connections", order: 6 }
---

A useful connection should transfer a piece of reasoning, not merely attach a familiar name. For each connection below, ask two questions: **what becomes easier to see?** and **where does the analogy stop?**

## 7.6.1 Prediction and compression share the same arithmetic

Suppose a next-token model assigns probability $q(x_t\mid x_{<t})$ to the observed token. An ideal arithmetic coder can encode that token in approximately

$$
-\log_2 q(x_t\mid x_{<t})
$$

bits. Summing over positions gives the sequence code length; averaging in nats gives the cross-entropy used for training, with $1$ nat $=1/\ln2\approx1.443$ bits. A probability improvement and a compression improvement are therefore two units for the same mathematical event.

This viewpoint explains why prediction can learn surprisingly broad structure. Any reliably exploitable regularity—spelling, syntax, recurring facts, document format or a repeated error—can shorten the code. A model need not be told which regularity a human would call “grammar” or “knowledge”; gradient descent only sees whether probability moved toward the observed token.

It also exposes two limits.

1. **Compression follows the source, not truth.** A frequent misconception is compressible too. If a truth label never changes the observable text distribution, log-loss has no signal from which to recover it.
2. **Compression does not guarantee a usable interface.** A model may encode a regularity in parameters without making it reliable under a new prompt, a linear probe or a shifted deployment distribution.

Be careful with a tempting but false shortcut: “two statements occur equally often, so the model cannot distinguish them.” Equal marginal frequency is insufficient. Their surrounding contexts may differ, and a conditional compressor can exploit that difference. The correct impossibility claim requires the relevant observable distributions to be identical.

## 7.6.2 Statistical learning is broader than next-token prediction

Human learners can extract distributional regularities without receiving an explicit label for every observation. In a classic experiment, eight-month-old infants used transition statistics in a short continuous stream of artificial syllables to distinguish word-like sequences ([Saffran, Aslin & Newport 1996](/references/#saffran-1996)). The relevant connection is modest but important: structured exposure itself can create prediction problems and learning signals.

That result does **not** imply that an infant is optimizing Transformer cross-entropy, or that next-token prediction is a complete theory of cognition. Biological learning is active, embodied, multimodal and shaped by development, attention, action and social interaction. A language model receives a designer-chosen corpus, tokenizer, architecture and fixed objective. The analogy helps us notice learning from regularity; it cannot license claims that the mechanisms or outcomes are the same.

A second lesson travels in both directions. The learner only observes distinctions present in its experience. If two meanings always appear in indistinguishable contexts, exposure alone cannot identify which is which. Conversely, actions and interventions can create new evidence—one reason passive text prediction and embodied learning need not have the same limits.

## 7.6.3 Web search turned duplicate detection into a scalable pipeline

Near-duplicate detection arose as a practical search problem: mirrored and lightly edited pages consumed index space and crowded results. Broder's resemblance method connected document shingles, Jaccard similarity and MinHash ([Broder 1997](/references/#broder-1997)). The same three-stage pattern is useful for pretraining data:

1. **represent:** turn each document into a set of shingles;
2. **retrieve candidates:** use MinHash band collisions to avoid all-pairs comparison;
3. **decide:** compute exact similarity for candidates and apply a documented threshold and retention rule.

This division is more important than the particular algorithm. Candidate generation should be cheap and high-recall; final decisions can be slower and more precise. Confusing the two turns an approximate collision into an unjustified deletion.

Retrieval-augmented generation uses a related retrieve-then-score architecture (Unit 05), but usually with learned dense embeddings and approximate nearest-neighbour indexes rather than MinHash over word shingles. The shared engineering idea is sublinear candidate search. The similarity notion is different: lexical resemblance is useful for copies, while semantic embeddings aim to retrieve related meaning. Neither dominates the other for every audit.

## 7.6.4 Conditional fitting and classification connect to classical statistics

### Masking and pseudo-likelihood

Classical pseudo-likelihood replaces a difficult joint likelihood by a product of full conditionals,

$$
\operatorname{PL}(x)=\prod_{t=1}^{T}p(x_t\mid x_{-t}).
$$

Single-site masked prediction has this form. BERT-style multi-token corruption is a relative, not an exact synonym: several positions may be hidden simultaneously, and the corruption symbol becomes part of the conditioning information. The connection is useful because it asks the right consistency question—do the fitted conditionals correspond to one joint model?—while §7.5.1 explains why the answer is not automatically yes.

### Pair classification and density ratios

Create a binary classification problem. Let $D=1$ examples be real pairs from $p(x,y)$ and $D=0$ examples be shuffled pairs from $p(x)p(y)$, with equal class priors. The Bayes log-odds are

$$
\log\frac{P(D=1\mid x,y)}{P(D=0\mid x,y)}
=\log\frac{p(x,y)}{p(x)p(y)}.
$$

So a calibrated optimal classifier estimates a log density ratio. This gives a reusable recipe: turn “are these distributions different?” into classification, then read the ratio from the logit. Noise-contrastive estimation uses this idea to fit an unnormalized model against a known noise distribution ([Gutmann & Hyvärinen 2010](/references/#gutmann-2010)); classifier-based two-sample tests use classification accuracy as evidence of distribution shift (Unit 10).

The assumptions travel with the connection. Unequal class priors add a prior-odds constant. A different noise distribution changes the denominator. A restricted or uncalibrated classifier may separate samples well without returning an accurate numerical ratio.

## 7.6.5 The history is a sequence of choices about what to hide

The field is easier to remember as design choices than as a list of model names.

| Milestone | Automatically constructed target | Context made available | New interface |
|---|---|---|---|
| predictive word embeddings (Unit 02) | nearby word or context identity | a local window | one vector per word type |
| generative language-model pretraining ([Radford et al. 2018](/references/#radford-2018)) | next token | left context | generate, then fine-tune |
| BERT ([Devlin et al. 2019](/references/#devlin-2019)) | corrupted token | left and right context | bidirectional encoder |
| text-to-text transfer ([Raffel et al. 2020](/references/#raffel-2020)) | target text | task-formatted input text | one generative interface for many tasks |
| SimCLR ([Chen et al. 2020](/references/#chen-2020)) | matching augmented view | two image views plus batch negatives | transferable image representation |
| masked autoencoding ([He et al. 2022](/references/#he-2022)) | missing image patches | visible patches | reconstructive visual pretraining |
| CLIP ([Radford et al. 2021](/references/#radford-2021)) | matched image or caption | paired modalities plus negatives | shared retrieval space |

No row simply “replaced” every earlier row. Autoregressive and masked objectives expose different context; contrastive and reconstructive objectives preserve different information; encoder and generator interfaces serve different uses. The durable historical pattern is: construct a target from abundant structure, pretrain on it, then test whether the resulting state adapts to a separately specified task.

## 7.6.6 How the connections return later in the book

| Idea from this unit | Question it unlocks later |
|---|---|
| code length equals predictive log-loss | Unit 08: how does decoding transform a learned distribution into samples? |
| the source distribution contains truth, error and style together | Unit 09: what additional feedback can change behaviour rather than merely imitate it? |
| classifier logits can estimate density ratios | Unit 10: can a classifier detect train–test or deployment shift? |
| retrieval separates candidate generation from final scoring | Units 05 and 10: what does approximate search miss, and how is recall audited? |
| deduplication changes empirical weights | Unit 10: which evaluation population does a reported average represent? |
| pretraining creates a checkpoint, not a finished agent | Units 08–12: which later abilities were present, elicited, added or lost? |

The practical habit is to carry both halves of every connection: use the borrowed mathematics, and carry its assumptions with it.
