---
title: "2.2 First principles"
description: Features and linear read-outs, XOR, one-hot codes as count tables, embeddings as learned shared coordinates, the distributional hypothesis, tied input and output embeddings, and longer contexts without exponential tables.
sidebar: { label: "2.2 First principles", order: 2 }
---

## 2.2.1 A representation is a choice of coordinates

A learner never sees the world; it sees numbers. A **representation** is the map $\phi$ that turns a raw input — a character, an image, a sound — into a vector $\phi(x)\in\mathbb{R}^d$. Everything the learner can do afterwards is limited by what $\phi$ makes easy.

To isolate the effect, fix the simplest possible learner: a **linear read-out** that computes a score $w^{\top}\phi(x) + b$ for each output and predicts the highest score (or, for probabilities, applies a softmax to the scores). A linear read-out can only draw flat boundaries — lines, planes, hyperplanes — in the space of $\phi(x)$. If the right answer is a flat function of the representation, learning is easy; if not, no amount of data helps.

## 2.2.2 XOR: impossible, then trivial

Take four inputs with two binary coordinates and the label "exactly one coordinate is 1":

| $x_1$ | $x_2$ | label |
|---|---|---|
| 0 | 0 | 0 |
| 0 | 1 | 1 |
| 1 | 0 | 1 |
| 1 | 1 | 0 |

With $\phi(x) = (x_1, x_2)$, a linear classifier predicts 1 when $w_1x_1 + w_2x_2 + b > 0$. The four rows require $b \le 0$, $w_2 + b > 0$, $w_1 + b > 0$ and $w_1 + w_2 + b \le 0$. Adding the middle two gives $w_1 + w_2 + 2b > 0$, so $w_1 + w_2 + b > -b \ge 0$, contradicting the last. **No line separates XOR** ([Minsky & Papert 1969](/references/#minsky-1969) made this famous as a limit of the single-layer perceptron).

Now use $\phi(x) = (x_1, x_2, x_1x_2)$. The weights $w = (1, 1, -2)$, $b = -0.5$ give scores $-0.5, 0.5, 0.5, -0.5$ for the four rows: perfectly separated (Worked Example 2.1). The learner did not become smarter; the representation made the target linear.

Two lessons generalize:

1. **A good representation makes the target simple — ideally linear — in its coordinates.**
2. **Representations can be built by combining raw coordinates.** The product $x_1x_2$ is a *nonlinear feature*. A neural network's hidden layer computes many such features and *learns which ones to compute*; that is what makes it more than a linear model (§2.5.1).

## 2.2.3 One-hot codes: the representation that shares nothing

How should a character be represented? The obvious answer gives each of the $V$ characters its own coordinate: character $a$ is the **one-hot** vector $e_a\in\mathbb{R}^V$, with a 1 in position $a$ and 0 elsewhere.

One-hot vectors are all orthogonal and all at the same distance $\sqrt 2$ from each other. The representation asserts that `3` is exactly as different from `4` as it is from `T`. Look at what a linear read-out does with it: scores $z = W^{\top}e_a = W[a]$ — the read-out simply **looks up row $a$** of its weight matrix. A softmax over that row gives $q(\cdot\mid a)$. So a linear model on one-hot inputs *is* a table with one independent row per input. Trained by maximum likelihood, it converges to exactly the count-based bigram of Unit 01 (that is the content of spine Stage 03, which trains this model with gradients and recovers the counted table).

The bigram's inability to share, diagnosed in §1.7.7, is therefore not a property of counting. **It is a property of the one-hot representation.** Change the representation and sharing becomes possible.

## 2.2.4 Embeddings: learned coordinates that can be shared

Replace each one-hot vector by a short dense vector $e_a \in \mathbb{R}^d$ with $d \ll V$, stored as row $a$ of an **embedding table** $E\in\mathbb{R}^{V\times d}$. The table is *learned* along with everything else.

What changes? Suppose the model must predict what follows `3` and what follows `4`. If their embeddings are close, $e_{\texttt 3}\approx e_{\texttt 4}$, any function of the embedding gives them similar predictions — so evidence about `3` is automatically evidence about `4`. And the training signal pushes embeddings together exactly when the characters behave alike: two characters followed by similar distributions receive similar gradients and drift toward each other. Nothing tells the model which characters are digits; the data does.

This is the idea [Bengio et al. (2003)](/references/#bengio-2003) used to build the first neural language model: "associate with each word a distributed feature vector … learn simultaneously the feature vectors and the parameters of the probability function". The point of the word *distributed* is that each character is described by $d$ shared coordinates rather than one private one.

There is an accounting benefit as well. A bigram table has $V^2$ parameters. A bigram that scores pairs by $z_{ab} = e_a^{\top}Me_b + c_b$ (§2.3.3) has $Vd + d^2 + V$ — for $V = 50$, $d = 16$: 1 106 instead of 2 450. On the spine corpus it comes within 0.03 nats of the full table (Worked Example 2.4), and the structure it keeps is the structure that generalizes.

## 2.2.5 The distributional hypothesis

Why should "occurs in similar contexts" mean "should be represented similarly"? The linguist Zellig Harris argued in 1954 that differences in meaning correlate with differences in distribution ([Harris 1954](/references/#harris-1954)); J. R. Firth's later slogan was "you shall know a word by the company it keeps". For a predictive model the argument is simpler and needs no appeal to meaning: **if two symbols are interchangeable in every context, any optimal predictor treats them identically, so nothing is lost by giving them the same representation — and statistical strength is gained by pooling their evidence.**

The spine tests check this directly. They generate text from a Markov chain in which symbols 1 and 2 have identical predecessors and successors, train the factorized bigram, and require the cosine similarity of $e_1$ and $e_2$ to exceed 0.95 and to exceed every other pair's (Experiment 2.3). On real text the interchangeability is partial — `3` and `4` are interchangeable in most contexts, not all — and the geometry reflects degrees of it (Worked Example 2.5).

## 2.2.6 One table for reading and writing: tied embeddings

A language model uses characters twice: as **inputs** (what came before) and as **outputs** (what comes next). With separate tables, the input embedding of `3` and the output weights for `3` are learned independently. But the reasons for `3` and `4` to be similar as inputs (they are followed by similar things) and as outputs (they are preceded by similar things) are closely related. **Tying** uses one table $E$ for both: the score for "next character is $b$" is a dot product with $e_b$ ([Press & Wolf 2017](/references/#press-2017)). It halves the embedding parameters and often improves held-out loss; the spine model ties its embeddings from Stage 02 onward (Unit 04's Transformer reads out with $E^{\top}$).

## 2.2.7 Longer contexts without exponential tables

The decisive advantage of embeddings appears when the context grows. A count model over the previous $k$ characters needs a row for each of up to $V^k$ contexts — 6.25 million for $V = 50$, $k = 4$ — and only a tiny fraction will ever be observed. A **neural n-gram** represents the context by **concatenating** the $k$ embeddings and computing the prediction with a small network:

$$
h = \tanh\big(W_1[e_{x_{t-k}};\ \dots;\ e_{x_{t-1}}] + b_1\big),\qquad z = W_2h + b_2 .
$$

Its size grows *linearly* in $k$. Two contexts that differ only in a character's identity — `ad 3 ` versus `ad 4 ` — get nearby inputs, so the network can generalize from one to the other. This is how the neural 4-gram beats the count 4-gram on scarce data (§2.1, Worked Example 2.6).

Concatenation is still a crude way to combine positions: it fixes the window size and treats position 1 and position 4 with entirely separate weights. How information from different positions *should* be combined is the question of Unit 04, and the answer — attention — keeps the embedding table of this unit as its first layer.

## 2.2.8 What we have so far

| Representation | What it shares | Parameters for $k$ characters of context | Spine result |
|---|---|---|---|
| one-hot + table | nothing | up to $V^{k+1}$ | bigram 2.12 nats (Unit 01) |
| embedding + factorized bigram | similar characters | $Vd + d^2 + V$ | 2.15 nats with $d = 16$ (1 106 parameters) |
| embedding + neural n-gram | similar characters *and* similar contexts | $Vd + kdH + H + HV + V$ | 0.98 vs 1.11 for counts at 5 000 characters, $k = 4$ |

The common thread: **a representation decides what the learner treats as the same**. One-hot treats nothing as the same; embeddings learn what to treat as similar; Unit 04's attention and Unit 05's recurrence build in assumptions about *positions* as well.
