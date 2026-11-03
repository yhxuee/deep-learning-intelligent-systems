---
title: "2.2 First principles"
description: Features and linear read-outs, XOR, one-hot codes as count tables, embeddings as learned shared coordinates, the distributional hypothesis, tied input and output embeddings, and longer contexts without exponential tables.
sidebar: { label: "2.2 First principles", order: 2 }
---

## 2.2.1 A representation is a choice of coordinates

A model never receives “a character” or “an image” in the abstract; it receives numbers. Formally, a **representation** is a map

$$
\phi:x\longmapsto\phi(x)\in\mathbb R^d
$$

that turns an input $x$ into a vector of $d$ numbers. Here $x$ might be a character, image or sound; $\phi(x)$ is what the next computation sees; and $d$ is the representation's dimension. The map may be designed by a person, as in the XOR feature below, or learned from data, as in an embedding table.

To see what the representation contributes, hold the downstream rule fixed. We will begin with the simplest useful rule: a **linear read-out**. For binary classification it computes

$$
s(x)=w^{\top}\phi(x)+b
$$

and predicts class 1 when the score crosses a threshold. With several classes, it computes one score per class and either chooses the largest or applies softmax to obtain probabilities. Such a read-out can draw only flat decision boundaries—lines in two dimensions, planes in three, and hyperplanes in higher dimensions.

So the precise beginner's slogan is:

> Learning with a linear read-out is easy when the target is linear in the chosen representation.

This is a claim about the **pair** consisting of a representation and a read-out. A one-hot vector followed by an unrestricted table behaves differently from the same vector followed by a multilayer network; we will keep the read-out explicit rather than credit every effect to the representation alone.

## 2.2.2 XOR: impossible, then easy

Take four inputs with two binary coordinates and the label "exactly one coordinate is 1":

| $x_1$ | $x_2$ | label |
|---|---|---|
| 0 | 0 | 0 |
| 0 | 1 | 1 |
| 1 | 0 | 1 |
| 1 | 1 | 0 |

With $\phi(x)=(x_1,x_2)$, suppose a linear classifier predicts 1 when $w_1x_1+w_2x_2+b>0$. Reading the four rows in order gives four requirements:

$$
b\le0,\qquad w_2+b>0,\qquad w_1+b>0,\qquad w_1+w_2+b\le0.
$$

Add the two strict middle inequalities: $w_1+w_2+2b>0$. Therefore $w_1+w_2+b>-b\ge0$, which contradicts the last requirement. The contradiction is the proof: **no line separates XOR**. [Minsky & Papert (1969)](/references/#minsky-1969) made this limitation of a single-layer perceptron famous.

Now use $\phi(x)=(x_1,x_2,x_1x_2)$. The weights $w=(1,1,-2)$ and $b=-0.5$ give scores $-0.5,0.5,0.5,-0.5$ for the four rows, so the same kind of linear read-out separates them perfectly (Worked Example 2.1). The new coordinate records whether both inputs are 1. It supplies exactly the interaction that the original two coordinates could not express linearly.

Two lessons generalize:

1. **A good representation makes the target simple — ideally linear — in its coordinates.**
2. **Representations can combine raw coordinates.** The product $x_1x_2$ is a *nonlinear feature*. A hidden layer computes many nonlinear features and learns which are useful; the final layer may then remain linear (§2.5.1).

Do not overread the example. We chose $x_1x_2$ because we already knew the rule. Deep learning replaces that hand choice with an optimization problem: search for internal features that reduce predictive loss. Whether that search succeeds is a separate question, taken up in Unit 03.

## 2.2.3 One-hot codes: the representation that shares nothing

Suppose the vocabulary contains $V$ characters. The most literal encoding gives each character its own coordinate. Character $a$ becomes the **one-hot** vector $u_a\in\mathbb R^V$: position $a$ is 1 and every other position is 0. For a three-symbol vocabulary,

$$
u_1=(1,0,0),\qquad u_2=(0,1,0),\qquad u_3=(0,0,1).
$$

Every pair of distinct one-hot vectors is orthogonal and lies at distance $\sqrt2$. The code itself provides no notion that `3` resembles `4` more than `T`. Now let $W\in\mathbb R^{V\times V}$ be an unrestricted linear read-out. For input $a$,

$$
z=W^{\top}u_a=W[a],
$$

where $W[a]$ denotes row $a$. Multiplication by a one-hot vector is therefore just a **row lookup**. Softmax turns that row into $q(\cdot\mid a)$, so a one-hot input plus an unrestricted linear read-out is exactly a table with one independently adjustable row per input. Maximum-likelihood training recovers the count-based bigram from Unit 01. The executable spine later verifies the equivalence by fitting the table with gradients rather than counts.

The careful conclusion is not that one-hot vectors can never be used in a model that shares parameters. It is that **one-hot plus an unrestricted linear read-out introduces no sharing across symbols**. Sharing must come from an additional constraint, a different parameterization, or a learned lower-dimensional representation.

## 2.2.4 Embeddings: learned coordinates that can be shared

Replace the length-$V$ one-hot code by a shorter vector $e_a\in\mathbb R^d$, with $d\ll V$. Store all of them as the rows of an **embedding table** $E\in\mathbb R^{V\times d}$, so looking up symbol $a$ returns $E[a]=e_a$. These entries are trainable parameters, not predetermined semantic attributes.

What can this buy us? Suppose one shared, reasonably smooth prediction function consumes every embedding. If $e_{\texttt3}$ and $e_{\texttt4}$ are close and the function's weights are not extreme, their predictions will also be close. An update that improves how the shared function handles one region of embedding space can then help several symbols in that region. This is **statistical sharing**.

The direction of explanation matters. Similar behaviour can give two symbols similar training signals and may lead the optimizer to place them nearby; merely declaring an embedding dimension does not guarantee that useful geometry will emerge. The learned geometry also depends on the task, data, objective, model and optimization path. Nothing labels a coordinate “is a digit.”

This is the idea [Bengio et al. (2003)](/references/#bengio-2003) used to build the first neural language model: "associate with each word a distributed feature vector … learn simultaneously the feature vectors and the parameters of the probability function". The point of the word *distributed* is that each character is described by $d$ shared coordinates rather than one private one.

There is also a parameter bottleneck. A full bigram stores $V^2$ logits, although adding a constant to every logit in one row changes no probability, leaving $V(V-1)$ effective degrees of freedom. The factorized score

$$
z_{ab}=e_a^{\top}Me_b+c_b
$$

uses $Vd+d^2+V$ trainable numbers (§2.3.3). With $V=50$ and $d=16$, that is 1 106 stored numbers instead of 2 500 logits in the full table (2 450 effective degrees of freedom). On the spine corpus its held-out loss is within 0.03 nats of the full table (Worked Example 2.4). The smaller model is not automatically better; its low-dimensional bottleneck is an inductive bias that helps only when the omitted distinctions are less useful than the sharing it creates.

## 2.2.5 The distributional hypothesis

Why should “occurs in similar contexts” suggest “use a similar representation”? Zellig Harris argued that differences in meaning correlate with differences in linguistic distribution ([Harris 1954](/references/#harris-1954)); J. R. Firth later summarized the idea through the company a word keeps. For prediction we need a narrower claim, not a complete theory of meaning:

> If two symbols are interchangeable in every context relevant to the task, an optimal predictor treats them identically. Pooling their evidence can then reduce estimation noise without sacrificing task-relevant information.

The qualification is essential. Words can occur in similar contexts yet differ in a way another task needs; antonyms are a familiar warning. An embedding records similarity *for the training objective*, not an all-purpose semantic truth.

The spine tests check this directly. They generate text from a Markov chain in which symbols 1 and 2 have identical predecessors and successors, train the factorized bigram, and require the cosine similarity of $e_1$ and $e_2$ to exceed 0.95 and to exceed every other pair's (Experiment 2.3). On real text the interchangeability is partial — `3` and `4` are interchangeable in most contexts, not all — and the geometry reflects degrees of it (Worked Example 2.5).

## 2.2.6 One table for reading and writing: tied embeddings

A language model uses characters twice: as **inputs** (what came before) and as **outputs** (what comes next). With separate parameters, the input embedding of `3` and the output vector used to score `3` need not be related. **Weight tying** reuses the same table $E$ on both sides: the score for “next character is $b$” contains an inner product with $e_b$ ([Press & Wolf 2017](/references/#press-2017)). This reduces parameters and imposes the belief that useful input and output geometry should be related.

Tying is a design choice, not a theorem that the two roles must be identical. It also requires compatible dimensions, sometimes supplied by an extra projection. The factorized bigram in spine Stage 02 ties $E$ in $e_a^{\top}Me_b+c_b$; the separate neural n-gram used later in this section does **not** tie its output matrix. Unit 04's Transformer again reads out with $E^{\top}$.

## 2.2.7 Longer contexts without exponential tables

The pressure for sharing becomes strongest as context grows. A count model over the previous $k$ characters may need one row for each of $V^k$ contexts—6.25 million when $V=50$ and $k=4—and only a small fraction may be observed. A **neural n-gram** instead looks up the $k$ embeddings, **concatenates** them into one vector of length $kd$, and sends that vector through a small network:

$$
h = \tanh\big(W_1[e_{x_{t-k}};\ \dots;\ e_{x_{t-1}}] + b_1\big),\qquad z = W_2h + b_2 .
$$

Read the equation from inside out. The bracketed expression joins the $k$ vectors; $W_1$ mixes their coordinates; $\tanh$ supplies a nonlinear hidden representation $h$; and $W_2h+b_2$ produces one logit per possible next character. Holding $d$, hidden width $H$ and vocabulary size $V$ fixed, its parameter count grows *linearly* rather than exponentially with $k$.

Contexts such as `ad 3 ` and `ad 4 ` can now enter the shared network at nearby points. A smooth learned function can transfer evidence between them instead of requiring two unrelated rows. That is the mechanism behind the neural 4-gram's advantage over the count 4-gram when training data are scarce (§2.1, Worked Example 2.6); it is a useful bias, not a guarantee for every dataset.

Concatenation is still a crude way to combine positions: it fixes the window size and treats position 1 and position 4 with entirely separate weights. How information from different positions *should* be combined is the question of Unit 04, and the answer — attention — keeps the embedding table of this unit as its first layer.

## 2.2.8 What we have so far

| Representation | What it shares | Parameters for $k$ characters of context | Spine result |
|---|---|---|---|
| one-hot + unrestricted table | nothing across contexts | up to $V^{k+1}$ logits | bigram 2.12 nats (Unit 01) |
| embedding + factorized bigram | similar characters | $Vd + d^2 + V$ | 2.15 nats with $d = 16$ (1 106 parameters) |
| embedding + neural n-gram | similar characters *and* similar contexts | $Vd + kdH + H + HV + V$ | 0.98 vs 1.11 for counts at 5 000 characters, $k = 4$ |

The common thread is now more precise: **a representation, together with the function that consumes it, determines which cases can share evidence and which distinctions remain available.** One-hot plus a free table shares nothing across contexts. A low-dimensional embedding plus a shared function can learn graded similarity. Unit 04's attention and Unit 05's recurrence add assumptions about *positions* as well.

Before continuing, check that you can answer four questions in plain language:

1. Why can more XOR data not rescue a linear classifier on $(x_1,x_2)$?
2. Why is $W^{\top}u_a$ a lookup rather than a genuine mixture of rows?
3. What assumption allows nearby embeddings to share statistical strength?
4. What useful information might a low-dimensional embedding discard?

If one answer is unclear, return to that subsection before beginning the matrix derivations in §2.3.
