---
title: "7.3 Mathematical construction"
description: The autoregressive objective and teacher forcing, masked language modelling as pseudo-likelihood, contrastive learning and the InfoNCE bound, Jaccard similarity, MinHash and locality-sensitive hashing, and n-gram contamination checks.
sidebar: { label: "7.3 Construction", order: 3 }
---

## 7.3.1 The autoregressive objective

For a sequence $x_{1:T}$ and a model $q_\theta(x_t\mid x_{<t})$, the per-token loss averaged over a corpus is

$$
\hat L(\theta) = -\frac{1}{\sum_s T_s}\sum_{s}\sum_{t=1}^{T_s}\ln q_\theta\big(x^{(s)}_t\mid x^{(s)}_{<t}\big).
$$

By the chain rule (§7.2.2), $\sum_t\ln q_\theta(x_t\mid x_{<t}) = \ln q_\theta(x_{1:T})$, so minimizing $\hat L$ is maximum likelihood for the sequence model, and by Unit 01 (§1.3.3) it minimizes the KL divergence from the empirical distribution of sequences to the model.

**Teacher forcing.** During training every conditional is computed with the *true* prefix $x_{<t}$, never the model's own predictions. This is what lets a Transformer compute all $T$ losses in one parallel pass under the causal mask. It also creates a mismatch: at generation time the model conditions on its own samples, which may lie outside the training distribution of prefixes (a cause of degenerate repetition, Unit 08).

**Windows.** In practice $T$ is the context length and long documents are cut into windows; the spine model samples random 128-character windows from the concatenated stories. A window that starts mid-story still trains every position, but the first positions see little context; a held-out estimate should be computed per story (Unit 10).

## 7.3.2 Masked language modelling

Choose a random subset $M$ of positions (BERT uses 15%), replace their tokens by a mask symbol (or, in 10% of cases each, by a random or unchanged token), and minimize

$$
\hat L_{\text{MLM}}(\theta) = -\mathbb{E}_{M}\sum_{t\in M}\ln q_\theta\big(x_t\mid x_{\setminus M}\big),
$$

where $x_{\setminus M}$ is the sequence with the masked positions hidden ([Devlin et al. 2019](/references/#devlin-2019)). Each prediction uses context on both sides, so the representation at position $t$ can encode its whole sentence. But the conditionals $q(x_t\mid x_{\setminus M})$ for different $M$ need not be consistent with any single joint distribution: the objective is a **pseudo-likelihood**, and a masked model does not define a normalized probability of a sequence or an obvious way to sample one. Only 15% of positions are trained per pass, against 100% for the autoregressive objective — part of why autoregressive pretraining extracts more signal per token.

## 7.3.3 Contrastive learning and the InfoNCE bound

Let $(x, y)$ be a **positive pair** drawn from a joint distribution $p(x, y)$ — two augmentations of one image, an image and its caption, two halves of one story — and let $y_2,\dots,y_K$ be **negatives** drawn from the marginal $p(y)$. A critic $f(x, y)$ (typically a dot product of embeddings divided by a temperature) is trained to pick the positive:

$$
\mathcal L_{\text{NCE}} = -\mathbb{E}\Big[\ln\frac{e^{f(x, y_1)}}{\sum_{k=1}^{K}e^{f(x, y_k)}}\Big].
$$

This is a $K$-way softmax classification with the positive as the correct class ([van den Oord et al. 2018](/references/#oord-2018); compare noise-contrastive estimation, [Gutmann & Hyvärinen 2010](/references/#gutmann-2010)).

**Mutual information.** $I(X; Y) = \mathbb{E}_{p(x,y)}\big[\ln\frac{p(x,y)}{p(x)p(y)}\big]$ measures, in nats, how much knowing $x$ reduces uncertainty about $y$.

**Proposition 7.1 (InfoNCE bound).** For any critic, $I(X; Y)\ \ge\ \ln K - \mathcal L_{\text{NCE}}$.

*Proof ([Poole et al. 2019](/references/#poole-2019)).* Draw $Y_1\sim p(y\mid x)$ and $Y_2,\dots,Y_K\sim p(y)$ independently. The extra variables are independent of everything else, so $I(X; Y_{1:K}) = I(X; Y_1) = I(X;Y)$. For *any* conditional density $q(y_{1:K}\mid x)$, Gibbs' inequality (Unit 01, Theorem 1.1) gives the variational bound $I(X; Y_{1:K})\ge\mathbb{E}\big[\ln q(Y_{1:K}\mid X) - \ln p(Y_{1:K})\big]$. Choose
$$
q(y_{1:K}\mid x) = p(y_{1:K})\,\frac{e^{f(x,y_1)}}{\frac1K\sum_{k}e^{f(x,y_k)}} .
$$
It is a valid density: $p(y_{1:K}) = \prod_k p(y_k)$ is symmetric in the $y_k$, so its integral equals the integral of $p(y_{1:K})$ times the average over $i$ of $e^{f(x,y_i)}/\frac1K\sum_k e^{f(x,y_k)}$, which is 1. Substituting, $I(X;Y)\ge\mathbb{E}\big[\ln\frac{e^{f(X,Y_1)}}{\frac1K\sum_k e^{f(X,Y_k)}}\big] = \ln K - \mathcal L_{\text{NCE}}$. $\blacksquare$

Two consequences matter. Minimizing the contrastive loss maximizes a lower bound on the information the two views share, so the embeddings keep what is common to both views and may discard the rest. And the bound **cannot exceed $\ln K$**: with 1 024 negatives, no more than $\ln 1024 = 6.93$ nats can be certified, however much information the views share (Worked Example 7.7) — one reason contrastive methods use very large batches.

**Why negatives.** Without them, the loss "make $f(x, y_1)$ large" is minimized by mapping every input to the same point — **collapse**. Negatives force embeddings of different items apart. In the limit of many negatives the loss rewards *alignment* (positives close) and *uniformity* (embeddings spread over the sphere) ([Wang & Isola 2020](/references/#wang-isola-2020)). Non-contrastive methods avoid collapse by other means — asymmetric networks and stopped gradients ([Grill et al. 2020](/references/#grill-2020)) — whose analysis is subtler (§7.5.3).

## 7.3.4 Duplicates and Jaccard similarity

Represent a document by its set $S$ of **shingles**: here, the word 5-grams it contains. Two documents' resemblance is their **Jaccard similarity**

$$
J(S_1, S_2) = \frac{|S_1\cap S_2|}{|S_1\cup S_2|}\in[0,1].
$$

Identical documents have $J = 1$; a one-word edit of a 60-word story changes about five shingles and leaves $J$ near 0.8; unrelated stories from the same template share few 5-grams. Comparing every pair is $O(n^2)$ — impossible for billions of documents. MinHash makes it nearly linear.

## 7.3.5 MinHash and locality-sensitive hashing

Pick a random hash function $h$ that orders all possible shingles uniformly at random, and summarize a set by its minimum: $m(S) = \min_{s\in S}h(s)$.

**Proposition 7.2 ([Broder 1997](/references/#broder-1997)).** $\Pr\big[m(S_1) = m(S_2)\big] = J(S_1, S_2)$.

*Proof.* Consider the element of $S_1\cup S_2$ with the smallest hash. Each of the $|S_1\cup S_2|$ elements is equally likely to be it. The two minima are equal exactly when that element lies in $S_1\cap S_2$, which happens with probability $|S_1\cap S_2|/|S_1\cup S_2|$. $\blacksquare$

With $n$ independent hash functions, the fraction of agreeing minima estimates $J$ with standard error $\sqrt{J(1-J)/n}$. A **signature** of $n = 64$ minima summarizes each document in 64 numbers.

**Banding** finds candidate pairs without comparing all of them. Split the signature into $b$ bands of $r$ rows ($br = n$); hash each band; two documents become a candidate if *any* band matches exactly. A band matches with probability $J^r$, so

$$
\Pr[\text{candidate}] = 1 - (1 - J^r)^b .
$$

This S-shaped curve rises steeply around $J\approx(1/b)^{1/r}$. Stage 07 uses $b = 16$, $r = 4$: pairs with $J = 0.7$ become candidates with probability 0.988, pairs with $J = 0.3$ with probability 0.122 (Worked Example 7.3). Candidates are then confirmed by computing their exact Jaccard similarity, so the false positives of banding cost time, not correctness.

## 7.3.6 Contamination checks

A held-out item is **contaminated** if (a near copy of) it occurs in the training data. The standard check measures, for each evaluation item, the fraction of its word $n$-grams that also occur anywhere in the training data, and flags items above a threshold; GPT-3's evaluation used 13-grams ([Brown et al. 2020](/references/#brown-2020)). Two design choices dominate:

- **$n$ and the threshold** trade false positives against false negatives. Templated or formulaic text shares many short $n$-grams by construction: in the spine corpus, a typical clean held-out story shares 40% of its word 8-grams with the training text, and any threshold between 0.68 and 1.0 catches every planted copy with no false alarms (Worked Example 7.5).
- **What counts as a copy.** Paraphrases, translations and reformatted answers evade exact $n$-gram matching. Detection from the model's side — for example, testing whether a model assigns systematically higher likelihood to a benchmark's canonical ordering than to shuffled orderings ([Oren et al. 2024](/references/#oren-2024)) — catches some of these.

Stage 07 removes the training copies of contaminated stories and leaves the held-out set untouched, so that held-out numbers stay comparable across stages.

:::note[Claim check]
- **Claim.** An evaluation item that shares few $n$-grams with the training data was not trained on.
- **Assumptions.** Copies are near-verbatim, and $n$ and the threshold were set on items known to be clean and items known to be copied (Worked Example 7.5).
- **Test.** Plant known copies in the training data and measure how many the check flags, and how many clean items it flags (Worked Example 7.5; Experiment 7.1).
- **Limitation.** Paraphrases, translations and reformatted answers evade it (§7.7.3): a low overlap is evidence against verbatim copies only.
:::

