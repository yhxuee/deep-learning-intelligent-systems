---
title: "7.3 Mathematical construction"
description: Autoregressive maximum likelihood and teacher forcing, masked conditional reconstruction, contrastive classification and the InfoNCE bound, then Jaccard, MinHash, locality-sensitive hashing and directional n-gram contamination checks.
sidebar: { label: "7.3 Construction", order: 3 }
---

This section builds two pieces that are often taught separately. Sections 7.3.1–7.3.3 construct **learning objectives**. Sections 7.3.4–7.3.6 construct **data-audit estimators**. They belong together: a loss is an expectation over data, so an unexamined dataset silently changes the mathematical objective.

## 7.3.1 Autoregressive maximum likelihood

Let story $s$ contain tokens $x^{(s)}_{1:T_s}$. Add a begin symbol so the first token also has a context, and optionally an end symbol so sequence termination is modelled. An autoregressive model supplies one normalized distribution

$$
q_\theta(x^{(s)}_t\mid x^{(s)}_{<t})
$$

at every position. The token-weighted empirical loss is

$$
\hat L(\theta)
=-\frac{1}{\sum_s T_s}
\sum_s\sum_{t=1}^{T_s}
\ln q_\theta\!\left(x^{(s)}_t\mid x^{(s)}_{<t}\right).
$$

By the chain rule,

$$
\sum_{t=1}^{T_s}\ln q_\theta(x^{(s)}_t\mid x^{(s)}_{<t})
=\ln q_\theta(x^{(s)}_{1:T_s}),
$$

so the numerator is the corpus negative log-likelihood. Minimizing it is maximum likelihood for the declared sequence factorization. Equivalently, it minimizes a forward KL divergence from the **token-weighted empirical distribution** to the model, up to the empirical entropy constant (Unit 01, §1.3.3). If every story rather than every token should receive equal weight, the averaging rule must be changed; long stories currently contribute more targets.

### Teacher forcing is a computation rule

During training, the conditional at $t$ receives the true prefix $x_{<t}$. This is **teacher forcing**. A causal Transformer can process all positions in one call: the input contains the full shifted sequence, while the mask prevents position $t$ from reading its target or future tokens.

During generation, the prefix contains the model's earlier samples. A mistake can therefore create a prefix uncommon in training and alter later predictions. This train–generation context difference is often called **exposure bias**. It is a possible contributor to generation failures, not a complete explanation for repetition or hallucination; decoding, data, capacity and post-training also matter.

### Windows change the context distribution

Long streams are trained in length-$T$ windows. In the spine code, cleaned stories are joined with newline tokens and random 128-character windows are sampled. A window can start mid-story or cross a story boundary, so early positions have truncated context and the newline becomes the learned separator. The objective is therefore over sampled windows of this constructed stream, not an abstract collection of isolated stories.

For checkpoint monitoring, `model.evaluate` scores fixed non-overlapping windows of the concatenated held-out stream. For final uncertainty, Stage 10 scores each held-out story separately and bootstraps stories—the independent sampling unit. Both numbers can be useful, but they estimate differently weighted quantities.

## 7.3.2 Masked and denoising objectives

Let $M$ be a sampled set of hidden positions and let $\tilde x=C_M(x)$ be a corrupted sequence. A masked reconstruction loss is

$$
\hat L_{\text{mask}}(\theta)
=-\mathbb E_M\sum_{t\in M}
\ln q_\theta(x_t\mid \tilde x,M).
$$

BERT selects 15% of positions; among selected positions, 80% are replaced by `[MASK]`, 10% by a random token and 10% left unchanged ([Devlin et al. 2019](/references/#devlin-2019)). The target is always the original token. Random and unchanged cases reduce the train–use mismatch caused by a special mask symbol that never appears downstream.

If one position at a time were hidden and every full conditional $q(x_t\mid x_{-t})$ were trained, the product of those conditional terms would be a classical **pseudo-likelihood**. BERT's multi-position random corruption is better described as pseudo-likelihood-style conditional reconstruction: masked positions may hide information from one another, and its learned conditionals are not automatically guaranteed to be the full conditionals of one normalized joint distribution.

Only selected positions receive direct token losses in one corrupted view, whereas a shifted autoregressive window supplies a loss at almost every position. That is an accounting difference, not a universal data-efficiency ranking. Bidirectional context may make each masked prediction more informative, and different architectures, corruption rates and compute budgets change the comparison.

Two limiting cases expose the design:

- If almost nothing is masked, the task becomes easy and may teach little global structure.
- If almost everything is masked, context carries too little information and the model falls back toward marginal token frequencies.

The useful corruption rate lies between these extremes and depends on the object and target use.

## 7.3.3 Contrastive classification and the InfoNCE bound

Let $(X,Y)$ be a positive pair drawn from a joint distribution $p(x,y)$: two augmentations of one image, an image and its caption, or two related spans. Given $X=x$, draw

$$
Y_1\sim p(y\mid x),\qquad Y_2,ldots,Y_K\overset{\text{iid}}{\sim}p(y).
$$

There are $K$ candidates: one positive and $K-1$ negatives. A critic $f(x,y)$ scores each pair. The InfoNCE loss is the cross-entropy for identifying candidate 1:

$$
\mathcal L_{\text{NCE}}
=-\mathbb E\left[
\ln\frac{e^{f(X,Y_1)}}{\sum_{k=1}^{K}e^{f(X,Y_k)}}
\right].
$$

Often $f(x,y)=g(x)^\top h(y)/\tau$, where $g,h$ are encoders and $\tau>0$ is a temperature. Smaller $\tau$ sharpens score differences and gradients; it does not add information by itself.

### What mutual information means here

Mutual information is

$$
I(X;Y)
=\mathbb E_{p(x,y)}
\left[\ln\frac{p(x,y)}{p(x)p(y)}\right]
=\mathbb E\left[\ln\frac{p(Y\mid X)}{p(Y)}\right].
$$

It is zero when $X$ and $Y$ are independent and positive when observing one changes the distribution of the other. The log density ratio is exactly what distinguishes a true pair from a candidate drawn independently from $p(y)$.

**Proposition 7.1 (InfoNCE bound).** Under the sampling scheme above, for any measurable critic for which the expectations exist,

$$
I(X;Y)\ge \ln K-\mathcal L_{\text{NCE}}.
$$

*Proof, with the auxiliary distribution made explicit.* The negative variables are independent of $(X,Y_1)$, so

$$
I(X;Y_{1:K})=I(X;Y_1)=I(X;Y).
$$

The marginal distribution of $Y_{1:K}$ is $p(y_{1:K})=\prod_{k=1}^{K}p(y_k)$. For any normalized conditional density $r(y_{1:K}\mid x)$, non-negativity of KL gives the variational inequality

$$
I(X;Y_{1:K})
\ge \mathbb E\left[
\ln\frac{r(Y_{1:K}\mid X)}{p(Y_{1:K})}
\right].
$$

Choose

$$
r(y_{1:K}\mid x)
=p(y_{1:K})
\frac{e^{f(x,y_1)}}{\frac1K\sum_{k=1}^{K}e^{f(x,y_k)}}.
$$

Why is this normalized? Under the symmetric product measure $p(y_{1:K})$, each index has the same expected ratio. The average over all indices is identically

$$
\frac1K\sum_{i=1}^{K}
\frac{e^{f(x,y_i)}}{\frac1K\sum_k e^{f(x,y_k)}}=1,
$$

so each ratio integrates to 1. Substitution into the variational inequality gives

$$
I(X;Y)
\ge\mathbb E\left[
\ln\frac{e^{f(X,Y_1)}}{\frac1K\sum_k e^{f(X,Y_k)}}
\right]
=\ln K-\mathcal L_{\text{NCE}}.
\qquad\blacksquare
$$

The bound cannot certify more than $\ln K$ nats because $\mathcal L_{\text{NCE}}\ge0$. Thus $K=1{,}024$ candidates—one positive and 1,023 negatives—has a ceiling of $\ln1024=6.93$ nats or 10 bits. More candidates raise the ceiling but also cost memory and computation.

The assumptions matter. In-batch “negatives” may include another valid match, creating a **false negative**. Correlated batches need not be iid draws from $p(y)$. A high mutual-information lower bound also says only that the views share distinguishable information, not that the retained information is useful or causal downstream.

### Collapse and other ways to avoid it

A loss that only pulls positive embeddings together admits the constant representation $g(x)=c$ as a trivial solution. Contrastive negatives make that solution bad because all candidates then receive equal scores. They encourage positive **alignment** and separation or **uniformity** among other examples ([Wang & Isola 2020](/references/#wang-isola-2020)). Non-contrastive methods can avoid collapse using predictor asymmetry, stopped gradients, variance constraints or covariance penalties; “no negatives” does not mean “no anti-collapse mechanism” (§7.5.3).

## 7.3.4 From documents to sets: Jaccard similarity

Exact hashes find byte-identical or normalized-identical documents. Near duplicates need a representation that tolerates small edits. Tokenize a document, form every consecutive word 5-gram, and keep the set of resulting **shingles** $S$.

For nonempty sets,

$$
J(S_1,S_2)=\frac{|S_1\cap S_2|}{|S_1\cup S_2|}\in[0,1].
$$

$J=1$ means the shingle sets are identical; $J=0$ means they share none. Because these are sets, repeating the same phrase inside one document does not increase its weight. Using multisets or containment similarity would answer different questions.

Shingle length controls tolerance. Short shingles produce chance matches in formulaic text; long shingles miss heavier edits. A one-word edit affects several overlapping 5-grams, so its effect is larger than one divided by the word count but much smaller than replacing the document.

Comparing every pair of $n$ documents costs $n(n-1)/2=O(n^2)$ exact similarities. MinHash supplies a compact estimator; locality-sensitive hashing uses those estimates to avoid enumerating every pair.

## 7.3.5 MinHash, then banding

Imagine a uniformly random permutation of the universe of possible shingles. Define

$$
m_h(S)=\arg\min_{s\in S}h(s),
$$

the member of $S$ appearing first under that ordering. In practice, a seeded hash approximates a random ordering; independence and collision behaviour of the hash family are engineering assumptions.

**Proposition 7.2 ([Broder 1997](/references/#broder-1997)).** For nonempty finite sets,

$$
\Pr[m_h(S_1)=m_h(S_2)]=J(S_1,S_2).
$$

*Proof.* Look at the first element of $S_1\cup S_2$ in the random ordering. Every union element is equally likely to be first. The two MinHashes agree exactly when this first element belongs to the intersection. Therefore the probability is $|S_1\cap S_2|/|S_1\cup S_2|$. $\blacksquare$

With $n_h$ independent permutations, agreement indicators are Bernoulli variables with mean $J$. Their average $\hat J$ is unbiased and has

$$
\operatorname{SE}(\hat J)=\sqrt{\frac{J(1-J)}{n_h}}.
$$

The standard error describes the similarity estimate; Stage 07 uses the signature differently, to propose candidate pairs.

Split $n_h=br$ signature values into $b$ bands of $r$ rows. Hash every band. If all $r$ rows of one band agree, that band collides; if any band collides, propose the pair. Under independent rows and bands,

$$
P_{\text{cand}}(J)=1-(1-J^r)^b.
$$

For Stage 07, $b=16,r=4$. The curve is intentionally permissive: a candidate is **not yet a duplicate**. The code computes exact Jaccard for candidates and removes a document only when $J\ge0.7$. Banding false positives therefore cost comparisons; final false removals depend on the exact threshold and shingle representation. Banding false negatives remain possible because a truly similar pair may collide in no band.

## 7.3.6 Directional contamination checks

Deduplication asks whether two training documents represent the same content. Contamination asks a directional question: **how much of this evaluation item is already covered by training?** For held-out document $h$, define word-$n$-gram sets $G_n(h)$ and a training index $U_{\text{train}}=\bigcup_{d\in\text{train}}G_n(d)$. Stage 07 measures

$$
C_n(h)=\frac{|G_n(h)\cap U_{\text{train}}|}{|G_n(h)|}.
$$

The denominator is the held-out item, so $C_n(h)=1$ means every distinct held-out $n$-gram appears somewhere in training—even if no one training document contains the whole item. This differs from Jaccard, whose denominator is a union. It is also vulnerable to a “mosaic” false alarm in highly formulaic corpora.

The design variables are not cosmetic:

- **$n$:** small values detect loose reuse but collide by chance; large values demand near-verbatim overlap;
- **threshold:** lower catches more copies and more innocent formulaic text;
- **normalization:** case, punctuation, Unicode, numbers and whitespace change the $n$-grams;
- **aggregation:** matching against the union of all training grams is more sensitive but less attributable than requiring one matching document;
- **short items:** if a document has fewer than $n$ words, the statistic is undefined unless a fallback is declared.

GPT-3 reported a 13-gram contamination analysis ([Brown et al. 2020](/references/#brown-2020)); Stage 07 uses word 8-grams because its stories are short. In the synthetic corpus, the planted six copies have overlap 1.0, while the most-overlapping clean held-out story is below 0.68, so the frozen threshold 0.8 separates them. That separation is a property of this dataset, not a universal threshold.

The safe intervention is to remove or quarantine matching **training** items while preserving the frozen evaluation set. Removing difficult or suspicious evaluation cases after looking at model scores changes the estimand. Keep the pre-clean and post-clean counts, document identifiers, thresholds and reasons.

:::note[Claim check]
- **Claim:** low $C_n(h)$ proves that $h$ was never used in training.
- **What the statistic actually supports:** little near-verbatim word-$n$-gram coverage under the declared normalization and threshold.
- **Known misses:** paraphrases, translations, screenshots, reordered answers and semantically equivalent code.
- **Validation:** plant known copies and known clean items, then report recall and false positives. Passing that test scopes the detector; it does not prove complete coverage of unknown transformations.
:::

**Next:** [§7.4 Worked examples](/core/unit-07/worked-examples/) computes each objective and audit probability before §7.5 asks what their optima preserve.
