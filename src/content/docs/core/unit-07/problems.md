---
title: "7.10 Problems and further study"
description: Twenty graded problems preserve the unit's mathematical depth while adding starting points, checkpoints and common traps for independent study.
sidebar: { label: "7.10 Problems", order: 10 }
---

Difficulty: ★ routine · ★★ combines ideas · ★★★ proof or open-ended. **[AI-off]** marks problems intended for independent practice or assessment.

Use the scaffolds in three passes: first try without them, then read **Start**, and only then use **Check**. A check tells you the shape or boundary of an answer, not the complete derivation.

## Computation

### 7.1 ★ [AI-off] From token probabilities to sequence probability

A model assigns probabilities $0.2,0.5,0.5,0.9$ to the four observed characters of a word, each conditioned on its prefix. Compute (a) the four per-character losses in nats, (b) their mean, (c) the probability assigned to the whole word and (d) its total NLL.

**Start.** Use $\ell_t=-\ln p_t$ and the autoregressive product rule. **Check.** The total NLL must equal four times the mean NLL, up to rounding.

### 7.2 ★ [AI-off] Jaccard and one MinHash

Two documents have 40 distinct shingles each and share 30. Compute their Jaccard similarity and the probability that one ideal MinHash value agrees.

**Start.** Draw two overlapping circles; the denominator is the union, not $40+40$. **Common trap.** “Share 30” means intersection size 30, so subtract it once from the sum.

### 7.3 ★ From Jaccard to LSH candidates

For $b=20$ bands of $r=5$ rows, compute $P_{\mathrm{cand}}$ at $J=0.5,0.7,0.9$. Also compute the common heuristic location $J_0=(1/b)^{1/r}$ and the candidate probability at $J_0$.

**Start.** One band misses with probability $1-J^r$; all bands miss with its $b$th power. **Check.** $J_0$ is not the exact 50% point or exact inflection point. At large $b$, its candidate probability is near $1-e^{-1}\approx0.632$.

### 7.4 ★ Precision of a MinHash similarity estimate

With 64 independent hash functions, what is the standard error of $\hat J$ when $J=0.5$? How many hash functions are required to halve that standard error? What changes if the hash rows are correlated?

**Start.** Treat agreement indicators as Bernoulli variables. **Check.** Standard error scales as $1/\sqrt{n_h}$, so halving it requires four times as many independent rows.

### 7.5 ★★ Reading an InfoNCE certificate

An InfoNCE run uses $K=4096$ candidates per classification problem and reports loss $2.0$ nats. Compute the mutual-information lower bound in nats and bits. List two sampling assumptions required before reporting it as a certificate.

**Start.** Apply $I(X;Y)\ge\ln K-\mathcal L$ and divide nats by $\ln2$ for bits. **Common trap.** A batch of 4,096 examples gives $K=4096$ only if the construction really supplies one positive and 4,095 valid marginal negatives for each anchor.

### 7.6 ★★ Duplicate weights

A raw corpus has $N$ records. Content class $A$ occurs 1,000 times and content class $B$ once. Compute $w_A/w_B$ under uniform record sampling. After exact deduplication, suppose one representative of each of $U$ unique classes is sampled uniformly; compute the new ratio. Explain why neither weighting is automatically the deployment distribution.

**Start.** Write $w_i=m_i/N$ before deduplication and $w_i=1/U$ afterward. **Extension.** What changes if the pipeline caps every class at at most ten copies instead of one?

## Derivation

### 7.7 ★ [AI-off] Chain rule and sequence NLL

Prove that the sum of next-token log-losses equals the negative log-likelihood of the sequence under an autoregressive model. Include start and, if modelled, end-of-sequence symbols in your statement.

**Start.** Write the probability chain rule first, then take $-\ln$ of both sides. **Check.** The claim is an identity for any normalized autoregressive model; it does not require the model to match the data distribution.

### 7.8 ★★ [AI-off] MinHash collision probability

Prove Proposition 7.2 for two nonempty finite shingle sets under a uniformly random permutation of their union. Then state which step becomes an approximation when a practical seeded hash replaces the permutation.

**Start.** Focus on the first element of $S_1\cup S_2$ in the random order. **Check.** Agreement occurs exactly when that element lies in the intersection.

### 7.9 ★★ The LSH curve and its exact inflection

Derive $P(J)=1-(1-J^r)^b$. For $r>1$, differentiate twice and show that its interior inflection point is

$$
J_{\mathrm{infl}}=left(\frac{r-1}{br-1}\right)^{1/r}.
$$

Compare it with the heuristic $(1/b)^{1/r}$ and the exact 50% point $[1-2^{-1/b}]^{1/r}$.

**Start.** It is easier to differentiate $P'(J)=brJ^{r-1}(1-J^r)^{b-1}$ by taking its logarithm. **Common trap.** “Threshold” is informal; these three locations answer different questions.

### 7.10 ★★ Bayes-optimal contrastive critic

One of $K$ candidates is drawn from $p(y\mid x)$ and the rest independently from $p(y)$; the positive index is uniform before observing candidates. Derive its posterior and show that a softmax critic can represent it with

$$
f^\star(x,y)=\ln\frac{p(y\mid x)}{p(y)}+c(x).
$$

Repeat the final line when negatives come from $\nu(y)$.

**Start.** Write the joint density under the hypothesis “index $i$ is positive,” cancel factors common to every index, then normalize. **Check.** Any term depending only on $x$ cancels from softmax.

### 7.11 ★★ InfoNCE gradients

For scores $s_1,ldots,s_K$ with candidate 1 positive, let

$$
\ell=-s_1+\log\sum_{k=1}^{K}e^{s_k}.
$$

Derive $\partial\ell/\partial s_1$ and $\partial\ell/\partial s_j$ for $j>1$. Explain how a hard negative changes the gradient, and why the gradients sum to zero.

**Start.** Define $\pi_k=e^{s_k}/\sum_j e^{s_j}$. **Check.** The positive derivative is $\pi_1-1$; every negative derivative is non-negative.

## Proof and counterexample

### 7.12 ★★★ Incompatible full conditionals

Let $X,Y\in\{0,1\}$. Consider

$$
P(X=1\mid Y=0)=0.2,
\quad P(X=1\mid Y=1)=0.8,
$$

and

$$
P(Y=1\mid X=0)=0.2,
\quad P(Y=1\mid X=1)=0.2.
$$

Prove that no strictly positive joint distribution has all four conditionals. Then state a positivity-and-compatibility condition under which full conditionals determine a joint, and explain the role of Brook's lemma.

**Start.** Each pair of binary conditionals implies an odds ratio for the $2\times2$ joint table. Compute it once by conditioning $X$ on $Y$ and once by conditioning $Y$ on $X$. **Check.** The two implied odds ratios disagree.

### 7.13 ★★ Normalize the InfoNCE auxiliary distribution

Complete the proof of Proposition 7.1 by verifying that

$$
r(y_{1:K}\mid x)=
\left[\prod_{k=1}^{K}p(y_k)\right]
\frac{e^{f(x,y_1)}}{K^{-1}\sum_k e^{f(x,y_k)}}
$$

integrates to 1. Clearly state the symmetry argument and then substitute it into the variational mutual-information inequality.

**Start.** Define the same ratio with each candidate $i$ in the numerator. Their average is identically one. Under the exchangeable product measure, all $K$ integrals are equal.

### 7.14 ★★ What negatives rule out—and when

(a) Prove that $g(v)=c$ globally minimizes the positive-pair alignment loss $\mathbb E\|g(V_1)-g(V_2)\|^2$.

(b) Show that a constant critic gives InfoNCE loss $\ln K$.

(c) For $K=2$ and $X=Y$ uniformly distributed on $\{0,1\}$, construct scores that give expected loss below $\ln2$.

(d) Give a counterexample showing that negatives do **not** guarantee a non-constant optimum when $X$ and $Y$ are independent.

**Start.** In (c), score matching binary values higher than mismatches and let the score gap grow. In (d), use the fact that the positive is statistically indistinguishable from the negative. **Check.** The valid conclusion is conditional on exploitable dependence, not “InfoNCE always forbids collapse.”

### 7.15 ★★★ When greedy decoding reproduces a sequence

Let a training sequence be $x_{1:T}$. Suppose there is an index $m$ such that for every $t>m$,

$$
q(x_t\mid x_{<t})>q(a\mid x_{<t})
\quad\text{for every }a\ne x_t.
$$

Prove by induction that greedy decoding from prefix $x_{1:m}$ reproduces $x_{m+1:T}$. Then explain why “the model assigned high likelihood to the sequence” alone is insufficient, and characterize a prefix property that can make a small $m$ sufficient.

**Start.** The induction must show that the generated prefix remains the training prefix at every step. **Common trap.** High product probability does not imply the observed token is the unique argmax at every position.

## Implementation and design

### 7.16 ★ [AI-off] Implement MinHash as a two-stage detector

From memory, implement normalization, 5-word shingling, exact hashing, MinHash signatures, banding and exact Jaccard confirmation. Reproduce the canonical manifest audit in Experiment 7.1.

**Required tests.** Empty/short documents, identical documents, one-token edit, deterministic seeded output, candidate-versus-final-decision separation and a planted pair missed by an intentionally weak banding configuration.

### 7.17 ★★ Attack a contamination detector

Implement a directional $n$-gram contamination check robust to case, whitespace and punctuation changes. Construct three attacks: paraphrase, translation and table reformatting. Measure both overlap and semantic preservation, then propose one provenance-based and one model-based complement.

**Checkpoint.** Do not call a successful attack “proof of no contamination.” It is evidence that the detector's target class is narrower than the leakage class.

### 7.18 ★★ Implement a masked-character objective

Add a bidirectional masked-character training path for the spine architecture and run Experiment 7.5. Preserve the original path. Add tests that masked targets cannot be copied directly, unselected positions do not contribute target loss and evaluation never tunes on the final held-out set.

**Report.** Parameter count, updates, total predicted targets, wall-clock time, native losses, shared last-character NLL and probe baselines. Explain which differences remain confounded by the attention mask.

### 7.19 ★★ Design a story-and-arithmetic mixture

Design a mixture for a spine model that must both tell coherent stories and give correct sums. Specify source distributions, weights $\alpha_j$, a fixed token budget and separate held-out metrics. Include one experiment with static weights and one with a schedule $\alpha_j(t)$.

**Start.** Draw a Pareto plot with story NLL on one axis and arithmetic accuracy/calibration on the other. **Check.** Choosing the run after looking at the test set invalidates the comparison; reserve validation data for weight selection.

### 7.20 ★★★ Does deduplication improve generalization or only efficiency?

Design a study that separates three claims:

1. deduplication reaches a target validation NLL in fewer updates;
2. at fixed updates, it improves performance on clean in-distribution items;
3. at fixed updates, it improves transfer to new templates or domains.

State the raw and deduplicated conditions, compute-matching rule, split and decontamination order, checkpoint rule, per-item metrics, seed protocol and uncertainty estimate. Include a result pattern that would support each claim and one pattern that would refute it.

**Start.** One learning curve can answer efficiency, but generalization requires frozen clean and shifted test sets. **Common trap.** If deduplication also changes vocabulary, total tokens or held-out membership, the treatment is no longer isolated.

## Selected answers and checkpoints

- **7.1:** losses $1.609,0.693,0.693,0.105$ nats; mean $0.775$; sequence probability $0.045$; total NLL $3.101$ nats.
- **7.2:** union size $40+40-30=50$, so $J=30/50=0.6$ and one ideal MinHash agrees with probability $0.6$.
- **7.3:** $1-(1-J^5)^{20}$ gives approximately $0.470,0.975,0.99999998$; $J_0=20^{-1/5}\approx0.549$, where $P_{\mathrm{cand}}=1-(1-1/20)^{20}\approx0.642$.
- **7.4:** $\sqrt{0.5(0.5)/64}=0.0625$; 256 independent rows halve it. Positive correlation reduces the effective sample size.
- **7.5:** $\ln4096-2=6.318$ nats, or about $9.11$ bits, provided negatives follow the declared marginal/iid construction and false negatives are handled as assumed.
- **7.6:** raw ratio $w_A/w_B=1000$; after one-per-class uniform sampling it is $1$. The deployment ratio requires a separate target-population argument.
- **7.9 checkpoint:** for $b=20,r=5$, $J_{\mathrm{infl}}=(4/99)^{1/5}\approx0.526$; it need not equal either $0.549$ or the 50% point.
- **7.11 checkpoint:** gradients sum to $(\pi_1-1)+\sum_{j>1}\pi_j=0$, reflecting invariance to adding one constant to every score.

## Further reading: read each source with a question

**Objectives.** Read [Radford et al. 2018](/references/#radford-2018) and [Devlin et al. 2019](/references/#devlin-2019) to compare which context each objective reveals. Read [van den Oord et al. 2018](/references/#oord-2018) beside [Poole et al. 2019](/references/#poole-2019) and ask which mutual-information claim follows from each bound.

**Data experiments.** [Lee et al. 2022](/references/#lee-2022) asks what deduplication changes; [Penedo et al. 2024](/references/#penedo-2024) and [Li et al. 2024](/references/#dclm-2024) ask how to make data decisions controlled and reproducible; [Broder 1997](/references/#broder-1997) supplies the compact MinHash derivation.

**Representations.** Compare pixel reconstruction with I-JEPA's target-space prediction ([Assran et al. 2023](/references/#assran-2023)), then identify the extra action-conditioned stage in V-JEPA 2 ([Assran et al. 2025](/references/#assran-2025)).

**Risks.** [Carlini et al. 2021](/references/#carlini-2021) and [Carlini et al. 2023](/references/#carlini-2023) operationalize extraction and memorization. [Shumailov et al. 2024](/references/#shumailov-2024) studies recursive generated-data replacement; read it while asking which result does—and does not—apply to synthetic augmentation.
