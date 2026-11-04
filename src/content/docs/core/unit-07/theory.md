---
title: "7.5 General theory"
description: Derive the population optimum of each self-supervised objective, identify the assumptions behind density-ratio learning, distinguish several kinds of collapse, and state when pretraining can and cannot support a downstream task.
sidebar: { label: "7.5 General theory", order: 5 }
---

The constructions in §7.3 tell us how to compute the losses. This section asks a different question: **if data, model capacity and optimization were ideal, what would each loss ask the model to learn?** That question gives a reference point, not a promise about a finite trained network.

Labels used below: **[D]** derivation, **[E]** experimental evidence, **[O]** engineering observation, **[H]** open hypothesis.

## 7.5.1 The population optimum of each objective

Start with a visible variable $V$, a target $Y$ and a predictor $q(y\mid v)$. Its population log-loss is

$$
L(q)=\mathbb E_{p(v,y)}[-\ln q(Y\mid V)].
$$

Add and subtract $-\ln p(Y\mid V)$ inside the expectation:

$$
L(q)
=H_p(Y\mid V)
+\mathbb E_{p(v)}\operatorname{KL}\!\left(
p(\cdot\mid v)\,\|\,q(\cdot\mid v)
\right).
$$

The conditional entropy is fixed by the data, and KL divergence is non-negative. Therefore an unrestricted predictor minimizes the loss at

$$
q^\star(\cdot\mid v)=p(\cdot\mid v)
$$

for every $v$ that has positive probability **[D]**. This is the conditional version of Unit 01's proper-scoring-rule result. Three qualifications matter:

1. it is a **population** statement about the declared data and corruption distributions;
2. a finite model may not represent every conditional, and optimization may not find its best representable one;
3. the result says what distribution is fitted, not whether that distribution is truthful, useful or safe.

Now substitute each self-supervised construction for $(V,Y)$.

| Objective | Visible variable and target | Ideal optimum | What is identified |
|---|---|---|---|
| autoregressive | $V=X_{<t},\ Y=X_t$ | $q^\star(x_t\mid x_{<t})=p(x_t\mid x_{<t})$ | the joint sequence distribution through the chain rule |
| masked / denoising | $V=(\widetilde X,M),\ Y=X_M$ | the true conditional of the hidden content for each sampled corruption | the conditionals queried by that corruption process |
| contrastive | $V=(X,Y_{1:K}),\ Y=$ index of the positive | Bayes posterior over the positive index | a density ratio relative to the negative distribution |

For autoregressive modelling, multiplying the optimal normalized conditionals gives

$$
\prod_{t=1}^{T}p(x_t\mid x_{<t})=p(x_{1:T}).
$$

So, with a specified start and termination convention, the optimum defines a normalized generative model. This is an identity about the factorization; it does not say a learned finite model reaches the optimum.

For masked modelling, the optimum is narrower. It recovers the hidden-token conditionals actually requested by the mask distribution. A collection of approximate conditionals need not be mutually compatible with one normalized joint distribution. Even exact full conditionals determine a unique joint only under regularity conditions such as common positive support; Problem 7.12 examines the failure case. Thus “BERT learns the joint distribution” is not a consequence of masked loss alone.

For contrastive learning, suppose one candidate is sampled from $p(y\mid x)$ and the other $K-1$ candidates independently from the marginal $p(y)$. Bayes' rule gives, for candidate $i$,

$$
P(I=i\mid x,y_{1:K})
=\frac{p(y_i\mid x)/p(y_i)}
{\sum_j p(y_j\mid x)/p(y_j)}.
$$

A softmax critic matches this posterior when

$$
f^\star(x,y)=\ln\frac{p(y\mid x)}{p(y)}+c(x),
$$

where $c(x)$ cancels from every softmax score **[D]**. The ratio answers a comparison question: how much more plausible is $y$ after seeing $x$ than under the background population? For example, changing a label probability from $0.5$ to $0.8$ gives a log density ratio $\ln(0.8/0.5)=0.47$ nats. It does **not** recover $p(y)$ by itself.

The denominator depends on how negatives are drawn. If negatives come from $\nu(y)$ rather than $p(y)$, the ideal score is $\ln[p(y\mid x)/\nu(y)]+c(x)$. Correlated batches, hard-negative mining and false negatives therefore change either the statistical problem or the validity of the simple derivation.

## 7.5.2 One scalar loss mixes many prediction tasks

A next-token model does not practise one homogeneous skill. Every context defines another supervised classification problem: after “Now Ben has” the missing information may be arithmetic; after “went to the” it may be a location; after “called K” it may require copying a name from far earlier.

Writing $a$ for a task type, the average gradient can be decomposed schematically as

$$
\nabla L(\theta)
=\sum_a P(A=a)\,
\mathbb E[\nabla\ell_\theta\mid A=a].
$$

The first factor is **frequency**; the second contains current difficulty, model confidence and gradient geometry. A rare task may have a large loss per occurrence and still contribute little to the total because it appears so seldom. Conversely, an easy token type can matter in aggregate because it occurs millions of times.

Consider a deliberately simple accounting example. Suppose 99% of targets have loss $0.20$ nats and 1% arithmetic targets have loss $3.0$ nats. Their contributions to mean loss are

$$
0.99(0.20)=0.198,
\qquad
0.01(3.0)=0.030.
$$

Halving the common-token loss improves the mean by $0.099$ nats. Solving arithmetic perfectly improves it by only $0.030$ nats. Average NLL can therefore fall substantially while arithmetic remains unreliable. This is not a defect in cross-entropy: it is exactly the weighting implied by the corpus and averaging rule.

Three design choices follow:

- report skill-specific evaluations rather than treating average loss as a complete ability score;
- change sampling or loss weights when the desired deployment mixture differs from the corpus mixture;
- check whether tokenization and windowing make the intended evidence visible at the target position.

Reweighting is not free: emphasizing one family of targets changes the estimand and may reduce performance on the original mixture. Unit 10 therefore evaluates capabilities separately instead of inferring them from validation loss.

## 7.5.3 Collapse is not one phenomenon

“Collapse” is used for several failures. Keeping them separate prevents a slogan from replacing a diagnosis.

### Constant-representation collapse

For a pure agreement loss

$$
L_{\text{agree}}=\mathbb E\|g(V_1)-g(V_2)\|^2,
$$

the constant encoder $g(v)=c$ achieves zero loss. The objective asked only for agreement, so it never required different items to remain distinguishable **[D]**.

Contrastive learning changes this. If every representation is constant, all $K$ candidates receive the same score and InfoNCE loss is $\ln K$. Whenever positive pairs contain usable dependence, a non-constant critic can do better. Negatives thus provide an explicit pressure against constant collapse and encourage both positive-pair alignment and population spread ([Wang & Isola 2020](/references/#wang-isola-2020)).

### Dimensional and shortcut collapse

An encoder can avoid being literally constant yet use only a few dimensions, or encode an easy nuisance feature rather than the semantics we wanted. Negatives alone do not rule these out. Variance floors and covariance penalties can keep dimensions active; careful augmentations and evaluations are needed to reveal shortcuts.

Reconstruction also supplies anti-collapse pressure only to the extent that the target varies predictably with the representation. A constant bottleneck cannot reconstruct input-dependent targets unless the decoder receives a side channel. But if the visible input contains no information about the target, the optimal predictor is precisely the target marginal—an input-independent output. Consequently, “the target is data” is not an unconditional proof that every internal representation is informative.

Asymmetric non-contrastive methods add stopped gradients, a predictor, a slowly updated target network or explicit variance/covariance constraints. BYOL, for example, is empirically non-collapsing under its full training recipe ([Grill et al. 2020](/references/#grill-2020)) **[E]**. A stopped gradient by itself is not a theorem that collapse is impossible; the explanation involves architecture and optimization dynamics **[H]**.

## 7.5.4 When can pretraining help a downstream task?

Let $Z=g(X)$ be a learned representation and $Y$ a downstream target. The clean ideal is **sufficiency**:

$$
p(y\mid x)=p(y\mid z).
$$

If this holds, replacing $X$ by $Z$ loses no information needed for Bayes-optimal prediction of $Y$. But a pretraining objective mentions its own targets, not every future $Y$, so sufficiency for an unseen task is not automatic.

A counterexample makes the gap concrete. Suppose two augmented views share both object identity and a unique background watermark. Matching views by watermark can solve the contrastive pretext task. If the downstream label is the object's species and watermarks change at deployment, that representation may transfer poorly despite excellent pretraining loss. The objective rewarded shared information, but not necessarily the shared information we cared about.

Transfer is favoured when four alignments hold:

1. **information:** the pretrained state retains variables predictive of $Y$;
2. **access:** the downstream learner—perhaps only a linear probe—can extract them;
3. **distribution:** the relationship survives the shift from pretraining to deployment;
4. **sample and optimization budget:** adaptation has enough evidence and can find a useful solution.

These distinctions explain why “frozen linear probe,” “full fine-tuning” and “prompting” answer different questions. A fact may be present but not linearly accessible; fine-tuning may create a new feature rather than merely reveal an old one. For language modelling, a task directly expressible as likely text continuation is well aligned with the pretraining interface, but reliability still depends on the prompt distribution and on whether the corpus conditional is the desired answer conditional.

Idealized contrastive theory can prove transfer when positive pairs arise from latent classes aligned with downstream labels. That is a conditional theorem, not a universal account of natural data **[D]/[H]**. Broad transfer is ultimately established by held-out experiments, including distribution-shift tests **[E]**.

## 7.5.5 ★ Deduplication changes the estimand

Let the collected corpus contain $N$ documents and let unique content class $i$ occur $m_i$ times. Ordinary empirical risk gives that class weight

$$
w_i^{\text{raw}}=\frac{m_i}{N}.
$$

Exact deduplication followed by uniform sampling over the $U$ retained classes instead gives

$$
w_i^{\text{dedup}}=\frac{1}{U}.
$$

The loss, gradient and maximum-likelihood target have all changed **[D]**. Deduplication is therefore not merely a way to save compute.

Which weighting is better cannot be answered from duplication counts alone:

- a page mirrored by 100 scraper domains is probably a collection artefact, so raw weighting exaggerates it;
- a safety warning repeated by many independent sources may reflect real deployment frequency or social importance;
- repeated practice examples can intentionally improve a narrow skill, even though they reduce corpus diversity.

Near-duplicate removal adds more decisions: the shingle definition, similarity threshold, graph clustering rule and retained representative. These can discard a newer correction while keeping an older copy unless provenance is consulted. A reproducible pipeline records those choices rather than reporting only “we deduplicated.”

The evaluation split creates a stricter rule. Split membership must be frozen first; if a training item overlaps a held-out item, remove or quarantine the **training** copy. Deleting the held-out example after observing overlap changes the test set and can make the reported score easier. Deduplication within training and decontamination against evaluation are related audits, but they answer different questions.

## 7.5.6 What has—and has not—been established

| Result | Established conclusion | Boundary |
|---|---|---|
| conditional log-loss optimum | an unrestricted population predictor recovers the declared conditional **[D]** | finite data, model class and optimization can miss it |
| autoregressive factorization | optimal next-token conditionals form a joint distribution **[D]** | the learned joint reproduces data errors as well as truths |
| masked optimum | recovers queried corruption conditionals **[D]** | compatibility with one joint is not automatic |
| contrastive optimum | recovers a ratio relative to the negative distribution **[D]** | does not recover the marginal or guarantee useful semantics |
| InfoNCE bound | $I(X;Y)\ge\ln K-\mathcal L_{\mathrm{NCE}}$ under its sampling assumptions **[D]** | certificate is capped at $\ln K$ and can track nuisance information |
| anti-collapse mechanisms | prevent particular trivial solutions under stated conditions **[D]/[E]** | do not by themselves guarantee downstream sufficiency |
| transfer | succeeds when useful information is retained, accessible and stable **[H]/[E]** | pretraining loss alone is not evidence of downstream reliability |
| deduplication | changes empirical weights and compute use **[D]** | whether the new target is preferable depends on provenance and deployment |

The common thread is simple: a self-supervised loss learns exactly from the statistical distinctions its construction exposes. The hard engineering and scientific work is deciding whether those distinctions match the behaviour we eventually need.
