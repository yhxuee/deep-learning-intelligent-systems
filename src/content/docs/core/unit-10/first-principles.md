---
title: "10.2 First principles"
description: Build an evaluation claim from its estimand, sampling unit, population, metric and uncertainty; then see how shift, leakage, aggregation and miscalibration break that claim.
sidebar: { label: "10.2 First principles", order: 2 }
---

## 10.2.1 From one result to one estimand

Suppose a model is evaluated on items $X_1,\ldots,X_n$. A metric function $m(f,X_i)$ converts the model $f$ and item $X_i$ into a number. The test average is

$$
\hat S_n=\frac1n\sum_{i=1}^n m(f,X_i).
$$

This number is an **estimate**. The population quantity it aims to estimate is the **estimand**

$$
S_q(f)=\mathbb E_{X\sim q}[m(f,X)],
$$

where $q$ is the evaluation population. The distinction is small in notation and enormous in reasoning:

- $\hat S_n$ changes if another finite sample is drawn;
- $S_q$ changes if the population $q$, model $f$ or metric $m$ changes;
- neither one automatically equals the deployment objective.

For example, “accuracy = 0.82” omits nearly everything. “This checkpoint answered 328 of 400 fixed arithmetic prompts correctly under greedy decoding” is a sample fact. “Its accuracy on future prompts drawn by the same generator is about 0.82” is a population inference. “It is a reliable reasoner” is a much broader claim that the experiment does not identify.

### The metric's sign and unit

Before interpreting a number, ask:

1. Is higher or lower better?
2. Is it a sum, a mean or a quantile?
3. What is the unit—errors per prompt, seconds, dollars, nats per character?
4. What was averaged first?

That last question matters whenever item sizes differ. If story $i$ has $L_i$ characters and total loss $T_i$, two legitimate quantities are

$$
\text{macro story loss}
=\frac1n\sum_i\frac{T_i}{L_i},
$$

and

$$
\text{micro character loss}
=\frac{\sum_iT_i}{\sum_iL_i}.
$$

The first gives every story equal weight; the second gives long stories more weight. They answer different questions. Stage 10 uses one per-story mean as its resampling unit because stories were independently generated, while still reporting the loss in nats per character.

## 10.2.2 Name the independent sampling unit

The familiar standard-error rule $\text{SD}/\sqrt n$ assumes the $n$ contributions are independent, or at least satisfy conditions that make their dependence negligible. Tokens within one story are not independent samples from deployment:

- they share the same characters and template;
- later tokens are conditioned on earlier ones;
- one unusual story can make many adjacent token losses large.

If 140 stories contain 42,000 characters, using $n=42{,}000$ in a simple independent-item formula can create false precision. The effective information is closer to the number and diversity of stories than to the raw token count.

Use the level at which sampling or assignment occurred:

| application | often appropriate independent unit | common wrong unit |
|---|---|---|
| language-model benchmark | prompt, document or conversation | each token |
| medical study | patient or clinic | each image slice from one patient |
| tutoring system | student or classroom | each click |
| recommender | user or session | each displayed item |
| repeated experiment | training seed or full run | every checkpoint from one run |

When dependence is hierarchical—patients inside hospitals, prompts inside topics—reporting uncertainty at only the lowest level can still be optimistic. A clustered or hierarchical analysis resamples the level about which the claim is made.

## 10.2.3 Failure 1: the wrong population

A held-out sample drawn from training-like distribution $q$ estimates $S_q$. Deployment may follow $p_{\mathrm{deploy}}\ne q$.

This is **distribution shift**. It is not one mysterious event; ask what changed:

- input features or styles;
- label prevalence;
- the rule mapping input to correct output;
- language, location, time, hardware or user behaviour;
- the mix of easy and hard cases.

A model validated at one hospital can fail at another because scanner types and patient populations differ ([Zech et al. 2018](/references/#zech-2018)). Models are vulnerable when a feature is a useful shortcut in training but unstable across environments ([Geirhos et al. 2020](/references/#geirhos-2020)). In the spine corpus, a finite name list is such a shortcut.

### A shifted input is not automatically a failed model

There are two separate questions:

1. **Did the input distribution change?** A two-sample test or domain classifier can help.
2. **Did the model's relevant performance change?** This requires labels, outcomes or another justified evaluation on the new population.

A shift detector can flag bright images becoming darker, but it cannot by itself say whether diagnostic accuracy fell. Conversely, performance can fall under a subtle concept change that an input-only detector misses. Detection is a trigger for investigation, not the final reliability verdict.

### Build a test to break one assumption

“Test on diverse data” is too vague to falsify anything. Write a specific invariance claim:

> Changing a person's name while keeping the story's arithmetic and syntax fixed should not substantially change loss outside the name itself.

Then intervene on that feature. The unseen-name set is useful because it has a known contrast. If every aspect changes at once, a failure is visible but its cause is hard to localize.

## 10.2.4 Failure 2: the test influenced the system

A test set is evidence only if its answers did not leak into the fitted system or the process that selected it. Leakage has several timescales.

### Before training: data leakage

- exact or near-duplicate test items occur in training;
- examples from one patient, user or story are split across both sides;
- normalization, vocabulary selection or feature extraction is fitted on all data before splitting;
- a feature encodes the target through collection procedure rather than the intended signal.

The safe order is **split by independent unit first, then fit every learned preprocessing step on training only**.

### During development: adaptive test reuse

Suppose twenty model variants are tried and the one with best test score is reported. Even if no gradient touched the test set, test information selected the model. The final score is optimistically biased because it is the maximum of noisy estimates.

The roles should be distinct:

- training data fit parameters;
- validation data choose hyperparameters and stopping rules;
- a final test set supports one bounded claim after choices are frozen.

After repeated public use, a benchmark behaves more like validation data for the research community. A new hidden or post-cutoff set can test whether progress transfers.

### “Too good” is a clue, not a proof

Unexpectedly high performance should trigger provenance and leakage checks, but genuinely easy tasks and strong models also produce high scores. Likewise, an ordinary score does not prove cleanliness. Data lineage, overlap analysis and split logic are evidence; surprise alone is not.

## 10.2.5 Failure 3: the metric measures the wrong quantity

Metrics compress behaviour. Compression discards information.

Accuracy turns every item into 0 or 1. It treats a harmless typo and a dangerous medical recommendation as equally wrong. Mean loss gives frequent easy events more influence than rare skills. A judge score may combine factuality, style and agreement without showing which changed. A benchmark can therefore improve while a deployment-relevant failure worsens.

### Disaggregate before averaging

For each important dimension, report both its prevalence and performance:

| dimension | useful split |
|---|---|
| difficulty | routine, compositional, adversarial |
| user or context | language, locale, accessibility need |
| outcome severity | harmless error, recoverable error, high-stakes error |
| skill | recall, calculation, refusal, citation |
| confidence | low-, middle- and high-confidence cases |

An overall score is then a weighted average:

$$
S=\sum_g P(G=g)S_g.
$$

If deployment group weights differ from the benchmark's, the same subgroup performances yield a different overall score. Always reporting only $S$ hides both the subgroup values $S_g$ and the weights $P(G=g)$.

### Metric and decision are different

A probability forecast is not yet a decision. If a false negative costs $C_{\mathrm{FN}}$ and a false positive costs $C_{\mathrm{FP}}$, the appropriate threshold depends on those costs and prevalence. A model with slightly worse average NLL may produce better decisions under the relevant cost function. Evaluation should include the operational rule, not only the upstream score.

## 10.2.6 Failure 4: uncertainty is ignored

Even after choosing the right population and metric, the test result is random. If another valid sample were drawn, the result would differ.

There are at least two sources:

1. **test-sample uncertainty:** a fixed model is evaluated on a finite sample;
2. **training-run uncertainty:** the training procedure produces a different model under another seed, data order or hardware path.

Stage 10's per-story bootstrap addresses the first for one fixed checkpoint. It does not make a claim about the average performance of the training method over seeds. Unit 12 adds that level.

### What a 95% interval says

In the usual frequentist reading, a 95% confidence-interval procedure covers the fixed estimand in 95% of repeated samples under its assumptions. After one interval is observed, it is not literally a 95% posterior probability that the fixed estimand lies inside—unless a Bayesian model has separately been specified.

The interval also inherits the evaluation population. A narrow interval around $S_q$ can be precise and irrelevant to $S_{\mathrm{deploy}}$. More test items reduce sampling uncertainty; they do not repair shift, leakage or the wrong metric.

### Pair whenever the question is a difference

If models A and B see the same items, compute

$$
d_i=m(f_A,X_i)-m(f_B,X_i)
$$

and form an interval for $\mathbb E[d_i]$. This asks the comparison directly. Separate intervals for A and B ignore their positive covariance on shared easy and hard items. “The intervals overlap” is not the correct test of whether their paired difference is zero.

## 10.2.7 Calibration is a conditional claim

Let $C$ be a reported confidence and $Y=1$ indicate that the corresponding prediction is correct. Perfect top-label calibration means

$$
P(Y=1\mid C=c)=c
$$

for relevant confidence values $c$.

This is not the same as

$$
\mathbb E[C]=\mathbb E[Y].
$$

The latter is only one average equality. In a multiclass task with at least five classes, consider four predictions:

| confidence | correct? |
|---:|---:|
| 0.8 | 0 |
| 0.8 | 0 |
| 0.2 | 1 |
| 0.2 | 1 |

Mean confidence and accuracy are both 0.5, yet the model is wrong whenever confident and right whenever unsure. Aggregate agreement hides conditional failure.

A **reliability table** groups cases by confidence and compares mean confidence with observed accuracy in each group. **Expected calibration error (ECE)** averages absolute bin gaps ([Guo et al. 2017](/references/#guo-2017)). It is useful but not canonical:

- changing bin edges changes ECE;
- small bins have noisy observed accuracies;
- a large easy bin can dominate the average;
- top-label calibration ignores probabilities assigned to non-top classes;
- calibration on one population need not survive shift.

Therefore report the table or diagram, sample counts and binning rule—not only ECE.

### Calibrated does not mean correct

A weather model predicting rain with probability 0.1 every day is calibrated if it rains on 10% of those days, but it may have little resolution: it does not distinguish rainy from dry days. Likewise, the spine arithmetic model can be calibrated about being wrong often. Calibration makes probabilities interpretable; it does not create skill.

## 10.2.8 Evaluation is a chain, not a leaderboard

The complete chain is

$$
\text{deployment question}
\to
\text{population and units}
\to
\text{metric and decision rule}
\to
\text{sample and split}
\to
\text{estimator and uncertainty}
\to
\text{bounded conclusion}.
$$

A failure early in the chain cannot be repaired by sophistication later. A perfect bootstrap of a contaminated sample gives a precise estimate of contaminated performance. A beautiful reliability diagram on the wrong population gives calibrated evidence for the wrong population.

Use a **claim ledger** for important results:

| field | question to answer |
|---|---|
| claim | exactly what conclusion is proposed? |
| estimand | what population quantity would support it? |
| sample | how were units selected and split? |
| metric | what behaviour is rewarded or hidden? |
| uncertainty | test items, seeds, annotators—which sources are included? |
| independence | what information influenced training and selection? |
| stress tests | which named assumptions were deliberately broken? |
| boundary | what does the result still not establish? |

Stage 10 is a small executable claim ledger. It does not certify the spine model. It makes each claim narrow enough that a reader can decide what the evidence actually supports.
