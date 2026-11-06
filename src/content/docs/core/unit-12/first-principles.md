---
title: "12.2 First principles"
description: Define the estimand and experimental unit, construct rival hypotheses, isolate causes, control randomness, separate exploration from confirmation, pre-register decisions, audit execution, replicate, and interpret inconclusive results.
sidebar: { label: "12.2 First principles", order: 2 }
---

Rigour is not a decorative checklist added after an experiment. It is the work of arranging the experiment so that the observed result answers the intended question.

## 12.2.1 Begin with an estimand, not a slogan

“Positional embeddings matter” is too vague. Matter to what model, trained how, measured on which distribution, compared under which intervention?

An **estimand** is the quantity the experiment intends to learn. For the capstone,

$$
\delta=
\mathbb E_{s\sim S}
\left[
L\bigl(\operatorname{Train}(\text{no-pos},s)\bigr)
-
L\bigl(\operatorname{Train}(\text{pos},s)\bigr)
\right],
$$

where $L$ is held-out NLL, `Train` includes the fixed architecture, data, optimizer, and 400-step budget, and $S$ is a declared distribution over training randomness.

This definition immediately reveals the scope:

- it concerns two training procedures, not two fixed checkpoints;
- randomness from training belongs inside the expectation;
- the outcome is NLL on one held-out distribution;
- it says nothing yet about longer training, wider models, or downstream tasks.

Before collecting data, write five fields:

| Field | Capstone example |
|---|---|
| population/system | width-32, 2-layer character Transformer |
| intervention | remove learned positional table |
| outcome | fixed-split held-out NLL in nats/character |
| randomness | initialization and data-order seeds |
| estimand | mean no-position minus position NLL across training randomness |

Without this step, different readers can attach different questions to the same number.

## 12.2.2 A useful hypothesis has a rival and a refuter

A hypothesis is not just a sentence that could technically be false. It should generate a prediction that differs from plausible alternatives.

For the capstone:

- $H_1$: learned positions provide a meaningful benefit, so $\delta>0.02$;
- $H_0$: any benefit is smaller than the design-relevant threshold, so $\delta<0.02$;
- $H_A$: the effect depends strongly on seed, training time, or context, so a single mean at 400 steps is unstable.

The threshold $0.02$ is the **smallest effect size of interest** (SESOI). It must be justified by a decision: perhaps $0.02$ is the smallest gain worth additional parameters or engineering complexity. If no decision changes until $0.10$, powering an experiment for $0.02$ wastes compute. If $0.005$ would matter, $0.02$ is too coarse.

A strong prediction states in advance:

> If $H_1$ is correct in the declared system, the entire uncertainty interval for $\delta$ should exceed $0.02$. If it lies below $0.02$, the registered meaningful-effect claim is not supported. If it crosses $0.02$, the experiment is unresolved.

The third outcome prevents “absence of significance” from being silently converted into “absence of effect.”

## 12.2.3 Identify the experimental unit

The **experimental unit** is the independently randomized entity to which a condition is applied. In a training ablation, it is usually a complete training run, not a token, prompt, or validation example.

This distinction controls the denominator of uncertainty. Suppose two conditions are each trained once and evaluated on 10,000 prompts. A narrow interval across prompts estimates uncertainty about those two fixed checkpoints:

$$
\operatorname{SE}_{\text{prompts}}
\propto\frac{1}{\sqrt{10{,}000}}.
$$

It does not estimate variability from retraining:

$$
\operatorname{Var}_{\text{train}}
=\operatorname{Var}_{s}
\left[L\bigl(\operatorname{Train}(c,s)\bigr)\right].
$$

Treating 10,000 prompts as 10,000 independent training runs is **pseudoreplication**. It produces very small error bars around the wrong source of variation.

Data are often hierarchical:

$$
\text{seed}
\supset
\text{checkpoint}
\supset
\text{document}
\supset
\text{token}.
$$

The uncertainty procedure must match the inferential target. Bootstrap documents for fixed-checkpoint performance; vary seeds for claims about training procedures; vary datasets or domains for broader generalization claims.

## 12.2.4 Change one cause—or vary several causes systematically

A controlled ablation changes one factor while holding all other causally relevant factors fixed:

$$
Y_s^{B}-Y_s^{A}

$$

should differ because of the intended factor, not because condition B saw different data, ran longer, or initialized shared weights differently.

“Change one thing” is a design principle, not a ban on multifactor experiments. A factorial design can vary several factors deliberately:

$$
Y=\beta_0+\beta_A A+\beta_B B+\beta_{AB}AB+\varepsilon,
$$

and estimate both main effects and interactions. What is invalid is changing several things while pretending the observed difference belongs to one.

### Controls answer specific alternative explanations

- **negative control:** a change predicted not to affect the target;
- **positive control:** a known effect confirming the pipeline can detect change;
- **implementation control:** an invariant such as identical shared initial weights;
- **data control:** identical train/validation splits and preprocessing;
- **budget control:** identical steps, examples, tuning effort, and compute when relevant.

A control is useful only if its failure diagnoses something.

## 12.2.5 Randomization, blocking, and pairing

Randomization prevents condition assignment from following a hidden pattern. **Blocking** groups experimental units expected to share nuisance variation, then compares conditions within each group.

Pairing by seed is a block design. For seed $s$,

$$
d_s=Y_s^{B}-Y_s^{A}.
$$

If the same seed creates similar shared initialization and data order, nuisance effects cancel. Algebraically,

$$
\operatorname{Var}(d)
=\sigma_A^2+\sigma_B^2-2\rho\sigma_A\sigma_B.
$$

Positive correlation $\rho$ makes pairing efficient. Negative or meaningless pairing can make it worse.

The capstone demonstrates a subtle failure. Parameter initialization draws the positional table before later weights. Removing the table changes how many random numbers are consumed, so “seed 3” no longer means the same shared initialization. The labels are paired; the causal nuisance variable is not.

The fix is concrete:

- give each parameter tensor a named random stream; or
- always draw the positional table and discard it in the no-position condition; and
- test that every shared initial tensor is bit-identical before training.

Pairing is an implementation property to verify, not a word to put in the methods section.

## 12.2.6 Treat randomness as data, not inconvenience

Initialization, minibatch order, augmentation, sampling, nondeterministic kernels, and hyperparameter search can all change results. Seed variation has materially altered conclusions in deep learning ([Henderson et al. 2018](/references/#henderson-2018); [Bouthillier et al. 2021](/references/#bouthillier-2021)).

Report per-seed values before the mean. With $n=5$, the vector

$$
(0.164,-0.039,-0.076,-0.055,-0.044)
$$

reveals more than its mean $-0.010$: one run points strongly in the opposite direction. A mean without the run-level distribution hides fragility.

Choose the seed count from the smallest effect of interest and a variance estimate. If variance is unknown, run a pilot whose declared purpose is to estimate it. Do not call the pilot a confirmatory test after seeing its result.

## 12.2.7 Separate exploration, model selection, and confirmation

Exploration is indispensable. It discovers metrics, bugs, subgroups, architectures, and hypotheses. The problem is reusing the same observations as if those choices had been fixed in advance.

An analyst may choose:

- final loss or best loss;
- five seeds or four after excluding an “unstable” run;
- all prompts or a subgroup;
- absolute difference or percentage improvement;
- step 400 or the visually best checkpoint.

Each choice can be reasonable. Choosing among them because one looks favourable creates a **garden of forking paths** ([Gelman & Loken 2014](/references/#gelman-2014)).

Use three partitions of work:

1. **exploration:** free analysis; clearly labelled; generates hypotheses;
2. **selection/tuning:** chooses hyperparameters, metrics, and thresholds;
3. **confirmation:** untouched data or fresh runs analysed by a fixed rule.

The same dataset can sometimes serve different roles through strict splitting, but no example should be both the reason a rule was invented and the independent evidence that confirms it.

## 12.2.8 Pre-registration freezes decisions, not curiosity

Pre-registration records choices before outcome data are inspected ([Nosek et al. 2018](/references/#nosek-2018)). A useful registration includes:

- question, estimand, and rival hypotheses;
- primary outcome and smallest effect of interest;
- experimental unit, conditions, controls, and randomization;
- seed/sample count and stopping rule;
- exclusions and failure handling;
- statistical analysis and interval;
- decision rule for every possible outcome;
- planned replication.

The spine writes JSON plus a SHA-256 digest. A digest detects later modification, but does not establish that the file existed before training. A trusted timestamp requires a public registry, pushed commit, signed artifact, or independent witness.

Deviations are allowed. Record what changed, when, why, and whether the affected analysis is now exploratory. Pre-registration protects the distinction between prediction and postdiction; it is not a ban on learning.

## 12.2.9 Audit the implementation before interpreting the estimate

Statistical correctness cannot rescue a program that runs the wrong experiment. An audit should test:

1. **determinism:** same condition and seed reproduce the same result where promised;
2. **contrast integrity:** shared parameters, data, and budgets match;
3. **metric integrity:** the code computes the registered outcome with correct units and sign;
4. **split integrity:** no validation or test leakage;
5. **registration integrity:** record and digest still match;
6. **decision integrity:** synthetic intervals exercise all decision branches;
7. **artifact integrity:** configs, code revision, logs, and outputs are linked.

The broken random-stream pairing is an implementation failure that no bootstrap can detect by itself. Tests comparing initial tensors could have caught it before the first expensive run.

## 12.2.10 Replication has several meanings

“We reproduced the result” is incomplete. Specify what was held fixed and what was made independent.

| Form | What changes | Main error challenged |
|---|---|---|
| computational reproducibility | rerun same code/data/environment | missing artifacts, nondeterminism |
| direct replication | fresh experimental units under same protocol | chance result |
| independent replication | different team or implementation | shared code and analyst errors |
| conceptual replication | different model, dataset, or operationalization | narrow scope of explanation |

The capstone's fresh seeds are a direct replication, not an independent implementation. Because both original and replication use the same broken initialization pairing, agreement would not remove that systematic error.

Replication is strongest when it introduces independence along the failure mode that matters.

## 12.2.11 “Inconclusive” is an informative outcome

With threshold $\tau$ and interval $[\ell,u]$:

- $\ell>\tau$: evidence that the effect exceeds the meaningful threshold;
- $u<\tau$: evidence that it lies below the threshold under the registered one-sided rule;
- otherwise: the experiment does not resolve the threshold.

An inconclusive result can reveal:

- insufficient experimental units;
- a noisy outcome;
- weak or invalid pairing;
- heterogeneous effects;
- a threshold finer than the available compute supports.

It does not license “there is no effect,” nor does it make the work worthless. A precise diagnosis of resolution is often the most useful result for designing the next study.

## 12.2.12 A failed prediction can be scientific progress

Unit 06's registered scaling prediction missed. That failure exposed a regime change. Without the frozen prediction, refitting the curve could have hidden the error.

The productive sequence is:

$$
\text{prediction fails}
\rightarrow
\text{locate broken assumption}
\rightarrow
\text{form new hypothesis}
\rightarrow
\text{test on new data}.
$$

Do not rewrite the original prediction as though the revised model had been specified all along. Version the evidence trail.

## 12.2.13 The complete claim template

> For system and data distribution $D$, we estimated estimand $\delta$ by comparing conditions $A$ and $B$ across experimental units $U$, using controls $C$ and registered analysis $M$. The estimate was $\hat\delta$ with uncertainty interval $I$. Under decision threshold $\tau$, the result was $R$. Implementation audit $Q$ passed except for limitations $L$. A replication independent in dimensions $J$ produced $R_2$. The evidence supports bounded claim $K$ but not broader claims $K'$.

If any field is missing, the reader has to guess which question the number answers.
