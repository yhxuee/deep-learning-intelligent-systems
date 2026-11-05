---
title: "11.9 Experiments"
description: Seven reproducible experiments that move from a known circuit to unknown trained mechanisms, with predictions, controls, metrics, failure criteria, and bounded conclusions.
sidebar: { label: "11.9 Experiments", order: 9 }
---

These experiments are arranged from **calibration**, where the mechanism is known, to **discovery**, where it is not. Do not begin with a visualization. Begin with a prediction that could be wrong.

Core code:

- [`spine/stage11_induction.py`](https://github.com/yhxuee/deep-learning-intelligent-systems/tree/main/spine) constructs, trains, scores, ablates, and probes the models;
- `head_mask` and `patch` in `spine/model.py` implement interventions;
- `spine/test_spine.py` contains executable assertions for the ground-truth circuit.

Run the fast Stage 11 checks with

```bash
.venv/bin/python -m pytest spine/test_spine.py -k stage11 -q
```

Run the demonstration with

```bash
.venv/bin/python spine/stage11_induction.py --steps 8000
```

## 11.9.1 Use the same record for every experiment

Before looking at results, fill in:

| Field | Entry |
|---|---|
| question | what uncertainty is this experiment meant to reduce? |
| input distribution | how are prompts generated and split? |
| outcome | NLL, logit margin, attention score, or accuracy? |
| intervention | exactly which tensor is replaced, by what value, and when? |
| prediction | numerical direction or threshold expected before running? |
| refuter | what result would count against the hypothesis? |
| controls | no-op, wrong donor, matched random component, or alternative replacement? |
| repetitions | seeds, prompts, and checkpoints? |
| uncertainty | standard deviation, bootstrap interval, or paired interval? |
| conclusion scope | which model, task, metric, and intervention does the result cover? |

Keep exploratory and confirmatory results in separate tables. If a plot suggested the hypothesis, it cannot also be the untouched confirmation of that hypothesis.

## Experiment 11.1 — Calibrate methods on a circuit with known ground truth

### Question

Do attention scores, direct logit attribution, ablation, and patching recover the two components that were deliberately constructed?

### Ground truth and prediction

The hand-built network has four heads. Head 0.0 writes the previous-token feature; head 1.0 uses it to copy the successor. Heads 0.1 and 1.1 are unused.

Predict before running:

| Method | Expected positive | Expected negative |
|---|---|---|
| previous-token attention score | head 0.0 | other heads |
| induction attention score | head 1.0 | other heads |
| zero-ablation NLL increase | heads 0.0 and 1.0 | heads 0.1 and 1.1 |
| direct correct-logit write | head 1.0 | head 0.0 |
| clean-to-corrupt recovery | head 1.0 output patch | irrelevant-head patch |

The distinction between the last two rows matters: head 0.0 is causally important but does not make the decisive direct logit write.

### Procedure

1. Call `build_induction_circuit(K=8, c=60, lam=10)`.
2. Generate 64 sequences with `repeated_batch`.
3. Record baseline second-half NLL.
4. Compute `head_scores`.
5. Zero each head separately and recompute NLL.
6. Corrupt the first copy while leaving the second copy fixed.
7. Patch head 1.0's clean output into the corrupted run.

### Required controls

- patch a run with its own activation: logits must not change;
- patch an unused head: clean behaviour should not be restored;
- use the same token batch for every head comparison;
- report raw NLL, not only a normalized recovery score.

### Interpretation

Passing this experiment validates the analysis **on one known circuit**. It does not validate every method on trained networks. Failure is informative: fix the method or implementation before applying it to an unknown model.

## Experiment 11.2 — Stress-test attention sharpness

### Question

Does a visual attention threshold fail before the behaviour itself fails?

The model scales dot products by $\sqrt{d_h}$. If the unscaled correct score is $c$ and the other $n-1$ scores are zero, the correct attention weight is

$$
a(c,n,d_h)=\frac{e^{c/\sqrt{d_h}}}{e^{c/\sqrt{d_h}}+n-1}.
$$

Here $d_h=18$. For $n=16$, predict $a(60,16,18)\approx1$ and $a(20,16,18)\approx0.881$. The tempting calculation $a(5,16)\approx0.908$ is **wrong** because it forgets the scale factor:

$$
a(5,16,18)=
\frac{e^{5/\sqrt{18}}}{e^{5/\sqrt{18}}+15}
\approx0.178.
$$

This comparison also reveals that $c=20$ should be even sharper than $c=5$; a reported reverse ordering indicates a calculation or indexing error.

### Procedure

1. Sweep $c\in\{0,1,2,5,10,20,60\}$ while holding $\lambda=10$ fixed.
2. For each $c$, record theoretical attention weight, measured attention weight, baseline NLL, and zero-ablation NLL.
3. Repeat while sweeping $\lambda\in\{0,1,2,5,10\}$ at fixed $c=60$.

### What each sweep isolates

- $c$ controls **routing sharpness**;
- $\lambda$ controls **write strength**;
- an attention map can be sharp while the value write is too weak;
- an imperfect attention map can still support good behaviour if the correct source dominates enough.

### Plot

Plot NLL and attention weight against $c$ on separate vertical axes. Do not infer a universal attention cutoff. Locate the region in which the behavioural conclusion changes.

## Experiment 11.3 — Test patching as information transfer

### Question

Does head 1.0 carry the clean-run information that distinguishes the correct successor from a corrupted one?

### Donor–recipient design

Let the clean sequence be

$$
[\mathrm{BOS},s,s],
$$

and construct a recipient whose first copy is replaced by a different permutation $r$ while its second copy remains $s$:

$$
[\mathrm{BOS},r,s].
$$

The corrupted context points toward the wrong successors. Patch the clean output of head 1.0 into the corrupted run.

### Pre-registered predictions

1. clean NLL $<0.01$ nat;
2. corrupted NLL $>1$ nat;
3. correct head-1.0 patch restores NLL to the clean value within numerical tolerance;
4. same-run patch changes logits by less than $10^{-10}$;
5. a batch-permuted wrong donor does not systematically restore the correct sequence.

The repository test implements the first four predictions. Add the fifth as an extension.

### Why the wrong-donor control matters

If any high-norm activation improves the result, recovery does not demonstrate content-specific transfer. The donor must carry the *right* successor information, not merely a generic activation scale.

### Bounded conclusion

Success supports: “in this constructed circuit and corruption, the clean output of head 1.0 is sufficient to restore clean prediction.” It does not by itself show that patching is on-manifold in a trained model.

## Experiment 11.4 — Observe copying during training

### Question

Does copying appear gradually or through a sharp transition, and do different random seeds implement it the same way?

### Design

Train the two-layer attention-only model on `shifted_batch`, which varies the gap between copies so a fixed absolute-position shortcut is insufficient.

Use at least five seeds. Every 100 steps record:

- held-out weighted NLL on the predictable part of the second copy;
- best induction attention score;
- all four single-head zero-ablation NLLs;
- optionally, pairwise ablations.

Define the transition step before running, for example

$$
t^*=\min\{t:L_{\text{copy}}(t)<1\text{ nat}\}.
$$

### Competing predictions

- **canonical-circuit hypothesis:** a high induction score appears near the loss transition, and one or two heads dominate ablation effects;
- **distributed-learning hypothesis:** loss improves without a single canonical score, and ablation effects are spread across heads;
- **positional-shortcut hypothesis:** held-out performance collapses when gaps or absolute positions move beyond training support.

### Analysis

Plot each seed separately before averaging. Then report the median and range of $t^*$. Aligning curves by $t^*$ can reveal a common transition shape, but also show the unaligned plot so alignment does not hide training-time variability.

Do not call a visually steep decline a “phase transition” without defining a quantitative comparator, such as maximum slope relative to a smoothed baseline.

## Experiment 11.5 — Separate decodability from use

### Question

Can a linear probe decode a property, and does the model causally use the decoded direction?

Use two labels on Stage 07 hidden states:

1. local label: “the next character is a digit”;
2. global label: “this story contains a pet.”

### Probe protocol

1. Split by story, not by token, so nearby states from the same story cannot leak across splits.
2. Collect the same number of positive and negative examples, or report balanced accuracy.
3. Fit the same regularized logistic regression for real labels and control labels.
4. For the control task, randomly assign labels by input type as in control-task probing, then preserve those assignments across train and test.
5. Report selectivity:

$$
S=\operatorname{Acc}_{\text{real}}-
\operatorname{Acc}_{\text{control}}.
$$

### Causal extension

Let $v$ be the normalized probe direction. Intervene with

$$
h'=h+\alpha v,
$$

and compare against norm-matched random directions and $-v$. Sweep $\alpha$ rather than selecting one favourable value.

### Possible outcomes

| Probe | Intervention | Interpretation |
|---|---|---|
| succeeds | changes relevant behaviour | feature is decodable and causally influential under this intervention |
| succeeds | no effect | information is decodable but use is not established |
| fails | effect along another discovered direction | this probe or representation was inadequate |
| control also succeeds | — | probe capacity or dataset permits memorization |

The original draft predicted that a successful probe meant the property “is used.” This experiment is designed specifically not to make that inference without an intervention.

## Experiment 11.6 — Discover the trained toy's mechanism

### Question

The trained model copies, all four tested zero ablations hurt, and no head has a large canonical induction score. What computation explains those facts?

This is an open investigation. Begin with at least three mutually distinguishable hypotheses:

- $H_1$: copying is distributed across several heads using relative position;
- $H_2$: the model combines token identity with absolute-position templates;
- $H_3$: the canonical metric misses an induction-like computation because the relevant information is in values or subspaces, not the maximum attention pattern.

### Discriminating experiments

| Test | $H_1$ | $H_2$ | $H_3$ |
|---|---|---|---|
| shift copies to unseen absolute positions | should mostly survive | should fail | depends |
| change gap while preserving tokens | should survive trained range | may show position-specific failures | may survive |
| permute positional embeddings | should damage relative-position machinery in a structured way | strong damage | depends |
| patch same tokens across different gaps | transferable subspaces possible | position mismatch hurts | value-level patch may recover |
| inspect QK only | incomplete | may reveal templates | likely misses key evidence |
| inspect OV/direct writes | distributed contributions | position-conditioned writes | should reveal hidden copy information |

### Reporting rule

Report hypotheses ruled out as carefully as the preferred one. “No canonical induction head” is a finding about a metric; “there is no copying mechanism” contradicts the observed copying behaviour.

## Experiment 11.7 — Compare post-training changes without inventing a mechanism

### Question

Did exact-reward and buggy-reward post-training change arithmetic behaviour through prompt-dependent or prompt-independent logit shifts?

Use the reference, exact-reward, and buggy-reward checkpoints from Stage 09. On 200 held-out prompts $a+b=$, compute

$$
\Delta z_k(x)=z_k^{\text{post}}(x)-z_k^{\text{ref}}(x)
$$

for each output digit $k$.

### Pre-registered predictions

- buggy reward: a broadly prompt-independent positive shift for digit 9;
- exact reward: a shift that depends on the correct result digit;
- both: changes may be distributed across layers, so final-logit evidence alone will not identify the internal path.

### Analysis

1. Plot $\Delta z_k$ by correct answer.
2. Fit a model with answer-digit indicators rather than regressing directly on numeric $a+b$; digits are categorical outputs, not a naturally linear scale.
3. Compare the variance explained by answer identity with a constant-only model.
4. Bootstrap prompts to obtain intervals for mean shifts.
5. Repeat direct-logit attribution by layer or component, then validate candidates with ablation or patching.

### Conclusion boundary

A constant shift toward 9 supports a prompt-insensitive output bias. It does not prove where that bias is stored. An answer-dependent shift supports task-conditional behaviour, not a unique arithmetic algorithm. The intervention stage is what turns a logit pattern into a mechanistic hypothesis.

## 11.9.8 Final lab report

For each experiment, end with four lines:

1. **Observed:** the numerical result with uncertainty.
2. **Supported:** the narrowest causal or representational claim licensed.
3. **Not supported:** the tempting stronger claim that remains unproven.
4. **Next discriminator:** one experiment on which surviving hypotheses disagree.

That format makes a negative result useful and prevents a successful visualization from silently becoming a mechanism claim.
