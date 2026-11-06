---
title: "11.6 Connections"
description: What mechanistic interpretability can learn from neuroscience, causal inference, system identification, software reverse engineering, scientific explanation, and the rest of this book.
sidebar: { label: "11.6 Connections", order: 6 }
---

Mechanistic interpretability did not invent the problem “infer an internal process from observations and interventions.” Neuroscience studies brains, causal inference studies interventions, control theory identifies dynamical systems, and software engineers debug programs. The analogies are useful only if we also mark where they break.

## 11.6.1 Neuroscience: recording, decoding, and lesioning

Imagine that a neuron fires strongly whenever a face appears. Three different experiments support three different claims.

| Experiment | Neural-network analogue | Supported claim |
|---|---|---|
| record activity while faces appear | cache activations | response correlates with faces |
| train a face decoder on population activity | fit a probe | face information is decodable |
| silence the neuron and retest recognition | ablate a component | the neuron has a causal role under that intervention |

The claims do not collapse into one another. A face-selective neuron may be unnecessary because other neurons carry redundant information. Conversely, a neuron with weak face selectivity may be causally important as part of a distributed computation.

This is the same reason attention visualization, probe accuracy, and ablation effects answer different questions. The lesson from neuroscience is not merely “perform lesions.” It is to combine evidence:

1. characterize the behaviour;
2. record candidate representations;
3. perturb them with matched controls;
4. test whether the result generalizes across stimuli and subjects—or, for models, prompts and checkpoints.

Neural networks offer unusually strong experimental access: we can save every activation, intervene at exact times, and replay the identical input. But “any tensor can be edited” does not imply “every edit is biologically or computationally natural.” Zeroing an entire residual stream is closer to replacing tissue with a zero vector than to a clean lesion. The intervention still needs a reference distribution and controls.

### Degeneracy is not failure of analysis

Biological systems often exhibit **degeneracy**: structurally different components can perform similar functions. Trained networks can too. If two heads each carry a backup copy signal, single-head ablation may show almost no damage:

$$
Y_{-1}\approx Y,
\qquad
Y_{-2}\approx Y,
\qquad
Y_{-\{1,2\}}\ll Y.
$$

The correct conclusion is not “neither head matters,” but “the single interventions do not separate the redundant paths.” Joint interventions and retraining experiments are needed.

## 11.6.2 Causal inference: from observation to do

Causal inference distinguishes seeing a variable from setting it ([Pearl 2009](/references/#pearl-2009)). In symbols,

$$
p(Y\mid H=h)
\quad\text{need not equal}\quad
p(Y\mid \operatorname{do}(H=h)).
$$

The left side observes examples whose activation happened to equal $h$; other upstream variables may differ too. The right side forces $H=h$ while leaving the remaining structural assignments intact.

For a neural network, the forced computation is executable. This removes many identification obstacles faced in medicine or economics, but not every conceptual obstacle:

- Which tensor or direction represents the proposed variable?
- Is zero, a mean activation, a resampled activation, or a donor activation the appropriate intervention?
- Does the edited state lie near states produced by ordinary inputs?
- Are there interactions that make an “effect of one component” context-dependent?
- Does the outcome metric correspond to the behavioural claim?

The important import from causal inference is therefore not the slogan “we can intervene.” It is the discipline of writing an estimand before computing it: treatment, outcome, baseline, population, intervention, and assumptions.

### Mediation and path questions

Suppose input corruption changes a logit margin by $-8$. Patching head $h$ recovers $+6$. It is tempting to say “head $h$ mediates $75\%$ of the effect.” That statement is safe only as the descriptive ratio $6/8$ for this corruption and patch. With nonlinear interactions or redundant paths, another head may also recover $6$, and the two percentages need not sum to $100\%$.

Causal language sharpens rather than inflates the conclusion:

> On this donor–recipient pair, setting head $h$'s output to its clean value recovers 6 of the 8 lost margin units.

That sentence is less dramatic and more reproducible.

## 11.6.3 System identification and control

Control theory often starts with an unknown dynamical system

$$
s_{t+1}=F(s_t,u_t),
\qquad
y_t=G(s_t),
$$

then applies inputs $u_t$ and observes outputs $y_t$ to infer a useful model. Mechanistic interpretability faces the same three choices:

- **state:** which activations summarize what matters?
- **input:** which prompts or internal interventions excite the system?
- **output:** which logits or behaviours reveal the computation?

An experiment can fail because the mechanism is absent, or because the probe inputs never **excite** it. A model may possess a copying circuit that stays dormant on ordinary prose. Conversely, a highly regular synthetic prompt may activate a shortcut absent in natural text.

This motivates a design principle: vary inputs so competing mechanisms make different predictions. For an induction hypothesis, change the token identities, gap length, number of distractors, and absolute positions independently. A single repeated string does not identify which feature the network used.

System identification also warns about non-unique internal coordinates. If $s'_t=As_t$ for invertible $A$, then an equivalent model can be written in the new basis. A feature may be stable as a **subspace** even when individual neurons change. This connects directly to superposition and basis dependence in §11.7.

## 11.6.4 Software reverse engineering and debugging

A reverse engineer does not begin by reading every machine instruction. They first reproduce a behaviour, trace data flow, set breakpoints, patch values, and test a hypothesis about the hidden algorithm.

| Software practice | Interpretability analogue |
|---|---|
| breakpoint | activation cache |
| watch expression | probe or feature readout |
| replace a variable at runtime | activation patch |
| turn an instruction into a no-op | ablation |
| unit test | behavioural and intervention test |
| differential test between builds | checkpoint or model comparison |

The analogy explains why interpretability should produce **executable tests**, not only pictures. If “head 1.0 passes the successor token” is the hypothesis, construct donor–recipient inputs for which the successor changes and assert the transferred prediction.

The analogy also has limits. Conventional software is designed from named abstractions; its variables and modules often preserve a programmer's intent. A trained network is selected by optimization. It may use distributed, approximate, redundant, or entangled procedures with no short source-code equivalent.

Even in software, a debugger can create impossible states. Changing a length field without changing the buffer does not reveal normal execution; it tests a malformed program state. Cross-run neural patches have the same risk. A strong patching experiment chooses donor and recipient states that preserve as much surrounding context as possible and includes negative-control patches.

## 11.6.5 Representation geometry: coordinates versus features

Suppose a residual representation is rotated by an invertible orthogonal matrix $R$:

$$
h'=Rh,
\qquad
W'_U=W_UR^\top.
$$

Then

$$
W'_Uh'=W_UR^\top Rh=W_Uh,
$$

so the model's logits are unchanged even though every neuron coordinate has changed. A statement such as “neuron 137 stores plurality” is therefore basis-sensitive. “Plurality is represented in this one-dimensional subspace and read by this downstream map” is closer to an invariant claim.

This connects interpretability to the linear algebra of Units 02–04:

- singular vectors reveal influential input/output directions;
- low-rank QK and OV factorizations constrain what a head can represent;
- superposition lets many features share a smaller space;
- change of basis can preserve function while redistributing neuron-level descriptions.

Directions are not automatically semantic features, either. A direction earns a label only through examples, counterexamples, and interventions on downstream behaviour.

## 11.6.6 Philosophy of science: explanation must support counterfactuals

A compact summary is not necessarily an explanation. “This head attends backward” compresses an attention map but may not say why the output occurs. A mechanistic explanation should support counterfactual questions:

- What changes if the earlier matching token is replaced?
- What changes if the successor is moved?
- Which internal value must be transferred to move the prediction?
- Under what conditions does the proposed algorithm fail?

This is why causal abstraction is a stronger target than visual resemblance. It requires the high-level story to agree with low-level interventions.

Whether a predictive model “understands” language is a separate and contested question ([Lipton 2018](/references/#lipton-2018)). This unit avoids settling the word *understanding*. It asks a narrower, answerable question: which computations are causally responsible for a defined behaviour over a defined domain?

## 11.6.7 Safety auditing: a mechanism is not a guarantee

Mechanistic evidence can improve safety work. It can reveal a trigger, identify a memorized association, or show that a refusal depends on a fragile direction. But finding one circuit does not certify the whole model.

Three gaps remain:

1. **coverage:** the audited prompts may not span deployment inputs;
2. **adaptivity:** another prompt may recruit a different path;
3. **completeness:** an explanation may capture one sufficient mechanism while missing backups.

Therefore “we found a refusal feature” is not equivalent to “the model will refuse all harmful requests.” Interpretability contributes evidence to an assurance case; it is not the assurance case by itself.

## 11.6.8 Connections inside this book

| Idea in this unit | Earlier foundation | Later use |
|---|---|---|
| directions and basis dependence | Unit 02: linear algebra and superposition | sparse feature methods in §11.8 |
| QK and OV circuits | Unit 04: attention as soft lookup | induction construction in §11.3 |
| memorization versus a learned rule | Units 01 and 07 | behavioural controls in §11.2 |
| behaviour before mechanism | Unit 10: measurement and uncertainty | experiment design in Unit 12 |
| interventions and counterfactuals | Unit 09: changing a policy changes behaviour | causal testing in §§11.3–11.5 |
| distribution shift | Units 06 and 10 | off-manifold intervention limits in §11.7 |
| pre-specified refuters | Unit 10: benchmark validity | falsification and replication in Unit 12 |

The recurring pattern is

$$
\text{define behaviour}
\rightarrow
\text{measure}
\rightarrow
\text{hypothesize mechanism}
\rightarrow
\text{intervene}
\rightarrow
\text{test new cases}.
$$

Skipping the first two steps produces an elegant explanation of an ill-defined phenomenon. Skipping the last two produces an untested story.

## 11.6.9 Takeaway

Each neighbouring field contributes one discipline:

- neuroscience: separate recording, decoding, and lesioning;
- causal inference: define the intervention and estimand;
- system identification: excite competing mechanisms and mind the choice of state;
- software engineering: turn explanations into executable tests;
- representation geometry: prefer invariant subspaces over arbitrary coordinates;
- philosophy of science: demand counterfactual and novel predictions;
- safety engineering: distinguish local evidence from system-wide guarantees.

Together they point to a modest but strong standard: **a mechanism claim should be operational, interventional, predictive, and explicit about its domain.**
