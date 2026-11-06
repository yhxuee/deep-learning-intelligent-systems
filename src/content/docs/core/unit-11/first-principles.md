---
title: "11.2 First principles"
description: Separate behaviour, decodable representation and causal mechanism; learn what observation, probes, attribution and intervention can establish; validate methods on known ground truth before interpreting unknown models.
sidebar: { label: "11.2 First principles", order: 2 }
---

Interpretability begins with a language problem: the phrase “the model learned X” can mean several different things. If those meanings are not separated, evidence for a weak claim is easily presented as evidence for a strong one.

## 11.2.1 Three levels of “learned”

### Level 1 — behaviour

A behavioural claim concerns observable input–output relations:

> When a novel name appears twice, the model predicts the later characters of the second occurrence better than a baseline.

Evidence comes from controlled tasks, held-out examples, uncertainty intervals and counterexamples. Behaviour is the right level for questions such as “does it copy?” or “does accuracy survive a name change?”

Behaviour does not identify an internal algorithm. A lookup table and a general addition algorithm can agree on every tested sum while differing on untested ones.

### Level 2 — representation

A representation claim concerns information available in an internal state:

> A linear classifier can decode the next character from the residual stream at this position.

This is stronger than an input–output observation, but weaker than a mechanism. The information may be present because it is carried along incidentally. A downstream computation may ignore it. A powerful probe may also reconstruct the label from correlations instead of reading a simple representation.

“Decodable” therefore means:

$$
\text{some decoder in the tested class can recover the label},
$$

not

$$
\text{the model itself uses that decoder or feature}.
$$

### Level 3 — mechanism

A mechanistic claim describes a causally relevant internal computation:

> At the second occurrence of token A, head 1.0 uses information written by head 0.0 to attend to the token following the earlier A and writes that token toward the output logits.

This statement predicts more than an attention picture. It predicts what should happen when either component is changed, which activation should restore behaviour, and which control interventions should do little.

Mechanisms can be described at different scales: a neuron, feature direction, head, layer, circuit or algorithm. A correct head-level account need not identify individual neurons; a correct algorithmic account may have several equivalent low-level implementations.

### The logical relation

The three levels are not a ladder of automatic implications:

$$
\text{successful behaviour}
\centernot\Rightarrow
\text{one unique representation}
\centernot\Rightarrow
\text{one unique mechanism}.
$$

A mechanism should predict behaviour, but observing the behaviour does not invert that mapping uniquely. This many-to-one relation is the central identifiability problem of interpretability.

## 11.2.2 Four evidence types answer four questions

### Observation — what patterns are present?

Observation records activations without changing the computation:

- attention maps;
- activation distributions;
- inputs that maximally activate a unit;
- similarity between representations;
- direct logit contributions in architectures where the decomposition is defined.

Observation is excellent for generating candidate hypotheses. It establishes correlations and descriptive structure. It does not establish that the observed pattern is needed or used.

### Probing — what can a decoder recover?

A **probe** is a supervised model trained on frozen activations. Linear probes are popular because their limited capacity makes the result easier to interpret ([Alain & Bengio 2017](/references/#alain-2017); [Belinkov 2022](/references/#belinkov-2022)).

A good probe study needs at least:

- a train/test split whose independent unit matches the claim;
- a simple baseline, such as decoding from the input embedding alone;
- a control task with randomized or permuted labels;
- a report of probe capacity and regularization;
- comparison across layers or positions chosen before looking at the test labels.

If a probe gets 92% and the matched control gets 81%, the relevant result is not merely “92% decodable.” The 11-point selectivity asks how much performance exceeds what the probe can obtain from memorization or incidental structure ([Hewitt & Liang 2019](/references/#hewitt-2019)).

Even high selectivity supports representation, not use.

### Attribution — where is the output locally sensitive?

Gradient attribution asks how an output score changes under a small perturbation:

$$
a_j=\frac{\partial s(x)}{\partial h_j}
\quad\text{or}\quad
a_j=h_j\frac{\partial s(x)}{\partial h_j}.
$$

Integrated gradients accumulates gradients along a path from a baseline to the input ([Sundararajan et al. 2017](/references/#sundararajan-2017)). Such methods are fast and can rank candidate inputs or components.

But a local derivative is not a complete causal story:

- a saturated component can be causally important while its local gradient is near zero;
- correlated components can share or reroute credit;
- the chosen baseline and path affect integrated attributions;
- some saliency methods can remain visually similar after model parameters are randomized ([Adebayo et al. 2018](/references/#adebayo-2018)).

Attribution is evidence about sensitivity under a mathematical perturbation, not automatically about the computation used across natural inputs.

### Intervention — what changes when an internal variable is changed?

An intervention modifies the forward computation and measures a behavioural effect.

**Ablation** replaces a component output with zero, its mean, a resampled value or another reference. **Activation patching** runs a clean and corrupted input, then inserts a clean activation into the corrupted run to see whether a specific behaviour is restored ([Vig et al. 2020](/references/#vig-2020); [Meng et al. 2022](/references/#meng-2022)).

Interventions address causal questions more directly than observation. Their conclusions remain conditional on the intervention:

> Replacing head 1.0's output with zero increases loss by 2.37 nats on this distribution.

is an observed intervention effect. It does not automatically imply:

> Head 1.0 is the unique natural implementation of copying in every context.

The zero state may be off the model's natural activation manifold; another component may compensate under a different intervention; the ablated head may carry several features.

## 11.2.3 Why an attention map is not yet an explanation

For one attention head,

$$
o_i=\sum_j A_{ij}v_j.
$$

The weight $A_{ij}$ tells us how strongly value vector $v_j$ is mixed into the update at position $i$. It does **not** tell us:

1. what information $v_j$ contains;
2. which direction the output projection writes into;
3. whether downstream layers use that direction;
4. whether another path carries redundant information;
5. whether the pattern is stable across inputs.

Two heads can share the same attention pattern and write different value content. Conversely, two heads with different patterns can write similar outputs.

The NLP debate over “attention as explanation” reached an appropriately bounded conclusion: attention patterns can be informative observations, but their explanatory force depends on the question, value content and supporting interventions ([Jain & Wallace 2019](/references/#jain-2019); [Wiegreffe & Pinter 2019](/references/#wiegreffe-2019)).

The trained copying model in this unit is a constructive counterexample. Its copying loss is low, and every head's zero ablation damages performance, yet no single head has the textbook induction pattern. Searching only for the familiar pattern would miss real behaviour; assigning a story to the highest score would overinterpret noise.

## 11.2.4 Controls make an intervention interpretable

Suppose ablating head 1.0 raises loss. At least four explanations remain:

- the head carries the hypothesized copying information;
- any same-sized ablation damages the network;
- the intervention creates an unnatural activation;
- the effect comes from a broad distribution shift rather than the chosen examples.

Controls target these alternatives.

| Control | Question it asks |
|---|---|
| ablate an unused or matched head | does any ablation cause the effect? |
| replace by mean instead of zero | is zero itself an artefact? |
| patch a run with its own activation | does the hook alter computation even when values are unchanged? |
| patch the wrong position or head | is restoration specific? |
| reverse clean and corrupted sources | does the behaviour follow the patched information? |
| repeat over many inputs | is the effect stable or one cherry-picked case? |

A large effect without controls is a result. A large, specific effect with controls is evidence for a mechanism claim.

Effect size should be paired within input:

$$
\Delta_i=L_i(\text{intervened})-L_i(\text{clean}).
$$

Report the distribution or interval of $\Delta_i$, not only one dramatic example.

## 11.2.5 Validate the method where ground truth is known

How can we test whether an interpretability method works when the real model's mechanism is unknown? Begin with a model whose mechanism is known because we constructed it.

Stage 11's attention-only Transformer has four heads:

| head | constructed role | expected observational signal | expected zero-ablation effect |
|---|---|---|---|
| 0.0 | previous-token head | attends to position $i-1$ | copying breaks |
| 0.1 | unused control | no special target | little change |
| 1.0 | induction/copy head | attends after earlier matching token | copying breaks |
| 1.1 | unused control | no special target | little change |

It also predicts a patching result: replacing the corrupted run's head-1.0 output with the clean output should restore the clean prediction, while self-patching should be a no-op.

This setup tests the entire measurement pipeline—caching, indexing, head selection, loss mask and patch hook. An off-by-one error can still draw a plausible attention heat map, so exact synthetic checks matter.

Passing this ground-truth test establishes **sensitivity to this circuit**. It does not establish universal sensitivity. A method designed to find single sparse heads may pass here and fail on a distributed trained mechanism.

## 11.2.6 Establish the phenomenon before explaining it

This principle sounds obvious, yet interpretability workflows often reverse it:

1. choose a fashionable mechanism;
2. search every head for its best resemblance;
3. tell a story about the maximum score.

With enough heads, one will always be the largest even when every score is noise. The correct order is:

1. define the behaviour and baseline;
2. show the model performs it on held-out controlled examples;
3. quantify uncertainty and boundary conditions;
4. only then search for an internal account;
5. test the account with predicted interventions.

For the spine model, loss on later letters of a repeated made-up pet name is 6.8–10.6 nats, compared with a uniform baseline near 3.9. This checkpoint does not exhibit the tested copying behaviour. The “best” induction-attention score of 0.02 therefore does not identify a weak copying mechanism; it is merely the maximum among small scores.

The absence of this behaviour also does not prove why it is absent. Familiar-name memorization, tokenization, limited context training and several other mechanisms remain possible. The correct conclusion stays at the measured level.

## 11.2.7 A template for mechanistic claims

Before writing “component C implements function F,” fill in:

| field | question |
|---|---|
| behaviour | what held-out phenomenon is present? |
| candidate | what observation nominated component C? |
| causal prediction | what should change under a specified intervention? |
| positive control | can the method recover a known effect? |
| negative control | what matched intervention should do little? |
| effect | what paired behavioural change occurred, with uncertainty? |
| specificity | do wrong components, positions or sources fail to reproduce it? |
| alternatives | which competing mechanisms remain? |
| scope | which inputs, checkpoint and intervention does the claim cover? |

Mechanistic interpretability becomes scientific when the explanation risks being wrong in advance. A story that fits every possible result is not an explanation; it is decoration.
