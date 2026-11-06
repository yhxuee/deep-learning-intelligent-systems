---
title: "Unit 11 · What has a model actually learned?"
description: A beginner-friendly path from behaviour to representation and causal mechanism, using a hand-built induction circuit, controlled interventions, a trained counterexample, and a model with no copying behaviour to explain.
sidebar: { label: "11.1 The problem", order: 1 }
unit:
  number: 11
  part: 4
  question: When does successful prediction justify a claim about mechanism or understanding?
  anchor: Ablation and activation patching, validated on a two-layer induction circuit built by hand, then applied to a trained attention-only model and to the spine model's missing ability to copy made-up names.
  spine: "Stage 11 — verify attention scores, ablation and patching on a hand-built induction circuit; train an attention-only model to copy and test the textbook induction story; check whether the spine model copies novel names before searching for a copying mechanism."
  checkpoint: Construct the weights of a two-layer induction circuit by hand; design an intervention that distinguishes two explanations of the same behaviour; explain why an attention pattern alone does not establish a mechanism.
  status: draft
  atlas: [method-activation-patching, mech-attention, mech-residual-connection, arch-transformer-block]
  answer: "A mechanistic claim is a causal claim about internal computation. Decodability shows that information is available, attribution associates components with an output, and controlled interventions test whether a component's state makes a difference under the intervention used."
  leaves: "Every interpretability result is an experiment with assumptions, controls and possible failures. Unit 12 asks how such findings become reliable discoveries."
  builds_on:
    - { unit: 4, uses: "attention heads and the residual stream" }
    - { unit: 10, uses: "evaluation with controls and bounded claims" }
---

## 11.1 The problem — a correct answer does not reveal the computation

Imagine two students who both answer $17+25=42$.

- One added the units and tens columns.
- One remembered this exact question.
- One copied from a neighbour.
- One guessed and happened to be right.

The visible behaviour is identical. The internal process is not.

A neural network creates the same scientific problem. Input–output tests can show **what** the network does on tested cases. They do not uniquely determine **how** it produced the answer. Many internal computations can implement the same function, and the same component can matter differently on different inputs.

Unit 10 ended with a concrete puzzle. The spine model performs well on held-out stories drawn from the familiar generator, but its loss rises sharply when names are replaced by unseen ones. A tempting explanation is that it memorized familiar name continuations instead of learning to copy a name from context. That explanation is plausible—but it is still a hypothesis.

This unit asks how to move from behaviour to a mechanism claim without turning a plausible story into a fact.

## Three questions that must not be merged

### 1. Can the model perform the behaviour?

Give it controlled inputs and measure the output. For copying, show a novel string once, repeat its prefix later, and ask whether the model predicts the remaining characters better than an appropriate baseline.

This is a **behavioural** question. If the behaviour is absent, searching for “the copying circuit” is premature.

### 2. Is relevant information present internally?

Train a probe to decode the next character from a hidden state, or inspect whether an attention head points toward an earlier occurrence.

This is a **representation or association** question. The information may be present without being used. A probe may also learn the task from incidental features or memorize examples.

### 3. Does a component causally contribute?

Change the component while controlling the rest: ablate a head, patch an activation from another run, or intervene on a feature direction. Measure a predeclared behavioural effect and compare with controls.

This is an **intervention** question. It is stronger, but not assumption-free. Zeroing an activation may create a state the model never encounters naturally; patching may alter several entangled features at once.

The evidence ladder is therefore not

$$
\text{attention picture}\Rightarrow\text{mechanism proved}.
$$

It is closer to

$$
\text{behaviour}
\longrightarrow
\text{candidate representation}
\longrightarrow
\text{controlled intervention}
\longrightarrow
\text{bounded causal claim}.
$$

Each arrow needs a test.

## The running example — induction as contextual copying

An **induction pattern** has the form

$$
[A][B]\;\ldots\;[A]\longrightarrow[B].
$$

After seeing “A followed by B” once, the model sees A again and predicts B. One well-studied Transformer mechanism uses two attention heads in sequence:

1. a **previous-token head** writes information about the preceding token into the residual stream;
2. an **induction head** matches the current token to that stored information, attends to what followed its earlier occurrence, and copies the associated token.

This circuit has been described in mechanistic-interpretability work ([Elhage et al. 2021](/references/#elhage-2021); [Olsson et al. 2022](/references/#olsson-2022)). It is an excellent teaching example because the proposed mechanism makes precise predictions about attention, ablation and activation patching.

It is not a universal law that every copying model must implement. A network can solve a copying task using another distributed computation. “The literature found an induction head” is a source of hypotheses, not permission to assume one exists in every model.

## Why we begin with a circuit whose answer is known

Interpretability tools are themselves measuring instruments. Before trusting them on an unknown trained network, we should test them on a case with ground truth.

Stage 11 therefore builds a two-layer, two-head attention-only Transformer **by hand**:

- head 0.0 implements the previous-token operation;
- head 1.0 implements induction and writes the copied token into the logits;
- heads 0.1 and 1.1 are deliberately unused controls.

Because the weights were constructed, we know the intended circuit. A valid method should recover the two active heads, show that control heads do little under the chosen intervention, and restore clean behaviour when the relevant clean activation is patched into a corrupted run.

This still does not prove that the method works on every trained model. It is a unit test: failure would invalidate the method immediately; success earns permission for a harder test.

## Three cases, three different conclusions

### Case A — the hand-built mechanism is known

The circuit predicts the repeated segment with loss about 0.0004 nats per token. Its attention scores, direct logit contributions, ablations and patches recover the components we installed.

**Permitted claim:** under this architecture and these interventions, the tools recover the constructed circuit.

### Case B — the behaviour is learned, but the familiar story fails

The same broad attention-only architecture is trained on repeated segments with random gaps. Its evaluation loss on the predictable part falls to about 0.26 nats, far below the uniform baseline $\ln 24\approx3.18$.

Yet the largest textbook induction-attention score is only about 0.02, and zero-ablating any of the four heads damages copying. This is evidence for a distributed, non-textbook solution under those measurements. It does not by itself identify the alternative mechanism, and zero ablation does not establish that each head is uniquely necessary under every possible intervention.

**Permitted claim:** the model copies, but the tested single-head attention pattern does not explain the behaviour.

### Case C — the target behaviour is absent

The pretrained spine model is shown a made-up pet name twice. On later letters of the second mention, its loss is worse than uniform guessing and its best induction-attention score is about 0.02.

The first conclusion is behavioural: this checkpoint does not exhibit the tested copying ability. A weak attention score is then unsurprising. These results are consistent with reliance on familiar name continuations, but do not prove a stored-name-list mechanism.

**Permitted claim:** there is no successful copying behaviour in this probe for which a copying mechanism needs to be located.

The order matters:

$$
\boxed{\text{Verify the behaviour before explaining its mechanism.}}
$$

## Predict before reading on

For the hand-built circuit, make three predictions.

1. If head 1.0 is truly the copying head, what should happen to repeat loss when it is zero-ablated?
2. What should happen when unused head 1.1 is ablated by the same procedure?
3. A clean run and a corrupted run differ only in the first copy of the sequence. What should happen if head 1.0's clean output is patched into the corrupted run?

Do not merely predict “up” or “down.” Rank the three losses and explain which alternative explanation each control rules out. Worked Examples 11.3–11.4 return to your predictions.

## What this unit builds

1. **First principles (§11.2):** behaviour, representation and mechanism; observation, attribution and intervention; method validation on known ground truth.
2. **Mathematical construction (§11.3):** residual-stream decomposition; QK and OV circuits; composition across layers; a hand-built induction circuit; ablation, patching, probes and gradient attribution.
3. **Worked examples (§11.4):** exact attention scores, direct logit contributions, controlled ablations, patching, learned counterexamples and probe selectivity.
4. **General theory and connections (§§11.5–11.6):** causal models, abstraction, neuroscience, causal inference, philosophy of science and reverse engineering.
5. **Limits and current practice (§§11.7–11.8):** off-manifold interventions, distributed mechanisms, sparse dictionaries, automated discovery and world-model claims.
6. **Experiments and problems (§§11.9–11.10):** prediction-first intervention studies and counterexamples.

## Prerequisites

- **Unit 04:** queries, keys, values, multi-head attention and the residual stream (§§4.3.5–4.3.9).
- **Unit 10:** controls, paired comparisons, independent units and bounded evidence claims.
- **Unit 02:** one-hot codes, embedding geometry and superposition (§2.5.3).

Every needed concept is briefly rebuilt before use. If matrix notation is new, focus first on the question each matrix answers: **where does the head read, what does it copy, and where does it write?**

## How to read and verify this unit

The main text is self-contained. Fixed numbers come from `spine/stage11_induction.py` and are checked in `spine/test_spine.py` or `labs/test_worked_examples_u09_u12.py`. Intervention hooks are implemented in `spine/model.py` as `head_mask` and `patch`.

Passing those tests establishes arithmetic and software behaviour for the tested cases. It does not certify that an interpretation is uniquely true; that requires the controls and alternative explanations developed throughout the unit.

| Section | Content | Approx. reading time |
|---|---|---|
| [11.2 First principles](/core/unit-11/first-principles/) | three levels of claim, four evidence types and ground-truth validation | 75 min |
| [11.3 Mathematical construction](/core/unit-11/construction/) | residual stream, QK/OV, composition, interventions, probes and attribution | 2–3 h |
| [11.4 Worked examples](/core/unit-11/worked-examples/) | eight complete mechanism audits | 90 min |
| [11.5 General theory](/core/unit-11/theory/) | causal models, abstraction and faithfulness | 90 min |
| [11.6 Connections](/core/unit-11/connections/) | neuroscience, causal inference, philosophy and reverse engineering | 45 min |
| [11.7 Limitations and counterexamples](/core/unit-11/limitations/) | illusions, distributed mechanisms and intervention artefacts | 50 min |
| [11.8 Modern developments](/core/unit-11/modern/) | sparse dictionaries, automated circuit discovery and world models | 50 min |
| [11.9 Experiments](/core/unit-11/experiments/) | six controlled intervention experiments | lab time |
| [11.10 Problems and further study](/core/unit-11/problems/) | graded calculations, proofs, audits and reading | — |
