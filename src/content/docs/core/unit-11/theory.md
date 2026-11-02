---
title: "11.5 General theory"
description: Interventions in the language of structural causal models, direct and indirect effects, causal abstraction as the standard for a mechanistic claim, and what such a claim must predict to be worth making.
sidebar: { label: "11.5 General theory", order: 5 }
---

## 11.5.1 A network is a causal model

A neural network is a **structural causal model** in the sense of [Pearl (2009)](/references/#pearl-2009): every activation is a deterministic function of the activations it reads, and the computation graph of Unit 03 is its causal graph. Interventions are well-defined operations on it — replace one node's function by a constant and recompute everything downstream — which is exactly what `head_mask` and `patch` do in `spine/model.py`. Unlike causal inference in the natural sciences, there is no confounding we cannot control: we can set any variable to any value and run the model again. The difficulty is not identification but **interpretation** — the graph has millions of nodes, and the right *level* of description is unknown.

## 11.5.2 Direct, indirect and total effects

For a component $c$ on the path from input to output, distinguish:

- the **total effect** of changing the input from $x_{\text{x}}$ to $x_{\text{c}}$: $f(x_{\text{c}}) - f(x_{\text{x}})$;
- the **indirect effect through $c$**: $f(x_{\text{x}}\mid h_c\leftarrow h_c(x_{\text{c}})) - f(x_{\text{x}})$ — activation patching;
- the **direct effect** of $c$ on the output, holding every other path fixed — in the residual stream, its direct logit contribution (§11.3.1).

Mediation analysis of this kind was introduced to language models to study how gender information flows to predictions ([Vig et al. 2020](/references/#vig-2020)) and to locate where factual associations are stored ([Meng et al. 2022](/references/#meng-2022)). The hand-built circuit illustrates the distinction exactly: head 0.0 has *no* direct effect on the logits (it writes only into the previous-token block, which the read-out ignores) and a large indirect effect through head 1.0 — ablate it and the loss rises to 2.84.

## 11.5.3 Causal abstraction: what a mechanistic claim asserts

A claim such as "head 1.0 implements induction" asserts that a *high-level* algorithm — "find the earlier occurrence of my token, output what followed it" — describes the network's computation, with high-level variables mapped onto low-level components. **Causal abstraction** makes the claim testable ([Geiger et al. 2021](/references/#geiger-2021)): the high-level model is a faithful abstraction if, for every intervention on its variables, the corresponding intervention on the mapped network components produces the same output change. Interchange interventions — patching a component's activation from one input into another and checking that the output changes as the high-level algorithm says it should — are the experimental test. Worked Example 11.4 is a single interchange intervention; a full validation runs many, over many inputs.

The standard is demanding, and it should be: a mechanistic claim is a prediction about *all* interventions on the named components, not a description of one attention map.

## 11.5.4 What a mechanistic claim must predict

A good mechanistic claim predicts behaviour that has **not been tested yet**. "The spine model has memorized a list of sixteen names and has no copying mechanism" predicts failure on unseen animal names, on familiar names in unfamiliar positions, and success after training on many random names — each a new experiment. "The trained toy uses a distributed positional mechanism" predicts that permuting positional embeddings while keeping token content should break copying, and that the effect should be spread over heads. A claim that predicts nothing new is a description, however detailed. This is the bridge from interpretation to the methodology of Unit 12: mechanistic claims are hypotheses, and the way to treat a hypothesis is to state, before testing, what result would refute it.

## 11.5.5 Summary

| Concept | Says | Tool |
|---|---|---|
| network as causal model (§11.5.1) | interventions are well-defined and cheap | `head_mask`, `patch` |
| indirect effect (§11.5.2) | patching measures what a component carries | activation patching |
| direct effect (§11.5.2) | what a component writes to the output | direct logit attribution |
| causal abstraction (§11.5.3) | a mechanism claim = a high-level model faithful under interventions | interchange interventions |
| prediction (§11.5.4) | a claim earns its keep by predicting untested behaviour | new experiments, pre-registered |
