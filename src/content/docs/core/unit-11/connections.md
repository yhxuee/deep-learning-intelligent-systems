---
title: "11.6 Connections"
description: Interpretation in neuroscience, causal inference, the philosophy of explanation and software reverse engineering — and within the book.
sidebar: { label: "11.6 Connections", order: 6 }
---

## 11.6.1 Neuroscience: recording versus lesioning

Neuroscience has the same three kinds of evidence. Recording a neuron's activity shows what it correlates with (observation); lesions and optogenetic silencing show what it is needed for (intervention); tuning curves and decoders show what information a population carries (probing). The field learned, sometimes painfully, that correlation and necessity come apart: a neuron that responds to faces need not be required for recognizing them. Interpretability can do what neuroscience cannot — ablate any unit, in any combination, on any input, at no cost — and should therefore hold itself to a higher standard of causal evidence.

## 11.6.2 Causal inference

The do-operator, mediation analysis and the distinction between direct and indirect effects come from causal inference ([Pearl 2009](/references/#pearl-2009)). In epidemiology or economics, interventions are expensive or impossible and much of the theory concerns when causal effects can be identified from observation. In a neural network every intervention is available, so the identification problem disappears and the remaining problems are choosing the right variables (§11.5.3) and the right reference distribution for ablations (§11.7.3).

## 11.6.3 Philosophy of science: explanation and understanding

What makes an explanation good? Philosophers of science distinguish explanations that *subsume* an event under a general law from explanations that describe the *mechanism* producing it, and argue that the latter support counterfactual reasoning — what would happen if a part were different. Mechanistic interpretability adopts the second view (§11.5.3). Whether a model that predicts well "understands" anything is a separate, contested question about the word ([Lipton 2018](/references/#lipton-2018)); this unit sidesteps it by asking only which computations are causally responsible for which behaviour.

## 11.6.4 Software reverse engineering

Reverse engineers recover algorithms from compiled binaries: they trace data flow, set breakpoints, patch instructions and watch what changes. Activation patching is a breakpoint with a patch; ablation is a NOP. The analogy has limits — a compiled program was written by someone who meant something by it, while a trained network's "algorithm" may have no clean description — but the working practice transfers: start from a behaviour, form a hypothesis about where it is computed, test it by intervention, and refine.

## 11.6.5 Inside the book

| This unit | Returns in |
|---|---|
| QK and OV circuits | Unit 04 (attention as soft lookup; the $QK^{\top}$ rank bound) |
| superposition, directions not units | Unit 02 (Johnson–Lindenstrauss, §2.5.3) |
| behaviour before mechanism | Unit 10 (measure first) |
| a mechanism claim must predict | Unit 12 (pre-registration, falsification) |
| memorized lists versus learned skills | Unit 01 (memorization vs generalization), Unit 07 (memorization) |
