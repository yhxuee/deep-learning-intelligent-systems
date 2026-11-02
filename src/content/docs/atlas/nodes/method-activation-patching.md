---
title: Ablation and activation patching
description: Interventions that test whether a model component is used — remove it (ablation) or replace its activation with one from another input (patching) — with the controls that make the result interpretable.
atlas:
  id: method-activation-patching
  type: method
  mastery: understand-and-verify
  core_units: [11]
  prerequisites: [mech-attention, arch-transformer-block]
  related: [method-preregistration]
  references: [meng-2022, wang-2023, zhang-nanda-2024]
  last_reviewed: 2026-11-02
---

## In one paragraph

Observational evidence — attention patterns, probes, attributions — shows what information is present or correlated. **Interventions** show what is used. *Ablation* replaces a component's output by zero (or its mean) and measures the change in behaviour. *Activation patching* runs the model on a "corrupted" input and replaces one component's activation with its value on a "clean" input; if the clean behaviour is restored, that component carries the difference. Both need controls: ablate components that should not matter, patch with activations from irrelevant inputs, and validate the whole pipeline on a model whose mechanism is known.

## Invariants

- Patching a run with its own activations changes nothing (spine `test_stage11_patching_a_run_with_its_own_activations_is_a_no_op`).
- Zero-ablating a head equals patching it with zeros (spine `test_ablating_a_head_equals_patching_it_with_zeros`).
- On the hand-built induction circuit, ablating either circuit head destroys copying and ablating the unused heads does not (spine `test_stage11_hand_built_circuit_is_found_used_and_needed`).

## Known values

| Quantity | Value |
|---|---|
| Hand-built circuit, loss on the repeated half: intact / circuit head ablated / unused head ablated | < 0.01 / > 1 / < 0.01 nats |

## Common bugs

- Concluding "this head implements X" from an attention pattern alone.
- Zero-ablation pushing activations far off-distribution, so the measured effect reflects the shock, not the mechanism; compare with mean-ablation.
- Patching from a corrupted input that differs from the clean one in more than the variable of interest.
