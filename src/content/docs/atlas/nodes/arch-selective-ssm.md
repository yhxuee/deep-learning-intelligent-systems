---
title: Selective state-space model
description: Input-dependent (selective) recurrences such as Mamba — what selectivity buys, what it costs, and how to check an implementation.
atlas:
  id: arch-selective-ssm
  type: architecture
  mastery: understand-and-verify
  core_units: [5]
  prerequisites: [mech-linear-recurrence]
  predecessors: [mech-linear-recurrence]
  alternatives: [mech-attention, arch-transformer-block]
  references: [gu-2022, gu-2024, arora-2024, jelassi-2024]
  last_reviewed: 2026-11-02
---

## In one paragraph

A selective SSM discretizes a continuous-time linear system with an input-dependent step $\Delta_t$ and input-dependent $B_t, C_t$: $h_t = \bar A_t h_{t-1} + \bar B_t x_t$, $y_t = C_t h_t$, with $\bar A_t = \exp(\Delta_t A)$ [Gu & Dao 2024](/references/#gu-2024). Selectivity lets the model choose what to write and when to forget — content dependence like attention — with a fixed-size state and linear cost in sequence length. It gives up the time-invariant convolution form, so training uses a parallel scan.

## Invariants

- With $\Delta, B, C$ independent of the input it reduces to a time-invariant SSM ([linear recurrence](/atlas/nodes/mech-linear-recurrence/)).
- For real negative diagonal $A$: $\Delta_t\to0$ gives $\bar A_t\to I$ (hold the state); $\Delta_t\to\infty$ gives $\bar A_t\to0$ (reset).
- Scan output equals the sequential loop.

## Known limitation (evidence)

Fixed-state models trail attention on associative recall and copying at matched size [Arora et al. 2024](/references/#arora-2024); [Jelassi et al. 2024](/references/#jelassi-2024), consistent with the capacity argument in [Unit 05](/core/unit-05/). Hybrid stacks mixing attention and SSM layers are an active research area — classify claims about them as *open hypothesis* unless a controlled comparison is cited.
