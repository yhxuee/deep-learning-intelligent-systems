---
title: Pre-registered ablation studies
description: Write the hypothesis, prediction, metric, analysis and decision rule before running; change one factor at a time across seeds; analyse exactly as registered; replicate.
atlas:
  id: method-preregistration
  type: method
  mastery: internalize
  core_units: [12]
  prerequisites: [method-bootstrap-ci]
  related: [method-activation-patching, method-scaling-law-fit]
  references: [nosek-2018, henderson-2018, pineau-2021]
  last_reviewed: 2026-11-02
---

## In one paragraph

Flexible analysis — choosing metrics, subsets, seeds or thresholds after seeing results — can make almost any hypothesis look supported. **Pre-registration** fixes those choices in advance and timestamps them. An **ablation** changes one factor at a time with everything else held fixed, over several seeds, and compares conditions with a paired analysis. A finding counts when the registered decision rule says so and an **independent replication** — fresh seeds at least, ideally another implementation — agrees.

## Invariants

- Same seed, same code, same data → identical result (spine `test_stage12_same_seed_same_result`).
- Editing a registered record is detected by its digest (spine `test_stage12_preregistration_detects_tampering`).
- The decision rule is applied mechanically: "needed", "not needed" or "inconclusive" from the interval alone (spine `test_stage12_decision_rule`).

## Known values

| Quantity | Value |
|---|---|
| Registered threshold in the spine capstone | 0.02 nats/char |
| Spine capstone outcome | see Unit 12, Worked Example 12.6 |

## Common bugs

- Registering the hypothesis but not the analysis, then choosing the analysis that works.
- Comparing conditions on different seeds (unpaired) and reading noise as an effect.
- Reporting only the best of several runs.
