---
title: Contamination and leakage checks
description: Detecting evaluation data inside training data — exact and n-gram matching, normalization, and what a clean check does not prove.
atlas:
  id: method-contamination-check
  type: method
  mastery: understand-and-verify
  core_units: [7, 10]
  related: [mech-cross-entropy]
  references: [sainz-2023, kapoor-2023, lee-2022]
  last_reviewed: 2026-11-02
---

## In one paragraph

Leakage — information about the test set reaching training — is a leading cause of irreproducible machine-learning results [Kapoor & Narayanan 2023](/references/#kapoor-2023). For pretrained language models the main form is benchmark **contamination** [Sainz et al. 2023](/references/#sainz-2023). Basic checks: normalize text (case, whitespace, punctuation), then look for exact matches and long n-gram overlaps between evaluation items and the training corpus; deduplicate the corpus itself [Lee et al. 2022](/references/#lee-2022).

## Invariants

- A planted exact copy of a test item must be detected (recall 1.0 on planted duplicates).
- Normalization must be identical for both corpora.
- Splits by *entity* (user, patient, document) must not share entities, not merely rows.

## What a clean check does not prove

Absence of n-gram overlap does not rule out paraphrased, translated or reformatted contamination, and gives no guarantee for closed-data models. Report the method and its blind spots alongside the result.

## Common bugs

- Different tokenization or normalization for the two sides.
- Checking rows instead of entities (e.g. the same patient in train and test).
- Running the check after hyperparameters were tuned on the contaminated split.
