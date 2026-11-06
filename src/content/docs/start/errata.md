---
title: Errata and review status
description: Every correction made after publication, with the finding it answers, and what kind of review each part of the book has had.
sidebar: { order: 4 }
---

A textbook that asks readers to verify claims must make its own corrections visible. This page lists every substantive correction since v2.0, and keeps two kinds of review apart.

## Two review signals

| Signal | What it establishes | What it does not |
|---|---|---|
| **Executed validation** — the test suite and the build pass in CI | every quoted number is reproduced by code; gradients agree with finite differences; links and citations resolve | that a theorem's hypotheses and quantifiers are right, that a proof is complete, that prose states the scope of a claim correctly |
| **Scientific editorial review** — a named, dated reading by a subject expert | that statements, proofs and scope are correct | that the code reproduces the numbers |

Neither implies the other. As of v2.3, every unit has executed validation and has received an internal beginner-oriented editorial rewrite; **no unit has completed named scientific editorial review or a learner pilot**. The external audit below is a partial review: it read the general-theory pages of all twelve units and selected other sections, and did not check every worked example or problem.

## Corrections in v2.1 (audit of 3 November 2026)

An independent scientific audit of v2.0 (revision `24928ea`) reported sixteen findings. All are corrected in both languages; findings with a computable counterexample have a regression test that fails if the error returns.

| ID | Severity | Where | Was | Now | Test |
|---|---|---|---|---|---|
| K01 | critical | [Unit 09 §9.3.2](/core/unit-09/construction/) | discounted policy-gradient theorem without $\gamma^t$ | factor $\gamma^t$ in theorem and proof; note on implementations that drop it | `spine/test_spine.py::test_stage09_discounted_policy_gradient_carries_gamma_t` |
| K02 | major | [Unit 09 §9.3.3, §9.5.1](/core/unit-09/theory/) | $V^\pi$ called the variance-minimizing baseline | $V^\pi$ is common, not optimal; counterexample with $b^\star = 0.2$, $V = 0.8$ | `…::test_stage09_value_baseline_is_not_the_variance_optimal_baseline` |
| K03 | major | `spine/stage09_post_training.py`; [Unit 09 §9.2.3, §9.3.7](/core/unit-09/construction/) | batch-mean baseline including the sample's own reward (bias factor $(B-1)/B$), described as unbiased | leave-one-out baseline; bias of the batch and group mean stated; Stage 09 re-run and numbers updated | `…::test_stage09_batch_mean_baseline_is_biased_and_leave_one_out_is_not` |
| K04 | major | [Unit 05 §5.5.1](/core/unit-05/theory/) | transfer-function poles "exactly" the eigenvalues of $A$ | every pole is an eigenvalue; equality only for minimal realizations; internal vs input–output stability | `labs/test_audit_regressions.py::test_k04_…` |
| K05 | major | [Unit 05 §5.5.4](/core/unit-05/theory/) | $O(TN)$ for a generic recurrence | $O(TN^2)$ dense; $O(TN)$ diagonal; FFT needs kernel precomputation and time invariance | `…::test_k05_…` |
| K06 | major | [Unit 09 §9.5.2](/core/unit-09/theory/) | exact KL-regularized optimum called mode-seeking | exponential tilting, restricted-family approximation and optimization collapse distinguished | — |
| K07 | major | [Unit 10 §10.3.4](/core/unit-10/construction/) | top-label score presented as the Brier score | multiclass Brier score defined; top-label variant labelled; example where they differ | `…::test_k07_…` |
| K08 | moderate | [Unit 09 §9.3.5](/core/unit-09/construction/) | PPO clipping "prevents" large updates | clipping removes an incentive, not a constraint; KL monitoring | — |
| K09 | moderate | [Unit 01 §1.5.1](/core/unit-01/theory/) | one-sided theorem used two-sided in the corollary | two-sided uniform bound stated and proved | `…::test_k09_…` |
| K10 | moderate | [Unit 02 §2.5.4](/core/unit-02/theory/) | permutation-equivariant layer "two parameters" in general | single-channel case; multichannel form $XA + \mathbf 1\mathbf 1^{\top}XB$ | — |
| K11 | moderate | [Unit 06 §6.6.4, Example 6.8](/core/unit-06/worked-examples/) | Landauer bound per bit compared with joules per FLOP | comparison removed; bound stated per erased bit | — |
| K12 | moderate | [Unit 10 Example 10.8](/core/unit-10/worked-examples/) | interval-precision sample size read as a powered comparison | precision (766) and 80% power (1 564) computed separately | `…::test_k12_…` |
| K13 | minor | [Unit 01 §1.3.3](/core/unit-01/construction/) | $n$ normalizing $n-1$ transitions | $(n-1)\hat L$ | `…::test_k13_…` |
| K14 | minor | [Unit 10 §10.5.3](/core/unit-10/theory/) | conformal quantile undefined when its index exceeds $n$ | $\hat q = +\infty$; ties; marginal coverage | `…::test_k14_…` |
| K15 | editorial | [Unit 05 §5.7.6](/core/unit-05/limitations/) | "the KV cache never forgets" | retention distinguished from recall | — |
| K16 | editorial | [Unit 09 §9.5.3](/core/unit-09/theory/) | zero gradient "if the model never produces a correct answer" | zero support vs zero successes in a finite sample | — |

## Corrections in v2.2

Found by the checks added in v2.2 rather than by a reader. Each is corrected in both Chinese editions as well.

| Where | Was | Now | Found by |
|---|---|---|---|
| [Unit 04 §4.3.6](/core/unit-04/construction/) | English cited Worked Example 4.3 for the backward pass through attention | Worked Example 4.4, as the Chinese text already did | `scripts/check-bilingual.mjs` |
| [Unit 05 §5.2.1](/core/unit-05/first-principles/) | "no state smaller than the window" can answer a sliding-window query | no state with fewer distinguishable values than there are possible windows (§5.5.2) | overstatement pass |
| Units 02, 06, 07, 08, 09, 10, 11 (limitations) | **[E]** labels with no named source | each names its publication (four new bibliography entries) or the book's own experiment | `scripts/check-evidence.mjs` |
| [Unit 08 Experiment 8.7](/core/unit-08/experiments/) (new in v2.2, before release) | cited §8.7.2 for few-step bias | §8.7.3 | proofreading |

## Review ledger

Each row records what kind of review a part of the book has had, at which revision, and with what outcome. A named scientific review is added here when it happens, with the reviewer's name, the date, the revision read, the pages covered and the issues it opened.

| Part | Executed validation (every commit, in CI) | Audit | Named scientific review |
|---|---|---|---|
| Units 01–12 | tests for every quoted number and gradient; links; citations; math typesetting; English–Chinese agreement of display mathematics and result numbers (from v2.2); **[E]** sources (from v2.2) | 3 November 2026, revision `24928ea`: general-theory pages of all units and selected other sections; K01–K16, all corrected in v2.1 | none yet |
| Appendices A–H | every example recomputed by `labs/test_appendices.py` | none (written after the audit) | none yet |
| Spine project | `spine/test_spine.py`, including the Stage 09 baseline and discount regressions | Stage 09 baseline (K03) | none yet |
| Traditional Chinese edition | the same checks; display mathematics and result numbers compared with English | — | no named translation reviewer yet |
| Simplified Chinese edition | generated from Traditional Chinese by `scripts/zh_cn.py`; build fails if it is out of date or if a listed Taiwan-usage term survives | — | no named translation reviewer yet |
| Labs and experiments | `labs/` test suites, including the beyond-language generators (from v2.2) | — | no learner pilot yet |

The v2.3 rewrite changed exposition throughout the Core, appendices and onboarding pages. It was checked for generated-edition consistency, displayed mathematics, numbered-result parity, cited evidence and executable examples; it was not a named external pedagogical review. This distinction prevents an extensive authorial revision from being reported as independent validation.

"None yet" is a statement of fact, not a judgment of quality: executed validation can show that the numbers are reproduced, not that the theorems are stated correctly.

## How to report an error

Open an issue on the [repository](https://github.com/yhxuee/deep-learning-intelligent-systems/issues) with the page, the claim, and — if you can — a counterexample or a computation. A confirmed error is corrected in both languages in the same change, listed here, and, where possible, given a test.
