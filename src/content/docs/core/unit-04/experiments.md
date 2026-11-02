---
title: "4.9 Experiments"
description: Eight experiments — soft lookup, fixed mixing versus attention, scaling and entropy, invariants and the bug hunt, convolution reach, quadratic scaling, the online softmax, and spine Stage 04.
sidebar: { label: "4.9 Experiments", order: 9 }
---

Write a prediction before running each experiment. Code: [`labs/unit-04-attention`](https://github.com/yhxuee/deep-learning-intelligent-systems/tree/main/labs/unit-04-attention) (AI-off construction in `attention.py`; invariants in `invariants.py`; bug hunt in `buggy_variants.py`), [`spine/`](https://github.com/yhxuee/deep-learning-intelligent-systems/tree/main/spine), and [`figures/make_figures.py`](https://github.com/yhxuee/deep-learning-intelligent-systems/tree/main/figures).

## Experiment 4.1 — Soft lookup and temperature

**Tests:** §4.2.7, Worked Example 4.1. **Predict:** the output for $\beta\in\{0, 1, 2, 4, 8\}$, and how it changes if a fourth pair $(d, 5)$ is added.

**Procedure.** Implement the hand-built $W_Q, W_K, W_V$ in NumPy and evaluate. Then shuffle the pairs and change the query.

**Look for:** the values of Figure 4.1; invariance to shuffling; a lower output at fixed $\beta$ with more distractors, because the match weight is $e^\beta/(e^\beta + n)$ with $n$ non-matching positions.

## Experiment 4.2 — Fixed mixing versus attention, trained

**Tests:** Theorem 4.1, Worked Example 4.2. **Predict:** the best mean-squared error a trained fixed-mixing model reaches on random lookup sequences with 3 pairs, and the error of a one-layer attention model.

**Procedure** (framework autograd allowed). Generate random lookup sequences with keys from 8 symbols and values in $\{0,\dots,9\}$. Train (a) a model with learned fixed mixing weights plus linear layers, (b) the same with an MLP after mixing, (c) one attention layer. Equal parameter budgets; three seeds.

**Look for:** (a) stuck near the error of predicting the mean value; (b) partial improvement that degrades as the number of pairs grows; (c) near-zero error. Report seed variance.

## Experiment 4.3 — Scaling, saturation and entropy

**Tests:** §4.2.8, §4.5.2. **Predict:** the mean attention entropy over 64 random keys for $d_k = 4$ and $d_k = 512$, with and without the $1/\sqrt{d_k}$ factor.

**Procedure.** Reproduce Figure 4.2 (`u04_softmax_saturation`); add entropy; then compute the Frobenius norm of the softmax Jacobian averaged over rows.

**Look for:** entropy near 3.7 nats with scaling for all $d_k$; collapse toward 0 without it; Jacobian norms that collapse with entropy — the vanishing-gradient mechanism.

## Experiment 4.4 — Invariants and the bug hunt

**Tests:** Propositions 4.1–4.2, §4.3.10, §4.7.7. **Predict:** for each of the five variants in `buggy_variants.py`, which invariants fail. Write your answers in `bug_hunt_predictions.py` first.

**Procedure.** Implement `attention.py` (AI-off) and run `python -m pytest` in the lab directory; then `python -m pytest test_bug_hunt.py -v -s`. Finally, give the five variants to an AI assistant *without* the tests and ask it to review them.

**Look for:** your implementation passing all invariants; each variant failing at least one; the reshape bug passing permutation equivariance and failing only reference comparison and KV-cache equivalence. Record which bugs the AI review missed.

## Experiment 4.5 — How far can convolutions see?

**Tests:** §4.2.5. **Predict:** the largest key–query distance at which a stack of $L$ width-3 convolutions (with nonlinearities) can solve lookup, for $L\in\{2, 4, 8\}$.

**Procedure.** Separated key–value lookup with controlled distance between the stored pair and the query; train convolution stacks and a one-layer attention model.

**Look for:** a sharp failure beyond distance $2L$ for convolutions; no distance dependence for attention within the context.

## Experiment 4.6 — Quadratic cost, measured

**Tests:** §4.5.6, Worked Example 4.8. **Predict:** the ratio of wall-clock time for one attention layer at $T = 4\,096$ versus $T = 1\,024$, for $d = 768$, in NumPy on a CPU.

**Procedure.** Time `mha` from the reference implementation for $T\in\{256, 512, \dots, 8192\}$; separately time the projections.

**Look for:** attention time growing roughly 4× per doubling of $T$ at large $T$, projections roughly 2×; deviations caused by memory effects (Unit 06).

## Experiment 4.7 — Online softmax

**Tests:** §4.8.1, Worked Example 4.9. **Predict:** the maximum absolute difference between blocked and direct attention for random scores of standard deviation 30, in float32 and float64.

**Procedure.** Implement the blocked update for one query with block sizes 1, 7 and 64; compare with direct softmax computed stably.

**Look for:** agreement to round-off at every block size; a naive implementation *without* the running maximum overflowing for large scores.

## Experiment 4.8 — Spine Stage 04: attention in the spine model

**Tests:** §4.3.9, the [spine project](/spine/). **Predict:** the validation loss of the spine model with one causal attention block versus the Stage 03 bigram, and its loss at initialization.

**Procedure.** Add one pre-LN Transformer block with learned positions to the spine model (a framework's autograd is allowed from here on; the reference `spine/model.py` keeps hand-derived gradients, checked by `spine/test_model.py`). Before training, run all attention invariants on your block. Train with AdamW; record validation loss.

**Look for:** loss at initialization near $\ln V$; all invariants passing; validation loss below the bigram's because the model can now use more than one previous character. If invariants fail, find the bug before interpreting any loss.
