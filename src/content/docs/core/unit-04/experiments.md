---
title: "4.9 Experiments"
description: Eight experiments — soft lookup, fixed mixing versus attention, scaling and entropy, invariants and the bug hunt, convolution reach, quadratic scaling, the online softmax, and spine Stage 04.
sidebar: { label: "4.9 Experiments", order: 9 }
---

Write a prediction before running each experiment. For every result, record the random seed, data split, dtype, hardware and the quantity actually measured; a curve without its conditions is not reproducible evidence. If the result disagrees with the prediction, first separate a mathematical contradiction from an optimization failure or a measurement artifact.

Code: [`labs/unit-04-attention`](https://github.com/yhxuee/deep-learning-intelligent-systems/tree/main/labs/unit-04-attention) (AI-off construction in `attention.py`; invariants in `invariants.py`; bug hunt in `buggy_variants.py`), [`spine/`](https://github.com/yhxuee/deep-learning-intelligent-systems/tree/main/spine), and [`figures/make_figures.py`](https://github.com/yhxuee/deep-learning-intelligent-systems/tree/main/figures). Experiments 4.1 and 4.3 can first be explored in the [interactive figures](/labs/interactive/).

## Experiment 4.1 — Soft lookup and temperature

**Tests:** §4.2.7, Worked Example 4.1. **Predict:** the output for $\beta\in\{0, 1, 2, 4, 8\}$, and how it changes if a fourth pair $(d, 5)$ is added.

**Procedure.** Implement the hand-built $W_Q, W_K, W_V$ in NumPy and evaluate. Then shuffle the pairs and change the query.

**Look for:** the values of Figure 4.1 and invariance to shuffling the key--value pairs. With $n$ zero-score non-matches, the matching weight is $e^\beta/(e^\beta+n)$, so another distractor always reduces the *matching weight*. The output itself need not always decrease: it moves toward the added distractor's value. For the specified value 5, which lies below the previous output, it decreases. This distinction between a weight and the weighted sum is exactly the caution of §4.7.6.

## Experiment 4.2 — Fixed mixing versus attention, trained

**Tests:** Theorem 4.1, Worked Example 4.2. **Predict:** the train and held-out mean-squared error of each model on random lookup sequences with 3 pairs, then how the errors change at 5 and 7 pairs.

**Procedure** (framework autograd allowed). Generate separate training and test sequences with keys from 8 symbols and values in $\{0,\dots,9\}$. Train (a) a model with learned, input-independent mixing weights plus linear layers, (b) the same model with an MLP after mixing, and (c) one attention layer. Match parameter budgets as closely as possible and keep the optimizer, update count and training examples fixed; use at least three seeds. Report both the achieved result and the budget mismatch rather than silently changing width.

**Look for:** model (a) approaching a mean-like predictor, model (b) improving on it but becoming less reliable as the number of pairs grows, and attention learning a much lower-error lookup rule. Report seed variance and failed runs. Theorem 4.1 proves an impossibility only for the affine fixed-mixing class; it does **not** guarantee these particular training outcomes, nor does it rule out every sufficiently large nonlinear fixed-connectivity network. This experiment measures the practical gap under the stated budget.

## Experiment 4.3 — Scaling, saturation and entropy

**Tests:** §4.2.8, §4.5.2. **Predict:** the mean attention entropy over 64 random keys for $d_k = 4$ and $d_k = 512$, with and without the $1/\sqrt{d_k}$ factor. Entropy here uses natural logarithms, so its unit is the **nat** and the maximum is $\ln64\approx4.159$ nats.

**Procedure.** Draw independent standard-normal queries and keys, reproduce Figure 4.2 (`u04_softmax_saturation`), add entropy, and average over enough rows and at least three seeds to show uncertainty. Then compute the Frobenius norm of each row's softmax Jacobian and average it. Keep the number of keys fixed when comparing dimensions.

**Look for:** with scaling, score variance and mean entropy remain similar as $d_k$ changes (about 3.7 nats for this simulation). Without scaling, entropy decreases as $d_k$ grows and is near zero at $d_k=512$; it should **not** already be near zero at $d_k=4$. The softmax-Jacobian norm becomes small in the highly saturated regime. This demonstrates one score-gradient mechanism, not the claim that every gradient in the model vanishes.

## Experiment 4.4 — Invariants and the bug hunt

**Tests:** Propositions 4.1–4.2, §4.3.10, §4.7.7. **Predict:** for each of the five variants in `buggy_variants.py`, which invariants fail. Write your answers in `bug_hunt_predictions.py` first.

**Procedure.** Implement `attention.py` (AI-off) and run `python -m pytest` in the lab directory; then `python -m pytest test_bug_hunt.py -v -s`. Finally, give the five variants to an AI assistant *without* the tests and ask it to review them.

**Look for:** your implementation passing all invariants and each planted variant failing at least one check. In the current fixtures, the reshape bug passes the tested symmetry checks but fails KV-cache equivalence and comparison with the independent reference. Record the assistant model, prompt and which bugs it misses; one review is an observation about that review, not a general accuracy estimate for AI systems.

## Experiment 4.5 — How far can convolutions see?

**Tests:** §4.2.5. **Predict:** the largest past key--query distance reachable by a stack of $L$ **causal** width-3 convolutions, for $L\in\{2,4,8\}$. Use offsets $\{0,1,2\}$, so the structural limit is $2L$ positions into the past. A centered width-3 convolution would instead expand by only one position on each side per layer.

**Procedure.** Construct separate training and test sets for key--value lookup with a controlled distance between the stored pair and query. Train the convolution stacks and a one-layer causal-attention model with matched widths, training examples and update budgets. Evaluate every distance, including values just below, at and above $2L$.

**Look for:** the convolution cannot use the stored value beyond distance $2L$, regardless of optimization, because no computational path connects it to the query. It may still fail *inside* that boundary because training is imperfect. Attention has a direct edge to every earlier position, so it has no analogous hard distance cutoff inside the context; its empirical accuracy may nevertheless vary with distance.

## Experiment 4.6 — Quadratic cost, measured

**Tests:** §4.5.6, Worked Example 4.8. **Predict:** the arithmetic model predicts a ratio near $16$ for the pairwise part at $T=4{,}096$ versus $T=1{,}024$, while the projection part predicts a ratio near $4$. Wall-clock ratios need not equal either number.

**Procedure.** First estimate memory. Merely holding one float64 score matrix and one weight matrix needs at least $2T^2\times8$ bytes: about 1.07 GB at $T=8{,}192$, before other arrays. Choose a safe maximum for your machine; $T\in\{256,512,1024,2048,4096\}$ is enough to expose the trend, and 8,192 is optional. After warm-up, time several repetitions of `mha` from the reference implementation and report the median; separately time the projections. Fix dtype and numerical-library thread settings, and do not run other large jobs simultaneously.

**Look for:** the pairwise work approaching 4× per doubling of $T$ and the projections approaching 2×, once fixed overhead is small. Cache, allocation, threading and memory-bandwidth effects can cause substantial deviations (Unit 06). A measured wall-clock curve supports a hardware-specific observation; the $T^2$ operation count is the derivation.

## Experiment 4.7 — Online softmax

**Tests:** §4.8.1, Worked Example 4.9. **Predict:** the maximum absolute difference between blocked and stable direct attention for random scores of standard deviation 30, in float32 and float64. Predict which dtype has the larger rounding discrepancy.

**Procedure.** Implement the blocked update for one query with block sizes 1, 7 and 64; compare with direct softmax computed by subtracting its maximum. Then exploit shift invariance: add 100 to every float32 score and 800 to every float64 score. The stable answers should stay the same, whereas a naive implementation that exponentiates raw scores is forced to overflow.

**Look for:** agreement to dtype-appropriate round-off at every block size, not bitwise identity. Random standard-deviation-30 scores alone do not guarantee overflow in every draw; the positive shifts make the failure deterministic while leaving the mathematical softmax unchanged. Also test an all-masked row and verify that your API either rejects it or handles it explicitly (§4.7.8).

## Experiment 4.8 — Spine Stage 04: attention in the spine model

**Tests:** §4.3.9, the [spine project](/spine/). **Predict:** the validation loss of the spine model with one causal attention block versus the Stage 03 bigram, and its loss at initialization. A uniform prediction over vocabulary size $V$ has cross-entropy $\ln V$; a random initialization should be near, but not exactly at, this value.

**Procedure.** Add one pre-LN Transformer block with learned positions to the spine model (a framework's autograd is allowed from here on; the reference `spine/model.py` keeps hand-derived gradients, checked by `spine/test_model.py`). Before training, run all attention invariants on your block. Use the same train/validation split and evaluation code as Stage 03, fix the token and update budgets, train at least three seeds with AdamW, and record initialization, training and validation losses.

**Look for:** initialization loss near $\ln V$, all invariants passing, and validation loss below the bigram baseline because the attention block has access to more than one previous character. Treat the last point as the experimental hypothesis, not a guarantee: an optimization bug, insufficient training or overfitting can reverse it. If an invariant fails, fix the implementation before using its loss as evidence about the architecture.
