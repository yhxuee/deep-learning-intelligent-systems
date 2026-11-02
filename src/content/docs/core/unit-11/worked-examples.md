---
title: "11.4 Worked examples"
description: Eight examples — attention scores of the hand-built circuit, its loss and direct logit contribution, ablations with controls, a patching experiment, a trained model that copies without the textbook pattern, a model that does not copy at all, a probe's selectivity, and what the KL of post-training does not say.
sidebar: { label: "11.4 Worked examples", order: 4 }
---

Numbers are produced by `spine/stage11_induction.py` and checked in `spine/test_spine.py`; the hand-built circuit uses $K = 8$, $c = 60$, $\lambda = 10$ ($d = 36$, $d_h = 18$).

## Worked Example 11.1 — How sharp is the hand-built attention?

**Setting.** Head 0.0's score for the intended key is $c/\sqrt{d_h} = 60/\sqrt{18} = 14.14$; every other key scores 0.

**Attention weight.** At a position with $n$ visible keys, $e^{14.14}/(e^{14.14} + n - 1)$: for $n = 16$, $1 - 1.1\times10^{-5}$.

**Reading.** The measured previous-token score of head 0.0 is 1.00 and the induction score of head 1.0 is 1.00; the unused heads, with zero weights, attend uniformly (0.08 = about 1/12 on average over positions). With $c = 20$ instead, the score would be only $4.7$, and the attention to the intended key falls to about 0.9 — the $1/\sqrt{d_h}$ scaling of Unit 04 at work.

## Worked Example 11.2 — The circuit's output

**Setting.** 64 sequences $[\text{BOS}, s, s]$; loss on the 2nd–8th tokens of the repeat.

**Numbers.** Loss 0.0004 nats per token. At one position, head 1.0 contributes exactly $\lambda = 10.0$ to the logit of the correct next token, and the embedding contributes $1.0$ to the logit of the *current* token; the correct token's probability is 0.9996.

**Reading.** The direct-logit decomposition (§11.3.1) accounts for the whole prediction: one head writes the answer, the embedding writes a distractor, and nothing else contributes. In a trained model the same decomposition is exact for attention-only networks and approximate with layer normalization.

## Worked Example 11.3 — Ablation, with controls

**Setting.** Zero-ablate each head of the hand-built circuit in turn.

| head ablated | role | loss on the repeat (nats) |
|---|---|---|
| none | — | 0.0004 |
| 0.0 | previous-token | 2.84 |
| 1.0 | induction | 2.37 |
| 0.1 | unused | 0.0004 |
| 1.1 | unused | 0.0004 |

**Reading.** Both circuit heads are necessary, and the circuit needs *both*: the induction head alone cannot match tokens without the previous-token information written by layer 0 (K-composition). The control ablations of the unused heads change nothing, which rules out "any ablation hurts". Without controls, the first two rows would be uninterpretable.

## Worked Example 11.4 — Patching restores the behaviour

**Setting.** Clean input: $[\text{BOS}, s, s]$. Corrupted input: the first copy of $s$ replaced by a different permutation, the second copy unchanged. Then: the corrupted run with head 1.0's output patched in from the clean run.

| run | loss on the repeat |
|---|---|
| clean | 0.0004 |
| corrupted | 8.77 |
| corrupted, head 1.0 patched from clean | 0.0004 |

**Reading.** The corruption destroys the information the second half needs (its "earlier occurrences" now point to the wrong next tokens), and the model confidently copies the wrong tokens — hence a loss far above uniform. Patching one head's output restores the clean behaviour completely: head 1.0 carries all of the information that distinguishes the two runs. Patching a run with its own activations changes nothing (the spine test asserts it), so the restoration is not an artefact of the hook.

## Worked Example 11.5 — A trained model that copies without the textbook pattern

**Setting.** The same architecture (2 layers × 2 heads, $d = 48$, learned positions, no MLP, no normalization) trained for 8 000 steps on sequences $[\text{BOS}, \text{noise}, s, \text{noise}, s, \text{noise}]$ with random gaps, 24 tokens, $|s| = 8$.

**Behaviour.** Loss on the predictable part of the second copy: 2.63 at step 1 000 … 0.26 at step 8 000 (uniform: $\ln24 = 3.18$).

**Attention.** The largest induction score of any head is 0.02; previous-token scores are at most 0.18.

**Ablation.**

| head ablated | 0.0 | 0.1 | 1.0 | 1.1 |
|---|---|---|---|---|
| loss on the second copy | 3.52 | 1.89 | 2.21 | 1.44 |

**Reading.** The model has learned to copy — the behaviour is real, and every head is needed for it. But no head attends from a token's second occurrence to the token that followed its first, which is what the induction-head account predicts. The mechanism is something else: distributed across all four heads, possibly reading positional information in ways the attention-to-target score cannot see. An attention-pattern search alone would conclude "no induction head, so no copying" — contradicted by the behaviour; a search that assumed the textbook mechanism and found the "best" head (score 0.02) would tell a false story. What the mechanism *is* remains open here (Experiment 11.4).

## Worked Example 11.6 — A model that does not copy

**Setting.** The pretrained spine model (Stage 07), prompts in which a made-up pet name occurs and then recurs: "… Max also had a cat called Raman. One sunny day Max went to the park with Sam. Raman ran to …".

**Behaviour.** Loss on the 2nd, 3rd and 4th letters of the second mention: 10.62, 6.76 and 6.88 nats (uniform: 3.91). The model predicts the continuation of a familiar name ("R" → "uby") with confidence.

**Reading.** There is no copying behaviour, so there is no copying mechanism to find. The best "induction score" of any head (0.02) is noise. This is also the mechanism-level explanation of Unit 10's unseen-names failure: the model has memorized a name list rather than learned to copy names from context — a hypothesis that predicts it will also fail on unseen animal names, and that a model trained with more varied names (Experiment 11.5) should not.

## Worked Example 11.7 — Probe selectivity

**Setting.** A linear probe trained on 200 hidden states to predict a binary property reaches 92% held-out accuracy. The same probe, trained on the same states with labels randomly reassigned per *input type* (a control task), reaches 81%.

**Computation.** Selectivity $= 92 - 81 = 11$ points.

**Reading.** Most of the probe's accuracy comes from the probe's own capacity to memorize input types, not from a simple encoding of the property. A small selectivity is weak evidence of representation, and no evidence of use ([Hewitt & Liang 2019](/references/#hewitt-2019)). (Illustrative numbers; Experiment 11.3 measures a real probe on the spine model.)

## Worked Example 11.8 — What changed in post-training?

**Setting.** Unit 09's REINFORCE and buggy-checker runs ended with KL to the reference of 1.37 and 1.32 nats at the answer position — almost the same — with true accuracy 0.82 and 0.06.

**Reading.** A single summary of how much the model changed says nothing about *what* changed. A mechanistic hypothesis does: if the buggy run learned a prompt-independent bias toward "9", its change in the direct logit contributions should be the same for every prompt; if the correct run learned to use the digits, its change should depend on them. Experiment 11.6 tests exactly this — a prediction the behaviour (both changed) and the KL (both 1.3) cannot make, and one that could turn out wrong.
