---
title: "11.4 Worked examples"
description: Eight worked mechanism audits — sharp attention, direct logits, controlled ablation, real activation patching, a trained non-textbook copier, absent copying, probe selectivity, and why equal KL does not imply equal change.
sidebar: { label: "11.4 Worked examples", order: 4 }
---

Exact calculations are checked in `spine/test_spine.py` and `labs/test_worked_examples_u09_u12.py`. Fixed circuit outputs come from `spine/stage11_induction.py`. The hand-built model uses

$$
K=8,\qquad c=60,\qquad\lambda=10,
\qquad d=36,\qquad d_h=18.
$$

Each example separates three statements:

- **result:** what was calculated or observed;
- **supported claim:** the strongest conclusion justified by that result;
- **remaining alternative:** what the result still cannot distinguish.

## Worked Example 11.1 — From an attention score to an attention weight

**Question.** How sharply does the hand-built previous-token head select its intended key?

The target query–key dot product is $c=60$. Scaled attention divides it by $\sqrt{d_h}$:

$$
s_*=\frac{60}{\sqrt{18}}=14.142\ldots
$$

All non-target scores are zero. If $n$ keys are visible, the target weight is

$$
A_*(n)=\frac{e^{s_*}}{e^{s_*}+n-1}.
$$

For $n=16$,

$$
A_*(16)=0.999989,
\qquad
1-A_*(16)\approx1.1\times10^{-5}.
$$

The same head is less sharp if $c=20$:

$$
s_*=20/\sqrt{18}=4.714,
\qquad
A_*(16)\approx0.881.
$$

**Sanity check.** Increasing the number of visible distractor keys reduces the target weight even when each distractor has score zero. At $n=1\,000$ and $c=60$, the missing mass is about $7.2\times10^{-4}$—still small, but much larger than at $n=16$.

**Supported claim.** The constructed head almost deterministically routes attention to the previous position on these sequence lengths.

**Remaining alternative.** A sharp routing pattern alone does not tell us what value content is moved or whether it affects the output. Example 11.2 inspects the write; Example 11.3 intervenes on it.

## Worked Example 11.2 — Reconstruct the output probability from logit contributions

**Question.** Why is repeat loss approximately 0.0004 nats?

At a scored position in the second copy:

- head 1.0 writes $\lambda=10$ toward the correct next-token logit;
- the tied embedding writes $1$ toward the **current** token, a distractor;
- the other seven vocabulary logits receive 0 in this idealized construction.

Softmax therefore assigns the correct token probability

$$
p_*
=\frac{e^{10}}{e^{10}+e^1+7e^0}
=0.99956\ldots
$$

and loss

$$
-\log p_*=0.00044\ldots\ \text{nats}.
$$

This agrees with the measured average loss 0.0004 over the predictable second-to-eighth tokens of the repeat.

**Why this decomposition is exact here.** The model is attention-only, has no layer normalization, and uses a linear tied readout. The final residual stream is literally a sum of writes.

**Supported claim.** In the hand-built model, head 1.0 supplies the decisive direct write for the correct token, while the embedding supplies a known competing logit.

**Remaining alternative.** In a normalized trained Transformer, using the same additive story without specifying an attribution convention would be an approximation. Also, a component with small direct contribution may still matter indirectly by feeding later components.

## Worked Example 11.3 — Ablation becomes informative only with controls

**Question.** Which heads matter under zero ablation?

| head set to zero | constructed role | repeat loss | change from clean |
|---|---|---:|---:|
| none | — | 0.0004 | — |
| 0.0 | previous-token write | 2.84 | +2.8396 |
| 1.0 | induction/copy write | 2.37 | +2.3696 |
| 0.1 | unused control | 0.0004 | approximately 0 |
| 1.1 | unused control | 0.0004 | approximately 0 |

Head 1.0 cannot perform its match when head 0.0 no longer writes the previous-token field. This is the K-composition predicted by the construction.

The two unused heads are critical controls. If all four ablations had raised loss, “the proposed circuit heads matter” would be confounded with “zeroing any head breaks the program.”

**Supported claim.** Heads 0.0 and 1.0 are necessary for the constructed copying behaviour **under this zero-ablation intervention**, while matched unused heads are not.

**Remaining alternative.** In an ordinary trained network, zero may be off-manifold and redundancy may allow a natural computation to be implemented in several ways. Repeat with mean or resample ablation before making a broad necessity claim.

## Worked Example 11.4 — A real patching test, not a renamed clean run

**Question.** Does head 1.0 transmit the information that distinguishes a clean input from a controlled corruption?

Start with

$$
x_{\mathrm{clean}}=[\mathrm{BOS},s,s].
$$

Construct $x_{\mathrm{corrupt}}$ by replacing the first copy of $s$ with another permutation while keeping the second copy fixed. Now the induction rule points toward wrong successor tokens.

| forward run | repeat loss |
|---|---:|
| clean | 0.0004 |
| corrupted | 8.77 |
| corrupted input, head 1.0 output patched from clean run | 0.0004 |

The recovery fraction is

$$
R=\frac{8.77-0.0004}{8.77-0.0004}\approx1.
$$

The test implementation matters: one must run the **corrupted input** while passing

```python
patch = {(1, 0): clean_cache["blocks"][1]["o"][:, 0]}
```

Recomputing the clean input and calling it “patched” would not test the intervention. The repository test explicitly performs the corrupted forward run and compares its patched loss with clean loss. A separate self-patching test verifies that the hook is a no-op when values are unchanged.

**Supported claim.** In this constructed circuit and corruption, replacing head 1.0's output with its clean value is sufficient to restore clean behaviour.

**Remaining alternative.** Patching the whole head output does not identify which coordinates carry the effect. In an unknown network, wrong-head, wrong-position and unrelated-source patches are needed to establish specificity. Here the hand construction supplies ground truth.

## Worked Example 11.5 — Copying without the textbook attention pattern

**Question.** If a trained model copies, must one head display the canonical induction pattern?

Train a two-layer, two-head attention-only model with learned positions, $d=48$, no MLP and no normalization. Training sequences contain two copies of an eight-token segment separated by random-length noise:

$$
[\mathrm{BOS},\mathrm{noise},s,\mathrm{noise},s,\mathrm{noise}].
$$

The vocabulary has 24 ordinary tokens, so uniform loss is

$$
\log24=3.178\ \text{nats}.
$$

### Behaviour

Loss on the predictable part of the second copy falls from 2.63 at step 1 000 to 0.26 at step 8 000. The model has learned substantial copying behaviour on the frozen evaluation generator.

### Observational search

The largest induction score of any head is only 0.02, and previous-token scores are at most 0.18. No single attention map resembles the hand-built two-head story.

### Zero ablations

| head set to zero | 0.0 | 0.1 | 1.0 | 1.1 |
|---|---:|---:|---:|---:|
| second-copy loss | 3.52 | 1.89 | 2.21 | 1.44 |

All four losses exceed the clean 0.26, so each zero-ablation damages the behaviour. The effects differ substantially, which already argues against treating heads as interchangeable.

**Supported claim.** This trained model copies on the tested distribution, the canonical single-head attention score does not locate that behaviour, and all four zero-ablation interventions damage it.

**Remaining alternative.** These results do not yet reveal the actual algorithm. It may use distributed position-dependent paths, several weak matches, or another feature basis. Nor do the ablations prove that every head is uniquely necessary under natural interventions. Experiment 11.4 must distinguish candidate mechanisms.

This is a counterexample to two shortcuts:

$$
\text{no familiar attention pattern}\not\Rightarrow\text{no behaviour},
$$

$$
\text{largest weak score}\not\Rightarrow\text{the mechanism}.
$$

## Worked Example 11.6 — When there is no copying behaviour to localize

**Question.** Does the pretrained Stage 07 spine checkpoint copy a made-up pet name from context?

A prompt introduces a generated pet name, inserts intervening text, and repeats the name. The probe scores predictions of letters 2–4 in the second occurrence.

| target within second mention | NLL (nats) |
|---|---:|
| second letter | 10.62 |
| third letter | 6.76 |
| fourth letter | 6.88 |
| uniform baseline | 3.91 |

Lower is better. The model is not merely a little worse than a copier; it assigns the true characters less probability than uniform on average. In one illustrative case, after “R” it confidently continues a familiar name such as “Ruby” instead of the earlier made-up name.

The largest measured induction-attention score is 0.02. Because no null distribution for the maximum score is supplied, call it “small under this metric,” not proven noise.

**Supported claim.** This checkpoint fails the specified contextual name-copying probe. There is no successful copying behaviour here that requires a copying circuit explanation.

**Remaining alternative.** The result is consistent with reliance on familiar continuations, but it does not prove that the model stores a finite name list. Tokenization, training coverage, positional structure and other mechanisms remain. Tests with unseen animal names, controlled token lengths and training interventions can separate them.

## Worked Example 11.7 — Probe accuracy needs a capacity control

**Question.** What does 92% probe accuracy mean if a matched control reaches 81%?

A linear probe predicts a binary property from 200 hidden states:

$$
\operatorname{acc}_{\mathrm{real}}=0.92.
$$

The same probe class is trained on labels randomly reassigned per input type, preserving the opportunity to memorize input types:

$$
\operatorname{acc}_{\mathrm{control}}=0.81.
$$

Selectivity is

$$
0.92-0.81=0.11,
$$

or 11 percentage points ([Hewitt & Liang 2019](/references/#hewitt-2019)).

The high control score warns that the probe and split permit substantial memorization or nuisance decoding. It is not valid to say exactly “81 of the 92 points came from probe capacity”; accuracies are not additive causal contributions.

**Supported claim.** The tested representation offers an 11-point advantage over this particular capacity control for this decoder and split.

**Remaining alternative.** A different control construction, regularization or group split may change selectivity. Even high selectivity would show decodability, not that the original model uses the property. That requires a feature intervention.

## Worked Example 11.8 — Similar KL, different task changes

**Question.** Does similar KL from a reference model imply similar learned computation?

Unit 09 produced two post-training runs:

| run | mean KL to reference at answer position | held-out greedy true accuracy | mean $p(\mathrm{correct})$ |
|---|---:|---:|---:|
| exact-reward REINFORCE | 1.347 nats | 0.82 | 0.79 |
| buggy $\ge$ checker | 1.43 nats | 0.09 | 0.07 |

The old shorthand “true accuracy 0.79” would confuse two different columns: 0.79 is mean probability assigned to the correct answer, while greedy true accuracy is 0.82.

KL measures an average distributional displacement:

$$
D_{\mathrm{KL}}(\pi\|\pi_{\mathrm{ref}}).
$$

It does not identify which logits changed, whether the change depends on the prompt, or whether the changed distribution solves the intended task.

A mechanistic hypothesis makes a sharper prediction:

- if the buggy run learned a nearly prompt-independent bias toward digit 9, its answer-logit change should be similar across digit inputs;
- if exact-reward training learned to use the two input digits, its changes should vary systematically with them.

**Supported claim.** Similar scalar KL can accompany opposite task outcomes, so KL magnitude alone cannot identify what computation changed.

**Remaining alternative.** The prompt-independent-bias story is still a hypothesis until direct-logit changes and interventions are measured. Experiment 11.6 is designed to risk that hypothesis.

## Evidence ledger

| example | evidence type | strongest justified conclusion |
|---|---|---|
| 11.1 | observation | routing is sharply concentrated |
| 11.2 | exact decomposition | head 1.0 directly writes the correct logit in the constructed model |
| 11.3 | controlled zero ablation | constructed heads matter under zero replacement |
| 11.4 | activation patching | clean head output restores behaviour in the constructed corruption |
| 11.5 | behaviour + ablation | copying exists without the searched single-head pattern |
| 11.6 | behavioural falsification | the spine checkpoint fails this copying probe |
| 11.7 | probe + control | limited decodability beyond a matched control |
| 11.8 | aggregate counterexample | equal change magnitude does not identify change content |
