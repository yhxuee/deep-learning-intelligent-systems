---
title: "11.10 Problems and further study"
description: Twenty graded problems on residual-stream decomposition, QK and OV circuits, hand-built circuits, ablation and patching, probes, attribution and causal abstraction, with selected answers and an annotated reading list.
sidebar: { label: "11.10 Problems", order: 10 }
---

Difficulty: ★ routine · ★★ combines ideas · ★★★ proof or open-ended. **[AI-off]** problems are for independent practice and assessment.

## Computation

**11.1 ★ [AI-off]** In the hand-built circuit with $c = 60$ and $d_h = 18$, compute the attention weight on the intended key at a position with 4, 16 and 1 000 visible keys.

**11.2 ★ [AI-off]** With $\lambda = 10$ and the current token's embedding contributing 1 to its own logit, compute the probability of the correct token when $V = 9$ and all other logits are 0. How does it change with $\lambda = 3$?

**11.3 ★** A probe scores 88% on real labels and 85% on a control task; another scores 75% and 52%. Which shows stronger evidence of representation, and why?

**11.4 ★** Ablating a head raises a loss from 0.20 to 0.35; ablating three randomly chosen heads in the same layer raises it to 0.24 on average. What can and cannot be concluded?

**11.5 ★★** Compute the rank of the QK and OV circuits of head 1.0 in the hand-built circuit, and write the $V\times V$ table $E\,W_QW_K^{\top}E^{\top}$ restricted to token embeddings. Why is it zero, and where does the induction head's selectivity come from?

**11.6 ★★** A corrupted run's loss is 8.77, the clean run's 0.0004, and patching component $c$ gives 4.4. What fraction of the effect does $c$ carry? What further patching would test whether the rest flows through another component?

## Derivation

**11.7 ★ [AI-off]** Derive the direct-logit decomposition of an attention-only Transformer with tied embeddings and no normalization.

**11.8 ★★ [AI-off]** Write the weights of the previous-token head of §11.3.4 explicitly as matrices and verify that position $t$ attends to $t-1$.

**11.9 ★★** Show that zero-ablating a head is equivalent to patching it with zeros, and that patching a run with its own activations is the identity.

**11.10 ★★** Show that the integrated-gradients attributions of a function sum to $f(z) - f(z_0)$.

**11.11 ★★** For a single attention head, derive how its output changes under an interchange intervention on its keys only, and on its values only.

## Proof

**11.12 ★★** Prove that without the BOS token, the hand-built circuit makes a systematic error at the first position of the repeat, and compute its loss there.

**11.13 ★★★** Construct a 2-layer attention-only model that copies with no head attending to the induction target (for example, using positional information), and show which ablation results it predicts.

**11.14 ★★★** Give two networks with identical input–output behaviour on all inputs and different internal mechanisms. What does this imply about the limits of behavioural evidence?

**11.15 ★★★** Formalize "head $h$ implements induction" as a causal abstraction and list the interchange interventions that would test it.

## Implementation and design

**11.16 ★ [AI-off]** Implement `build_induction_circuit` from memory and pass the spine tests.

**11.17 ★★** Implement direct logit attribution for the spine model (approximating the layer-norm scale as fixed) and report which head contributes most to predicting the second letter of familiar names.

**11.18 ★★** Implement resample ablation and compare it with zero ablation on the trained toy model.

**11.19 ★★** Build a 1-layer model by hand that solves "copy the previous token", and verify it with the methods of this unit.

**11.20 ★★★** Design an interpretability study of the Stage 09 post-trained model's arithmetic that would survive the criticisms of §11.7. Specify the behaviour, the hypotheses, the interventions, the controls and what would refute each hypothesis.

## Selected answers

- **11.1** Score $60/\sqrt{18} = 14.14$; weight $e^{14.14}/(e^{14.14} + n - 1)$: $1 - 2.2\times10^{-6}$ for $n = 4$, $1 - 1.1\times10^{-5}$ for $n = 16$, $1 - 7.2\times10^{-4}$ for $n = 1\,000$.
- **11.2** Logits: 10 (correct), 1 (current token), 0 (seven others): $p = e^{10}/(e^{10} + e + 7) = 0.9996$. With $\lambda = 3$: $e^3/(e^3 + e + 7) = 0.674$.
- **11.3** The second: selectivity 23 points against 3.
- **11.6** $(8.77 - 4.4)/(8.77 - 0.0004) = 0.50$: half the effect. Patch other components jointly with $c$ and individually to find the remainder.

## Further reading

**Foundations.** [Elhage et al. 2021](/references/#elhage-2021) — the residual stream, QK/OV circuits and composition. [Olsson et al. 2022](/references/#olsson-2022) — induction heads and in-context learning.

**Methods.** [Meng et al. 2022](/references/#meng-2022); [Wang et al. 2023](/references/#wang-2023); [Zhang & Nanda 2024](/references/#zhang-nanda-2024); [Heimersheim & Nanda 2024](/references/#heimersheim-2024) — patching, done carefully. [Hewitt & Liang 2019](/references/#hewitt-2019) — probes with controls.

**Theory and caution.** [Geiger et al. 2021](/references/#geiger-2021) — causal abstraction. [Adebayo et al. 2018](/references/#adebayo-2018); [Makelov et al. 2024](/references/#makelov-2024); [Jain & Wallace 2019](/references/#jain-2019) — how interpretations go wrong.
