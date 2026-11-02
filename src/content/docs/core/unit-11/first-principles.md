---
title: "11.2 First principles"
description: Behaviour, representation and mechanism; observation, attribution and intervention as kinds of evidence; why a method must be validated on a model with a known mechanism before it is trusted on one without.
sidebar: { label: "11.2 First principles", order: 2 }
---

## 11.2.1 Three levels of "what a model has learned"

A claim about what a model has learned can be made at three levels:

- **Behaviour** — what it outputs on which inputs. "The model completes the second mention of a name correctly." Units 01–10 measured behaviour.
- **Representation** — what information its internal states contain. "The hidden state at this position encodes the next letter of the name." A representation can be present and unused.
- **Mechanism** — which internal computation produces the behaviour. "Head 1.0 attends from the second mention to the position after the first mention and copies that token into the output." A mechanism is a causal claim: change the component and the behaviour changes in a predicted way.

The levels are not interchangeable. Successful behaviour does not reveal the mechanism (many mechanisms produce the same outputs); present information does not prove use (a probe can read what the model ignores). Each level needs its own evidence.

## 11.2.2 Three kinds of evidence

**Observation** looks at internal quantities without changing anything: attention patterns, activation statistics, which inputs maximally activate a unit, linear **probes** trained to read a property from hidden states ([Alain & Bengio 2017](/references/#alain-2017); [Belinkov 2022](/references/#belinkov-2022)). Observation shows what is there and what correlates with what.

**Attribution** distributes an output over inputs or components, usually through gradients: how much would the output change if this input changed slightly? Attributions are cheap and local; they measure sensitivity, which is neither necessary nor sufficient for use ([Sundararajan et al. 2017](/references/#sundararajan-2017); [Adebayo et al. 2018](/references/#adebayo-2018)).

**Intervention** changes a component and measures the effect on behaviour. **Ablation** removes it (sets its output to zero or to its average). **Activation patching** replaces its output with the output it had on a different input, and asks whether the behaviour follows the patched information ([Vig et al. 2020](/references/#vig-2020); [Meng et al. 2022](/references/#meng-2022)). Interventions are the only kind of evidence that tests the causal claim directly — and they need controls, because removing *anything* from a network can change its behaviour.

## 11.2.3 Why attention patterns are not explanations

An attention weight says how much of each value vector is mixed into a position's update. It does not say what the value vector carries, whether the mixed vector is used downstream, or whether the same information arrives by another route. Debates about attention as explanation in NLP reached the same conclusion from both sides ([Jain & Wallace 2019](/references/#jain-2019); [Wiegreffe & Pinter 2019](/references/#wiegreffe-2019)): attention patterns are observations, informative when combined with interventions and misleading alone. Worked Example 11.6 makes the point with a model whose behaviour depends on every head while no head attends to the position the textbook mechanism requires.

## 11.2.4 Ground truth first

How do we know an interpretability method works? The same way we knew the spine's deduplicator worked: run it where the answer is known. Unit 11's ground truth is a two-layer attention-only Transformer whose weights are set **by hand** to implement induction (§11.3.4). Every method in this unit is first checked on it:

| Method | Must find on the hand-built circuit |
|---|---|
| previous-token attention score | head 0.0 attends to the previous position (score 1.00) |
| induction attention score | head 1.0 attends to the position after the earlier occurrence (score 1.00) |
| ablation | removing head 0.0 or 1.0 destroys copying; removing heads 0.1 or 1.1 does not |
| activation patching | patching head 1.0's clean output into a corrupted run restores the clean behaviour |

Only after a method passes on known ground truth is it applied to a model whose mechanism is unknown. A method that fails here is broken; a method that passes here can still mislead elsewhere, but at least not trivially.

## 11.2.5 Behaviour before mechanism

The last principle is the simplest and the most often skipped: **establish the behaviour before explaining it.** The spine model's loss on the second mention of a made-up pet name is 6.8–10.6 nats per letter — worse than the 3.9 of uniform guessing (Worked Example 11.7). It does not copy novel names. Searching its heads for an induction mechanism would find the head with the highest "induction score" (0.02) and invite a story about a weak induction head; the behavioural test says there is nothing to explain. Interpretability results start from a measured behaviour, with its uncertainty, exactly as in Unit 10.
