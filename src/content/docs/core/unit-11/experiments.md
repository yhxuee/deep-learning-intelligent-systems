---
title: "11.9 Experiments"
description: Six experiments — validating methods on the hand-built circuit, the training dynamics of copying, a probe with a control task, finding the toy model's mechanism, testing the name-list hypothesis, and comparing two post-trained models.
sidebar: { label: "11.9 Experiments", order: 9 }
---

Write the prediction first. Code: [`spine/stage11_induction.py`](https://github.com/yhxuee/deep-learning-intelligent-systems/tree/main/spine), the `head_mask` and `patch` arguments of `spine/model.py`, and the Stage 11 tests.

## Experiment 11.1 — Validate the methods on known ground truth

**Tests:** §11.2.4, Worked Examples 11.1–11.4. **Predict:** what each method reports on the hand-built circuit when $c$ is lowered from 60 to 5, and when the BOS token is removed.

**Procedure.** Rebuild the circuit with different $c$, $\lambda$ and without BOS; run attention scores, ablations and patching each time.

**Look for:** the methods agree when the circuit is sharp; as $c$ falls, attention scores become ambiguous before ablations do; without BOS a spurious match appears at the first repeat position. Write down which method failed first and why.

## Experiment 11.2 — When does copying appear?

**Tests:** §11.8.3, Worked Example 11.5. **Predict:** the step at which the toy model's loss on the second copy falls below 1 nat, and whether it falls gradually or suddenly.

**Procedure.** Train the toy with `log_every=100` and record loss and all four heads' ablation effects at each checkpoint, over three seeds.

**Look for:** whether the drop is a phase change; whether different seeds use the heads differently (the ablation profile at the end).

## Experiment 11.3 — A probe with a control task

**Tests:** §11.3.6, Worked Example 11.7. **Predict:** the selectivity of a linear probe for "the next character is a digit" on the Stage 07 model's final hidden state.

**Procedure.** Collect hidden states at 2 000 positions; train logistic regression on real labels and on a control task (labels randomly reassigned per input character); compare held-out accuracy.

**Look for:** high selectivity — this property is simple and used; then repeat for "this story contains a pet" and compare.

## Experiment 11.4 — What does the toy model actually do?

**Tests:** §11.7.1, Worked Example 11.5. An open problem. **Predict** — and register (Unit 12) — one hypothesis about the trained toy's copying mechanism and one experiment that would refute it.

**Procedure.** Suggested tools: attention maps averaged by *relative offset to the earlier occurrence* rather than by absolute target; patching each head's output between sequences with the same tokens but different gaps; shuffling positional embeddings at test time; direct logit contributions per head.

**Look for:** a mechanism that predicts the ablation table of Worked Example 11.5. Report what you found and what you ruled out, including hypotheses that failed.

## Experiment 11.5 — Test the name-list hypothesis

**Tests:** §11.5.4, Worked Example 11.6. **Predict:** if the spine model memorized a name list and cannot copy, its loss on unseen *animal* names will also be high; if trained on a corpus where every story introduces a fresh random name used three times, it will learn to copy.

**Procedure.** (a) Score stories with unseen animals. (b) Modify `make_tales.py` so all names are random; pretrain a small model; repeat the pet probe; look for a head with a high induction score; ablate it.

**Look for:** (a) confirms or refutes the hypothesis's first prediction; (b) whether copying behaviour and an induction-like head appear together, and whether ablating the head removes the behaviour.

## Experiment 11.6 — What changed in post-training?

**Tests:** Worked Example 11.8. **Predict** (register first): the buggy-checker model's change in direct logit contributions is prompt-independent; the correct model's depends on the digits.

**Procedure.** For 200 arithmetic prompts, compute the difference in final-position logits between each post-trained model and the reference; regress the difference for the logit of each digit on the prompt's $a + b$.

**Look for:** a flat regression for the buggy model's "9" logit and a strong dependence for the correct model's correct-digit logit — or a result that refutes the prediction.
