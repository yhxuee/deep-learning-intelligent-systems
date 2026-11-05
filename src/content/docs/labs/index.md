---
title: The Laboratory
description: Laptop-scale labs with invariant tests — the executable half of the Atlas verification kits.
---

Labs turn mathematical ideas into predictions and measurements. You are not expected to produce a correct implementation in one attempt. Each lab begins with an **AI-off construction** file containing signatures but no solution, plus a test suite drawn from the Atlas invariants. The same tests run against a reference implementation in CI, so a fresh failure tells you what remains to be built rather than whether the tests are viable.

The aim is not merely to make the progress bar green. It is to connect four things: a mathematical claim, a prediction, a program behavior and a test that can distinguish the intended mechanism from a plausible bug.

| Lab | Unit | You implement (AI-off) | Tests check |
|---|---|---|---|
| [`labs/unit-03-autodiff`](https://github.com/yhxuee/deep-learning-intelligent-systems/tree/main/labs/unit-03-autodiff) | [03](/core/unit-03/) | scalar reverse-mode autodiff `Value` | worked example, fan-out accumulation, finite differences on random graphs, linearity |
| [`labs/unit-04-attention`](https://github.com/yhxuee/deep-learning-intelligent-systems/tree/main/labs/unit-04-attention) | [04](/core/unit-04/) | causal multi-head self-attention in NumPy | row sums, causality, permutation equivariance, KV-cache equivalence, √d scaling, GPT-2 parameter count; **bug hunt** over five planted variants |
| [`labs/unit-05-memory`](https://github.com/yhxuee/deep-learning-intelligent-systems/tree/main/labs/unit-05-memory) | [05](/core/unit-05/) | linear recurrence (loop + closed form), linear-attention memory | closed form, spectral decay, non-normal transient growth, exact recall ≤ $d_k$ vs. interference beyond |
| [`spine/`](https://github.com/yhxuee/deep-learning-intelligent-systems/tree/main/spine) | 01–12 | the spine language model | stage verification targets |
| [`labs/test_worked_examples.py`](https://github.com/yhxuee/deep-learning-intelligent-systems/blob/main/labs/test_worked_examples.py) | 03–05 | — | recomputes every number quoted in the units' worked examples, so text and computation cannot drift apart |

## Running a lab

Requirements: Python ≥ 3.10, NumPy, pytest.

```bash
pip install -r labs/requirements.txt
cd labs/unit-04-attention
python -m pytest                       # tests YOUR implementation (attention.py)
LAB_IMPL=reference python -m pytest    # the reference passes the same tests
python -m pytest test_bug_hunt.py -v   # which invariant catches which planted bug?
```

A fresh checkout's `engine.py` / `attention.py` / `memory.py` contain only signatures and `NotImplementedError`, so tests fail until you implement them. That is intended.

## Your first lab, step by step

Use Unit 03 autodiff if you have not written numerical software before.

1. Read the corresponding worked example and calculate its expected output by hand.
2. Open the starter file and identify the smallest missing operation. Do not read the complete reference yet.
3. Run one named test, for example `python -m pytest -k add -v`, and read the assertion from left to right: input, observed value, expected property.
4. Implement only that operation. Rerun the same test, then a neighboring test that exercises fan-out or a random graph.
5. When a test fails, preserve the smallest failure and state the mechanism in words before asking an AI or changing code.
6. After the suite passes, compare with the reference implementation and write one case that your current tests might still miss.

Use AI after the independent attempt to request a hint, an alternative derivation or a hostile test. Do not paste a generated solution into the starter and treat passing tests as understanding. You should be able to explain the data shapes, predict a simple output and identify which invariant would catch a proposed bug without the tool present.

## Reading test outcomes

- **A single focused test fails:** localize the operation and reduce the input.
- **Finite differences fail but forward values pass:** suspect the derivative rule, accumulation at fan-out, or an in-place mutation.
- **Reference and learner implementations both fail:** first check environment and versions; then treat it as a possible test or documentation defect and preserve the evidence.
- **Everything passes:** the listed invariants hold on the tested domain. Performance, numerical behavior at extreme values and usefulness on a new task remain separate questions.

## Reproducibility record

Every experiment you report records: hypothesis and prediction; environment (`python --version`, package versions); exact command and code revision; seeds; compute budget; evaluation protocol; results with uncertainty; negative results; and the conclusion's validity boundary. A compact record is better than a polished story reconstructed after the fact.
