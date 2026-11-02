---
title: The Laboratory
description: Laptop-scale labs with invariant tests — the executable half of the Atlas verification kits.
---

Labs turn mathematical ideas into predictions and measurements. Each lab has an **AI-off construction** file you write yourself and a test suite drawn from the Atlas invariants. The same tests run against a reference implementation in CI, so the tests themselves are verified.

| Lab | Unit | You implement (AI-off) | Tests check |
|---|---|---|---|
| [`labs/unit-03-autodiff`](https://github.com/yhxuee/deep-learning-intelligent-systems/tree/main/labs/unit-03-autodiff) | [03](/core/unit-03/) | scalar reverse-mode autodiff `Value` | worked example, fan-out accumulation, finite differences on random graphs, linearity |
| [`labs/unit-04-attention`](https://github.com/yhxuee/deep-learning-intelligent-systems/tree/main/labs/unit-04-attention) | [04](/core/unit-04/) | causal multi-head self-attention in NumPy | row sums, causality, permutation equivariance, KV-cache equivalence, √d scaling, GPT-2 parameter count; **bug hunt** over five planted variants |
| [`labs/unit-05-memory`](https://github.com/yhxuee/deep-learning-intelligent-systems/tree/main/labs/unit-05-memory) | [05](/core/unit-05/) | linear recurrence (loop + closed form), linear-attention memory | closed form, spectral decay, non-normal transient growth, exact recall ≤ $d_k$ vs. interference beyond |
| [`spine/`](https://github.com/yhxuee/deep-learning-intelligent-systems/tree/main/spine) | 01–12 | the spine language model | stage verification targets |

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

## Reproducibility record

Every experiment you report records: hypothesis and prediction; environment (`python --version`, package versions); seeds; compute budget; evaluation protocol; and negative results.
