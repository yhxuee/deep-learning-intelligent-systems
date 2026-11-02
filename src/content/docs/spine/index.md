---
title: The spine project
description: One small language model built across all twelve units — each stage adds or interrogates one component and has a verification target.
---

Abstract questions need a concrete object to be asked about. The spine project is **one small language model that you build, extend and interrogate across the whole Core**. Stanford's CS336 shows that building a language model from scratch is a viable organizing thread for a course [CS336](/references/#cs336-2026); the spine applies the same idea at laptop scale, using a TinyStories-style corpus on which small models already produce coherent text [Eldan & Li 2023](/references/#eldan-2023).

Every stage has a **verification target** — a number or invariant that tells you, independently of any AI, whether the stage is correct.

| Stage | Unit | You add | Verification target | Status |
|---|---|---|---|---|
| 01 | [01](/core/unit-01/) | Count-based bigram, add-α smoothing | held-out NLL < uniform $\ln V$; α→∞ recovers $\ln V$ exactly | **Code + tests** |
| 02 | [02](/core/unit-02/) | Learned embeddings, tied unembedding | lookup gradient = one-hot matmul gradient | Spec |
| 03 | [03](/core/unit-03/) | Neural bigram trained with your derived gradient $p-e_y$ | init loss $=\ln V$; train loss → count-MLE optimum (±0.01 nats), never below | **Code + tests** |
| 04 | [04](/core/unit-04/) | Causal self-attention block (framework autograd from here on) | all attention invariants; init loss ≈ $\ln V$ | Spec |
| 05 | [05](/core/unit-05/) | KV-cache decoding; recall stress test | cached logits = full recompute | Spec |
| 06 | [06](/core/unit-06/) | Five model sizes; scaling-law fit | pre-registered prediction for a 6th size | Spec |
| 07 | [07](/core/unit-07/) | Pretraining on a TinyStories subset; dedup | no held-out story in training (planted-duplicate recall 1.0) | Spec |
| 09 | [09](/core/unit-09/) | Post-training on a checkable task | reward ↑ with bounded KL; hacking audit | Spec |
| 10 | [10](/core/unit-10/) | Evaluation harness | calibration + bootstrap CIs on synthetic known cases | Spec |
| 11 | [11](/core/unit-11/) | Find an induction head | patching self = no-op; ablation removes in-context copying | Spec |
| 12 | [12](/core/unit-12/) | Capstone: reproduce → challenge → extend | oral defense | Spec |

## Running what exists

```bash
cd spine
python stage01_bigram.py              # count-based bigram on data/sample.txt
python stage03_neural_bigram.py       # neural bigram with hand-derived gradients
python -m pytest                      # verification targets for stages 01 and 03
```

Both stages need only NumPy and run in seconds on the bundled sample corpus. For real runs, download a TinyStories subset and pass `--data path/to/text.txt`.

## Design rules

- **Earned abstraction.** Stages 01–03 use only NumPy and your own gradients. From Stage 04 the spine uses a framework's autograd, because you have built one.
- **One change per stage**, with a prediction written before the run.
- **Laptop first.** Every stage must run on a laptop CPU in minutes on the sample corpus; consumer-GPU runs are optional extensions.
