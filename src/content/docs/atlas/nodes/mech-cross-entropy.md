---
title: Cross-entropy and likelihood
description: The default training objective — negative log-likelihood of the observed data — and the numbers it must start at.
atlas:
  id: mech-cross-entropy
  type: mechanism
  mastery: internalize
  core_units: [1, 3, 7, 10]
  prerequisites: [math-softmax]
  references: [goodfellow-2016, bengio-2003]
  last_reviewed: 2026-11-02
---

## In one paragraph

For a model distribution $q$ and observed outcome $y$, the loss $-\log q(y)$ is the negative log-likelihood. Averaged over data it estimates the cross-entropy $H(p, q) = -\mathbb{E}_{y\sim p}\log q(y) = H(p) + \mathrm{KL}(p\,\|\,q)$, so minimizing it minimizes the KL divergence to the data distribution. For language models it is reported per token, in nats (natural log) or bits ($\div \ln 2$); perplexity is $\exp$ of the mean per-token loss in nats.

## Invariants

- $-\log q(y) \ge 0$, with equality only if $q(y) = 1$.
- The cross-entropy of a model is never below the entropy of the data: $H(p,q)\ge H(p)$.
- A uniform model over $V$ outcomes scores exactly $\ln V$ nats, whatever the data.

## Known values

| Vocabulary $V$ | Uniform loss $\ln V$ (nats) | bits |
|---|---|---|
| 2 | 0.6931 | 1.0000 |
| 65 (Shakespeare characters) | 4.1744 | 6.0224 |
| 50 257 (GPT-2 BPE) | 10.8249 | 15.6170 |

A freshly initialized language model should start **near** $\ln V$. Much higher means the initial logits are too large; much lower means information is leaking (e.g. a broken causal mask).

## Common bugs

- Averaging over padding tokens.
- Passing probabilities where the function expects logits (double softmax).
- Comparing losses across tokenizers: nats/token is not comparable when tokens differ; use bits per byte or per character.
