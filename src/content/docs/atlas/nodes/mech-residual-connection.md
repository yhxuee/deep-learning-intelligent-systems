---
title: Residual connections
description: x + F(x) — an identity path for signal and gradient that makes deep stacks trainable, and the residual-stream view of Transformers.
atlas:
  id: mech-residual-connection
  type: mechanism
  mastery: internalize
  core_units: [3, 4, 11]
  prerequisites: [math-chain-rule]
  related: [arch-transformer-block]
  references: [he-2016, radford-2019, elhage-2021]
  last_reviewed: 2026-11-02
---

## In one paragraph

A residual block computes $x + F(x)$. Its Jacobian is $I + \partial F/\partial x$, so even when $\partial F$ is small the gradient has an identity path back to early layers [He et al. 2016](/references/#he-2016). In Transformers every sub-layer reads from and adds to the same per-position vector — the *residual stream* — which is why sub-layer outputs can be analyzed as additive contributions [Elhage et al. 2021](/references/#elhage-2021).

## Invariants

- If $F \equiv 0$ the block is exactly the identity.
- In a pre-LN stack the residual stream is a **sum** of sub-layer outputs, so its variance tends to grow with depth unless output projections are scaled; GPT-2 scales residual-layer weights at initialization by $1/\sqrt{N}$ for $N$ residual layers [Radford et al. 2019](/references/#radford-2019).

## Common bugs

- Normalizing the stream itself (post-LN) when the reference is pre-LN, or vice versa — both train, but behave differently at depth.
- Applying dropout to the skip path rather than to $F(x)$.
- Shape mismatch between $x$ and $F(x)$ hidden by broadcasting.
