---
title: Multivariate chain rule
description: Jacobians compose by matrix multiplication; derivatives along all paths add.
atlas:
  id: math-chain-rule
  type: mathematics
  mastery: internalize
  core_units: [3]
  related: [mech-reverse-mode-autodiff]
  references: [griewank-2008, baydin-2018]
  last_reviewed: 2026-11-02
---

## In one paragraph

For $g:\mathbb{R}^n\to\mathbb{R}^m$ and $f:\mathbb{R}^m\to\mathbb{R}^k$, the Jacobian of the composition is the product of Jacobians: $J_{f\circ g}(x) = J_f(g(x))\,J_g(x)$. When a variable influences the output along several paths, the contributions **add**. Reverse-mode differentiation is this rule evaluated right-to-left for a scalar output.

## Invariants

- **Shapes.** $J_f(g(x))$ is $k\times m$, $J_g(x)$ is $m\times n$; the product is $k\times n$. A derivation whose shapes do not compose is wrong.
- **Path sum.** If $y = f(u(x), v(x))$ then $\frac{dy}{dx} = \frac{\partial f}{\partial u}u'(x) + \frac{\partial f}{\partial v}v'(x)$.
- **Order.** Matrix Jacobians do not commute; reversing the order changes the answer unless the shapes force it.

## Known values

- $\frac{d}{dx}\sin(x^2) = 2x\cos(x^2)$; at $x = 1$: $1.0806046$.
- $y = x\cdot x$ via two paths: $\frac{dy}{dx} = x + x = 2x$.

## Common bugs

- Dropping one path when a variable is reused (the root cause of the `=` vs `+=` autodiff bug).
- Transposed Jacobians in hand-derived matrix gradients — catch with the shape rule.
