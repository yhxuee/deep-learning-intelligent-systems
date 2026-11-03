---
title: "6.6 Connections"
description: How computational limits connect to complexity theory, computer architecture, economics, thermodynamics and biology, and to the rest of the book.
sidebar: { label: "6.6 Connections", order: 6 }
---

## 6.6.1 Complexity theory: constants matter

Complexity theory classifies algorithms by asymptotic growth: attention is $O(T^2d)$, a recurrence $O(Td^2)$. This unit adds the engineering corollary that **constants and memory traffic often decide**. Two $O(T^2d)$ attention implementations differ by several times in speed depending on how often they read the $T\times T$ matrix from slow memory (Unit 04, §4.8.1); an $O(Td^2)$ recurrence can be slower than $O(T^2d)$ attention in practice if it cannot be parallelized over time (Unit 05, §5.3.13). Asymptotics say which method wins eventually; the roofline says which wins at the sizes you will actually run.

## 6.6.2 Computer architecture: the memory wall

For decades, processor arithmetic has improved faster than memory bandwidth — the "memory wall". The roofline model ([Williams et al. 2009](/references/#williams-2009)) was introduced for scientific computing to reason about exactly this gap, and the same reasoning dominates deep-learning systems: matrix units grew much faster than memory bandwidth, so the hardware ratio $\pi/\beta$ keeps rising and more operations become memory-bound. Fusing operations (so intermediate results stay in fast on-chip memory), choosing layouts that read memory sequentially, and recomputing rather than storing (§6.3.2) are standard responses in both fields.

## 6.6.3 Economics: marginal cost and marginal value

Proposition 6.2 is a textbook constrained optimization — a consumer splitting a budget between two goods with diminishing returns, where the optimum equates the marginal loss reduction per FLOP spent on parameters with that spent on data. The economic framing also exposes what the proposition leaves out: inference costs (a model is trained once and used many times), latency requirements, the cost of *acquiring* data rather than just processing it (Unit 07), and the value of a loss reduction, which is not linear in nats. Real decisions optimize total cost of ownership, not training loss per FLOP.

## 6.6.4 Thermodynamics: a floor under erasure, not under FLOPs

Erasing one bit of information dissipates at least $kT\ln2$ of heat ([Landauer 1961](/references/#landauer-1961)) — about $2.9\times10^{-21}$ J at room temperature. A 16-bit floating-point operation on current accelerators costs around $10^{-12}$ J including memory traffic (§6.3.6). The two numbers are not directly comparable: Landauer's bound applies to each *irreversible erasure of a bit*, and how many erasures a multiply–add requires depends on the circuit, the number representation and how much of the computation is made reversible; a floating-point operation does not have a known physical floor in joules. What the comparison does show is that present-day energy per operation is dominated by engineering — switching, wires and data movement — not by any thermodynamic limit we are close to. Most of the energy goes to *moving* data rather than computing on it, which is the roofline's lesson in another unit.

## 6.6.5 Biology: a different trade-off

The human brain runs on roughly 20 W. Comparisons of "brain FLOPs" with GPU FLOPs are unreliable — neurons do not perform floating-point operations — but the energy gap is a reminder that the trade-offs of §6.2 are features of current hardware, not laws of intelligence. The brain keeps memory and computation in the same place and communicates sparsely, avoiding the memory wall by construction; neuromorphic and in-memory computing pursue the same idea.

## 6.6.6 Inside the book

| This unit | Returns in |
|---|---|
| $6ND$ and the compute-optimal $D^\star$ | Unit 07: compute-optimal training needs more tokens than can be labelled |
| decoding is memory-bound | Unit 08: sampling cost; speculative decoding |
| FLOPs of a forward pass | Unit 09: the cost of sampling many candidate answers for RL; test-time compute |
| scaling laws as fitted predictions | Unit 10: evaluating whether a gain is real; Unit 12: pre-registered predictions |
| irreducible loss $E$ | Unit 01: the entropy floor |
| KV-cache memory | Units 04–05: architectures designed around it |
