---
title: "6.6 Connections"
description: How computational limits connect to complexity theory, computer architecture, economics, thermodynamics and biology, and to the rest of the book.
sidebar: { label: "6.6 Connections", order: 6 }
---

The equations of this unit are not unique to deep learning. They are versions of questions asked in algorithms, hardware design, economics and physics: how does cost grow, which resource is scarce, and what should one more unit of budget buy? The connections are useful only if their different units and assumptions remain visible.

## 6.6.1 Complexity theory: constants matter

Complexity notation asks how cost grows when an input dimension becomes large. Dense attention has a $T^2d$ term; a typical dense recurrence has a $Td^2$ term. Ignoring constants, their arithmetic crossover occurs around $T\sim d$. This is already more informative than saying “attention is quadratic”: quadratic in *which variable*, while what else is held fixed?

Asymptotics intentionally discard constants, memory traffic, parallel depth and hardware efficiency. Engineering must put them back. Two mathematically equivalent $O(T^2d)$ attention implementations can differ substantially because one writes a $T\times T$ matrix to slow memory and another tiles it on chip (Unit 04, §4.8.1). A recurrence with fewer total FLOPs may take longer if its time steps are sequential while attention exposes a large parallel product (Unit 05, §5.3.13).

The safe division of labour is:

- asymptotic analysis identifies growth and eventual bottlenecks under stated scaling assumptions;
- exact counts locate constants and crossover sizes;
- the roofline and measurement determine what the present implementation does on the present hardware.

No one of the three replaces the others.

## 6.6.2 Computer architecture: the memory wall

The **memory wall** names a persistent imbalance: arithmetic throughput can grow faster than the rate at which large memories deliver operands. The roofline model ([Williams et al. 2009](/references/#williams-2009)) turns the imbalance into the ratio $\pi/\beta$. As that ratio rises, an operation needs more FLOPs of reuse per byte to stay compute-bound.

Computer architecture is hierarchical, not one box labelled “memory.” Registers and on-chip SRAM are small and fast; high-bandwidth device memory is larger and slower; host memory and storage are farther away again. An optimization can reduce traffic at one boundary while increasing it at another. Flash-style attention keeps tiles on chip; kernel fusion avoids writing intermediates; contiguous layouts turn scattered accesses into efficient transfers; checkpointing recomputes values instead of preserving them (§6.3.2). The shared principle is **place reuse near the arithmetic**.

This also explains why algorithmic equivalence does not imply systems equivalence. Two programs can perform the same FLOPs yet cross the hierarchy a different number of times.

## 6.6.3 Economics: marginal cost and marginal value

Proposition 6.2 is a resource-allocation problem with diminishing returns. Spend more of a fixed training budget on parameters and the finite-model term falls, but fewer tokens can be processed and the finite-data term rises. At the stationary point, the exponent-weighted marginal changes balance:

$$
\alpha\frac{A}{N^\alpha}=\beta\frac{B}{D^\beta}.
$$

This resembles an economic optimum: move one small unit of budget from one use to the other until the marginal benefit no longer improves.

The analogy exposes missing objectives. Training FLOPs are not money: data must be acquired, cleaned and governed (Unit 07); experiments consume engineer time; latency can have contractual value; a model trained once may be served billions of times. Nor is a reduction of 0.01 nats worth the same amount in every application. A real project may minimize

$$
\text{training cost}+\text{lifetime inference cost}+\text{data and labour cost}
$$

subject to quality, latency, capacity and risk constraints. “Compute-optimal” is one slice of total cost of ownership.

## 6.6.4 Thermodynamics: a floor under erasure, not under FLOPs

Landauer's principle states that an irreversible erasure of one bit in a system at temperature $\Theta$ dissipates at least $k\Theta\ln2$ of heat ([Landauer 1961](/references/#landauer-1961))—about $2.9\times10^{-21}$ J near room temperature. Worked Example 6.8 estimated roughly $10^{-12}$ J per **useful model FLOP** for one illustrative accelerator, utilization and system boundary.

The units are both joules, but the denominators differ: one is per bit erased, the other per accounted floating-point operation. Relating them requires a circuit model that says how many irreversible bit erasures, transitions and data transfers implement that operation. A FLOP therefore has no universal Landauer energy obtained by simple substitution.

The useful connection is methodological. Energy must be attached to a physical process and a declared boundary. Lowering precision, improving reuse, shortening wires or avoiding memory transfers can reduce energy even when the abstract model computes the same function. The example shows a large engineering gap; it does not quantify a universal “distance from physics.”

## 6.6.5 Biology: a different trade-off

The adult human brain is often estimated to consume roughly 20 W. Converting neural activity into “brain FLOPs” is not well defined: spikes, synapses, dendrites and biochemical adaptation do not implement the floating-point convention of §6.2. Comparisons that divide one guessed operation count by another can therefore create precision without a shared unit of work.

Biology is still instructive. Neural systems use sparse events, massive locality, mixed time scales and memory that is partly colocated with computation. They also pay heavily for communication and maintenance; long axons and spikes are not free, so the brain has not abolished a memory wall. It embodies a different architecture and objective. Neuromorphic and in-memory computing explore related principles—bring state closer to operations and communicate selectively—without implying that silicon should literally copy a neuron.

## 6.6.6 Inside the book

| This unit | Returns in |
|---|---|
| $6ND$ and the compute-optimal $D^\star$ | Unit 07: compute-optimal training needs more tokens than can be labelled |
| decoding is memory-bound | Unit 08: sampling cost; speculative decoding |
| FLOPs of a forward pass | Unit 09: the cost of sampling many candidate answers for RL; test-time compute |
| scaling laws as fitted predictions | Unit 10: evaluating whether a gain is real; Unit 12: pre-registered predictions |
| fitted asymptote $E$ versus entropy | Unit 01: the true conditional-entropy floor and why a fitted model floor may sit above it |
| KV-cache memory | Units 04–05: architectures designed around it |

**Next:** [§6.7 Limitations and counterexamples](/core/unit-06/limitations/) turns each connection into a boundary case: same FLOPs but different time, smooth loss but discontinuous metric, and a good interpolation with a failed extrapolation.
