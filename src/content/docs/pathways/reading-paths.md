---
title: Reading paths
description: How to read a unit on a first pass and on a deep pass, how the reading order differs from the mathematical dependencies between units, and which appendix to keep at hand for each unit.
sidebar: { order: 3 }
---

The book is written to be read from Unit 01 to Unit 12, and every unit in the same ten parts. That order is one path through the material, not the only one. This page separates three things that are easy to confuse: the order in which to read the parts of a unit, the order in which the units depend on one another mathematically, and the mathematics each unit assumes.

## Within a unit: first pass and deep pass

**First pass — the argument.** Aim to be able to state the unit's question, its central construction and its main limitation.

1. Part 1 (the problem) and part 2 (first principles), in full.
2. Part 3 (mathematical construction), **interleaved with part 4**: every worked example names the sections of part 3 it uses, so read it as soon as you reach the last of those sections, and do it on paper before reading its solution. Skip sections marked ★ Advanced.
3. Part 7 (limitations and counterexamples), at least its first and last sections — the last limitation is the one that motivates the next unit.
4. Run the unit's spine stage and its tests.

**Deep pass — the evidence.** Aim to be able to defend each claim and to say what would refute it.

1. Part 3 again, now with the ★ sections, re-deriving each displayed result.
2. Part 5 (general theory), with the appendix sections it cites (table below).
3. Part 7 in full, then part 8 (modern developments), reading each **[E]** label's source where a claim matters to you ([writing standard](/project/writing-standard/#rules-for-the-main-text)).
4. Part 9 (experiments): write the prediction before running anything.
5. Part 10 (problems), then part 6 (connections) to place the unit in the Atlas.

A first pass of a unit is roughly a third of the reading of a deep pass; the [undergraduate pathways](/pathways/undergraduate/) are built from first passes plus selected experiments, the [graduate pathway](/pathways/graduate/) from deep passes.

## Between units: reading order and mathematical dependency

Each unit declares what it **builds on** — the earlier units it uses, and for what. The reading order 01 → 12 respects these dependencies, but most dependencies are narrow: a unit typically uses one or two results of an earlier unit, not all of it. When you skip or reorder units, the right-hand column is what you need to know from the skipped unit; it is taken verbatim from each unit's metadata, which the build checks against the units themselves.

| Unit | Builds on | Uses from it |
|---|---|---|
| [01](/core/unit-01/) Learning | — | — |
| [02](/core/unit-02/) Representation | 01 | empirical risk, cross-entropy and held-out evaluation |
| [03](/core/unit-03/) Optimization | 01 | the cross-entropy objective and the ln V baseline |
| | 02 | embedding tables and linear read-outs |
| [04](/core/unit-04/) Interaction | 02 | embeddings and invariance |
| | 03 | backpropagation through softmax and residual stacks |
| [05](/core/unit-05/) Memory | 03 | gradient flow through time, eigenvalues and stability |
| | 04 | attention and the KV cache |
| [06](/core/unit-06/) Computation | 01 | held-out loss as the yardstick |
| | 03 | the cost of a backward pass and optimizer state |
| | 04 | the FLOPs of attention and the MLP |
| | 05 | KV-cache memory |
| [07](/core/unit-07/) Self-supervision | 01 | likelihood and held-out evaluation |
| | 02 | which representations transfer |
| | 06 | the data demands of scaling laws |
| [08](/core/unit-08/) Generation | 03 | training by gradient descent |
| | 05 | ODE discretization and stability |
| | 07 | pretrained conditional distributions |
| [09](/core/unit-09/) Goals and feedback | 03 | stochastic gradients and their variance |
| | 07 | the pretrained model as the initial policy |
| | 08 | sampling and the KL divergence |
| [10](/core/unit-10/) Failure | 01 | the i.i.d. argument and held-out evaluation |
| | 07 | contamination |
| | 09 | reward hacking |
| [11](/core/unit-11/) Mechanism | 04 | attention heads and the residual stream |
| | 10 | evaluation with controls |
| [12](/core/unit-12/) Knowledge | 01 | generalization from finite samples |
| | 10 | statistical uncertainty and evaluation validity |
| | 11 | interventional experiments |

Three consequences for planning:

- **Units 01, 03 and 04 are on almost every path.** Every later unit depends on at least one of them directly.
- **The evaluation thread is short.** Unit 10 needs from Unit 09 only the idea of reward hacking (§9.7.1), and from Unit 07 only contamination (§7.8.5); a course can go 01 → 03 → 04 → 07 → 10 with Unit 02's essentials, as the [compact undergraduate course](/pathways/undergraduate/) does.
- **The generation thread needs Unit 05's numerical half.** Unit 08 uses ODE discretization and stability from Unit 05, not its recurrent architectures; a reader heading for Units 08–09 can read §5.2, §5.3.1–§5.3.4 and §5.3.9 (stability and exact discretization) and leave the rest of Unit 05 for later.

## Mathematics to keep at hand

The units teach the mathematics they need where they need it; the [appendices](/appendix/) collect it by subject. Before a unit, skim the listed appendices; during a deep pass, follow the unit's links into them.

| Unit | Appendices |
|---|---|
| 01 | [C](/appendix/c-probability/) probability · [D](/appendix/d-information-theory/) information theory · [E](/appendix/e-statistics/) statistics |
| 02 | [A](/appendix/a-linear-algebra/) linear algebra · [B](/appendix/b-calculus/) calculus |
| 03 | [A](/appendix/a-linear-algebra/) · [B](/appendix/b-calculus/) · [F](/appendix/f-optimization/) optimization |
| 04 | [A](/appendix/a-linear-algebra/) · [B](/appendix/b-calculus/) |
| 05 | [A](/appendix/a-linear-algebra/) · [G](/appendix/g-linear-systems/) linear systems |
| 06 | [F](/appendix/f-optimization/) · [A](/appendix/a-linear-algebra/) |
| 07 | [C](/appendix/c-probability/) · [D](/appendix/d-information-theory/) |
| 08 | [B](/appendix/b-calculus/) · [C](/appendix/c-probability/) · [D](/appendix/d-information-theory/) |
| 09 | [H](/appendix/h-markov-decision/) decision processes · [F](/appendix/f-optimization/) · [D](/appendix/d-information-theory/) |
| 10 | [C](/appendix/c-probability/) · [E](/appendix/e-statistics/) |
| 11 | [A](/appendix/a-linear-algebra/) · [B](/appendix/b-calculus/) |
| 12 | [E](/appendix/e-statistics/) · [C](/appendix/c-probability/) |

## Paths through the whole book

| Path | Units | For |
|---|---|---|
| **Linear** | 01 → 12 | A first reading; the narrative assumes it |
| **Evaluation first** | 01 → 02 (essentials) → 03 → 04 → 07 → 10 → 12 | Readers who must judge models before building them |
| **Architecture** | 01 → 02 → 03 → 04 → 05 → 06 → 11 | Readers who build and inspect models |
| **Generation and control** | 01 → 03 → 05 (§5.2, §5.3.1–§5.3.4, §5.3.9) → 07 → 08 → 09 | Readers heading for generative models and post-training |
| **Frontier systems for EE** | 04 (§4.8) → 05 (memory limits) → 06 → 09 (§9.8.4–§9.8.5) → 10 | Readers connecting long context, inference, accelerators and evidence |

The frontier-systems path is deliberately not an LLM survey. Keep one worksheet with four columns—arithmetic, memory, communication and measurement—and place every new method in the column whose constraint it changes. Every path ends where the book does: with [Unit 12](/core/unit-12/)'s question of how a claim becomes a finding, which no shortcut skips.
