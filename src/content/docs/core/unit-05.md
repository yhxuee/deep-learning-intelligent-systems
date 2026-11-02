---
title: "Unit 05 · How does a model remember?"
description: Recurrence, state-space models, linear attention and the KV cache as different answers to what a finite system can retain from an unbounded history.
sidebar: { label: "05 · Memory", order: 5 }
unit:
  number: 5
  part: 2
  question: What can a finite state retain from an arbitrarily long history?
  anchor: "The linear recurrence h_t = A h_{t-1} + B x_t, contrasted with the KV cache as uncompressed memory."
  spine: "Stage 05 — implement KV-cache decoding for the spine model; build an associative-recall stress test and measure where it fails."
  checkpoint: Expand a linear recurrence, relate its spectrum to memory decay, and design a memory stress test — without AI.
  status: draft
  atlas: [mech-linear-recurrence, mech-kv-cache, arch-selective-ssm, mech-attention]
---

:::caution[Draft]
This unit is a complete first draft awaiting subject and pedagogical review. Section structure and learning activities are final; some prose will change.
:::

## 1 · The problem

A battery-powered device processes a continuous sensor stream and must respond to rare events long after they occur. It cannot store the entire stream. What must it keep, and what can it discard?

Three candidate designs: **(a)** a fixed window of the most recent inputs; **(b)** unrestricted access to stored past inputs; **(c)** a compressed state updated at every step.

:::note[Predict before reading on]
For each design, state its memory cost as a function of stream length $T$, and give one input pattern on which it fails.
:::

## 2 · The linear recurrence

The simplest compressed state is

$$
h_t = A h_{t-1} + B x_t,\qquad h_t\in\mathbb{R}^N .
$$

Unrolling it gives the closed form you must be able to derive unaided:

$$
h_t = A^t h_0 + \sum_{k=1}^{t} A^{\,t-k} B\, x_k .
$$

The influence of input $x_k$ on the state at time $t$ is the matrix $A^{t-k}B$. If $A = V\Lambda V^{-1}$ is diagonalizable with spectral radius $\rho(A) < 1$, that influence decays roughly like $\rho^{\,t-k}$ — **memory fades exponentially**. With $\rho > 1$ the state explodes.

Two assumptions hide in "roughly": diagonalizability, and the conditioning of $V$. **Non-normal** matrices can amplify an input transiently before decaying even when $\rho < 1$ — a counterexample the lab asks you to construct.

## 3 · Memory and gradients are the same problem

Because $\partial h_t/\partial h_k = A^{t-k}$, the gradient signal connecting a loss at time $t$ to an input at time $k$ obeys the same decay ([Unit 03](/core/unit-03/), transfer task). With nonlinear recurrences the Jacobian product varies by step, but the conclusion holds: gradients vanish or explode over long spans [Bengio et al. 1994](/references/#bengio-1994); [Pascanu et al. 2013](/references/#pascanu-2013). Gated recurrences such as the LSTM [Hochreiter & Schmidhuber 1997](/references/#hochreiter-1997) learn when to hold the state nearly constant.

## 4 · State-space models and selectivity

Structured state-space models parameterize $A$ so that long-range kernels are stable and fast to compute [Gu et al. 2022](/references/#gu-2022). When $A, B$ are time-invariant the whole recurrence is a convolution and can be trained in parallel. **Selective** SSMs such as Mamba [Gu & Dao 2024](/references/#gu-2024) make $A_t, B_t$ depend on the current input, which lets the model decide *what to write* — content-dependence, as in attention — while keeping a fixed-size state and linear cost in $T$. See [selective SSM](/atlas/nodes/arch-selective-ssm/).

## 5 · Linear attention is a recurrence

Drop the softmax from attention and write $y_t = \sum_{j\le t} v_j (k_j^\top q_t)$. Then

$$
M_t = M_{t-1} + v_t k_t^\top,\qquad y_t = M_t q_t ,
$$

a recurrence with a **matrix-valued state** $M_t \in \mathbb{R}^{d_v\times d_k}$. It stores key–value pairs by superposition. With orthonormal keys, up to $d_k$ pairs can be retrieved exactly; beyond $d_k$ the pairs interfere (cross-talk) and retrieval error is unavoidable. That is the finite-state capacity limit in its cleanest form — and an invariant the lab tests.

## 6 · The KV cache: memory without compression

Softmax attention with a [KV cache](/atlas/nodes/mech-kv-cache/) keeps *every* past key and value. Retrieval can be exact at any distance, but memory and per-step compute grow linearly with $T$. Empirically, fixed-state models lag attention on associative recall and copying tasks [Arora et al. 2024](/references/#arora-2024); [Jelassi et al. 2024](/references/#jelassi-2024) — consistent with the capacity argument above.

| Design | Memory | Per-step cost | Exact recall at distance $T$? |
|---|---|---|---|
| Fixed window $W$ | $O(W)$ | $O(W)$ | only if $T \le W$ |
| Linear recurrence / SSM | $O(N)$ | $O(N)$ | no — decays or interferes |
| Linear attention | $O(d_k d_v)$ | $O(d_k d_v)$ | up to $\approx d_k$ items |
| Softmax attention + KV cache | $O(T d)$ | $O(T d)$ | yes, until the context ends |

The durable lesson is the **trade-off surface**, not the ranking. Hybrid architectures choose points on it.

---

## Learning activities

### Construction (AI-off) — `labs/unit-05-memory`

Implement the recurrence both as a loop and in closed form, and a linear-attention associative memory. Tests check closed-form equivalence, decay governed by $\rho(A)$, a non-normal transient-growth example, and exact recall for $\le d_k$ orthonormal keys versus unavoidable error beyond it.

### Challenge (Critic mode)

Test recall of one isolated token at distance $T$. When a model fails, decide whether the failure reflects **capacity**, **data construction**, **optimization**, or the **evaluation procedure** — and design one control for each.

### Experiment (Collaborator mode)

Build data with short-range regularities plus rare, distant, relevant signals. With matched budgets, compare prediction quality, throughput and memory for a windowed model, a recurrent model and the spine Transformer with a KV cache.

### Spine increment — Stage 05

Implement KV-cache decoding. **Verification target:** cached generation produces the same logits as full recomputation (to float tolerance) for every prefix; measure tokens/second with and without the cache.

### Final artifact

A short paper: *How should sequence models store and retrieve history under a fixed budget?* — hypotheses, derivations, methods, uncertainty, negative results and limitations.

## Key sources

[Bengio et al. 1994](/references/#bengio-1994) · [Hochreiter & Schmidhuber 1997](/references/#hochreiter-1997) · [Pascanu et al. 2013](/references/#pascanu-2013) · [Gu et al. 2022](/references/#gu-2022) · [Gu & Dao 2024](/references/#gu-2024) · [Arora et al. 2024](/references/#arora-2024) · [Jelassi et al. 2024](/references/#jelassi-2024) · [Pope et al. 2023](/references/#pope-2023)
