---
title: "5.8 Modern developments"
description: A November 2026 evidence-labeled map of structured and selective SSMs, delta and linear attention, test-time learned state, gated recurrences and hybrids.
sidebar: { label: "5.8 Modern developments", order: 8 }
---

This section maps a **selective literature snapshot through November 2026** onto the principles of the unit; it is not a leaderboard or an exhaustive catalog. It will age faster than the rest of the chapter, so every claim carries an evidence label: **[D]** derivation, **[E]** experimental evidence from a stated comparison, **[O]** engineering observation or author-reported result without enough matched controls for a broad conclusion, **[H]** open hypothesis. Even **[E]** is scoped to the paper's data, scale and implementation. Read the original papers for details; the goal here is to identify each design's state, write rule and computation form.

## 5.8.1 Structured state-space models

| Model | Principle from this unit | What it adds |
|---|---|---|
| **S4** ([Gu et al. 2022a](/references/#gu-2022)) | exact discretization (§5.3.9); convolution view (§5.3.2) | HiPPO-motivated structure ([Gu et al. 2020](/references/#gu-2020)) that makes long convolution kernels tractable; FFT-based training **[D]** for the algorithm, with task gains reported in the paper **[E]** within those comparisons |
| **Diagonal SSMs** ([Gu et al. 2022b](/references/#gu-2022b)) | diagonalization (§5.3.3) | suitably initialized diagonal complex modes can match more structured parameterizations on many reported tasks **[E]**; negative-real modes are decaying filters, while complex modes can also oscillate—they are not all literal EMAs |
| **Mamba** ([Gu & Dao 2024](/references/#gu-2024)) | selectivity (§5.3.10); parallel scan | input-dependent $\Delta_t, B_t, C_t$; hardware-aware scan; reported language-modelling quality competitive with Transformers of similar size at the scales tested **[O]** |
| **Mamba-2 / structured state-space duality** ([Dao & Gu 2024](/references/#dao-2024)) | the matrix view (§5.5.4) | a restricted selective SSM whose matrix $M$ is also expressible as a masked attention-like product, so it can be computed either as a scan or with matrix multiplications **[D]** |

## 5.8.2 Linear attention and its descendants

| Model | Principle | What it adds |
|---|---|---|
| **Linear Transformers** ([Katharopoulos et al. 2020](/references/#katharopoulos-2020)) | matrix-state recurrence (§5.3.12) | kernel feature maps; constant-memory decoding **[D]** |
| **Retention / RetNet** ([Sun et al. 2023](/references/#sun-2023)) | decaying matrix state $S_t = \gamma S_{t-1} + v_tk_t^\top$ | explicit exponential decay — an EMA of outer products — with parallel, recurrent and chunked forms **[D]**; quality claims **[O]** |
| **RWKV** ([Peng et al. 2023](/references/#peng-2023)) | per-channel learned decay (§5.3.1) | an RNN-style formulation trainable in parallel; scaled to billions of parameters **[O]** |
| **Delta-rule models / DeltaNet** ([Schlag et al. 2021](/references/#schlag-2021); [Yang et al. 2024](/references/#yang-2024)) | delta rule (§5.3.12, Theorem 5.3) | error-correcting writes toward a new association rather than blind accumulation; parallel training over sequence length **[D]**; improved recall over additive linear attention in the reported comparisons **[E]** |
| **Gated DeltaNet** ([Yang et al. 2025](/references/#gated-deltanet-2025)) | input-dependent forgetting plus delta updates | separates rapid erasure from targeted association updates and supplies a parallel algorithm **[D]**; gains over the tested Mamba-2 and DeltaNet baselines are **[E]** for that setup, not a universal ranking |
| **Kimi Linear / KDA** ([Kimi Team 2025](/references/#kimi-linear-2025)) | finer-grained gated delta state plus occasional attention | a large hybrid using Kimi Delta Attention and latent attention; identical-recipe comparisons and released implementations are useful evidence **[E]**, while broad “replacement for full attention” claims remain **[O]** outside the evaluated regimes |

## 5.8.3 The state becomes an online learner

The matrix state of linear attention already looks like fast-changing weights. A further step is to make the state a small model and define each token's write as an optimization step on a self-supervised objective.

- **TTT layers** make the recurrent state a linear model or MLP and update its parameters during the forward pass ([Sun et al. 2024](/references/#ttt-2024)). This is still a recurrence—“training” here is the state-update rule, not a second global fine-tuning run. Linear-time sequence scaling follows from bounded work per token **[D]**; reported long-context comparisons are **[E]** within the tested scales, while hardware efficiency depends strongly on chunking and kernels **[O]**.
- **Titans** combines short-context attention with a neural long-term memory updated by a test-time learning rule ([Behrouz et al. 2025](/references/#titans-2025)). It changes *what the state is* from a vector of activations to parameters of a learned memory. The update equations provide the mechanism **[D]**; multi-domain and very-long-context results are author-reported evidence **[O]** until matched independent replications establish how broadly they transfer.

This direction does not evade capacity questions. It enlarges and restructures the state, and it changes the write rule from a fixed algebraic update to an optimizer. Precision, interference, optimization stability and the cost of updating that state still have to be measured.

## 5.8.4 Gated recurrences revisited

**xLSTM** ([Beck et al. 2024](/references/#beck-2024)) revisits the LSTM with exponential gating and a matrix-valued memory with covariance-style (outer-product) updates — the gated EMA of §5.3.8 combined with the associative memory of §5.3.12. Its reported results at moderate scale are **[O]**. The broader trend is convergence: selective SSMs, gated linear attention and modern LSTMs all implement "a matrix or vector state, updated by an input-dependent decay and an outer-product or additive write".

## 5.8.5 Hybrids

Attention provides a growing set of per-token records and potentially full cut rank, but not guaranteed exact recall; recurrences provide bounded state and cheaper length scaling, but must compress (§5.5.4). A natural design mixes them: mostly recurrent layers with a few attention layers. **Jamba** ([Lieber et al. 2024](/references/#lieber-2024)) interleaves Mamba and Transformer layers with mixture-of-experts and reports strong long-context behavior **[O]**. In the experiments of [Arora et al. (2024)](/references/#arora-2024), a small amount of attention closes much of the associative-recall gap **[E]**. Gated DeltaNet and Kimi Linear continue the same hybrid trend at different scales and with different attention schedules.

Whether hybrids dominate pure architectures at frontier scale, and what ratio of attention to recurrence is optimal, is an **open hypothesis [H]**: published comparisons rarely match data, compute, tuning effort and evaluation, and results at one scale do not reliably transfer to another (Unit 06).

## 5.8.6 What to take from this section

1. Every model above is a choice of **state structure** (vector, matrix, cache), **write rule** (fixed decay, gated decay, additive, delta) and **computation form** (recurrence, convolution, scan, matrix product). You can place any new architecture on these three axes.
2. Apply the right capacity statement: bit-counting bounds require finite precision; cut-rank bounds require a conditional linear map. A nonlinear or optimizer-valued state is not exempt, but it is not covered by an $N$-dimensional rank slogan alone.
3. A claim that “architecture X matches Transformers” is never architecture-only. Look for matched data, tokens, active parameters, training and inference compute, tuning effort, kernels, multiple seeds and recall-specific controls (§5.7.9). Label the conclusion **[E]** only for the comparisons actually performed; extrapolation beyond them remains **[O]** or **[H]**.
