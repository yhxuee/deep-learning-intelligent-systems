---
title: "5.8 Modern developments"
description: How recent sequence architectures extend the unit's principles — structured and selective state-space models, linear-attention variants, gated recurrences and hybrids — with the evidence status of each claim.
sidebar: { label: "5.8 Modern developments", order: 8 }
---

This section maps recent architectures onto the principles of the unit. It will age faster than the rest of the chapter, so every claim carries an evidence label: **[D]** derivation, **[E]** experimental evidence from controlled comparisons, **[O]** engineering observation (reported results without matched controls), **[H]** open hypothesis. Read the original papers for details; the point here is to see each design as an answer to the question of §5.1.

## 5.8.1 Structured state-space models

| Model | Principle from this unit | What it adds |
|---|---|---|
| **S4** ([Gu et al. 2022a](/references/#gu-2022)) | exact discretization (§5.3.9); convolution view (§5.3.2) | a structured $A$ (from HiPPO, [Gu et al. 2020](/references/#gu-2020)) that is stable and cheap to exponentiate; FFT-based training on long sequences **[D]** for the algorithm, **[E]** for long-range benchmark gains reported there |
| **Diagonal SSMs** ([Gu et al. 2022b](/references/#gu-2022b)) | diagonalization (§5.3.3) | shows diagonal $A$ with suitable initialization matches structured $A$ on many tasks — a bank of continuous-time EMAs with learned time constants **[E]** |
| **Mamba** ([Gu & Dao 2024](/references/#gu-2024)) | selectivity (§5.3.10); parallel scan | input-dependent $\Delta_t, B_t, C_t$; hardware-aware scan; reported language-modelling quality competitive with Transformers of similar size at the scales tested **[O]** |
| **Mamba-2 / structured state-space duality** ([Dao & Gu 2024](/references/#dao-2024)) | the matrix view (§5.5.4) | a restricted selective SSM whose matrix $M$ is also expressible as a masked attention-like product, so it can be computed either as a scan or with matrix multiplications **[D]** |

## 5.8.2 Linear attention and its descendants

| Model | Principle | What it adds |
|---|---|---|
| **Linear Transformers** ([Katharopoulos et al. 2020](/references/#katharopoulos-2020)) | matrix-state recurrence (§5.3.12) | kernel feature maps; constant-memory decoding **[D]** |
| **Retention / RetNet** ([Sun et al. 2023](/references/#sun-2023)) | decaying matrix state $S_t = \gamma S_{t-1} + v_tk_t^\top$ | explicit exponential decay — an EMA of outer products — with parallel, recurrent and chunked forms **[D]**; quality claims **[O]** |
| **RWKV** ([Peng et al. 2023](/references/#peng-2023)) | per-channel learned decay (§5.3.1) | an RNN-style formulation trainable in parallel; scaled to billions of parameters **[O]** |
| **Delta-rule models / DeltaNet** ([Schlag et al. 2021](/references/#schlag-2021); [Yang et al. 2024](/references/#yang-2024)) | delta rule (§5.3.12, Theorem 5.3) | overwrite instead of accumulate; a parallel training algorithm over sequence length **[D]**; improved recall over additive linear attention in reported experiments **[E]** |

## 5.8.3 Gated recurrences revisited

**xLSTM** ([Beck et al. 2024](/references/#beck-2024)) revisits the LSTM with exponential gating and a matrix-valued memory with covariance-style (outer-product) updates — the gated EMA of §5.3.8 combined with the associative memory of §5.3.12. Its reported results at moderate scale are **[O]**. The broader trend is convergence: selective SSMs, gated linear attention and modern LSTMs all implement "a matrix or vector state, updated by an input-dependent decay and an outer-product or additive write".

## 5.8.4 Hybrids

If attention buys exact recall at quadratic cost and recurrences buy linear cost with bounded recall (§5.5.4), a natural design mixes them: mostly recurrent layers with a few attention layers. **Jamba** ([Lieber et al. 2024](/references/#lieber-2024)) interleaves Mamba and Transformer layers (with mixture-of-experts) and reports strong long-context behaviour **[O]**. Arora et al. found that adding a small amount of attention closes much of the associative-recall gap in their synthetic and language-modelling experiments **[E]** ([Arora et al. 2024](/references/#arora-2024)).

Whether hybrids dominate pure architectures at frontier scale, and what ratio of attention to recurrence is optimal, is an **open hypothesis [H]**: published comparisons rarely match data, compute, tuning effort and evaluation, and results at one scale do not reliably transfer to another (Unit 06).

## 5.8.5 What to take from this section

1. Every model above is a choice of **state structure** (vector, matrix, cache), **write rule** (fixed decay, gated decay, additive, delta) and **computation form** (recurrence, convolution, scan, matrix product). You can place any new architecture on these three axes.
2. The capacity bounds of §5.5 apply to all fixed-state models, whatever their name.
3. Claims of the form "architecture X matches Transformers" are almost always **[O]**. To upgrade them to **[E]**, look for matched compute, matched data, multiple seeds, and recall-specific evaluations (§5.7.9).
