---
title: "11.8 Modern developments"
description: Sparse features, transcoders, attribution graphs, attention tracing, automated and formally verified circuit discovery, benchmarks, and models trained for interpretability—as of November 2026.
sidebar: { label: "11.8 Modern developments", order: 8 }
---

This page is a dated map of a fast-moving field, current to **November 2026**. It separates what a method constructs from what experiments have established. Labels indicate **[D]** derivation, **[E]** empirical evidence, **[O]** engineering observation, and **[H]** open hypothesis.

The main shift since early head-by-head circuit analysis is a move through four stages:

$$
\text{learn candidate features}
\rightarrow
\text{trace their interactions}
\rightarrow
\text{validate interventions}
\rightarrow
\text{benchmark or certify the claim}.
$$

Each arrow solves one bottleneck and introduces another.

## 11.8.1 Sparse autoencoders: change the units of analysis

### The problem

Transformer neurons are often polysemantic, while meaningful variables may be directions distributed across many neurons. Sparse autoencoders (SAEs) replace the original coordinate system with a learned overcomplete dictionary.

For an activation $h\in\mathbb R^d$, an SAE uses $m>d$ latent features:

$$
f(h)=\sigma\!\left(W_{\mathrm{enc}}(h-b)+b_{\mathrm{enc}}\right),
\qquad
\hat h=b+W_{\mathrm{dec}}f(h),
$$

and approximately minimizes

$$
\mathcal L_{\mathrm{SAE}}
=\mathbb E\left[\lVert h-\hat h\rVert_2^2
+\lambda\lVert f(h)\rVert_1\right].
$$

The reconstruction term keeps information; the sparsity penalty makes only a small fraction of the $m$ features active on each token. Overcompleteness permits more candidate features than original dimensions ([Bricken et al. 2023](/references/#bricken-2023); [Templeton et al. 2024](/references/#templeton-2024)) **[E]**.

### What the result means

If feature $j$ activates on French text, its decoder direction is $W_{\mathrm{dec}}[:,j]$, and clamping $f_j$ changes French-related output, we have three pieces of evidence: correlated examples, a learned direction, and an intervention. That is stronger than naming one neuron.

It still does not prove that feature $j$ is the model's unique internal concept “French.” The dictionary is learned by an auxiliary objective. Different widths, sparsity penalties, random seeds, and datasets can split one concept into several features or absorb one feature into another **[O]**.

The unexplained residual

$$
e(h)=h-\hat h
$$

is not a bookkeeping nuisance. Any behaviour that depends on $e(h)$ is absent from an analysis performed only in SAE coordinates **[D]**. Good reports therefore include reconstruction loss, downstream loss added by replacing $h$ with $\hat h$, feature sparsity, and intervention tests.

## 11.8.2 Transcoders: model a computation, not only a state

An SAE reconstructs a representation from itself. A **transcoder** instead approximates a model component's output $y=F(h)$ from its input:

$$
f(h)=\sigma(W_{\mathrm{enc}}h+b),
\qquad
\hat y=W_{\mathrm{dec}}f(h)+b_y.
$$

This places sparse features inside an input–output map. It is attractive for MLP analysis because a dense MLP can be replaced by a wider sparse module whose active features and connections are easier to trace ([Dunefsky et al. 2024](/references/#dunefsky-2024)) **[E]**.

Comparisons reported in 2025 found higher automated interpretability scores for transcoders and skip transcoders than for matched SAEs in the studied settings ([Paulo et al. 2025](/references/#paulo-2025)) **[E]**. The bounded reading is “these transcoder variants did better on these reconstruction–sparsity–interpretability evaluations,” not “transcoders are universally superior.” SAEs can operate directly on residual streams; transcoders are tied to a component map.

### Cross-layer transcoders

A cross-layer transcoder (CLT) lets a feature read at one layer and write predictions into multiple later layers. This can represent long-range influence more compactly than replacing each MLP independently. But it also creates a **replacement model**:

$$
F_{\text{original}}
\approx
F_{\text{replacement}}.
$$

The approximation sign matters. A beautiful circuit in the replacement model is evidence about the original only to the extent that reconstruction and intervention tests transfer.

## 11.8.3 Attribution graphs: a prompt-specific computational map

Circuit tracing with CLTs builds a graph for a particular prompt ([Ameisen et al. 2025](/references/#ameisen-2025)) **[E]**:

- nodes are active sparse features at token positions, input embeddings, outputs, or reconstruction-error terms;
- edges estimate how one node influences another;
- paths summarize chains of influence toward selected logits.

A simplified local edge score has the form

$$
A_{s\rightarrow t}
=a_s\,v_{\mathrm{out},s}^{\top}
J_{s\rightarrow t}
v_{\mathrm{in},t},
$$

where $a_s$ is source-feature activation, $v_{\mathrm{out},s}$ its write direction, $v_{\mathrm{in},t}$ the target's read direction, and $J_{s\rightarrow t}$ a locally linearized influence through intervening computation.

The graph is not a literal dump of every nonlinear operation. It depends on

1. the learned replacement model;
2. a local linearization around the prompt;
3. graph pruning needed to make millions of possible edges readable;
4. human labels assigned from activating examples.

Case studies applied these graphs to factual recall, multi-step reasoning, planning, refusals, and some hallucination examples in Claude 3.5 Haiku ([Lindsey et al. 2025](/references/#lindsey-2025)) **[E]**. The authors also reported reconstruction gaps, mechanistic-unfaithfulness risks, and overwhelming graph complexity. The correct conclusion is that attribution graphs generate and sometimes validate detailed hypotheses for studied prompts—not that they completely explain a frontier model.

### Validation remains interventional

If the graph predicts

$$
\text{Dallas}\rightarrow\text{Texas}
\rightarrow\text{Austin},
$$

then inhibiting the “Dallas” feature group should reduce “Texas” and downstream “Austin” features while leaving a matched unrelated path relatively unchanged. Output effects in the original model are stronger validation than internal agreement within the replacement model.

## 11.8.4 Explaining why attention looks where it does

Early attribution graphs could trace information carried through attention while leaving the attention pattern itself partly unexplained. Feature-level QK attribution extends the graph to the score

$$
s_{ij}=q_i^\top k_j.
$$

If query and key vectors are decomposed into feature contributions,

$$
q_i=\sum_a \alpha_a q_a,
\qquad
k_j=\sum_b \beta_b k_b,
$$

then

$$
s_{ij}
=\sum_{a,b}\alpha_a\beta_b q_a^\top k_b.
$$

The pairwise terms propose which query feature and key feature make position $i$ attend to position $j$. Work in 2025 integrated such QK attributions with feature graphs ([Kamath et al. 2025](/references/#kamath-2025)) **[E]**.

This is a direct modern extension of §11.3: OV analysis asks **what information is moved**; QK analysis asks **why that source is selected**. The decomposition is still prompt- and basis-dependent and requires intervention checks.

## 11.8.5 Automated circuit discovery: search first, explain second

Manual circuit discovery does not scale to thousands of components. Automated circuit discovery (ACD) methods assign importance to nodes or edges, then prune the computation graph while trying to preserve a task metric. A generic objective is

$$
\min_{C\subseteq G}|C|
\quad\text{subject to}\quad
\mathbb E_{x\sim D}
\left[d\bigl(f_C(x),f_G(x)\bigr)\right]\le\varepsilon.
$$

$G$ is the full graph, $C$ a proposed circuit, $d$ an output discrepancy, and $D$ the evaluation distribution. Activation patching can estimate edge importance; gradient-based attribution makes the search cheaper but inherits linearization error.

Automated Circuit Discovery recovered several previously hand-studied circuits under its evaluation setup ([Conmy et al. 2023](/references/#conmy-2023)) **[E]**. What it returns is a small predictive subgraph under a metric and ablation rule. Human-readable semantics and causal completeness are additional claims.

### Benchmarks replace one-off demonstrations

MIB introduced standardized tracks for circuit localization and causal-variable localization across tasks and models, with fixed splits and private-test evaluation ([Mueller et al. 2025](/references/#mueller-2025-mib)) **[E]**. In its reported experiments, attribution and mask-optimization methods led circuit localization; supervised distributed alignment search led causal-variable localization; SAE features did not outperform ordinary hidden dimensions on those benchmark tasks.

This is evidence about the included tasks, methods, metrics, and model sizes. It is not a general disproof of SAEs. Its broader contribution is methodological: a new interpretability method should beat baselines on held-out causal tests, not only produce appealing examples.

## 11.8.6 Formal guarantees: from sampled faithfulness to a domain claim

Ordinary circuit evaluation samples inputs. Even perfect agreement on $n$ prompts does not guarantee agreement between them. Formal mechanistic interpretability asks for a property over a continuous domain $\mathcal X$, for example

$$
\forall x\in\mathcal X,
\qquad
\lVert f_C(x)-f_G(x)\rVert_\infty\le\varepsilon.
$$

Work published at ICLR 2026 used neural-network verification to construct circuits with guarantees concerning input-domain robustness, patching robustness, and forms of minimality ([Hadad et al. 2026](/references/#hadad-2026)) **[D]/[E]**.

The experiments were on verification-scale vision models, not frontier language models. The significance is conceptual: “faithful on tested examples” and “certified throughout a defined region” are different standards. The open engineering problem is whether useful guarantees can scale to large sequence models and semantically meaningful domains **[H]**.

## 11.8.7 Train for interpretability rather than recover it afterward

Most methods accept a dense trained model and try to untangle it. Another direction modifies training so that circuits are sparse from the start.

Weight-sparse transformers constrain most connections to zero. Experiments reported that pruned circuits for several hand-crafted tasks contained more human-readable neurons, residual channels, and connections, while also showing a capability–interpretability trade-off and difficulty scaling beyond tens of millions of nonzero parameters ([Gao et al. 2025](/references/#gao-2025-sparse)) **[E]**.

This changes the optimization problem from

$$
\min_\theta \mathcal L_{\text{task}}(\theta)
$$

to a multi-objective problem such as

$$
\min_\theta
\mathcal L_{\text{task}}(\theta)
+\lambda\,\Omega_{\text{complexity}}(\theta).
$$

The promise is a model whose learned algorithm is easier to inspect. The cost may be capability, compute, or a sparsity pattern that is simple without being semantically meaningful. “Interpretable by construction” remains a research goal, not a solved product property **[H]**.

## 11.8.8 Three durable case studies

New tools do not erase earlier case studies; they clarify what made those studies convincing.

### Induction heads

[Olsson et al. (2022)](/references/#olsson-2022) connected the formation of induction heads with a sharp improvement in in-context learning, using stronger causal evidence in small attention-only models and more correlational evidence at larger scale **[E]/[H]**. It remains a mechanism for some copying and in-context behaviour, not a theorem that all in-context learning uses induction heads.

### Othello world representation

A sequence model trained on legal Othello moves developed a board-state representation that could be decoded and edited; interventions changed move predictions consistently with the edited board ([Li et al. 2023](/references/#li-2023)) **[E]**. Intervention distinguishes the result from probe-only evidence. Generalization from a closed board game to open-web language remains an additional hypothesis.

### Grokking

For small modular-addition networks, reverse engineering found Fourier features and trigonometric structure; mechanism-derived progress measures changed before test accuracy jumped ([Nanda et al. 2023](/references/#nanda-2023)) **[E]**. This is a strong explanation of a bounded training phenomenon, not a universal theory of grokking.

## 11.8.9 What has changed—and what has not

| Earlier bottleneck | Modern response | Remaining gap |
|---|---|---|
| neurons are polysemantic | SAEs and sparse dictionaries | learned features may split, absorb, or omit computation |
| MLPs are hard to trace | transcoders and CLTs | replacement-model error |
| circuits are found by hand | automated discovery and attribution graphs | metric dependence and incomplete semantics |
| attention source choice is opaque | feature-level QK attribution | local and basis-dependent decomposition |
| methods use bespoke examples | benchmarks such as MIB | benchmark scope and contamination |
| faithfulness is sampled | formal verification | severe scaling limits |
| dense models are tangled | sparsity-aware training | capability–interpretability trade-off |

The durable standard from this unit has not changed:

1. define a behaviour and outcome;
2. choose candidate variables;
3. distinguish observation, decoding, attribution, and intervention;
4. measure approximation error introduced by interpretability tools;
5. validate on the original model and held-out inputs;
6. state the domain and surviving alternatives.

Modern methods make larger hypotheses visible. They do not make those hypotheses self-validating.
