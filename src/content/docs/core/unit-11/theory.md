---
title: "11.5 General theory"
description: A careful theory of neural-network interventions, effect measures, mediation, causal abstraction, redundancy, and the evidence required for a mechanistic claim.
sidebar: { label: "11.5 General theory", order: 5 }
---

The preceding sections developed the foundational toolset of mechanistic interpretability: residual stream accounting, attention matrix decomposition, component ablation, activation patching, linear probing, and gradient attribution. In modern deep learning experimentation, extracting numbers or rendering colorful attention heatmaps is straightforward. The profound theoretical challenge lies in epistemology and causal boundaries:

> What does a measured numerical quantity mathematically and causally license us to assert about the internal computations of a network?

A mechanistic explanation is never a poetic or anthropomorphic narrative of internal activations; it is fundamentally a **predictive causal model**—one that must withstand rigorous counterfactual interventions and accurately predict internal behavior on novel, unseen distributions.

## 11.5.1 From a forward pass to a structural causal model

Holding the learned parameters of a neural network fixed, the entire forward execution program can be formalized as a set of deterministic assignment equations:

$$
Z_j := g_j(\operatorname{pa}(Z_j)),
$$

where $\operatorname{pa}(Z_j)$ denotes the parent variables in the computational graph directly read by node $j$. Input token embeddings, positional encodings, attention head outputs, MLP intermediate states, residual stream vectors, and output logits all function as structural nodes. In the computational sense of [Pearl (2009)](/references/#pearl-2009), these directed acyclic assignments define a deterministic **Structural Causal Model (SCM)**.

Consider a minimal two-step forward computation:

$$
H := Wx,
\qquad
Y := u^\top H.
$$

Standard forward execution answers a passive observational query: "Given input $x$, what is the value of output $Y$?"
In contrast, an active **causal intervention** $\operatorname{do}(H=h^*)$ severs the natural assignment $H:=Wx$, clamps intermediate variable $H$ to an externally specified vector $h^*$, and re-evaluates the downstream program $Y=u^\top h^*$.
In modern Transformer research:
- **Head Zero-Ablation**: corresponds to the intervention $\operatorname{do}(o^{(\ell,h)} = 0)$, clamping a head's output to the zero vector;
- **Activation Patching**: corresponds to the intervention $\operatorname{do}(H_c = H_c(x_{\text{clean}}))$, substituting activations recorded during a clean forward pass into a corrupted execution graph.

Both operations construct a **modified forward program**.

Direct access to computational graphs and parameter matrices grants interpretability researchers **flawless low-level experimental control**: we need not rely on observational correlations to speculate whether an attention head influences downstream logits; we can intervene upon its tensor, re-execute the graph, and measure the exact counterfactual shift.

However, three critical epistemological caveats constrain mechanistic claims:
1. **Variable boundaries reflect analytical choices**: Defining a variable as a scalar neuron, a multi-dimensional polysemantic direction, an entire attention head, or a semantic subspace produces fundamentally distinct causal graphs.
2. **Interventions can induce out-of-distribution artifacts**: Setting residual vectors to zero or splicing activations across divergent contexts may generate anomalous internal states never encountered on the natural data manifold. Downstream performance drops may reflect fragility to numerical corruption rather than genuine computational dependency.
3. **The low-level computational graph is not an explanation**: Listing millions of floating-point multiply-accumulate operations is mathematically exact but conceptually vacuous. Mechanistic interpretability seeks a **sparse, high-level algorithmic abstraction** that faithfully preserves the system's causal dynamics.

## 11.5.2 Choose the outcome before measuring an effect

In experimental interpretability, asserting that "the output changed" is scientifically incomplete. Every causal study must pre-register a well-defined scalar **outcome metric** $Y$ aligned with the specific theoretical hypothesis.

In autoregressive language models, the most standard scalar metrics include:
1. **Logit Margin**:
$$
Y_{\text{margin}}(x)=z_{y^*}(x)-z_{y^-}(x),
$$
where $y^*$ is the correct target token and $y^-$ is the top competing distractor token. A larger margin indicates stronger model preference for the correct continuation.
2. **Negative Log-Likelihood (NLL)**:
$$
Y_{\text{NLL}}(x)=-\log p(y^*\mid x).
$$
This corresponds directly to cross-entropy loss, where smaller values reflect higher predictive confidence.

Given a clean input $x_c$ (e.g. an in-distribution prompt prompting a factual completion) and a corrupted baseline $x_b$ (e.g. where the subject entity is replaced with a distractor), the **Total Effect (TE)** across the full system is:

$$
\operatorname{TE}=Y(x_c)-Y(x_b).
$$

The Total Effect measures the macro-level shift induced by changing the entire input, without isolating intermediate pathways. Letting $H_c(x)$ denote the internal activation of component $c$, the **Patching Effect (PE)** achieved by substituting clean activations into the corrupted run is:

$$
\operatorname{PE}_c=
Y\!\left(x_b;\operatorname{do}\bigl(H_c=H_c(x_c)\bigr)\right)-Y(x_b).
$$

If patching component $c$ alone recovers the bulk of the total effect, then component $c$ carries sufficient differential information to restore the target behavior under this corruption. We define the dimensionless **Recovery Score**:

$$
R_c=\frac{\operatorname{PE}_c}{\operatorname{TE}},
\qquad \operatorname{TE}\neq0.
$$

- $R_c \approx 1$: Component $c$ independently restores near-complete causal performance for this input pair and metric;
- $R_c \approx 0$: Component $c$ transmits negligible differential signal for this outcome;
- $R_c < 0$: Patching moves the output in the incorrect direction;
- $R_c > 1$: Feasible under nonlinear synergies, where hybrid states suppress downstream inhibitory paths, causing the patched run to outperform clean execution.

Crucially: $R_c$ is not a universal anatomical fraction of "how much computation component $c$ performs." It depends fundamentally on the chosen clean/corrupted pair, baseline corruption design, intervention site, outcome metric, and background network state.

## 11.5.3 Patching is not automatically a natural indirect effect

Classical causal mediation analysis investigates what proportion of an exposure's effect operates through a mediator. Neural activation patching structurally mirrors this mediation chain:

$$
x\longrightarrow H_c\longrightarrow Y,
$$

motivating extensive literature tracing factual retrieval and information routing ([Vig et al. 2020](/references/#vig-2020); [Meng et al. 2022](/references/#meng-2022)). However, topological resemblance to mediation graphs does not guarantee that patching recovers a uniquely defined "Natural Indirect Effect."

Consider a toy multiplicative system with output $Y = H_1 H_2$. In the clean execution, $(H_1, H_2) = (1, 1) \implies Y = 1$; in the corrupted execution, $(H_1, H_2) = (0, 0) \implies Y = 0$, giving $\operatorname{TE} = 1 - 0 = 1$.
Patching $H_1=1$ alone yields $Y = 1 \times 0 = 0$, producing $\operatorname{PE}_1 = 0 \implies R_1 = 0$.
Patching $H_2=1$ alone similarly yields $Y = 0 \times 1 = 0$, producing $\operatorname{PE}_2 = 0 \implies R_2 = 0$.
Yet patching both simultaneously causes output to jump to 1!
In such nonlinear synergistic systems, **no context-free linear additive decomposition exists** that can partition total causal responsibility into isolated shares.

Real networks exhibit three confounding phenomena:
- **Multiplicative Synergies**: An activation's functional utility often depends on whether downstream gating heads are open;
- **Functional Redundancy**: Redundant backup heads maintain performance when any single head is ablated, masking functional importance;
- **Hybrid Off-Distribution States**: Splicing clean features into corrupted contexts can push intermediate representations off the natural activation manifold.

Unless formal non-interaction assumptions are verified, experimental findings should be described as *patching effects* rather than general causal mediation shares.

## 11.5.4 Algebraic contribution is not causal effect

In Transformer linear readouts, attention head $c$'s output $o_c \in \mathbb R^d$ projects onto output logits via the unembedding matrix $W_U \in \mathbb R^{V\times d}$ as:

$$
\Delta z_c=W_Uo_c.
$$

This quantity is **Direct Logit Attribution (DLA)**. It constitutes an exact algebraic projection enabled by the linear residual stream, answering: "What geometric vector does this component write directly along the unembedding direction?"

Crucially, **algebraic contribution does not equal causal effect**. It cannot answer the counterfactual question: "What would the model output if this component were removed?"

Ablating $o_c$ can alter subsequent LayerNorm scaling, reshape attention routing in later layers, and toggle nonlinear MLP activations. Thus, in general:

$$
W_Uo_c\neq z(x)-z\bigl(x;\operatorname{do}(o_c=0)\bigr).
$$

Induction circuits provide the canonical demonstration of this distinction:
In a two-layer attention network, Layer 0 Previous-Token Heads (Head 0.0) write previous-token features into the residual stream to serve as keys for Layer 1 Induction Heads (Head 1.0). Because the final unembedding layer ignores these transition features, Head 0.0's direct logit attribution $\Delta z_{0.0}$ is virtually zero!
Yet, performing zero-ablation $\operatorname{do}(o^{(0,0)} = 0)$ causes model loss on repeated sequences to explode from $0.0004$ to $2.84$. Head 0.0 has **zero direct algebraic contribution, yet exerts an indispensable indirect causal role**.

| Interpretability Metric | Mathematical Statement Type | Core Scientific Question Answered | Key Analytical Limitation |
|---|---|---|---|
| Direct Logit Attribution (DLA) | Endpoint static algebraic projection | What does this component write directly along the output direction? | Blind to all indirect multi-layer causal routing |
| Gradient-Input Attribution | First-order local sensitivity | Which infinitesimal perturbation changes output fastest? | Confounds sensitivity with finite causal sufficiency |
| Zero-Ablation Effect | Counterfactual intervention outcome | What happens if this component is excised from the network? | Vulnerable to unnatural off-manifold corruption and backup circuits |
| Activation Patching Effect | Cross-run substitution counterfactual | Is this component's activation sufficient to restore target behavior? | Highly sensitive to corrupted baseline choice |
| Linear Probing Accuracy | Geometric separability correlation | Does this intermediate representation linearly encode a feature? | Decodability does not prove downstream utilization |

These analytical tools belong to distinct epistemic tiers and cannot be treated as interchangeable substitutes.

## 11.5.5 Necessity, sufficiency, and redundancy

In causal interpretability, asserting that a component is "necessary" or "sufficient" must always be conditioned on a specified **intervention family** and **input distribution**.

**Interventional Necessity.** Let $B$ denote a binary behavioral success criterion (e.g. achieving correct next-token prediction on repeated sequences). A component set $c$ is interventionally necessary under ablation policy $a$ across distribution $D$ if:

$$
\Pr_{x\sim D}\!\left[B\bigl(f(x;\operatorname{do}(c=a))\bigr)\right]
\ll
\Pr_{x\sim D}[B(f(x))].
$$

This does not imply that no alternative computational mechanism could perform $B$, nor that the network could not adapt if retrained without $c$.

**Interventional Sufficiency.** If within an evaluation context $r$, clamping component $c$ to clean activation $h_c^*$ restores successful behavior in an otherwise failing model:

$$
B\bigl(f(r;\operatorname{do}(H_c=h_c^*))\bigr)=1,
$$

then state $h_c^*$ is **sufficient** under context $r$. Sufficiency does not imply uniqueness: parallel pathways may each be independently sufficient, and distributed mechanisms may require joint interventions.

**Joint Ablation and Redundancy.** For a subset of components $S$, define the joint ablation effect:

$$
\Delta(S)=Y(x)-Y\bigl(x;\operatorname{do}(H_S=a_S)\bigr).
$$

If single ablations of heads $i$ and $j$ yield near-zero effects ($\Delta(\{i\}) \approx 0$ and $\Delta(\{j\}) \approx 0$), yet joint ablation triggers catastrophic performance loss ($\Delta(\{i,j\}) \gg 0$), the network possesses **functional backup redundancy**.
This yields an essential methodological warning: **scanning attention heads one-by-one can falsely suggest that no single head is important.** True circuit discovery requires exploring combinatorial subgraphs.

## 11.5.6 Causal abstraction: what mechanistic claims actually assert

Consider a high-level algorithmic model $H$ for sequence induction:
1. Search input history for earlier occurrences of the current token;
2. Retrieve the token immediately following that earlier match;
3. Boost the corresponding output logit for that subsequent token.

Asserting that "Head 1.0 implements induction" is a formal claim of **Causal Abstraction**: it asserts that low-level tensor operations correspond systematically to high-level algorithmic transitions.

Let $L$ denote the low-level neural forward execution, $H$ the high-level algorithm, $\tau$ an abstraction map mapping continuous neural states to discrete algorithmic variables, $i$ a high-level intervention, and $\omega(i)$ the corresponding low-level tensor intervention.

A **Faithful Causal Abstraction** requires the following diagram to commute approximately across domain $\mathcal X$:

$$
\boxed{
\tau\!\left(L_{\omega(i)}(x)\right)
\approx
H_i\!\left(\tau(x)\right)
}
\qquad
x\in\mathcal X.
$$

The mathematical meaning is transparent: **intervening at the low-level tensor layer and abstracting the outcome produces the identical state as abstracting first and executing the intervention in the high-level symbolic algorithm.** This criterion establishes the mathematical foundation for testing circuits via **Interchange Interventions** ([Geiger et al. 2021](/references/#geiger-2021)).

A mechanistic claim achieves rigorous scientific status only when its high-level abstraction reliably predicts neural outputs across thousands of novel token identities, sequence lengths, and adversarial counterfactual controls.

## 11.5.7 The epistemic hierarchy of evidence

The strength of an interpretability claim must match the causal depth of its empirical evidence:

### Level 1: Black-Box Behavioral Observation
*Example*: The model achieves 98% accuracy on in-context copying benchmarks, but degrades on rare names.
*Limitation*: Identifies macro-level phenomena; reveals nothing about internal algorithmic structure.

### Level 2: Internal Representation Probing
*Example*: A linear probe decodes the preceding token from residual stream activations with 95% accuracy.
*Limitation*: Establishes linear separability; fails to prove downstream layers read this feature.

### Level 3: Static Attribution and Heatmap Correlation
*Example*: Attention matrices display diagonal striping, and DLA assigns high projections to target logits.
*Limitation*: Highlights candidate circuits, but cannot eliminate correlation artifacts or track indirect mediation.

### Level 4: Controlled Component Interventions
*Example*: Patching or zero-ablating specific heads systematically restores or degrades performance under controlled baselines.
*Limitation*: Establishes causal necessity/sufficiency under specific interventions, but does not formalize high-level algorithmic abstractions.

### Level 5: Faithful Causal Abstraction and Circuit Discovery
*Example*: A closed-loop circuit model predicts low-level responses across comprehensive interchange interventions, successfully anticipating failure modes on out-of-distribution prompts.
*Limitation*: Represents the highest attainable tier of mechanistic scientific explanation.

## 11.5.8 Mechanisms must make risky, falsifiable predictions

A scientifically meaningful mechanistic explanation must make precise, falsifiable predictions on **distributions and test cases not used during circuit discovery**:
- If an induction head depends causally on a previous-token head, destroying the previous-token subspace must impair induction matching while leaving unrelated attention heads intact;
- If copying relies on relative position encoding rather than absolute position memorization, shifting token sequences to unprecedented absolute offsets must preserve copying capabilities;
- If two heads form a redundant backup pair, pre-registered experiments must reveal near-zero single-ablation effects accompanied by catastrophic joint-ablation drops.

Formulating explicit falsification criteria **prior to experimentation** protects interpretability from descending into post-hoc storytelling. A narrative capable of rationalizing any experimental outcome explains nothing.

## 11.5.9 Mechanistic claim writing template

Vague assertions such as "Component $c$ is the mechanism for capability $X$" should be replaced with standardized, bounded causal statements:

> Over input distribution $D$, observing scalar outcome $Y$, intervention $I$ applied to component set $S$ produces an effect size $\Delta$ relative to baseline $B$. Controlled activation patching and interchange interventions conform to the predictions of high-level algorithm $A$ under controls $C$. This establishes causal role $R$ within the evaluated domain, while remaining uncommitted to untested out-of-domain alternative hypotheses $A_1$ and $A_2$.

## 11.5.10 Summary of core theoretical concepts

| Theoretical Concept | Scientific Fact Established | Inherent Epistemic Boundary |
|---|---|---|
| Computational SCM | Interventions are executed with 100% computational fidelity | Does not dictate natural semantic variable choices |
| Total Effect (TE) | Macro differences between clean and corrupted inputs | Does not reveal which internal pathways transmit the effect |
| Patching Effect (PE) | Transplanted activations locally restore measured outcomes | Does not represent a universal context-free causal share |
| Direct Logit Attribution (DLA) | Direct algebraic projection onto unembedding vectors | Ignores downstream non-linearities and multi-layer mediation |
| Interventional Necessity | Target behavior relies on component under specified ablation | Does not prove uniqueness or adaptation invariance |
| Interventional Sufficiency | Spliced activation alone restores target behavior | Does not prove necessity in natural unperturbed inference |
| Causal Abstraction | High-level algorithm reliably predicts low-level interventions | Scope remains bounded by evaluated data manifolds |

The guiding rule of mechanistic interpretability is: **always make variables, metrics, interventions, baselines, domains, and competing hypotheses explicit.** This discipline elevates intuitive circuit stories into rigorous, cumulative scientific knowledge.
