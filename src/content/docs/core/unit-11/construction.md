---
title: "11.3 Mathematical construction"
description: Build a mechanistic claim from residual-stream accounting, QK and OV matrices, cross-layer composition, a hand-constructed induction circuit, controlled ablation and patching, probes, and gradient attribution.
sidebar: { label: "11.3 Construction", order: 3 }
---

This section follows one question through increasing mathematical detail:

> At a position where the model predicts the next token, which earlier information is read, how is it transformed, and how does it affect the output?

The three verbs map to three objects:

| question | mathematical object |
|---|---|
| where does a head read? | QK scores and attention weights |
| what does it move? | OV transformation |
| does the moved information matter? | output decomposition and intervention effect |

## 11.3.1 The residual stream is a shared workspace

In a residual network, each block adds an update to a running vector. For an attention-only Transformer without normalization,

$$
r_t^{(0)}=e_{x_t}+p_t,
$$

$$
r_t^{(\ell+1)}
=r_t^{(\ell)}+sum_h o_t^{(\ell,h)},
$$

where $e_{x_t}$ is the token embedding, $p_t$ the position embedding and $o_t^{(\ell,h)}$ one head's output after its output projection.

After $L$ layers,

$$
r_t^{(L)}
=e_{x_t}+p_t+sum_{\ell=0}^{L-1}\sum_h o_t^{(\ell,h)}.
$$

With a linear unembedding matrix $U\in\mathbb R^{V\times d}$,

$$
z_t=Ur_t^{(L)}+b.
$$

Linearity gives an exact ledger:

$$
z_t
=Ue_{x_t}+Up_t+sum_{\ell,h}Uo_t^{(\ell,h)}+b.
$$

The vector $Uo_t^{(\ell,h)}$ is the head's **direct logit contribution**. Its coordinate for token $v$ says how much this head directly raises or lowers logit $z_{t,v}$ at the final readout ([Elhage et al. 2021](/references/#elhage-2021)).

### Direct is not total

Head 0.0 can write information that head 1.0 later reads. Its direct contribution to the final logit may be small even though removing it destroys the computation. Direct-logit attribution records the final write, not every indirect causal path.

### When the ledger stops being exact

If a nonlinear operation lies between the residual sum and the readout—especially layer normalization—the final logits are not simply the sum of separately unembedded writes. One can freeze the normalization scale, linearize locally or use other attribution conventions, but the result must be labelled as a convention or approximation. The hand-built model removes normalization so the accounting is exact.

## 11.3.2 One attention head contains a routing rule and a content rule

Use row-vector conventions. Stack residual vectors into $X\in\mathbb R^{T\times d}$. For one head of width $d_h$,

$$
Q=XW_Q,\qquad K=XW_K,\qquad V=XW_V,
$$

with $W_Q,W_K,W_V\in\mathbb R^{d\times d_h}$. The causal attention matrix is

$$
A=\operatorname{softmax}_{\text{row}}
\left(\frac{QK^\top}{\sqrt{d_h}}+M\right),
$$

where $M_{ij}=-\infty$ for future positions $j>i$. The residual update is

$$
O=AVW_O,
\qquad W_O\in\mathbb R^{d_h\times d}.
$$

Expanding one score and one output,

$$
\operatorname{score}_{ij}
=\frac{x_iW_QW_K^\top x_j^\top}{\sqrt{d_h}},
$$

$$
o_i=\sum_{j\le i}A_{ij}\,x_jW_VW_O.
$$

This exposes two low-rank maps:

$$
W_{QK}=W_QW_K^\top\in\mathbb R^{d\times d},
$$

$$
W_{OV}=W_VW_O\in\mathbb R^{d\times d}.
$$

- **QK circuit:** decides which query–key pairs receive high scores—**where to read**.
- **OV circuit:** transforms the selected residual content—**what to copy and where to write it**.

Both have rank at most $d_h$. This factorization limits the independent relationships one head can express.

### Read matrices in a meaningful basis

If token embedding $E\in\mathbb R^{V\times d}$ is used as a basis, then

$$
EW_{QK}E^\top\in\mathbb R^{V\times V}
$$

scores token-to-token query–key relationships, while

$$
EW_{OV}U^\top
$$

describes how an input-token direction is written toward output logits when shapes and conventions align. These tables can be easier to inspect than raw parameter matrices.

But basis projections inherit assumptions. A feature represented as a superposition of directions may not align with a single token vector. A table that looks sparse in one basis can look distributed in another.

## 11.3.3 Cross-layer composition creates a circuit

A later head reads a residual stream containing earlier head outputs. Substitute

$$
x_j^{(1)}=x_j^{(0)}+o_j^{(0,h)}
$$

into the later head's key:

$$
k_j^{(1,g)}
=x_j^{(1)}W_K^{(1,g)}
=x_j^{(0)}W_K^{(1,g)}
+o_j^{(0,h)}W_K^{(1,g)}.
$$

The second term is a path from head $(0,h)$'s output into head $(1,g)$'s keys. This is **K-composition**.

Similarly:

- **Q-composition:** an earlier write changes a later query;
- **K-composition:** it changes which positions a later head selects;
- **V-composition:** it changes the content a later head transports.

The terms describe paths, not exclusive categories. One earlier head can influence all three channels, and several paths can cooperate.

Induction uses a particularly clear K-composition:

1. layer 0 writes “the token immediately before this position”;
2. a layer-1 key reads that field;
3. the layer-1 query contains the current token;
4. their dot product is large where a position's previous token matches the current token.

That selected position is one step after an earlier occurrence of the current token—the location containing the token to copy.

## 11.3.4 Construct induction one coordinate block at a time

Let the ordinary vocabulary contain $K$ tokens and add one beginning-of-sequence token, so $V=K+1$. Each sequence is

$$
[\mathrm{BOS},s_0,s_1,\ldots,s_{K-1},s_0,s_1,\ldots,s_{K-1}],
$$

where $s$ is a permutation of the $K$ ordinary tokens.

Use a two-layer, two-head, attention-only model. Partition its residual stream into explicit coordinate blocks:

$$
r_t=
[
\underbrace{\text{current token}}_{V}
\mid
\underbrace{\text{absolute position}}_{2K+1}
\mid
\underbrace{\text{previous token}}_{V}
\mid
\underbrace{0}_{1}
].
$$

Therefore

$$
d=V+(2K+1)+V+1=2V+2K+2=4K+4.
$$

With two heads, $d_h=d/2=2K+2$, large enough to route all $2K+1$ position coordinates.

### Embeddings

Token $v$ writes one-hot vector $e_v$ into the current-token block. Position $t$ writes one-hot vector $u_t$ into the position block. Initially the previous-token block is zero.

### Head 0.0 — write the previous token

At query position $t$, choose

$$
q_t=c\,u_t.
$$

At key position $j$, choose

$$
k_j=u_{j+1}.
$$

Then

$$
q_t\cdot k_j=c\,[j=t-1].
$$

After division by $\sqrt{d_h}$ and softmax, the head attends almost entirely to $j=t-1$ when $c$ is large. Its value reads the current-token block at $j$, and its output writes that vector into the previous-token block at $t$.

After layer 0, position $t$ therefore contains a field saying which token occurred at $t-1$.

### Head 1.0 — match and copy

At position $i$, its query reads the current-token block:

$$
q_i=c\,e_{x_i}.
$$

At candidate position $j$, its key reads the previous-token block:

$$
k_j=e_{x_{j-1}}.
$$

Thus

$$
q_i\cdot k_j=c\,[x_i=x_{j-1}].
$$

On the second occurrence of $s_k$, the earlier matching $s_k$ occurs at position $k+1$. Therefore the selected $j$ is $k+2$, one position after that occurrence. Position $k+2$ contains $s_{k+1}$, the correct next token, except at the end where no within-repeat target is scored.

The value reads the selected current-token vector, and the output writes $\lambda$ times that vector back into the current-token block. With tied unembedding, the correct token logit rises by $\lambda$.

### Heads 0.1 and 1.1 — negative controls

All their weights are zero. They remain present so that “ablate a head” has a matched intervention that should not damage the constructed computation.

### Numerical scale

For $K=8$,

$$
V=9,\qquad d=36,\qquad d_h=18.
$$

With $c=60$, the intended attention score before softmax is

$$
\frac{c}{\sqrt{d_h}}
=\frac{60}{\sqrt{18}}
\approx14.14.
$$

If $n$ keys are visible and all non-target scores are zero, the target attention weight is

$$
\frac{e^{14.14}}{e^{14.14}+n-1}.
$$

At $n=16$ this is about $0.999989$. With $\lambda=10$, the correct token receives probability about 0.9996 and loss about 0.0004 nats on the scored repeat positions.

The construction is implemented by `stage11_induction.build_induction_circuit`. Its value is not realism; it gives every interpretability calculation an exact answer.

## 11.3.5 Define interventions as modified forward programs

Let the ordinary model be a deterministic structural computation with internal activation $h_c(x)$ and output $f(x)$. Define

$$
f_{c\leftarrow a}(x)
$$

as a **modified forward program** that computes the model normally up to component $c$, replaces its activation by $a$, and continues downstream.

This resembles a structural-causal-model intervention $do(h_c=a)$ ([Pearl 2009](/references/#pearl-2009)), but the causal interpretation is conditional on treating the chosen activations as variables in an adequate causal model. Neural activations can be entangled, and an arbitrary replacement may be off-manifold.

Common choices are:

| intervention | replacement $a$ | intended comparison | main risk |
|---|---|---|---|
| zero ablation | $0$ | remove the component's write | zero may be unnatural |
| mean ablation | $\mathbb E[h_c(X)]$ | replace by typical average | mean may blur distinct modes |
| resample ablation | $h_c(X')$ | preserve realistic marginal values | imports unrelated information |
| noise ablation | matched random noise | destroy information while matching scale | noise distribution is arbitrary |

For a loss $L$, the paired ablation effect on input $x_i$ is

$$
\Delta_i^{\mathrm{abl}}
=L(f_{c\leftarrow a}(x_i),y_i)-L(f(x_i),y_i).
$$

A positive mean says this replacement harms performance on the tested distribution. Necessity language should name the intervention: “necessary under zero ablation” is more defensible than “necessary in all implementations.”

## 11.3.6 Activation patching tests an information-specific prediction

Choose a clean input $x_{\mathrm c}$ where the behaviour succeeds and a corrupted input $x_{\mathrm x}$ that changes one controlled feature and breaks the behaviour.

Run both and cache activation $h_c$:

$$
h_c^{\mathrm c}=h_c(x_{\mathrm c}),
\qquad
h_c^{\mathrm x}=h_c(x_{\mathrm x}).
$$

Then evaluate

$$
f_{c\leftarrow h_c^{\mathrm c}}(x_{\mathrm x}).
$$

If the clean behaviour is restored, changing this activation is sufficient to transmit some causally relevant difference between the runs under the patching setup ([Zhang & Nanda 2024](/references/#zhang-nanda-2024); [Heimersheim & Nanda 2024](/references/#heimersheim-2024)). It does not prove that the feature is located only in $c$ or that every coordinate of $h_c$ is relevant.

For a loss where smaller is better, define a descriptive recovery fraction

$$
R=\frac{L_{\mathrm{corrupt}}-L_{\mathrm{patched}}}
{L_{\mathrm{corrupt}}-L_{\mathrm{clean}}}.
$$

- $R=0$: no recovery;
- $R=1$: clean loss restored;
- $R<0$: patching makes the result worse;
- $R>1$: it improves beyond the clean reference.

The fraction is not constrained to $[0,1]$ and can be unstable when the denominator is small.

### Implementation invariants

Two exact checks catch many hook and indexing bugs:

$$
f_{c\leftarrow h_c(x)}(x)=f(x)
$$

self-patching is a no-op, and

$$
f_{c\leftarrow0}(x)
$$

must match the implementation's zero-ablation path. Stage 11 tests both.

### Scientific controls

Also patch wrong heads, wrong positions and unrelated clean sources. If every patch restores behaviour, the result is not specific enough to support the proposed information path.

## 11.3.7 Probes estimate decodability, with a capacity correction

Given frozen activations $h_i$ and labels $y_i$, a linear probe learns

$$
P_\phi(y\mid h)=\operatorname{softmax}(Wh+b).
$$

Held-out probe accuracy estimates how well this decoder class recovers $y$ on the sampled population. The split must prevent leakage: if many positions come from the same sequence type, split by sequence or type rather than random positions.

A control task assigns randomized labels in a way that preserves nuisance structure. Train the same probe and define

$$
\operatorname{selectivity}
=\operatorname{score}_{\mathrm{real}}
-\operatorname{score}_{\mathrm{control}}.
$$

High real accuracy with low selectivity suggests that probe capacity or input-type memorization explains much of the result ([Hewitt & Liang 2019](/references/#hewitt-2019)).

Selectivity is not a universal correction. Its interpretation depends on how the control labels are constructed and which structure they preserve. A probe should also be compared with baselines from earlier layers or raw inputs.

Finally, decodability is still observational. To test use, intervene on the decoded feature while controlling other information and measure the model's task behaviour.

## 11.3.8 Gradient attribution is sensitivity along a chosen geometry

For scalar score $s(z)$, gradient-times-input assigns

$$
a_j=z_j\frac{\partial s}{\partial z_j}.
$$

Integrated gradients chooses a baseline $z_0$ and straight-line path

$$
\gamma(\alpha)=z_0+\alpha(z-z_0),
$$

then assigns

$$
\operatorname{IG}_j(z)
=(z_j-z_{0,j})
\int_0^1
\frac{\partial s(\gamma(\alpha))}{\partial z_j}
\,d\alpha.
$$

Under regularity conditions, the attributions satisfy completeness:

$$
\sum_j\operatorname{IG}_j(z)=s(z)-s(z_0)
$$

([Sundararajan et al. 2017](/references/#sundararajan-2017)).

Completeness is accounting relative to a baseline; it is not proof that each coordinate is a causal module. The baseline and path define which counterfactual journey is being summarized. A straight line in activation space may pass through states the model never naturally visits.

The hand-built circuit also illustrates saturation. Once an attention softmax is almost one-hot, its local gradient can be tiny even though the selected path is essential. Conversely, a large gradient marks local sensitivity and may not predict the effect of a large ablation.

Sanity checks should randomize labels or model parameters and verify that attribution changes appropriately. Some saliency methods failed this test by producing similar-looking maps for trained and randomized networks ([Adebayo et al. 2018](/references/#adebayo-2018)).

## 11.3.9 From calculation to claim

The mathematics produces several distinct outputs:

| calculation | direct conclusion | stronger claim that still needs evidence |
|---|---|---|
| high attention to position $j$ | the head routes weight toward $v_j$ | the routed content causes the prediction |
| large direct logit contribution | the component writes toward a token at readout | it is necessary for the full computation |
| high probe selectivity | a tested decoder recovers the property | the model uses the property |
| large zero-ablation effect | zero replacement damages behaviour | the component is uniquely necessary naturally |
| specific patch recovery | patched activation transmits a causal difference | the feature is exclusively localized there |
| large gradient attribution | output is locally sensitive in chosen coordinates | a large realistic intervention has the same effect |

Interpretability becomes reliable when the rightmost column is treated as the next experiment, not as a synonym for the calculation on the left.
