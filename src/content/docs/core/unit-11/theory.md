---
title: "11.5 General theory"
description: A careful theory of neural-network interventions, effect measures, mediation, causal abstraction, redundancy, and the evidence required for a mechanistic claim.
sidebar: { label: "11.5 General theory", order: 5 }
---

The previous sections supplied tools: residual-stream accounting, attention decomposition, ablation, patching, probes, and attribution. The difficult question is no longer **how to obtain a number**, but **what that number licenses us to say**.

> A mechanism is not a colourful description of an activation. It is a causal model that survives controlled interventions and predicts new cases.

## 11.5.1 From a forward pass to a causal graph

Fix the model parameters and write a feed-forward network as assignments

$$
Z_j := g_j(\operatorname{pa}(Z_j)),
$$

where $\operatorname{pa}(Z_j)$ are the variables read by node $j$. Input tokens, embeddings, head outputs, MLP outputs, residual states, and logits can all be nodes. These assignments give a deterministic **structural causal model** (SCM) in the computational sense of [Pearl (2009)](/references/#pearl-2009).

For a tiny two-step calculation,

$$
H := Wx,
\qquad
Y := u^\top H,
$$

the ordinary forward pass asks “what is $Y$ when the input is $x$?” The intervention $\operatorname{do}(H=h^*)$ removes $H:=Wx$, fixes $H$ to $h^*$, and recomputes $Y=u^\top h^*$. A head mask replaces a head output by zero; activation patching replaces it by a value recorded on another run. Both create a **modified forward program**.

This is powerful because the graph and parameters are available. We need not infer whether head 0.0 caused a downstream change from passive correlations: we can alter its output and run the network again. But three qualifications matter.

1. **The variables are chosen by the analyst.** A scalar neuron, an entire head output, a direction in the residual stream, and a semantic feature are different causal descriptions.
2. **An intervention may be unnatural.** Setting a residual vector to zero can produce a state that no ordinary input would create. The output change is real for that modified program, but may not describe normal operation.
3. **A computational graph is not yet an explanation.** Listing millions of assignments is exact but useless. Interpretability seeks a smaller high-level model that preserves the relevant causal behaviour.

Access to the network largely solves *low-level experimental control*. It does not solve variable choice, semantic interpretation, or abstraction.

## 11.5.2 Choose the outcome before measuring an effect

“The output changed” is incomplete. First choose a scalar outcome $Y$ appropriate to the claim. Common choices include the logit margin

$$
Y_{\text{margin}}(x)=z_{y^*}(x)-z_{y^-}(x),
$$

and negative log-likelihood

$$
Y_{\text{NLL}}(x)=-\log p(y^*\mid x).
$$

The sign of “improvement” differs: a larger margin is better, whereas a smaller NLL is better.

Let $x_c$ be a clean input and $x_b$ a corrupted baseline. The **total effect** is

$$
\operatorname{TE}=Y(x_c)-Y(x_b).
$$

It describes changing the entire input, not where that difference travels. Let $H_c(x)$ be component $c$'s activation. A clean-to-corrupt patch gives the **patching effect**

$$
\operatorname{PE}_c=
Y\!\left(x_b;\operatorname{do}\bigl(H_c=H_c(x_c)\bigr)\right)-Y(x_b).
$$

If it recovers much of the total effect, the clean activation is sufficient—under this corruption and intervention—to restore the measured outcome. A descriptive recovery score is

$$
R_c=\frac{\operatorname{PE}_c}{\operatorname{TE}},
\qquad \operatorname{TE}\neq0.
$$

- $R_c\approx1$: near-complete recovery for this pair and metric;
- $R_c\approx0$: little recovery;
- $R_c<0$: the patch moves the outcome the wrong way;
- $R_c>1$: possible when nonlinear interactions make the hybrid outperform the clean run.

$R_c$ is not a universal “fraction of computation performed by $c$.” It depends on the clean input, corruption, patch site, outcome, and the rest of the network.

## 11.5.3 Patching is not automatically a natural indirect effect

Classical mediation asks how much of an input's effect travels through a mediator. Neural-network patching resembles

$$
x\longrightarrow H_c\longrightarrow Y,
$$

and has been used to study information flow and factual associations ([Vig et al. 2020](/references/#vig-2020); [Meng et al. 2022](/references/#meng-2022)). But resemblance does not make every patch a uniquely defined “indirect effect.”

Suppose $Y=H_1H_2$. One run has $(H_1,H_2)=(1,1)$ and another $(0,0)$. Patching only $H_1=1$ into the second run leaves $Y=0$; patching only $H_2=1$ also leaves $Y=0$; patching both produces $Y=1$. No additive division gives each component a context-free share.

Three issues recur:

- **interaction:** one activation's usefulness depends on others;
- **redundancy:** either of two components may suffice, so ablating one reveals little;
- **hybrid states:** a clean activation combined with corrupted context may be off-distribution.

Call the measured quantity a *patching effect* unless the assumptions for a formal mediation estimand have been stated. The result remains valuable; its scope is simply narrower.

## 11.5.4 Algebraic contribution is not causal effect

In a linear readout, head output $o_c$ contributes

$$
\Delta z_c=W_Uo_c
$$

to the logits. This **direct logit attribution** is an exact algebraic decomposition at a chosen point. It answers “what did this component write in the direction read by the unembedding?” It does not by itself answer “what would happen if this component did not exist?”

Removing $o_c$ can alter later layer normalization, attention patterns, MLP gates, and downstream heads. Thus, in general,

$$
W_Uo_c\neq z(x)-z\bigl(x;\operatorname{do}(o_c=0)\bigr).
$$

The hand-built induction circuit makes this visible. Head 0.0 writes a previous-token feature that the final readout ignores, so its direct logit contribution is small or zero. Head 1.0 later reads that feature. Zero-ablating head 0.0 raises loss from roughly $0.0004$ to $2.84$: a large **indirect causal role** without a decisive direct write.

| Quantity | Type of statement | Typical question |
|---|---|---|
| direct logit attribution | algebraic | what did this component write toward the output? |
| gradient attribution | local sensitivity | which small change alters the outcome fastest? |
| zero-ablation effect | causal for a zero intervention | what happens in the zeroed program? |
| patching effect | causal for a cross-run replacement | can this activation restore the outcome? |
| probe accuracy | predictive/decoding | is information recoverable from this representation? |

None is a drop-in substitute for another.

## 11.5.5 Necessity, sufficiency, and redundancy

**Necessary** and **sufficient** are relative to an intervention family and input distribution. A component $c$ is interventionally necessary for behaviour $B$ under ablation $a$ and distribution $D$ if

$$
\Pr_{x\sim D}\!\left[B\bigl(f(x;\operatorname{do}(c=a))\bigr)\right]
\ll
\Pr_{x\sim D}[B(f(x))].
$$

This does not prove that no alternative implementation could perform $B$, or that the model could not adapt after retraining.

A state $h_c^*$ is sufficient under context $r$ if

$$
B\bigl(f(r;\operatorname{do}(H_c=h_c^*))\bigr)=1.
$$

That does not show $c$ is necessary. Two parallel paths can each be sufficient; a distributed mechanism may require several components even though no single one is sufficient.

For a component set $S$, compare

$$
\Delta(S)=Y(x)-Y\bigl(x;\operatorname{do}(H_S=a_S)\bigr).
$$

If $\Delta(\{i\})$ and $\Delta(\{j\})$ are small but $\Delta(\{i,j\})$ is large, the pair has redundancy or synergy that single ablations miss. Searching only one head at a time can therefore produce the false conclusion that “nothing matters.”

## 11.5.6 Causal abstraction: what a mechanism claim asserts

Consider a high-level induction algorithm:

1. find an earlier occurrence of the current token;
2. retrieve the token that followed it;
3. increase that token's next-token logit.

Saying “head 1.0 implements induction” claims that variables and interventions in this algorithm correspond to variables and interventions in the network—not merely that an attention map looks similar.

Let $L$ be the low-level computation, $H$ the proposed high-level algorithm, $\tau$ a mapping from low- to high-level states, $i$ a high-level intervention, and $\omega(i)$ its proposed low-level implementation. A faithful abstraction approximately satisfies

$$
\boxed{
\tau\!\left(L_{\omega(i)}(x)\right)
\approx
H_i\!\left(\tau(x)\right)
}
\qquad
x\in\mathcal X.
$$

Intervene upstairs and predict, or intervene downstairs and translate: the results should agree. This is **causal abstraction** tested by **interchange interventions** ([Geiger et al. 2021](/references/#geiger-2021)).

Patching becomes strong evidence when the donor and recipient runs are chosen so that the high-level model predicts a specific transferred value. One successful patch is one point in this diagram. Serious validation varies token identities, positions, lengths, distractors, donor–recipient pairs, intervention values, and alternative component mappings.

A circuit story that works for one prompt may be a local explanation. A mechanism claim intended to cover a task must survive a domain of interventions.

## 11.5.7 From evidence to calibrated claims

### Level 1: behaviour

The model succeeds on repeated-token sequences and fails on unseen names. This constrains what it does, not how.

### Level 2: representation

A probe decodes the previous token from a residual state. The information is available to the probe; this does not show that the model uses that direction.

### Level 3: association and attribution

An attention map points to the expected source, or an attribution method assigns a large score. This identifies a candidate, not a complete causal mechanism.

### Level 4: component intervention

A controlled ablation damages behaviour, and a patch restores it. This supports a causal role under named interventions and controls.

### Level 5: abstract mechanism

A high-level algorithm predicts many interchange interventions and new behavioural cases while plausible alternatives fail. This is the strongest evidence considered here.

These levels are complementary. Behaviour identifies the phenomenon; decomposition proposes candidates; interventions test use; abstraction tests the algorithmic story.

## 11.5.8 A mechanism must make risky predictions

A useful mechanism predicts observations not used to invent it.

- If a previous-token head feeds an induction head, disrupting the previous-token representation should impair the later match.
- If copying depends on absolute positions, moving a pattern to unseen positions should hurt even with familiar tokens.
- If a representation contains a feature downstream computation ignores, a probe may succeed while interventions along that feature do nothing.
- If two heads are redundant, single ablations should be weak but their joint ablation strong.

State predictions **before** the decisive experiment, and state a refuter: what result would make the proposed mechanism less plausible? A story that can absorb every outcome explains none.

This is the bridge to Unit 12. Mechanistic interpretability is empirical science on an unusually accessible object. Its advantage is control over the program; its obligations remain explicit hypotheses, discriminating experiments, uncertainty, and replication.

## 11.5.9 Claim-writing template

Replace “component $c$ is the mechanism” with a bounded statement:

> On input distribution $D$, for outcome $Y$, intervention $I$ on component set $S$ changes the outcome by $\Delta$ relative to baseline $B$. Donor–recipient patches behave as predicted by high-level algorithm $A$ across controls $C$. This supports causal role $R$ within the tested domain, but does not yet distinguish alternatives $A_1$ and $A_2$ outside it.

| Field | Question to answer |
|---|---|
| $D$ | which prompts, positions, and token types were tested? |
| $Y$ | loss, probability, logit margin, accuracy, or another metric? |
| $I$ | zero, mean, resample, noise, patch, or another intervention? |
| $S$ | neuron, direction, head, layer, or component set? |
| $B$ | which clean and corrupted baselines? |
| $C$ | what matched controls rule out generic damage? |
| $A$ | what algorithm predicts the intervention result? |
| alternatives | which other mechanisms remain compatible? |

## 11.5.10 Summary

| Concept | What it establishes | What it does not establish by itself |
|---|---|---|
| computational SCM | interventions are precisely executable | the right semantic variables |
| total effect | clean and corrupted outcomes differ | where the difference is mediated |
| patching effect | a donor activation changes a recipient run | a unique context-free causal share |
| direct logit attribution | an exact direct write at a chosen point | the result of removing the component |
| necessity test | behaviour depends on a component under one intervention | uniqueness or irreplaceability |
| sufficiency test | an inserted state can produce or restore behaviour | necessity |
| causal abstraction | a high-level algorithm matches low-level interventions | validity beyond the tested domain |
| novel prediction | the story risks being wrong | truth without replication and alternatives |

The practical rule is: **name the variable, outcome, intervention, baseline, domain, and surviving alternatives.** That turns an appealing circuit narrative into a testable scientific claim.
