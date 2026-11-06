---
title: "11.7 Limitations and counterexamples"
description: A diagnostic guide to distributed mechanisms, basis ambiguity, misleading probes and attributions, invalid interventions, feature-label errors, incomplete circuits, and researcher degrees of freedom.
sidebar: { label: "11.7 Limitations", order: 7 }
---

Interpretability methods fail in recognizable ways. Learning those failures is more useful than memorizing a slogan such as “correlation is not causation.” For each limitation below, ask three questions:

1. What tempting conclusion would a weak analysis draw?
2. What counterexample breaks that conclusion?
3. Which additional control would discriminate the alternatives?

Labels indicate the status of statements: **[D]** derivation, **[E]** empirical evidence, **[O]** engineering observation, and **[H]** open hypothesis.

## 11.7.1 The localization assumption can fail in two directions

The question “which head performs copying?” assumes that the behaviour is localized. That assumption can fail through **distribution** or **redundancy**.

### Distributed mechanism

Suppose four components contribute additively,

$$
Y=h_1+h_2+h_3+h_4.
$$

If each contributes $1/4$, no component looks decisive, although all four together implement the behaviour. The trained toy model in Worked Example 11.5 is a concrete warning: zeroing any one of its four heads damages copying, yet none displays the canonical hand-built induction pattern **[E]**. The bounded conclusion is “all four zero ablations hurt under this metric,” not “each head is uniquely necessary.”

### Redundant mechanism

Now suppose

$$
Y=\max(h_1,h_2),
$$

with $h_1=h_2=1$. Ablating either component leaves $Y=1$; ablating both makes $Y=0$. Single ablations falsely suggest that neither matters **[D]**.

**Diagnostic.** Test component sets, not only individuals. Compare $\Delta(S)$ over carefully chosen sets $S$, and examine whether a sparse set explains behaviour across prompts. Localization is a hypothesis, not a premise.

## 11.7.2 Coordinates are not invariant explanations

For any orthogonal rotation $R$, a hidden state and downstream matrix can be rewritten as

$$
h'=Rh,
\qquad
W'=WR^\top,
$$

without changing $W'h'=Wh$. A neuron-level description can therefore change while the network function remains identical **[D]**.

This creates two risks:

- a semantic feature spread across many coordinates looks absent when inspecting neurons one at a time;
- a single neuron appears interpretable in one basis even though the invariant object is a subspace or computation.

**Diagnostic.** Ask whether the claim survives a change of basis. Prefer statements about subspaces, input–output maps, or interventionally validated features over coordinate labels. When a single neuron is named, report how selective and complete it is: how often does it fire without the concept, and how often does the concept appear without it?

## 11.7.3 A probe can manufacture an easy readout

A high-capacity probe may decode a label from information that the model itself never uses. If a residual vector contains a faint linear trace of sentence length, a trained classifier may recover length accurately even when all downstream weights are orthogonal to that direction.

Formally, probe success establishes that there exists a decoder $q$ such that

$$
q(h(x))\approx y.
$$

It does not establish that the model's downstream computation $g$ depends on the same feature:

$$
g(h+\alpha v)\neq g(h)

$$

for interventions along the probe direction $v$.

A probe may also memorize the probing dataset, exploit class imbalance, or benefit from greater capacity than the representation being compared.

**Diagnostic.** Use held-out data, simple probes, selectivity controls, shuffled-label baselines, and matched probe capacity. Most importantly, intervene along the proposed feature while controlling vector norm and unrelated directions. Decodability plus behavioural effect is stronger than either alone.

## 11.7.4 Attention and attribution can look meaningful when they are not

An attention map is a data-dependent routing weight, not a complete explanation of the output. A large weight can carry a near-zero value vector; a small weight can matter if its value is large; later components can cancel either contribution.

Saliency methods face a related problem. Some produce visually plausible maps even after model parameters are randomized ([Adebayo et al. 2018](/references/#adebayo-2018)) **[E]**. Smooth pictures can reflect the method's prior rather than the learned computation.

**Counterexample.** Let two source positions receive weights $0.99$ and $0.01$, but values $v_1=0$ and $v_2=100u$. Their contributions are $0$ and $u$: the low-attention position supplies the entire output **[D]**.

**Diagnostic.** Decompose QK and OV, inspect actual value-weighted contributions, randomize weights or labels when appropriate, compare multiple attribution methods, and validate candidates with interventions. A method should fail on negative controls.

## 11.7.5 Ablation measures the replacement shock as well as the function

Zero ablation asks a precise question:

> What does the modified program do when this activation is replaced by zero?

It does not automatically ask:

> What role did this component play during normal computation?

If trained activations have mean $20$ and standard deviation $1$, inserting $0$ is a twenty-standard-deviation event. Downstream failure may reflect an out-of-distribution shock. Mean ablation, resample ablation, noise injection, and learned counterfactual replacements ask different questions; they can produce different rankings in practice ([Zhang & Nanda 2024](/references/#zhang-nanda-2024)) **[E]**.

| Replacement | Advantage | Main risk |
|---|---|---|
| zero | simple; exact absence in some constructed circuits | off-manifold shock |
| global mean | keeps scale closer to typical | erases condition-dependent structure |
| conditional mean | preserves selected context | depends on conditioning model |
| resampled activation | uses a real activation | donor changes multiple latent features |
| Gaussian noise | measures sensitivity | may still be unnatural |
| clean/corrupt patch | tests information transfer | creates a hybrid state |

**Diagnostic.** Repeat the claim with several defensible replacements, measure activation norms and downstream distribution shift, include same-run no-op patches, and use position- or class-matched donors. If the conclusion flips, report the dependence rather than selecting the preferred result.

## 11.7.6 A successful patch can use a dormant pathway

Suppose a network normally solves a task through path $A$, while path $B$ has downstream weights capable of producing the answer but is never naturally activated. A patch into $B$ may restore behaviour. The intervention proves sufficiency of the patched state in the modified program, not normal use of path $B$.

Low-dimensional subspace interventions can create this kind of convincing but spurious localization ([Makelov et al. 2024](/references/#makelov-2024)) **[E]**.

**Diagnostic.** Pair sufficiency with necessity-style evidence and normal-activation evidence:

- Does blocking the path harm clean behaviour?
- Does the proposed feature naturally vary with the task variable?
- Does patching a matched irrelevant subspace fail?
- Does the mechanism predict donor–recipient transfers not used to choose the subspace?

One successful steering direction is not yet a normal-computation mechanism.

## 11.7.7 Nonlinearity prevents simple effect bookkeeping

In a nonlinear system, individual effects need not add:

$$
\Delta(\{1,2\})
\neq
\Delta(\{1\})+\Delta(\{2\}).
$$

Layer normalization, softmax, gating, and residual interactions all create context dependence. Consequently, statements such as “head 1 explains 40% and head 2 explains 70%” need not be contradictory; both may be sufficient under different patches. Nor do two accuracy drops add as causal shares.

**Diagnostic.** Report joint as well as marginal interventions, preserve the raw outcome scale, and avoid presenting normalized recovery ratios as additive percentages unless an additive model has been justified. Interaction plots or a factorial intervention design make the dependence visible.

## 11.7.8 Feature labels can outrun the evidence

Researchers often inspect top-activating examples and name a feature: “legal language,” “deception,” or “refusal.” This is useful hypothesis generation, but the label can be too broad, too narrow, or based on a correlated surface cue.

For example, a proposed “Python code” feature might really respond to indentation, ASCII punctuation, or the token `def`. The difference matters when the feature is used to explain behaviour on unseen code styles.

**Diagnostic.** Build contrast sets:

- concept present, suspected cue absent;
- cue present, concept absent;
- paraphrases and translations;
- hard negatives from neighbouring concepts;
- examples chosen before inspecting activation.

Then test both precision and recall. A few vivid top examples establish neither.

## 11.7.9 Polysemanticity and superposition complicate component stories

Individual neurons and heads often respond to multiple apparently unrelated features—**polysemanticity**—consistent with feature superposition (Unit 02, §2.5.3; [Elhage et al. 2022](/references/#elhage-2022); [Bricken et al. 2023](/references/#bricken-2023)) **[E]**.

Two distinct problems follow:

1. ablating a polysemantic unit damages several functions, so the behavioural change cannot be assigned to one label;
2. intervening on a proposed feature direction may unintentionally move correlated features.

Sparse dictionaries and autoencoders (§11.8) try to recover cleaner variables, but sparsity does not guarantee semantic truth or causal use **[H]**.

**Diagnostic.** Measure feature co-activation, inspect diverse positive and negative examples, test downstream interventions, and report reconstruction error. An interpretable dictionary that omits important activation variance can miss the very mechanism being studied.

## 11.7.10 Local completeness is not global completeness

A circuit can explain a narrow prompt family extremely well and still fail elsewhere. Let $D_1$ be the discovery distribution and $D_2$ a shifted distribution. High faithfulness on $D_1$ does not imply

$$
\mathbb E_{x\sim D_2}
\bigl[\lvert Y_{\text{circuit}}(x)-Y_{\text{model}}(x)\rvert\bigr]
\text{ is small}.
$$

Large-model circuits characterized in detail usually cover bounded behaviours and input distributions; IOI is an influential example of a carefully scoped study ([Wang et al. 2023](/references/#wang-2023)) **[E]**.

Completeness has several axes:

- **component completeness:** were important heads, MLPs, embeddings, and normalization effects included?
- **input completeness:** which prompts, languages, positions, and lengths were covered?
- **outcome completeness:** does the circuit explain only one logit difference or the full output distribution?
- **checkpoint completeness:** does it survive another seed, training step, or model version?

**Diagnostic.** State the domain in the claim, maintain held-out intervention tests, and search deliberately for inputs on which the abstraction fails.

## 11.7.11 Rigour and scale pull in opposite directions

The hand-built circuit is nearly exhaustible. A two-layer trained toy already resists the obvious story. A production model may have thousands of heads, millions of neurons, long contexts, and an enormous input space. If $m$ components are considered, even all pairwise ablations require

$$
\binom{m}{2}=\frac{m(m-1)}{2}
$$

experiments. For $m=10{,}000$, that is $49{,}995{,}000$ pairs before varying prompts or replacement rules **[D]**.

Automation is necessary, but automated scores inherit the assumptions of their metric and search procedure **[O]**. A scalable candidate finder is not a scalable proof generator.

**Diagnostic.** Separate discovery from confirmation. Use automated methods to rank candidates on one dataset, then test pre-specified claims with held-out prompts, alternative interventions, and uncertainty estimates.

## 11.7.12 Researcher degrees of freedom create false stories

An analyst can choose prompts, layers, heads, metrics, corruption rules, patch sites, colour scales, thresholds, and stopping points. If enough choices are tried, some compelling pattern will appear by chance.

Worked Example 11.5 could be framed as “no canonical induction head was found” or “copying is distributed,” depending on which evidence is emphasized. The data support the narrower statement—copying occurs, every tested zero ablation hurts, and the largest measured canonical induction score is small under that metric—while the full mechanism remains unresolved.

Interpretability findings can be sensitive to methodological choices ([Zhang & Nanda 2024](/references/#zhang-nanda-2024)) **[E]**.

**Diagnostic.** Record the analysis path, distinguish exploratory from confirmatory work, correct for multiple searches where applicable, retain null results, and replicate on new prompts or checkpoints. A figure chosen after extensive search is exploratory evidence, even if the final caption sounds definitive.

## 11.7.13 Failure-mode checklist

| Observed result | Tempting claim | Live alternative | Next control |
|---|---|---|---|
| one head has high attention | it performs the task | value is zero or cancelled | inspect OV and ablate |
| probe is accurate | model uses the feature | downstream ignores it | feature intervention |
| zero ablation hurts | component is normally necessary | off-manifold shock | mean/resample ablation |
| patch restores output | component carries the normal mechanism | dormant path is activated | necessity + natural variation |
| single ablation does nothing | component is irrelevant | redundancy | joint ablation |
| one neuron has clear examples | neuron represents the concept | correlated surface cue | contrast set |
| circuit matches one prompt family | mechanism is understood | distribution-specific shortcut | held-out shifts |
| automated score is high | explanation is correct | metric/search bias | confirmatory intervention |

## 11.7.14 The limitation that motivates Unit 12

Every result in this unit is an experiment: someone chose the input, outcome, corruption, intervention, components, and interpretation. Mechanistic access does not remove sampling error, multiple comparisons, confirmation bias, or irreproducible workflows.

The central question therefore becomes:

> How does a plausible interpretability story become a reproducible finding?

The answer requires the same machinery as the rest of empirical deep learning: explicit hypotheses, frozen evaluation sets, uncertainty, ablations, negative controls, complete reporting, and independent replication. That is the subject of [Unit 12](/core/unit-12/).

The limitation is not that mechanistic interpretation is impossible. It is that **powerful access to internals raises the number of experiments we can run faster than it raises our discipline in interpreting them**.
