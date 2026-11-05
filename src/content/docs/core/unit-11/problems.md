---
title: "11.10 Problems and further study"
description: Twenty-four graded problems on evidence types, residual-stream decomposition, QK and OV circuits, interventions, probes, attribution, causal abstraction, implementation, and study design, with hints and selected solutions.
sidebar: { label: "11.10 Problems", order: 10 }
---

Difficulty: ★ routine · ★★ combines ideas · ★★★ proof, implementation, or open-ended. **[AI-off]** marks problems intended for independent assessment. A ★★★ problem may not have one short correct answer; it does have criteria for a strong answer.

Before solving, label the requested conclusion as **behavioural**, **representational**, **attributional**, or **causal**. Many mistakes come from answering a stronger question than the evidence supports.

## A. Read the evidence correctly

**11.1 ★ [AI-off] Evidence labels.** For each observation, name the strongest claim it supports and one claim it does not support.

1. A linear probe decodes tense with 94% held-out accuracy.
2. One attention head puts 0.97 weight on the previous token.
3. Zeroing a head raises NLL by 1.3 nats.
4. A clean-to-corrupt patch restores the clean logit margin.
5. A high-level algorithm predicts 500 held-out interchange interventions.

**11.2 ★ Selectivity.** Probe A scores 88% on real labels and 85% on a control task. Probe B scores 75% and 52%. Compute selectivity and explain why raw accuracy and representational evidence rank the probes differently. What extra experiment would test use rather than decodability?

**11.3 ★ Ablation controls.** Ablating head $h$ raises loss from $0.20$ to $0.35$. Ablating each of 30 matched random heads raises it by $0.04$ on average, with standard deviation $0.03$. State a bounded conclusion. Why do these data not prove that $h$ is uniquely necessary?

**11.4 ★ Patching language.** A clean run has loss $0.0004$, a corrupted run $8.77$, and a patched run $4.40$.

1. Compute loss recovery $R=(L_b-L_p)/(L_b-L_c)$.
2. Rewrite “component $c$ carries half the computation” as a defensible sentence.
3. Give one interaction pattern for which two components each recover more than 50% when patched separately.

**11.5 ★★ Baseline reversal.** A component appears important under zero ablation but unimportant under conditional-mean ablation. Give two explanations: one in which zero ablation is misleading and one in which conditional-mean ablation is misleading. Design a discriminator.

**11.6 ★★ Multiple search.** An analyst examines 2,000 neurons and reports the one whose activation correlates $r=0.31$ with a concept, without a held-out set. Explain why the ordinary single-test intuition is invalid. Specify a discovery/confirmation split and a negative control.

## B. Calculate the hand-built circuit

**11.7 ★ [AI-off] Attention sharpness.** The hand-built circuit has $c=60$ and $d_h=18$. If one permitted key has score $c/\sqrt{d_h}$ and the other $n-1$ keys have score zero, compute the intended attention weight for $n=4$, $16$, and $1{,}000$.

**11.8 ★ [AI-off] Exact output probability.** The induction head contributes $lambda=10$ to the correct-token logit. The current token's embedding contributes $1$ to a competing logit; seven other logits are zero. Compute the correct-token probability and NLL. Repeat for $\lambda=3$.

**11.9 ★★ Routing versus writing.** Hold $lambda=10$ and reduce $c$ from 60 to 5. Then hold $c=60$ and reduce $\lambda$ from 10 to 1. Predict qualitatively how attention weight, correct-logit contribution, and NLL change in each intervention. Which measurement separates routing from writing?

**11.10 ★★ QK table.** For head 1.0, write the token-to-token table

$$
E W_QW_K^\top E^\top.
$$

Why is it zero when both arguments contain only raw token embeddings? Show how K-composition with head 0.0 makes the current-token query match the previous-token field at a source position.

**11.11 ★★ OV map.** Derive the OV action of head 1.0 on a token one-hot vector. Which token receives a positive logit write, and why does this not by itself prove that the head attends to the correct source?

**11.12 ★★ Direct and indirect paths.** Head 0.0 has no direct correct-logit write, yet its zero ablation raises loss to about 2.84 nats. Draw the computational paths that explain this result. Give another network in which direct attribution is large but zero-ablation effect is small.

## C. Derive the tools

**11.13 ★★ [AI-off] Residual ledger.** For an attention-only Transformer with tied embeddings and no normalization, derive

$$
z_t=W_Ur_t^{(0)}+
\sum_{\ell,h}W_Uo_t^{(\ell,h)}.
$$

List exactly which architectural changes make this equality incomplete or require qualification.

**11.14 ★★ Zero and self patching.** Prove from the modified forward program that:

1. zero-ablating a head equals patching its output with zeros;
2. patching every position of a deterministic run with its own recorded head output is an identity;
3. patching only some positions need not be an identity if indexing is wrong.

**11.15 ★★ Keys versus values.** A single head outputs

$$
o_i=\sum_j\operatorname{softmax}_j(q_i^\top k_j/\sqrt{d_h})v_jW_O.
$$

Derive the output after an interchange intervention on keys only, then on values only. Explain which intervention changes routing and which changes transported content.

**11.16 ★★ Integrated gradients.** For differentiable $f$ and path $z(\alpha)=z_0+\alpha(z-z_0)$, prove

$$
\sum_i (z_i-z_{0,i})
\int_0^1\frac{\partial f(z(\alpha))}{\partial z_i}\,d\alpha
=f(z)-f(z_0).
$$

Which step fails if the numerical implementation uses too few integration points?

**11.17 ★★★ Interaction counterexample.** Construct a two-component network in which each component's individual ablation has zero effect, but joint ablation destroys the output. Construct another in which each individual patch recovers the full output. Explain why normalized recovery percentages are not additive.

**11.18 ★★★ Behavioural equivalence.** Give two networks with identical outputs for every input in a stated domain but different internal mechanisms. One solution may use an invertible change of basis; another should use redundant computation. What can no behavioural test on that domain distinguish?

## D. Causal abstraction and counterexamples

**11.19 ★★★ Formalize induction.** Define a high-level induction algorithm with variables for current token, earlier match, successor, and output. Propose a low-level mapping $\tau$ and intervention map $\omega$. Write at least four donor–recipient interchange interventions whose predicted outputs differ across plausible alternative mechanisms.

**11.20 ★★★ Positional shortcut.** Construct or describe a two-layer attention-only model that copies a repeated block at one fixed distance without attending to the content-defined induction target. Predict its behaviour when the gap changes, positions shift, token identities change, and individual heads are ablated.

**11.21 ★★★ Dormant-path illusion.** Construct a network with normal path $A$ and normally inactive path $B$. Show algebraically how patching $B$ can restore an output even though ablating $B$ on clean runs changes nothing. Give the necessity, natural-activation, and wrong-subspace controls that expose the illusion.

## E. Implementation and research design

**11.22 ★★ [AI-off] Rebuild and test.** Reimplement `build_induction_circuit` without copying its source. Your implementation must pass tests for baseline NLL, previous-token and induction scores, relevant and irrelevant head ablations, same-run no-op patching, and clean-to-corrupt recovery. Add a wrong-donor test.

**11.23 ★★★ Resample ablation study.** Implement zero, global-mean, position-conditional-mean, and resample ablation for the trained toy model. Freeze prompts and seeds before comparing them. Report:

- raw and paired NLL changes;
- activation norm and Mahalanobis-distance diagnostics;
- single and pairwise interventions;
- whether the component ranking changes with replacement rule.

If rankings disagree, do not choose a winner by intuition; explain which causal question each replacement asks.

**11.24 ★★★ Capstone study.** Design an interpretability study of Stage 09 arithmetic post-training that survives §11.7. Include:

1. a frozen behavioural phenomenon and outcome;
2. at least three competing mechanisms;
3. discovery and confirmation prompt sets;
4. candidate-finding methods;
5. necessity, sufficiency, no-op, wrong-donor, and replacement controls;
6. uncertainty and multiple-search handling;
7. a causal-abstraction test;
8. a claim template stating what remains unresolved.

A strong answer makes it possible to obtain an inconclusive result. A design in which every outcome “supports the mechanism” is not a valid study.

## Hints

- **11.4:** recovery is measured from corrupted toward clean, so the numerator is $L_b-L_p$.
- **11.10:** the induction head's key reads the previous-token block, not the raw token block.
- **11.12:** follow $\text{head 0.0}\rightarrow\text{previous-token field}\rightarrow\text{head 1.0 key}\rightarrow\text{logit}$.
- **11.17:** try $Y=\max(h_1,h_2)$ for redundancy and $Y=h_1h_2$ for interaction.
- **11.18:** if $h'=Rh$ and $W'=WR^{-1}$, then $W'h'=Wh$.
- **11.21:** let $Y=A+B$ and make a gate keep $B=0$ on all ordinary inputs.

## Selected solutions

### 11.2

Probe A has $88-85=3$ percentage points of selectivity; Probe B has $75-52=23$. Probe A has higher raw accuracy, but much of it is also obtainable on the control task. Probe B supplies stronger evidence that the representation, rather than probe memorization alone, supports the real labels. Neither establishes model use; intervene along the decoded direction with matched controls.

### 11.4

$$
R=\frac{8.77-4.40}{8.77-0.0004}
\approx0.4983.
$$

Defensible wording: “For this donor, recipient, outcome, and patch site, inserting $c$'s clean activation recovers about 50% of the loss difference.” Two redundant components can each recover nearly 100% separately, so recovery is not an additive ownership share.

### 11.7

The scaled score is

$$
s=\frac{60}{\sqrt{18}}\approx14.142,
\qquad
a_n=\frac{e^s}{e^s+n-1}.
$$

This gives approximately

| $n$ | intended weight |
|---:|---:|
| 4 | $0.9999978$ |
| 16 | $0.9999892$ |
| 1,000 | $0.9992799$ |

The number of distractors matters even when each distractor score is zero.

### 11.8

For $\lambda=10$,

$$
p=\frac{e^{10}}{e^{10}+e^1+7}
\approx0.999559,
\qquad
-\log p\approx0.000441\text{ nat}.
$$

For $\lambda=3$,

$$
p=\frac{e^3}{e^3+e+7}
\approx0.674,
\qquad
-\log p\approx0.395\text{ nat}.
$$

### 11.14

Let the original structural assignment be $O_h:=g_h(R)$. Zero ablation replaces it by $O_h:=0$; patching with a zero tensor makes the same replacement, so all downstream deterministic assignments agree. A self patch uses the recorded value $g_h(R)$ and therefore recreates the original downstream state by induction over the computation graph. A position-index mismatch inserts a different tensor entry and breaks the premise.

### 11.17

For redundancy, let $h_1=h_2=1$ and $Y=\max(h_1,h_2)$. Either individual ablation leaves $Y=1$; joint ablation gives $0$. For separate full recovery, begin from $(0,0)$ with the same $Y$ and patch either $h_1=1$ or $h_2=1$. Each patch recovers 100%, so the two recovery values sum to 200%. They describe counterfactual sufficiency in different hybrid states, not additive shares.

## Further reading by question

**How are Transformer circuits decomposed?** [Elhage et al. 2021](/references/#elhage-2021) develops residual-stream, QK, OV, and composition language. [Olsson et al. 2022](/references/#olsson-2022) studies induction heads and in-context learning.

**How should patching and circuit discovery be performed?** [Wang et al. 2023](/references/#wang-2023), [Conmy et al. 2023](/references/#conmy-2023), [Zhang & Nanda 2024](/references/#zhang-nanda-2024), and [Heimersheim & Nanda 2024](/references/#heimersheim-2024) move from a manual circuit to automation and methodological controls.

**How should representations be tested?** [Hewitt & Liang 2019](/references/#hewitt-2019) introduces control tasks for probes. [Bricken et al. 2023](/references/#bricken-2023) and [Templeton et al. 2024](/references/#templeton-2024) develop sparse feature dictionaries.

**What can go wrong?** [Adebayo et al. 2018](/references/#adebayo-2018) gives saliency sanity checks; [Makelov et al. 2024](/references/#makelov-2024) constructs a subspace-patching illusion; [Geiger et al. 2021](/references/#geiger-2021) states the stronger causal-abstraction target.

**How are newer methods compared?** [Mueller et al. 2025](/references/#mueller-2025-mib) benchmarks circuit and causal-variable localization; [Hadad et al. 2026](/references/#hadad-2026) studies formal guarantees on bounded model classes and domains.
