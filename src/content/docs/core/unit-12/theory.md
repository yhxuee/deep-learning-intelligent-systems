---
title: "12.5 General theory"
description: "Evidence as error-correcting inquiry: positive predictive value, selection, falsification, strong inference, Bayesian updating, replication, cumulative evidence, and bounded roles for AI."
sidebar: { label: "12.5 General theory", order: 5 }
---

The practical rules of §§12.2–12.4 rest on a deeper idea: science is not a machine that converts one significant result into truth. It is an **error-correcting process** in which claims remain exposed to alternative explanations and new tests.

## 12.5.1 Why a significant result need not be true

Suppose a fraction $\pi$ of tested hypotheses are true, a valid test has power $1-\beta$ when the hypothesis is true, and false-positive rate $\alpha$ when it is false. Among significant results, the positive predictive value is

$$
\operatorname{PPV}
=\Pr(H\text{ true}\mid +)
=\frac{(1-\beta)\pi}
{(1-\beta)\pi+\alpha(1-\pi)}.
$$

The numerator counts true positives; the denominator counts all positives.

For $\pi=0.1$, $\alpha=0.05$, and power $0.8$,

$$
\operatorname{PPV}=\frac{0.08}{0.08+0.045}=0.64.
$$

At power $0.2$ it falls to

$$
\frac{0.02}{0.02+0.045}\approx0.31.
$$

This stylized calculation explains how low power, low base rates of correct ideas, and flexible selection can make a literature unreliable ([Ioannidis 2005](/references/#ioannidis-2005)). It is not an estimate that “36% of ML papers are false.” Its assumptions are strong:

- every true hypothesis is assigned the same power;
- every false hypothesis has valid type-I error $\alpha$;
- hypotheses and studies are independent;
- there is no selective reporting, p-hacking, or measurement error;
- “true” is binary and well defined.

In real research, effect sizes vary and claims are rarely simply true or false. Treat the formula as a causal diagram for reliability, not a census.

## 12.5.2 Selection acts like an extra likelihood term

Let $S$ mean “the result was selected for reporting.” Readers observe not just a positive result $+$, but $(+,S)$. The relevant probability is

$$
\Pr(H\mid +,S),
$$

not $\Pr(H\mid +)$. If surprising positives are much more likely to be selected than nulls, selection changes the evidence.

This is why “we tried 20 analyses and report the significant one” cannot be interpreted as one pre-specified 5% test. Pre-registration, complete result tables, registered reports, and publication of null results reduce uncertainty about the selection mechanism.

## 12.5.3 Falsification is a direction, not a mechanical verdict

A scientific claim must expose itself to observations that would count against it ([Popper 1959](/references/#popper-1959)). But probabilistic models are rarely destroyed by one noisy contradiction. When a prediction fails, at least four explanations compete:

1. the hypothesis is wrong;
2. an auxiliary assumption is wrong;
3. the measurement is invalid;
4. an unlikely random outcome occurred.

Therefore a refuter must specify which assumptions are being tested and which controls make measurement failure unlikely. Unit 06's scaling miss did not logically prove a regime change; it made the old extrapolation inadequate and motivated a new, testable regime-change account.

## 12.5.4 Strong inference compares rivals

Strong inference asks researchers to formulate alternative hypotheses and choose experiments on which they disagree ([Platt 1964](/references/#platt-1964)). For the trained copying model:

- relative-position mechanism predicts transfer across absolute shifts;
- absolute-position template predicts failure after shifts;
- value-subspace mechanism predicts that QK inspection alone will miss the transferable content.

The best next experiment maximizes separation among these predictions. Evidence usually changes their relative plausibility rather than instantly “eliminating” all but one. Unit 11, Experiment 11.6 follows this structure.

## 12.5.5 Frequentist error control and Bayesian updating answer different questions

A confidence procedure controls long-run coverage under repeated datasets. It does not assign a probability that this fixed interval contains the parameter. A Bayesian posterior begins with prior $p(H)$ and updates by a likelihood:

$$
p(H\mid D)
=\frac{p(D\mid H)p(H)}{p(D)}.
$$

Neither framework removes design problems. A posterior based on a misspecified likelihood or a confidence interval over pseudoreplicated tokens can both be precise and wrong. Randomization, valid measurement, and transparent selection come first.

The PPV calculation is itself Bayesian in form, while its $alpha$ and power inputs come from frequentist operating characteristics. The frameworks can illuminate different parts of one workflow.

## 12.5.6 Evidence accumulates across different kinds of independence

Confidence increases when a claim survives tests whose errors are not shared:

- new seeds challenge chance training outcomes;
- new data challenge item selection and contamination;
- new code challenges implementation bugs;
- new teams challenge analyst expectations;
- new models challenge architecture-specific explanations;
- new interventions challenge merely correlational stories.

Ten replications using the same buggy helper are not equivalent to one clean-room implementation. “Replication count” matters less than **independence from the suspected failure mode**.

Evidence can be synthesized quantitatively when studies estimate compatible quantities, but heterogeneity is scientific information. If effects differ by model size or training time, a single pooled mean can hide the mechanism.

## 12.5.7 A finding is bounded and graded

A defensible finding states:

1. the system, distribution, intervention, and estimand;
2. the experimental units and uncertainty;
3. whether analysis was confirmatory or exploratory;
4. the estimate and decision without binary hype;
5. implementation and data audits;
6. replications and the dimensions in which they were independent;
7. alternatives and domains not tested.

Replication strengthens a finding; it is not a magical threshold between “result” and “knowledge.” The capstone supports a finding about the experiment's resolution and broken pairing. It does not support a directional finding about positional embeddings.

## 12.5.8 AI changes the speed and shape of the search

AI can propose hypotheses, write code, search program spaces, summarize literature, analyse outputs, and draft reports. Systems coupled to machine-checkable objectives have contributed verifiable constructions in combinatorics ([Romera-Paredes et al. 2024](/references/#romera-paredes-2024)) and matrix multiplication ([Fawzi et al. 2022](/references/#fawzi-2022)); learned systems have also transformed scientific prediction such as protein structure ([Jumper et al. 2021](/references/#jumper-2021)). These are different achievements and should not be merged into “AI autonomously does science.”

AI also multiplies researcher degrees of freedom. It can generate a hundred plausible hypotheses, metrics, exclusions, and narratives before lunch. If only the best-looking output is reported, selection bias grows with productivity.

### Delegate according to verifier strength

| Verifier | Suitable AI role | Required safeguard |
|---|---|---|
| exact checker or proof assistant | broad candidate search | checker integrity and specification audit |
| deterministic test suite | code generation and repair | adversarial tests and hidden cases |
| noisy experiment | hypothesis and protocol drafting | pre-registration, controls, replication |
| expert judgement | literature synthesis and idea generation | blinded review, source verification, disagreement records |

No evaluator is literally “unable to be fooled.” A program can encode the wrong objective; a wet-lab assay can be biased; a human rubric can reward fluency. Verification itself must be audited.

The practical rule is:

> The weaker and more gameable the evaluator, the less autonomy should be granted to search, select, and report within the same loop.

Keep proposer, executor, evaluator, and reporter roles separable. Preserve prompts, code diffs, tool outputs, exclusions, and human overrides so that AI assistance remains auditable.

## 12.5.9 Science as a loop back to generalization

Every study observes a finite sample of seeds, data, models, and interventions, then asks about unobserved cases. This is Unit 01 again:

$$
\text{finite evidence}
\longrightarrow
\text{bounded generalization}
\longrightarrow
\text{new counterexample search}.
$$

No method ends the loop. Good methodology makes the leap visible, measures some of its uncertainty, and creates artifacts that let others challenge it.

## 12.5.10 Summary

| Principle | Practical consequence |
|---|---|
| PPV depends on base rate, power, and error rate | significance alone is weak evidence |
| selection changes the observed evidence | preserve all searches and distinguish exploration |
| falsification tests a bundle of assumptions | use controls and diagnose failures |
| strong inference compares rivals | choose experiments with divergent predictions |
| replication must target shared errors | seek independence along likely failure modes |
| findings are bounded and graded | state scope, uncertainty, and unresolved alternatives |
| AI accelerates both discovery and selection bias | scale autonomy with verifier strength |

Verifiable knowledge is not certainty. It is a claim placed in an error-correcting system strong enough to discover when the claim stops working.
