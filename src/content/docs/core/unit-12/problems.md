---
title: "12.10 Problems and further study"
description: Twenty-six graded problems on estimands, experimental units, intervals, power, selection, replication, factorial design, registration, and verifier-first AI-assisted research, with selected answers.
sidebar: { label: "12.10 Problems", order: 10 }
---

Difficulty: ★ routine · ★★ combines ideas · ★★★ proof or open-ended. **[AI-off]** marks independent practice. For every numerical answer, write one sentence interpreting it in the units and scope of the problem. A correct number with the wrong estimand is not a correct scientific answer.

## Foundations and diagnosis

**12.1 ★ [AI-off] — Result, finding, or knowledge?** Classify each statement and say what additional evidence would raise its grade.

1. “Seed 3 achieved validation NLL 1.42.”
2. “Across five independent runs, removing positions changed NLL by $-0.010$ with a wide interval.”
3. “Positional embeddings are unnecessary for language models.”
4. “The test suite passes.”

**12.2 ★ [AI-off] — Name the estimand.** A team trains five models per condition on one dataset and evaluates every model on 10,000 examples. Write three different estimands concerning (a) new evaluation items for these fixed checkpoints, (b) a new training run under this pipeline, and (c) a new dataset. For each, name the experimental or sampling units needed to estimate uncertainty.

**12.3 ★★ — Pseudoreplication audit.** An article reports $n=50{,}000$ because five checkpoints were tested on 10,000 tokens each. Draw the sampling hierarchy. Under what target could token-level resampling be valid, and under what target is the effective number of training units five?

**12.4 ★★ — Missing runs.** Method A completes 20/20 runs with mean score 0.70. Method B completes 12/20 with mean score 0.78 among survivors. Give two estimands for which different summaries are appropriate. Design a report that does not hide B's failures.

**12.5 ★★ — A metric can be reproducible and wrong.** Construct a deterministic metric that rewards the wrong behaviour for one task in this book. Give a positive control, negative control, and adversarial case that would expose the construct-validity failure.

**12.6 ★★ — Same seed, broken pair.** Two model variants call one global random-number generator. Variant B initializes an extra tensor before the shared layers. Explain why equal integer seeds do not create a paired initialization. Specify a test that detects the problem without training.

## Computation and interpretation

**12.7 ★ [AI-off] — Positive predictive value.** Compute the PPV of a significant result for $\alpha=0.05$ under (a) $\pi=0.1$, power $0.8$; (b) $\pi=0.1$, power $0.2$; (c) $\pi=0.5$, power $0.8$. State two assumptions that prevent these numbers from being a census of true papers.

**12.8 ★ [AI-off] — Power and seed count.** Paired differences have $\sigma_d=0.05$. Using the normal approximation, how many pairs give 80% power at two-sided 5% level for $\delta=0.02$? For $\delta=0.05$? Why should the answers be rounded up?

**12.9 ★ — Multiplicity.** Ten independent null analyses are each tested at 0.05 and the first significant result is reported. Compute the family-wise false-positive probability. Give a Bonferroni per-test level and explain why independence is unnecessary for the Bonferroni guarantee.

**12.10 ★★ [AI-off] — Three questions, one interval.** Let $\tau=0.02$. For each interval $[0.03,0.08]$, $[-0.01,0.015]$, and $[0.01,0.05]$, answer separately:

1. Does the entire interval exceed $+\tau$?
2. Is the entire interval inside the equivalence region $[-\tau,+\tau]$?
3. Does the interval exclude zero in either direction?

Explain why “does not exceed $+\tau$” is not synonymous with “equivalent to zero.”

**12.11 ★★ — Original and replication.** An original gives $\hat\delta_1=0.08$, $s_1=0.03$; an independent replication gives $\hat\delta_2=0.01$, $s_2=0.02$. Compute approximate 95% intervals, the difference statistic in §12.3.10, and the inverse-variance fixed-effect estimate. State the assumption needed for pooling.

**12.12 ★★ — Value of pairing.** Both conditions have seed-to-seed SD 0.06. Compute the SD of their paired difference for correlation $r=0.8,0,-0.5$. Give the approximate sample-size ratio relative to $r=0$. Explain the negative-correlation case in words.

## Derivation and proof

**12.13 ★ [AI-off] — Derive PPV.** Starting from Bayes' rule, derive

$$
\operatorname{PPV}
=\frac{(1-\beta)\pi}{(1-\beta)\pi+\alpha(1-\pi)}.
$$

Show that it increases with power and $\pi$ and decreases with $\alpha$, away from degenerate endpoints.

**12.14 ★★ [AI-off] — Derive the power approximation.** Assume independent paired differences $d_i\sim\mathcal N(\delta,\sigma_d^2)$ and known $\sigma_d$. Derive

$$
n\approx
\left(\frac{z_{1-\alpha/2}+z_{1-\beta}}{\delta_1/\sigma_d}\right)^2.
$$

Identify what changes for a one-sided test, unknown variance, or a minimum effect defined on another scale.

**12.15 ★★ — Bonferroni without independence.** Prove with the union bound that

$$
\Pr\!\left(\bigcup_{j=1}^{m}\{p_j\le\alpha/m\}\right)\le\alpha
$$

under valid marginal $p$-values, regardless of dependence. Under independence, compare this bound with the exact probability.

**12.16 ★★ — Directional interval error.** A procedure reports an equal-tailed 95% confidence interval by inverting tests with 2.5% in each tail. Show that, when $\delta\le\tau$, the probability that its lower endpoint exceeds $\tau$ is at most 2.5% under the procedure's assumptions. Explain why “95% interval” alone, without the equal-tail/directional property, guarantees only a 5% noncoverage bound.

**12.17 ★★ — Winner's curse.** Let $\hat\delta\sim\mathcal N(\delta,s^2)$ and report only when $\hat\delta>1.96s$. Using the mean of a lower-truncated normal, show

$$
\mathbb E[\hat\delta\mid\hat\delta>1.96s]
=\delta+s\frac{\phi(a)}{1-\Phi(a)},
\qquad a=1.96-\delta/s.
$$

Evaluate the conditional mean when $\delta=s$. Why is an unbiased estimator before selection biased after selection?

**12.18 ★★★ — False-discovery control.** Prove the Benjamini–Hochberg FDR guarantee for independent valid $p$-values ([Benjamini & Hochberg 1995](/references/#benjamini-1995)). Then identify which step fails under arbitrary dependence and find a correction that restores a general guarantee.

## Simulation and implementation

**12.19 ★★ — Optional stopping.** Under a true zero effect, simulate paired normal observations. Starting at $n=5$, recompute an ordinary fixed-$n$ 95% interval after every new pair and stop when it excludes zero or at $n=50$. Estimate the false-positive probability over at least 20,000 simulations. Compare with a single look at $n=50$ and with a confidence sequence or registered group-sequential rule.

**12.20 ★★ — Tiny-pilot instability.** Draw three observations from a distribution with $\sigma_d=0.05$, estimate $s_d$, and plug it into the 80%-power formula for $\delta=0.02$. Repeat 10,000 times. Plot the implied $n$ and report median and 5th/95th percentiles. Repeat with pilot sizes 10 and 30. Explain the skew.

**12.21 ★★ — Named random streams.** Implement an experiment-local initializer in which each tensor's generator is derived from `(master_seed, parameter_name)`. Avoid Python's process-randomized `hash()` unless it is converted through a stable digest. Test invariance of shared tensors under optional components and test that different names do not accidentally share streams.

**12.22 ★★ — Semantic registration and trusted time.** Implement canonical serialization and SHA-256 verification. Write tests for whitespace, key order, value changes, added fields, and simultaneous record-plus-digest replacement. Add an external anchor and write a threat model: accidental edit, repository collaborator, and compromised machine.

## Study design and scientific judgement

**12.23 ★★ — Factorial attribution.** Unit 07 changed four factors at once. Design a $2^4$ factorial study. Specify the 16 cells, a primary main effect, at least one interaction, randomization or blocking, $n$ per cell under stated variance and target effect, total runs, and how multiplicity will be handled. Which assumptions allow fractional designs to save compute?

**12.24 ★★★ — Perfectly replicated and wrong.** Construct three cases in which an original and exact replication agree yet the claim is wrong because of (a) shared code, (b) shared data, and (c) shared construct measurement. For each, propose a replication independent along the relevant dimension.

**12.25 ★★★ — Inconclusive registration versus unregistered significance.** Is a perfectly registered but inconclusive study more informative than an unregistered significant one? Argue both directions using selection, power, effect size, interval width, and the PPV model. End by stating what additional evidence would dominate both.

**12.26 ★★★ — Verifier-first AI research policy.** Write a policy for an AI assistant that proposes hypotheses, edits code, runs experiments, analyses outputs, and drafts reports. For every stage, specify:

- permitted autonomy and maximum budget;
- artifacts and failed attempts that must be logged;
- verifier, its independence, and known blind spots;
- escalation condition;
- whether output is exploratory, confirmatory, or not creditable.

Test the policy on one experiment from Units 06–11. Include one seeded fault, demonstrate that the intended check catches it, and identify one consequential fault the check still would not catch.

## Selected answers and checks

- **12.1** (1) a result; repeat-run uncertainty and scope are absent. (2) a bounded finding if the design and interval are valid. (3) unsupported generalization far beyond the tested systems. (4) software evidence, not by itself an empirical finding.
- **12.2** Fixed-checkpoint item performance uses evaluation items as sampling units; training-procedure performance uses independent training runs; cross-dataset generalization needs datasets or a justified task population. More rows inside one run do not create more runs.
- **12.4** Report completion $20/20$ versus $12/20$ and survivor performance. A deployment estimand may combine failure and score; a conditional estimand reports score given successful completion. Neither licenses hiding eight failures.
- **12.6** B's extra draw shifts every later draw. Compare same-named shared tensors before training; they should be bitwise equal in a correctly paired initializer.
- **12.7** (a) $0.08/(0.08+0.045)=0.64$; (b) $0.02/(0.02+0.045)\approx0.31$; (c) $0.4/(0.4+0.025)\approx0.94$.
- **12.8** $\lceil(2.8\times0.05/0.02)^2\rceil=49$; $\lceil(2.8\times0.05/0.05)^2\rceil=8$. Fractions of an independent pair are unavailable, and rounding down loses planned power.
- **12.9** $1-0.95^{10}\approx0.401$ under independence. Bonferroni uses $0.005$; its union-bound guarantee does not assume independence.
- **12.10** $[0.03,0.08]$: exceeds $\tau$, not equivalent, excludes zero. $[-0.01,0.015]$: does not exceed $\tau$, is inside the stated equivalence region, does not exclude zero. $[0.01,0.05]$: inconclusive for superiority and equivalence, but excludes zero positively. The decision questions differ.
- **12.11** Approximate intervals are $[0.021,0.139]$ and $[-0.029,0.049]$. The difference statistic is $(0.01-0.08)/\sqrt{0.03^2+0.02^2}\approx-1.94$, near but not beyond the conventional two-sided 5% cutoff. Fixed-effect pooling gives about $0.0315$ with SE $0.0166$ and interval about $[-0.001,0.064]$, assuming a common effect and independent estimates.
- **12.12** $0.06\sqrt{2(1-r)}$ gives about $0.038$, $0.085$, and $0.104$. Relative sample sizes are $0.2$, $1$, and $1.5$. Negative correlation makes within-pair differences more variable than unrelated outcomes.
- **12.16** The event $L>\tau\ge\delta$ implies $L>\delta$, a directional lower-tail coverage failure. Equal-tail inversion allocates at most 2.5% to that event. Generic 95% coverage only bounds total noncoverage by 5%.
- **12.17** For $\delta=s$, $a=0.96$ and $\phi(a)/(1-\Phi(a))\approx1.49$, so the reported mean is about $2.49s$, roughly 2.49 times the true effect.

## Further reading

**Why findings fail.** [Ioannidis 2005](/references/#ioannidis-2005); [Gelman & Loken 2014](/references/#gelman-2014); [Open Science Collaboration 2015](/references/#osc-2015).

**Design and remedies.** [Nosek et al. 2018](/references/#nosek-2018) on registration and registered reports; [Platt 1964](/references/#platt-1964) on rival hypotheses; [Popper 1959](/references/#popper-1959) on falsifiability; [Wasserstein & Lazar 2016](/references/#wasserstein-2016) on $p$-value interpretation.

**Machine-learning evidence.** [Henderson et al. 2018](/references/#henderson-2018); [Melis et al. 2018](/references/#melis-2018); [Lucic et al. 2018](/references/#lucic-2018); [Bouthillier et al. 2021](/references/#bouthillier-2021); [Agarwal et al. 2021](/references/#agarwal-2021); [Dodge et al. 2019](/references/#dodge-2019); [Pineau et al. 2021](/references/#pineau-2021); [Kapoor & Narayanan 2023](/references/#kapoor-2023).

**AI in the research loop.** [Romera-Paredes et al. 2024](/references/#romera-paredes-2024); [Si et al. 2024](/references/#si-2024); [Lu et al. 2024](/references/#lu-2024); [Yamada et al. 2025](/references/#yamada-2025); [Reizinger & Brendel 2026](/references/#reizinger-2026).
