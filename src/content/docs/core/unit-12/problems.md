---
title: "12.10 Problems and further study"
description: Twenty graded problems on positive predictive value, power and seed counts, forking paths, decision rules, paired designs, replication, false discovery control and the design of ablations, with selected answers and an annotated reading list.
sidebar: { label: "12.10 Problems", order: 10 }
---

Difficulty: ★ routine · ★★ combines ideas · ★★★ proof or open-ended. **[AI-off]** problems are for independent practice and assessment.

## Computation

**12.1 ★ [AI-off]** Compute the positive predictive value of a significant result when $\alpha = 0.05$ and (a) $\pi = 0.1$, power 0.8; (b) $\pi = 0.1$, power 0.2; (c) $\pi = 0.5$, power 0.8.

**12.2 ★ [AI-off]** Paired differences have $\sigma_d = 0.05$. How many seed pairs are needed for 80% power at the 5% level to detect $\delta = 0.02$? $\delta = 0.05$?

**12.3 ★** An analyst tries ten independent analyses at level 0.05 and reports the first that is significant. What is the false-positive rate of the procedure? What per-analysis level would keep it at 0.05?

**12.4 ★ [AI-off]** Apply the three-way rule with $\tau = 0.02$ to the intervals $[0.03, 0.08]$, $[-0.01, 0.015]$ and $[0.01, 0.05]$.

**12.5 ★** An original study reports $\hat\delta_1 = 0.08$ (s.e. 0.03) and a replication $\hat\delta_2 = 0.01$ (s.e. 0.02). Apply the three summaries of §12.3.5. Do the two disagree more than chance allows?

**12.6 ★★** Two conditions each have seed-to-seed sd 0.06. Compute the sd of a paired difference when their correlation across seeds is 0.8, 0 and −0.5, and the ratio of seeds needed relative to the uncorrelated case.

## Derivation

**12.7 ★ [AI-off]** Derive the PPV formula from Bayes' rule, and show that PPV rises with power and with $\pi$ and falls with $\alpha$.

**12.8 ★★ [AI-off]** Derive the sample-size formula of §12.3.2 from the normal approximation to the distribution of $\bar d$.

**12.9 ★★** Show that $1 - (1-\alpha)^k \le k\alpha$, and use it to justify the Bonferroni correction for any dependence between the $k$ tests.

**12.10 ★★** The **winner's curse**: among experiments with true effect $\delta$ and standard error $s$, condition on $\hat\delta > 1.96\,s$. Show that $\mathbb{E}[\hat\delta \mid \text{significant}] > \delta$, and compute the ratio for $\delta = s$.

**12.11 ★★** A registration fixes $n$; an analyst instead stops as soon as the interval excludes zero, checking after every seed. Explain why this inflates the false-positive rate, and estimate it by simulation for up to 50 seeds.

## Proof

**12.12 ★★** Prove that a three-way rule based on a 95% interval concludes "effect exceeds $\tau$" with probability at most 2.5% when the true effect is $\le \tau$.

**12.13 ★★★** Prove that the Benjamini–Hochberg procedure controls the false discovery rate at level $q$ for independent tests ([Benjamini & Hochberg 1995](/references/#benjamini-1995)).

**12.14 ★★★** Construct a situation in which an original study and an exact replication agree perfectly and both are wrong. What kind of replication would have detected the error?

**12.15 ★★★** Is a perfectly pre-registered study with an inconclusive result more or less informative than an unregistered one with a significant result? Argue both sides precisely, using the PPV of §12.5.1.

## Implementation and design

**12.16 ★ [AI-off]** Implement `decide` and `preregister`/`verify_preregistration` from memory and pass the Stage 12 tests.

**12.17 ★★** Rewrite `init_params` so each tensor has its own random stream, keyed by seed and parameter name, and test that ablating any parameter group leaves the initialization of all other groups unchanged.

**12.18 ★★** Implement the Benjamini–Hochberg procedure and apply it to the per-head ablation effects of Unit 11's toy model, with $p$-values from a paired bootstrap over sequences.

**12.19 ★★** Design a $2^4$ factorial study attributing Unit 07's 0.106-nat gain to its four changes. Specify the registration, the primary contrast, the number of seeds per cell for $\delta = 0.05$, and the total compute in training runs.

**12.20 ★★★** Write a policy for an AI research assistant working on this book's spine: which steps it may perform unsupervised, what it must log, which outputs need human verification, and how that verification is itself checked. Test the policy on one experiment from Units 06–11.

## Selected answers

- **12.1** (a) $0.08/(0.08 + 0.045) = 0.64$; (b) $0.02/(0.02 + 0.045) = 0.31$; (c) $0.4/(0.4 + 0.025) = 0.94$.
- **12.2** $(2.8 \times 0.05/0.02)^2 = 49$; $(2.8 \times 0.05/0.05)^2 = 7.84$, so 8.
- **12.3** $1 - 0.95^{10} = 0.401$; Bonferroni: $0.05/10 = 0.005$.
- **12.4** Exceeds $\tau$; below $\tau$; inconclusive.
- **12.5** Not both significant (original $z = 2.7$; the replication's interval $[-0.03, 0.05]$ contains zero); the original estimate 0.08 lies outside the replication's interval; the difference $0.07$ against $\sqrt{0.03^2 + 0.02^2} = 0.036$ gives $z \approx 1.94$ — borderline disagreement.
- **12.6** $\mathrm{sd} = 0.06\sqrt{2(1-\rho)}$: 0.038, 0.085, 0.104. Seeds needed scale with $1-\rho$: 0.2, 1, 1.5 times the uncorrelated number.

## Further reading

**Why findings fail.** [Ioannidis 2005](/references/#ioannidis-2005); [Gelman & Loken 2014](/references/#gelman-2014); [Open Science Collaboration 2015](/references/#osc-2015).

**Remedies.** [Nosek et al. 2018](/references/#nosek-2018) — the case for pre-registration; [Platt 1964](/references/#platt-1964) — strong inference; [Popper 1959](/references/#popper-1959); [Wasserstein & Lazar 2016](/references/#wasserstein-2016).

**In machine learning.** [Lipton & Steinhardt 2019](/references/#lipton-2019); [Henderson et al. 2018](/references/#henderson-2018); [Melis et al. 2018](/references/#melis-2018); [Lucic et al. 2018](/references/#lucic-2018); [Bouthillier et al. 2021](/references/#bouthillier-2021); [Agarwal et al. 2021](/references/#agarwal-2021); [Dodge et al. 2019](/references/#dodge-2019); [Pineau et al. 2021](/references/#pineau-2021); [Kapoor & Narayanan 2023](/references/#kapoor-2023).

**AI in the research loop.** [Romera-Paredes et al. 2024](/references/#romera-paredes-2024); [Si et al. 2024](/references/#si-2024); [Lu et al. 2024](/references/#lu-2024).
