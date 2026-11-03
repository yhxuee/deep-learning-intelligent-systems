---
title: "1.10 Problems and further study"
description: Twenty graded problems on log-loss, maximum likelihood, smoothing, concentration, held-out evaluation and generalization theory, with starting hints, selected answers and an annotated reading list.
sidebar: { label: "1.10 Problems", order: 10 }
---

Difficulty: ★ direct use of one definition · ★★ combines ideas from more than one section · ★★★ proof, experimental design, or open-ended reasoning. **[AI-off]** problems are checkpoints for independent practice and assessment.

Use the page in three passes. First attempt a problem without notes. If you are stuck, use the italicized **Start** sentence to identify the first step, not the final answer. Only then compare with the selected answers. For numerical work, keep at least four decimal places until the last step and write the unit—nats, bits, probability, or number of samples. For proofs, state where every assumption enters; a correct formula without its conditions is incomplete.

## Computation — turn definitions into numbers

**1.1 ★ [AI-off]** A forecaster predicts rain with probability 0.8 on each of five days; it rains on three. Compute the average log-loss and the accuracy, using the decision rule “predict rain when $q>0.5$.” Repeat for a forecaster who always reports 0.6. Which forecaster does each score prefer, and what information does accuracy discard? *Start: write one loss for a rainy day and one for a dry day.*

**1.2 ★ [AI-off]** Redo every probability and loss in Worked Example 1.2 with $\alpha=0.5$. Is the held-out loss below the uniform baseline $\ln3$? *Start: each row denominator becomes $c(a)+0.5V$; do not reuse the add-one denominators.*

**1.3 ★** Convert a held-out loss of 1.2 nats per character to bits per character and perplexity. Then work backwards: what loss in nats and bits corresponds to perplexity 2? *Start: bits $=$ nats$/\ln2$ and perplexity $=e^{L_{\text{nats}}}$.*

**1.4 ★** For the coin in Worked Example 1.1, find the true head probability $p$ for which the expected test loss of the fitted prediction $q=0.7$ equals its training loss 0.6109. Explain why your answer makes the KL excess vanish. *Start: write $-p\ln0.7-(1-p)\ln0.3=0.6109$.*

**1.5 ★★** A bigram is fitted on a text in which character `z` occurs exactly once, followed by `e`. Write all two kinds of entries in the smoothed row $q_\alpha(\cdot\mid\texttt z)$ for $V=50$: the entry for `e` and the 49 equal entries for every other character. For what $\alpha$ is $q_\alpha(\texttt e\mid\texttt z)=0.5$? Check that the row sums to one.

**1.6 ★★** Using Hoeffding's inequality, how many independent held-out examples are needed to estimate one fixed model's 0–1 error to within ±0.5 percentage points with 99% confidence? Now suppose 1 000 candidate models are all evaluated on that set and one is selected: how many examples make the same deviation guarantee hold simultaneously for all 1 000 via the union bound? *Start: $\varepsilon=0.005$, $\delta=0.01$; in the second calculation allocate failure probability $\delta/K$ to each model.*

## Derivation — reconstruct the machinery

**1.7 ★ [AI-off]** Derive the maximum-likelihood estimate of a categorical distribution with $V$ outcomes using a Lagrange multiplier (Proposition 1.1). State how zero counts are handled. *Start: maximize $\sum_b c_b\ln q_b$ subject to $\sum_bq_b=1$, then differentiate with respect to every $q_b$ and the multiplier.*

**1.8 ★★ [AI-off]** Prove $H(p,q)=H(p)+\mathrm{KL}(p\Vert q)$ and Gibbs' inequality, including the equality condition. Then give a two-outcome numerical example where $\mathrm{KL}(p\Vert q)\ne\mathrm{KL}(q\Vert p)$. *Start: add and subtract $-\sum_xp(x)\ln p(x)$; for Gibbs, use $\ln u\le u-1$ with $u=q(x)/p(x)$.*

**1.9 ★★** Show that the posterior of a categorical parameter under a symmetric Dirichlet($\alpha$) prior is Dirichlet($c+\alpha$), and that the posterior predictive probability of outcome $b$ equals the add-$\alpha$ estimate. Identify the normalizing factors that can be absorbed into proportionality. *Start: multiply $\prod_bq_b^{\alpha-1}$ by $\prod_bq_b^{c_b}$.*

**1.10 ★★** Prove that expected log-loss is a *strictly proper* score: for any $q\ne p$ on the support of $p$, $\mathbb E_p[-\ln q]>\mathbb E_p[-\ln p]$. Then show that the score $1-q(x)$ is not proper by finding a two-outcome $p$ and a report $q\ne p$ with a lower expected score. *Start: use the cross-entropy decomposition for the first part; for the second, write the expectation as $1-\sum_xp(x)q(x)$ and ask which $q$ maximizes the dot product.*

**1.11 ★★** For a bigram fitted by maximum likelihood, show that its training loss equals the empirical conditional entropy $\sum_a\frac{c(a)}nH(\hat p(\cdot\mid a))$. Explain what happens to contexts with $c(a)=0$. *Start: group the sum over observed transitions by their first character and substitute $q(b\mid a)=c(a\to b)/c(a)$.*

## Proof — make every assumption visible

**1.12 ★★ [AI-off]** Prove the bias–variance decomposition (Proposition 1.2), explicitly stating where zero-mean noise, independence from the training set, and the definition of $\bar f$ are used. *Start: add and subtract $\bar f(x)$ before expanding the square; list all three cross terms.*

**1.13 ★★** Prove the no-free-lunch theorem (Theorem 1.4) when the learner may randomize its prediction. Your conclusion should average over both target functions and the learner's internal randomness. *Start: condition on one realized randomized prediction rule; pair every target with the target obtained by flipping only the unseen test label, then average over the learner's randomness.*

**1.14 ★★★** Prove Hoeffding's lemma: if $\mathbb EX=0$ and $X\in[a,b]$, then $\mathbb Ee^{sX}\le e^{s^2(b-a)^2/8}$. Do not quote the lemma itself at any step. *Start: bound the convex function $e^{sx}$ by the chord joining its endpoint values, take expectations, then bound the second derivative of the resulting log expression.*

**1.15 ★★★** Show that the class of thresholds on the real line, $h_t(x)=\mathbb1[x\ge t]$, has VC dimension 1, while intervals $\mathbb1[a\le x\le b]$ have VC dimension 2. Each result needs a lower bound and an upper bound: exhibit a set that can be shattered, then show why no larger set can be. *Start: sort the points and look for a labelling pattern the class cannot realize.*

**1.16 ★★★** Let $\hat L_1,\dots,\hat L_K$ be validation losses of $K$ models whose true losses all equal $L$, with independent Gaussian measurement errors of standard deviation $\sigma$. Show that $\mathbb E[\min_k\hat L_k]<L$ and that the optimism has leading scale $\sigma\sqrt{2\ln K}$ for large $K$. State why independence is an idealization for ordinary hyperparameter searches. *Start: standardize the errors and convert a minimum into the negative of a maximum; bound the expected maximum with the exponential moment method from Appendix E.8.*

## Implementation and design — make the claims executable

**1.17 ★ [AI-off]** Without looking back at the reference implementation, implement `fit_bigram` and `nll`; you may inspect the function signatures and tests, but do not use NumPy's `add.at`. Verify on explicit assertions that every probability row sums to 1 and that $\alpha\to\infty$ gives $\ln V$. Also test one unseen transition and one context absent from training.

**1.18 ★★** Implement a trigram model with add-$\alpha$ smoothing on the spine corpus. At both 2 000 and 385 274 training characters, split the held-out stories into validation and test subsets, select $\alpha$ separately for the bigram and trigram on validation, and report training and untouched test loss. Also report the fraction of possible and test-needed contexts unseen in training. Explain the result with §1.7.7.

**1.19 ★★** Implement *interpolated* smoothing,
$$
q(b\mid a)=\lambda\hat p(b\mid a)+(1-\lambda)\hat p(b).
$$
Choose $\lambda$ on validation stories. Tune add-$\alpha$ on the same validation stories, then compare both once on a separate test set. Check normalization at $\lambda=0$, $1$, and one interior value, and explain what evidence each endpoint ignores.

**1.20 ★★★** Design an evaluation protocol for a language model that will be used on stories written next year. Specify the target population, sampling unit, time-based split, metric and uncertainty estimate; state the distributional assumption connecting this year's data to next year's stories; and name an observable result that would contradict it. Explain which decisions use validation data and which single measurement is reserved for final testing. This previews Unit 10.

## Selected answers — use only after attempting the problem

- **1.1** Forecaster 0.8: log-loss $(3\times0.2231 + 2\times1.6094)/5 = 0.7777$, accuracy 3/5. Forecaster 0.6: $(3\times0.5108 + 2\times0.9163)/5 = 0.6730$, accuracy 3/5. Accuracy cannot tell them apart; log-loss prefers the forecaster whose probability (0.6) equals the observed frequency.
- **1.3** 1.2 nats = 1.731 bits; perplexity $e^{1.2} = 3.32$. Perplexity 2 ⇔ loss $\ln 2 = 0.693$ nats = 1 bit.
- **1.5** $q_\alpha(\texttt e\mid\texttt z) = (1+\alpha)/(1+50\alpha)$; this equals 0.5 when $2 + 2\alpha = 1 + 50\alpha$, i.e. $\alpha = 1/48$.
- **1.6** $n \ge \ln(2/\delta)/(2\varepsilon^2) = \ln 200/(2\times0.005^2) = 105\,967$; with $K = 1\,000$, $n\ge(\ln 1000 + \ln 200)/(2\times 0.005^2) = 244\,122$.
- **1.11** $n\hat L = -\sum_{a,b}c(a\to b)\ln\frac{c(a\to b)}{c(a)} = \sum_a c(a)\big[-\sum_b \hat p(b\mid a)\ln\hat p(b\mid a)\big]$.
- **1.16** The minimum of $K$ i.i.d. $N(0,\sigma^2)$ variables has expectation $\approx-\sigma\sqrt{2\ln K}$ for large $K$; with $K = 1000$ that is about $3.7\sigma$, an upper bound that the exact value ($\approx 3.24\sigma$) approaches slowly.

## Further reading — choose a route

You do not need to read every source at once. Choose the route that matches the part of the unit that was least clear, and return to the others later.

**Foundations.** [Shalev-Shwartz & Ben-David 2014](/references/#shalev-shwartz-2014), Chs. 2–6 — the cleanest modern account of ERM, uniform convergence, VC dimension and no free lunch. [Cover & Thomas 2006](/references/#cover-2006), Chs. 2 and 5 — entropy, KL divergence and coding.

**Language modelling.** [Shannon 1951](/references/#shannon-1951) — still the best introduction to predicting text. [Chen & Goodman 1999](/references/#chen-1999) — what decades of smoothing research learned. [Bengio et al. 2003](/references/#bengio-2003) — the bridge to Unit 02.

**Scoring rules.** [Gneiting & Raftery 2007](/references/#gneiting-2007) — why log-loss and the Brier score are proper, and what that buys.

**The modern puzzle.** [Zhang et al. 2017](/references/#zhang-2017), [Belkin et al. 2019](/references/#belkin-2019), [Nakkiran et al. 2020](/references/#nakkiran-2020), [Bartlett et al. 2020](/references/#bartlett-2020) — read in this order.

**Evaluation practice.** [Recht et al. 2019](/references/#recht-2019) and [Dwork et al. 2015](/references/#dwork-2015) — what happens to test sets that are reused.
