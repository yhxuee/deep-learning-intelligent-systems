---
title: "1.10 Problems and further study"
description: Twenty graded problems on log-loss, maximum likelihood, smoothing, concentration, held-out evaluation and generalization theory, with selected answers and an annotated reading list.
sidebar: { label: "1.10 Problems", order: 10 }
---

Difficulty: ★ routine · ★★ combines ideas · ★★★ proof or open-ended. **[AI-off]** problems are for independent practice and assessment.

## Computation

**1.1 ★ [AI-off]** A forecaster predicts rain with probability 0.8 on each of five days; it rains on three. Compute the average log-loss and the accuracy (predicting rain whenever $q > 0.5$). Repeat for a forecaster who always says 0.6. Which forecaster does each score prefer?

**1.2 ★ [AI-off]** Redo Worked Example 1.2 with $\alpha = 0.5$. Is the held-out loss now below $\ln 3$?

**1.3 ★** Convert a held-out loss of 1.2 nats per character to bits per character and to perplexity. What loss corresponds to perplexity 2?

**1.4 ★** For the coin of Worked Example 1.1, find the true $p$ for which the expected test loss of $q = 0.7$ equals its training loss 0.6109.

**1.5 ★★** A bigram is fitted on a text in which character `z` occurs exactly once, followed by `e`. Write the smoothed row $q_\alpha(\cdot\mid\texttt z)$ for $V = 50$. For what α is $q_\alpha(\texttt e\mid\texttt z) = 0.5$?

**1.6 ★★** Using Hoeffding's inequality, how many held-out examples are needed to estimate a 0–1 error to within ±0.5 percentage points with 99% confidence? With 1 000 candidate models?

## Derivation

**1.7 ★ [AI-off]** Derive the maximum-likelihood estimate of a categorical distribution with $V$ outcomes using a Lagrange multiplier (Proposition 1.1).

**1.8 ★★ [AI-off]** Prove $H(p,q) = H(p) + \mathrm{KL}(p\,\Vert\,q)$ and Gibbs' inequality. Give an example where $\mathrm{KL}(p\,\Vert\,q)\ne\mathrm{KL}(q\,\Vert\,p)$.

**1.9 ★★** Show that the posterior of a categorical parameter under a Dirichlet($\alpha$) prior is Dirichlet($c + \alpha$), and that the posterior predictive probability of outcome $b$ equals the add-α estimate.

**1.10 ★★** Prove that the expected log-loss is a *strictly* proper score: for any $q\ne p$, $\mathbb{E}_p[-\ln q] > \mathbb{E}_p[-\ln p]$. Then show that the *absolute* score $1 - q(x)$ is not proper by finding $p$ and $q\ne p$ with a lower expected score for $q$.

**1.11 ★★** For a bigram fitted by maximum likelihood, show that the training loss equals the empirical conditional entropy $\sum_a \frac{c(a)}{n}H(\hat p(\cdot\mid a))$.

## Proof

**1.12 ★★ [AI-off]** Prove the bias–variance decomposition (Proposition 1.2), stating where each assumption is used.

**1.13 ★★** Prove the no-free-lunch theorem (Theorem 1.4) for a learner that is allowed to randomize its predictions.

**1.14 ★★★** Prove Hoeffding's lemma: if $\mathbb{E}X = 0$ and $X\in[a,b]$, then $\mathbb{E}e^{sX}\le e^{s^2(b-a)^2/8}$.

**1.15 ★★★** Show that the class of thresholds on the real line, $h_t(x) = \mathbb{1}[x\ge t]$, has VC dimension 1, and that intervals $\mathbb{1}[a\le x\le b]$ have VC dimension 2.

**1.16 ★★★** Let $\hat L_1,\dots,\hat L_K$ be validation losses of $K$ models whose true losses are all equal to $L$, each estimated with independent Gaussian noise of standard deviation $\sigma$. Show that $\mathbb{E}[\min_k\hat L_k] < L$ and that the optimism grows like $\sigma\sqrt{2\ln K}$ for large $K$.

## Implementation and design

**1.17 ★ [AI-off]** Implement `fit_bigram` and `nll` from memory, without NumPy's `add.at`. Verify that your table's rows sum to 1 and that α → ∞ gives $\ln V$.

**1.18 ★★** Implement a trigram model with add-α smoothing on the spine corpus. Compare training and held-out loss with the bigram's at 2 000 and at 385 274 training characters. Explain the result with §1.7.7.

**1.19 ★★** Implement *interpolated* smoothing, $q(b\mid a) = \lambda\,\hat p(b\mid a) + (1-\lambda)\,\hat p(b)$, choose λ on validation stories, and compare with add-α.

**1.20 ★★★** Design an evaluation protocol for a language model that will be used on stories written next year. State the distributional assumption you are making, how your split tests it, and what result would show the assumption is false (a preview of Unit 10).

## Selected answers

- **1.1** Forecaster 0.8: log-loss $(3\times0.2231 + 2\times1.6094)/5 = 0.7777$, accuracy 3/5. Forecaster 0.6: $(3\times0.5108 + 2\times0.9163)/5 = 0.6730$, accuracy 3/5. Accuracy cannot tell them apart; log-loss prefers the forecaster whose probability (0.6) equals the observed frequency.
- **1.3** 1.2 nats = 1.731 bits; perplexity $e^{1.2} = 3.32$. Perplexity 2 ⇔ loss $\ln 2 = 0.693$ nats = 1 bit.
- **1.5** $q_\alpha(\texttt e\mid\texttt z) = (1+\alpha)/(1+50\alpha)$; this equals 0.5 when $2 + 2\alpha = 1 + 50\alpha$, i.e. $\alpha = 1/48$.
- **1.6** $n \ge \ln(2/\delta)/(2\varepsilon^2) = \ln 200/(2\times0.005^2) = 105\,967$; with $K = 1\,000$, $n\ge(\ln 1000 + \ln 200)/(2\times 0.005^2) = 244\,122$.
- **1.11** $n\hat L = -\sum_{a,b}c(a\to b)\ln\frac{c(a\to b)}{c(a)} = \sum_a c(a)\big[-\sum_b \hat p(b\mid a)\ln\hat p(b\mid a)\big]$.
- **1.16** The minimum of $K$ i.i.d. $N(0,\sigma^2)$ variables has expectation $\approx-\sigma\sqrt{2\ln K}$ for large $K$; with $K = 1000$ that is about $3.7\sigma$, an upper bound that the exact value ($\approx 3.24\sigma$) approaches slowly.

## Further reading

**Foundations.** [Shalev-Shwartz & Ben-David 2014](/references/#shalev-shwartz-2014), Chs. 2–6 — the cleanest modern account of ERM, uniform convergence, VC dimension and no free lunch. [Cover & Thomas 2006](/references/#cover-2006), Chs. 2 and 5 — entropy, KL divergence and coding.

**Language modelling.** [Shannon 1951](/references/#shannon-1951) — still the best introduction to predicting text. [Chen & Goodman 1999](/references/#chen-1999) — what decades of smoothing research learned. [Bengio et al. 2003](/references/#bengio-2003) — the bridge to Unit 02.

**Scoring rules.** [Gneiting & Raftery 2007](/references/#gneiting-2007) — why log-loss and the Brier score are proper, and what that buys.

**The modern puzzle.** [Zhang et al. 2017](/references/#zhang-2017), [Belkin et al. 2019](/references/#belkin-2019), [Nakkiran et al. 2020](/references/#nakkiran-2020), [Bartlett et al. 2020](/references/#bartlett-2020) — read in this order.

**Evaluation practice.** [Recht et al. 2019](/references/#recht-2019) and [Dwork et al. 2015](/references/#dwork-2015) — what happens to test sets that are reused.
