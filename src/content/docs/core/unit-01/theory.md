---
title: "1.5 General theory"
description: Uniform convergence for finite classes and the VC dimension, a proof of Hoeffding's inequality, the bias–variance decomposition, the no-free-lunch theorem, and why classical capacity arguments do not explain modern networks.
sidebar: { label: "1.5 General theory", order: 5 }
---

Section 1.3 analysed one model at a time. Learning theory asks a stronger question: if we choose the model that minimizes training loss over a whole family, when is its held-out loss close to its training loss? This section gives the classical answers, proves the key ones, and then shows where they stop explaining practice.

## 1.5.1 Uniform convergence

Let $\mathcal{H}$ be a family of predictors and $\ell\in[0,1]$ a bounded loss. ERM returns $\hat h = \arg\min_{h\in\mathcal H}\hat L(h)$. Because $\hat h$ depends on the training sample, Hoeffding's inequality does not apply to it directly. It does apply to every *fixed* $h$, and if the bound holds **simultaneously for all** $h\in\mathcal H$, it holds in particular for $\hat h$.

**Theorem 1.3 (finite classes).** If $|\mathcal H| = K$, then with probability at least $1-\delta$ over an i.i.d. sample of size $n$, every $h\in\mathcal H$ satisfies, for a loss with values in $[0, 1]$,
$$
\big|L(h) - \hat L(h)\big| \le \varepsilon,\qquad \varepsilon = \sqrt{\frac{\ln K + \ln(2/\delta)}{2n}} = \sqrt{\frac{\ln(2K/\delta)}{2n}} .
$$

*Proof.* For each fixed $h$, the two-sided Hoeffding inequality (Theorem 1.2) gives $\Pr(|L(h) - \hat L(h)|\ge\varepsilon)\le 2e^{-2n\varepsilon^2} = \delta/K$. The union bound over the $K$ members (§1.3.6) makes the probability that *any* deviation reaches $\varepsilon$ at most $\delta$. $\blacksquare$

Both directions are needed below: the upper deviation bounds the risk of the chosen $\hat h$, the lower one bounds the empirical risk of the comparator $h^\star$. (A one-sided statement would need only $\ln(K/\delta)$, but would not support the corollary.)

**Corollary 1.2.** Let $h^\star$ minimize the population risk over $\mathcal H$. With probability at least $1-\delta$, $L(\hat h) \le L(h^\star) + 2\sqrt{(\ln K + \ln(2/\delta))/(2n)}$.

*Proof.* $L(\hat h)\le \hat L(\hat h) + \varepsilon \le \hat L(h^\star) + \varepsilon \le L(h^\star) + 2\varepsilon$, using the two-sided uniform bound once in each direction and $\hat L(\hat h)\le\hat L(h^\star)$ (ERM). $\blacksquare$

Two features deserve emphasis. The excess risk depends on the family only through $\ln K$ — a **capacity** measure — and shrinks as $1/\sqrt{n}$. And the argument never used anything about *which* $h$ ERM picks; it holds for the worst member of the family. That is its strength (it is assumption-light) and, as §1.5.4 shows, its weakness.

**Infinite families.** A bigram table has continuous parameters, so $K$ is infinite. The classical remedy measures capacity combinatorially. For binary classification, the **VC dimension** $d_{\mathrm{VC}}$ of $\mathcal H$ is the largest number of points that $\mathcal H$ can label in all $2^m$ possible ways ("shatter"). Linear classifiers in $\mathbb{R}^k$ have $d_{\mathrm{VC}} = k+1$. The fundamental theorem of statistical learning states that a class is learnable by ERM if and only if its VC dimension is finite, with excess risk of order $\sqrt{(d_{\mathrm{VC}} + \ln(1/\delta))/n}$ ([Vapnik 1998](/references/#vapnik-1998); [Shalev-Shwartz & Ben-David 2014](/references/#shalev-shwartz-2014), Ch. 6). The finite-class proof above is the template: replace $\ln K$ by a quantity that counts the *effectively distinct* behaviours of the class on $n$ points.

### ★ Advanced: a proof of Hoeffding's inequality

Let $Z_i\in[a,b]$ be independent, $\mu_i = \mathbb{E}Z_i$, and $S = \sum_i(Z_i - \mu_i)$. For any $s>0$, Markov's inequality applied to $e^{sS}$ gives
$$
\Pr(S\ge n\varepsilon) \le e^{-sn\varepsilon}\,\mathbb{E}[e^{sS}] = e^{-sn\varepsilon}\prod_i\mathbb{E}[e^{s(Z_i-\mu_i)}].
$$
**Hoeffding's lemma** bounds each factor: a zero-mean variable $X\in[a',b']$ has $\mathbb{E}[e^{sX}]\le e^{s^2(b'-a')^2/8}$. (Proof sketch: by convexity of $e^{sx}$, $e^{sX}\le\frac{b'-X}{b'-a'}e^{sa'} + \frac{X-a'}{b'-a'}e^{sb'}$; take expectations, write the result as $e^{\varphi(u)}$ with $u = s(b'-a')$, and check $\varphi(0) = \varphi'(0) = 0$, $\varphi''\le 1/4$, so $\varphi(u)\le u^2/8$ by Taylor's theorem.) Hence $\Pr(S\ge n\varepsilon)\le\exp(-sn\varepsilon + ns^2(b-a)^2/8)$. Minimizing over $s$ at $s = 4\varepsilon/(b-a)^2$ gives $\exp(-2n\varepsilon^2/(b-a)^2)$. The lower tail is identical; the factor 2 comes from adding both tails. $\blacksquare$

## 1.5.2 The bias–variance decomposition

For squared loss there is an exact decomposition of the expected error of a learning *procedure*, averaged over training sets. Let $y = f(x) + \varepsilon$ with $\mathbb{E}\varepsilon = 0$, $\mathrm{Var}\,\varepsilon = \sigma^2$, and let $\hat f_{\mathcal D}$ be the predictor trained on a random training set $\mathcal{D}$. At a fixed test point $x$, write $\bar f(x) = \mathbb{E}_{\mathcal D}[\hat f_{\mathcal D}(x)]$.

**Proposition 1.2.**
$$
\mathbb{E}_{\mathcal D,\varepsilon}\big[(y - \hat f_{\mathcal D}(x))^2\big] = \underbrace{\sigma^2}_{\text{noise}} + \underbrace{(f(x) - \bar f(x))^2}_{\text{bias}^2} + \underbrace{\mathbb{E}_{\mathcal D}\big[(\hat f_{\mathcal D}(x) - \bar f(x))^2\big]}_{\text{variance}} .
$$

*Proof.* Write $y - \hat f = \varepsilon + (f - \bar f) + (\bar f - \hat f)$. Square and take expectations. The cross terms vanish: $\varepsilon$ is independent of $\mathcal D$ with mean zero; $f - \bar f$ is a constant; and $\mathbb{E}_{\mathcal D}[\bar f - \hat f] = 0$ by definition of $\bar f$. $\blacksquare$

The **noise** term is the irreducible floor (the analogue of the entropy in Corollary 1.1). **Bias** measures how far the *average* trained model is from the truth — it shrinks as the family grows more flexible. **Variance** measures how much the trained model changes from one training set to another — it grows with flexibility and shrinks with more data. Smoothing in §1.2.7 is a bias–variance trade-off: α → ∞ gives zero variance (the uniform model ignores the data) and large bias; α → 0 gives zero bias in the large-data limit and high variance on small samples. The classical prediction from this picture is a U-shaped test error as capacity grows ([Geman et al. 1992](/references/#geman-1992); [Hastie et al. 2009](/references/#hastie-2009), §7.3).

## 1.5.3 No free lunch

Could some learning procedure be best on every problem? No.

**Theorem 1.4 (no free lunch, finite version).** Let $\mathcal{X}$ be finite, labels binary, and fix any learner. Average over *all* $2^{|\mathcal X|}$ target functions $f:\mathcal X\to\{0,1\}$, with the training set labelled by $f$. Then on any point not in the training set, the learner's expected accuracy is exactly $1/2$.

*Proof.* Fix a training set of inputs $S$ and a test point $x\notin S$. The learner's prediction at $x$ depends only on the labels of $S$. For each labelling of $S$, exactly half of the target functions consistent with it have $f(x) = 1$ and half $f(x) = 0$, because the value at $x$ is unconstrained. So whatever the learner predicts, it is right for exactly half of the targets. $\blacksquare$

The theorem does not say learning is impossible; it says learning is impossible **without assumptions about which targets are likely** ([Wolpert 1996](/references/#wolpert-1996)). Every successful learner wins on some problems by losing on others it implicitly considers unlikely. Smoothness, locality, sparsity, symmetry, and "similar inputs have similar outputs" are such assumptions — **inductive biases**. Unit 02 is about choosing them through representation; Unit 04 derives one (attention's permutation equivariance) from a symmetry argument.

## 1.5.4 Where the classical theory stops

Apply Theorem 1.3's logic to a modern network. A network with millions of parameters can fit random labels exactly (§1.1), so its effective capacity on $n = 50\,000$ points is at least the information in arbitrary labels; any bound of the form "training error + capacity term" is then vacuous — it exceeds 100% error. Yet the same network trained on true labels generalizes well. Three observations from the last decade sharpen the puzzle:

- **Memorization of random labels** ([Zhang et al. 2017](/references/#zhang-2017)): standard architectures and optimizers fit random labels and random pixels; explicit regularization is neither necessary nor sufficient for generalization.
- **Double descent** ([Belkin et al. 2019](/references/#belkin-2019); [Nakkiran et al. 2020](/references/#nakkiran-2020)): as model size grows past the point where the model can just interpolate the training data, test error first rises (the classical U) and then *falls again*, often below the classical minimum. The same shape appears as a function of training time and of dataset size.
- **Benign overfitting** ([Bartlett et al. 2020](/references/#bartlett-2020)): in linear regression with many more parameters than examples, the minimum-norm interpolating solution fits noise exactly and still predicts nearly optimally — provided the data covariance has many small directions that absorb the noise harmlessly.

The common thread is that worst-case capacity is the wrong quantity. What matters is **which** interpolating solution the training procedure finds. Gradient descent from small initialization has an *implicit bias* toward simple solutions (minimum norm in linear models; Unit 03 §3.5 and Unit 06 return to this), and real data is structured so that simple solutions exist. Generalization is a joint property of data, architecture and optimizer — not of the parameter count alone. The precise theory is an open research area (§1.8); what is *not* open is the practical rule this unit is built on: **measure on held-out data, drawn from the same source, never used for a decision.**

## 1.5.5 Summary of the theory

| Result | Says | Assumes |
|---|---|---|
| Corollary 1.1 | log-loss = irreducible entropy + KL excess | — |
| Hoeffding (Thm 1.2) | held-out estimate within $O(1/\sqrt n)$ | i.i.d., bounded loss, model fixed in advance |
| Uniform convergence (Thm 1.3) | ERM's excess risk $O(\sqrt{\ln K / n})$ | i.i.d., bounded loss, finite or finite-VC class |
| Bias–variance (Prop 1.2) | error = noise + bias² + variance | squared loss |
| No free lunch (Thm 1.4) | no learner is best on all targets | uniform average over targets |
| Modern evidence (§1.5.4) | capacity bounds are vacuous for large networks; what the optimizer finds matters | empirical, partially theoretical |
