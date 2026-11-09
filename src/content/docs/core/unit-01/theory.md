---
title: "1.5 General theory"
description: Uniform convergence for finite classes and the VC dimension, a proof of Hoeffding's inequality, the bias–variance decomposition, the no-free-lunch theorem, and why classical capacity arguments do not explain modern networks.
sidebar: { label: "1.5 General theory", order: 5 }
---

Section 1.3 analysed one model fixed in advance. In real machine learning, however, we search over an enormous collection of candidate models and select the one that performs best on the training set.

This raises the central question of statistical learning theory: **If a model was explicitly chosen to achieve low loss on the training data, why should we trust it to perform well on unseen future data?** This section presents the answers of classical learning theory, dissects their proofs and intuition, and reveals why these classical bounds encountered a profound crisis when confronted with modern over-parameterized deep neural networks.

## 1.5.1 Uniform convergence and capacity

### Why empirical risk minimization cannot directly use a single concentration bound

Let $\mathcal{H}$ be a family of candidate predictors (the hypothesis class), and $\ell\in[0,1]$ a bounded loss. The empirical risk minimization (ERM) principle selects the predictor with minimal training loss: $\hat h = \arg\min_{h\in\mathcal H}\hat L(h)$.

If $h$ were a model fixed *before* observing any data, Hoeffding's inequality (Theorem 1.2) would guarantee that its empirical loss $\hat{L}(h)$ concentrates exponentially around its true expected population loss $L(h)$.

However, **we cannot directly apply Hoeffding's inequality to $\hat{h}$**. The reason is severe **data snooping bias**: $\hat{h}$ is itself a random variable determined by the specific training sample.

Consider $K$ completely fair coins (each with heads probability 0.5), each flipped 10 times. For any single predetermined coin, the probability of obtaining 10 consecutive heads is only $(1/2)^{10} \approx 0.00098$. But if we examine 1,000 coins and pick the one with the most heads on our sample, the probability that *at least one* coin turns up all heads is $1 - (1 - 0.5^{10})^{1000} \approx 62.4\%$! Claiming that this selected coin will continue to produce 100% heads on future flips is obviously fallacious.

To guarantee that the chosen $\hat{h}$ generalizes, we must demand that empirical loss approximates true loss **simultaneously across the entire hypothesis class $\mathcal{H}$** — this is **uniform convergence**.

**Theorem 1.3 (finite classes).** If $|\mathcal H| = K$, then with probability at least $1-\delta$ over an i.i.d. sample of size $n$, every $h\in\mathcal H$ satisfies, for a loss with values in $[0, 1]$,
$$
\big|L(h) - \hat L(h)\big| \le \varepsilon,\qquad \varepsilon = \sqrt{\frac{\ln K + \ln(2/\delta)}{2n}} = \sqrt{\frac{\ln(2K/\delta)}{2n}} .
$$

*Proof.* For each fixed $h \in \mathcal{H}$, the two-sided Hoeffding inequality (Theorem 1.2) states that the failure event for that specific model satisfies $\Pr(|L(h) - \hat L(h)|\ge\varepsilon)\le 2e^{-2n\varepsilon^2}$.
We are interested in whether *any* model in the class fails. By the **union bound** (Boole's inequality), the probability of the union of finite events does not exceed the sum of their individual probabilities:
$\Pr\Big(\exists h\in\mathcal{H}: |L(h) - \hat L(h)|\ge\varepsilon\Big) \le \sum_{h\in\mathcal{H}} \Pr\Big(|L(h) - \hat L(h)|\ge\varepsilon\Big) \le K \cdot 2e^{-2n\varepsilon^2}$.
Setting this upper bound equal to the permitted total failure probability $\delta$, i.e., $2K e^{-2n\varepsilon^2} = \delta$, and solving for $\varepsilon$:
$-2n\varepsilon^2 = \ln\frac{\delta}{2K} \implies \varepsilon = \sqrt{\frac{\ln(2K/\delta)}{2n}} = \sqrt{\frac{\ln K + \ln(2/\delta)}{2n}}$. $\blacksquare$

This bound reveals a profound fact: **the worst-case statistical price of searching over a large hypothesis space grows only logarithmically as $\sqrt{\ln K}$.** Even if the search space $K$ is vast, as long as sample size $n$ scales linearly, uniform convergence guarantees the reliability of the empirical minimizer.

Both directions are needed below: the upper deviation bounds the risk of the chosen $\hat h$, the lower one bounds the empirical risk of the comparator $h^\star$. (A one-sided statement would need only $\ln(K/\delta)$, but would not support the corollary.)

**Corollary 1.2.** Let $h^\star$ minimize the population risk over $\mathcal H$. With probability at least $1-\delta$, $L(\hat h) \le L(h^\star) + 2\sqrt{(\ln K + \ln(2/\delta))/(2n)}$.

*Proof.* $L(\hat h)\le \hat L(\hat h) + \varepsilon \le \hat L(h^\star) + \varepsilon \le L(h^\star) + 2\varepsilon$, using the two-sided uniform bound once in each direction and $\hat L(\hat h)\le\hat L(h^\star)$ (ERM). $\blacksquare$

Two features deserve emphasis. The excess risk depends on the family only through $\ln K$ — a **capacity** measure — and shrinks as $1/\sqrt{n}$. And the argument never used anything about *which* $h$ ERM picks; it holds for the worst member of the family. That is its strength (it is assumption-light) and, as §1.5.4 shows, its weakness.

**Infinite families and VC dimension.** A bigram table has continuous parameters, so $K$ is infinite. The classical remedy measures capacity combinatorially. For binary classification, the **VC dimension** $d_{\mathrm{VC}}$ of $\mathcal H$ is the largest number of points that $\mathcal H$ can label in all $2^m$ possible ways ("shatter").
For instance, linear classifiers in 2D space ($\mathbb{R}^2$) can shatter any 3 non-collinear points, but cannot shatter 4 points configured as an XOR pattern (where diagonally opposite points share labels), giving $d_{\mathrm{VC}} = 3 = 2 + 1$. In general, hyperplanes in $\mathbb{R}^k$ have $d_{\mathrm{VC}} = k+1$.
The fundamental theorem of statistical learning states that a class is learnable by ERM if and only if its VC dimension is finite, with excess risk of order $\sqrt{(d_{\mathrm{VC}} + \ln(1/\delta))/n}$ ([Vapnik 1998](/references/#vapnik-1998); [Shalev-Shwartz & Ben-David 2014](/references/#shalev-shwartz-2014), Ch. 6). The finite-class proof above is the template: replace $\ln K$ by a combinatorial quantity that counts the *effectively distinct* behaviours of the class on $n$ points (Sauer's lemma).

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

Apply Theorem 1.3's logic to a modern network. A network with millions or billions of parameters can fit random labels exactly (§1.1), so its effective capacity on $n = 50\,000$ points is at least the information in arbitrary labels; any bound of the form "training error + capacity term" is then vacuous — it exceeds 100% error. Yet the same network trained on true labels generalizes remarkably well. Four modern discoveries illuminate this puzzle:

1. **Memorization of random labels** ([Zhang et al. 2017](/references/#zhang-2017)): Standard architectures and optimizers can fit completely randomized labels with 100% accuracy, showing that their worst-case capacity is vast enough to destroy classical generalization bounds, yet they generalize well on real labels. Explicit capacity measures cannot distinguish between these two regimes.
2. **Double descent** ([Belkin et al. 2019](/references/#belkin-2019); [Nakkiran et al. 2020](/references/#nakkiran-2020)): As parameter count increases, test error first follows the classical U-shaped curve; at the interpolation threshold where the model can just fit the training data, test error peaks sharply; but past this threshold, as the model grows even larger, test error *decreases monotonically again*, often outperforming the classical optimal point.
3. **Benign overfitting** ([Bartlett et al. 2020](/references/#bartlett-2020)): In over-parameterized regimes, even though the model fits noisy training samples to zero error, the noise is diluted across many small eigen-directions of high-dimensional space without corrupting predictions along the dominant feature directions.
4. **Implicit bias of optimizers:** In over-parameterized regimes, infinitely many parameter configurations achieve zero training loss. Gradient descent (SGD/Adam) intrinsically selects solutions with minimal geometric norm (the minimum-norm interpolant, explored in Units 03 and 06).

**Conclusion:** Classical learning theory focused on worst-case model capacity. Modern deep learning succeeds through the trinity of **data distribution geometry, architectural inductive bias, and optimization dynamics**. In practice, one foundational rule remains steadfast: **always measure performance on held-out data drawn from the same distribution, completely isolated from model selection and tuning decisions.**

## 1.5.5 Summary of the theory

| Result | Core Insight and Intuition | Key Assumptions | Implication for Modern Deep Learning |
|---|---|---|---|
| **Corollary 1.1** | Log-loss = irreducible data entropy + KL divergence to truth | — | Cross-entropy training drives model predictions toward the true empirical distribution |
| **Hoeffding's inequality (Thm 1.2)** | Deviation of sample mean on fixed model decays exponentially as $O(1/\sqrt n)$ | i.i.d., bounded loss, model fixed before evaluation | Validation set size provides physical guarantee; $4\times$ data yields $2\times$ precision |
| **Uniform convergence bound (Thm 1.3)** | ERM excess risk bounded by $O(\sqrt{\ln K / n})$ | i.i.d., bounded loss, finite hypothesis class | Logarithmic cost of searching hypothesis space grounds the statistical feasibility of ML |
| **Bias–variance decomposition (Prop 1.2)** | Generalization error = irreducible noise + bias² + variance | Squared loss | Classical U-curve modified by double descent in over-parameterized regimes |
| **No free lunch (Thm 1.4)** | No universally superior learner exists without assumptions | Uniform average over all possible targets | Inductive biases must be encoded via architecture (e.g. attention permutation equivariance) |
| **Modern generalization insights (§1.5.4)** | Worst-case capacity bounds fail; optimizer implicit bias and data geometry govern generalization | Empirical findings and high-dimensional convex/non-convex theory | Underpins scaling laws and over-parameterized deep learning foundations |
