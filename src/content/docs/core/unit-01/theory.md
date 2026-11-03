---
title: "1.5 General theory"
description: Uniform convergence for finite classes and the VC dimension, a proof of Hoeffding's inequality, the bias–variance decomposition, the no-free-lunch theorem, and why classical capacity arguments do not explain modern networks.
sidebar: { label: "1.5 General theory", order: 5 }
---

Section 1.3 analysed a model that had already been fixed. Real training does something more dangerous: it searches through many possible models and returns the one that looks best on the training sample. The selected model therefore depends on the same data used to report its training loss.

Learning theory asks whether that data-dependent choice is still trustworthy. If we minimize training loss over an entire family, when will the chosen model also have small population risk? This section builds the classical answer in four layers:

1. **Uniform convergence** controls every model in a family at the same time.
2. **Bias–variance decomposition** separates three sources of squared prediction error.
3. **No free lunch** explains why every learner must embody assumptions about the world.
4. **Modern overparameterized models** show where worst-case capacity alone becomes too crude.

The mathematical statements and proofs remain important even where the final bounds are loose: they teach us exactly which assumptions make a generalization claim possible.

## 1.5.1 Uniform convergence

Let $\mathcal{H}$ be a family of predictors and let the loss satisfy $\ell\in[0,1]$. Empirical risk minimization (ERM) returns

$$
\hat h = \arg\min_{h\in\mathcal H}\hat L(h).
$$

There is a subtle obstacle. Hoeffding's inequality controls the measured loss of a *fixed* predictor, but $\hat h$ is not fixed before seeing the data: the sample itself decides which $h$ wins. We cannot safely apply a fixed-model guarantee only after looking at the sample and choosing its most favorable model.

The solution is to demand a stronger event: the training and population losses are close **simultaneously for every** $h\in\mathcal H$. If that event holds for the whole family, it automatically holds for whichever member the data selects, including $\hat h$. This is **uniform convergence**.

**Theorem 1.3 (finite classes).** If $|\mathcal H| = K$, then with probability at least $1-\delta$ over an i.i.d. sample of size $n$, every $h\in\mathcal H$ satisfies, for a loss with values in $[0, 1]$,
$$
\big|L(h) - \hat L(h)\big| \le \varepsilon,\qquad \varepsilon = \sqrt{\frac{\ln K + \ln(2/\delta)}{2n}} = \sqrt{\frac{\ln(2K/\delta)}{2n}} .
$$

*Proof.* The argument has two steps.

1. Fix one $h$. The two-sided Hoeffding inequality (Theorem 1.2) gives
   $$
   \Pr\big(|L(h)-\hat L(h)|\ge\varepsilon\big)
   \le 2e^{-2n\varepsilon^2}.
   $$
   Choose $\varepsilon$ so that this failure probability equals $\delta/K$.
2. There are $K$ possible failures, one for each member of $\mathcal H$. The union bound (§1.3.6) says that the probability of *at least one* failure is at most $K(\delta/K)=\delta$.

Therefore, with probability at least $1-\delta$, none of the $K$ predictors deviates by more than $\varepsilon$. Solving $2e^{-2n\varepsilon^2}=\delta/K$ gives the stated expression. $\blacksquare$

The absolute value controls both directions: $L(h)\le\hat L(h)+\varepsilon$ and $\hat L(h)\le L(h)+\varepsilon$. The corollary needs the first direction for the chosen model $\hat h$ and the second for the best population model $h^\star$. A one-sided statement would need only $\ln(K/\delta)$, but would not justify the full comparison below.

**Corollary 1.2.** Let $h^\star$ minimize the population risk over $\mathcal H$. With probability at least $1-\delta$, $L(\hat h) \le L(h^\star) + 2\sqrt{(\ln K + \ln(2/\delta))/(2n)}$.

*Proof.* Read the following chain from left to right:

$$
\underbrace{L(\hat h)}_{\text{what we want}}
\le \underbrace{\hat L(\hat h)+\varepsilon}_{\text{uniform bound}}
\le \underbrace{\hat L(h^\star)+\varepsilon}_{\text{ERM chooses the smallest training risk}}
\le \underbrace{L(h^\star)+2\varepsilon}_{\text{uniform bound again}}.
$$

The first and third inequalities use the two opposite directions of the same uniform event. The middle inequality is the definition of ERM. $\blacksquare$

Two features deserve emphasis. First, searching a larger family is not free, but the penalty depends on the family through $\ln K$, a **capacity** measure, rather than through $K$ itself. Second, the penalty shrinks only as $1/\sqrt n$: halving it requires four times as much data.

The proof never used anything about *which* member ERM tends to choose. It protects us even if ERM selects the worst-behaved member that the family permits. This worst-case protection is the theorem's strength—it needs few assumptions—and, as §1.5.4 shows, also its weakness for modern neural networks.

**Infinite families.** A bigram table has continuous parameters, so it contains infinitely many predictors and we cannot substitute $K=\infty$ into the finite theorem. The classical remedy does not count parameter values; it counts how many distinct decisions the family can express on a finite sample.

For binary classification, a family **shatters** $m$ points if it can realize every one of the $2^m$ possible binary labellings of those points. The **VC dimension** $d_{\mathrm{VC}}$ is the largest $m$ that can be shattered. For example, linear classifiers in $\mathbb{R}^k$ have $d_{\mathrm{VC}}=k+1$: they can realize every labelling of some $k+1$ suitably placed points, but not of every set of $k+2$ points.

The fundamental theorem of statistical learning states that a binary classification class is learnable by ERM if and only if its VC dimension is finite, with excess risk of order $\sqrt{(d_{\mathrm{VC}}+\ln(1/\delta))/n}$ ([Vapnik 1998](/references/#vapnik-1998); [Shalev-Shwartz & Ben-David 2014](/references/#shalev-shwartz-2014), Ch. 6). Conceptually, the finite-class proof remains the template: replace $\ln K$, which counts named models, by a quantity that counts the family's *effectively distinct behaviours* on $n$ observations.

### ★ Advanced: a proof of Hoeffding's inequality

This proof uses a common probability technique: turn the difficult event “a sum is large” into an event about a non-negative exponential, apply Markov's inequality, and then choose the exponential scale that gives the tightest bound.

Let $Z_i\in[a,b]$ be independent, let $\mu_i=\mathbb{E}Z_i$, and define the centred sum $S=\sum_i(Z_i-\mu_i)$. We want to bound $\Pr(S\ge n\varepsilon)$. For any $s>0$, the exponential is increasing, so this event is equivalent to $e^{sS}\ge e^{sn\varepsilon}$. Markov's inequality applied to the non-negative variable $e^{sS}$ gives
$$
\Pr(S\ge n\varepsilon) \le e^{-sn\varepsilon}\,\mathbb{E}[e^{sS}] = e^{-sn\varepsilon}\prod_i\mathbb{E}[e^{s(Z_i-\mu_i)}].
$$
Independence is what changes the exponential of a sum into the product of expectations on the right.

**Hoeffding's lemma** now bounds each factor: if a zero-mean variable $X\in[a',b']$, then

$$
\mathbb{E}[e^{sX}]\le e^{s^2(b'-a')^2/8}.
$$

*Proof sketch of the lemma.* Convexity places the graph of $e^{sx}$ below the line segment joining its values at the two endpoints:

$$
e^{sX}\le\frac{b'-X}{b'-a'}e^{sa'} + \frac{X-a'}{b'-a'}e^{sb'}.
$$

Take expectations, use $\mathbb E X=0$, and write the result as $e^{\varphi(u)}$ with $u=s(b'-a')$. One checks that $\varphi(0)=\varphi'(0)=0$ and $\varphi''\le1/4$. Taylor's theorem then gives $\varphi(u)\le u^2/8$.

Applying the lemma to every factor yields

$$
\Pr(S\ge n\varepsilon)
\le \exp\!\left(-sn\varepsilon+\frac{ns^2(b-a)^2}{8}\right).
$$

This is true for every $s>0$, so choose the best one. The quadratic exponent is minimized at $s=4\varepsilon/(b-a)^2$, which gives $\exp(-2n\varepsilon^2/(b-a)^2)$. Repeating the same argument for $-S$ bounds the lower tail; adding the two tail probabilities produces the factor 2. $\blacksquare$

## 1.5.2 The bias–variance decomposition

Uniform convergence studies the gap between training and population loss. The bias–variance decomposition asks a different question: *where does prediction error come from?* For squared loss, the answer is an exact three-part identity.

Imagine repeatedly drawing a new training set $\mathcal D$ from the same source and rerunning the entire learning procedure. Each run may produce a different predictor $\hat f_{\mathcal D}$. Let the observed target be

$$
y=f(x)+\varepsilon,qquad \mathbb E\varepsilon=0,qquad \mathrm{Var}\,\varepsilon=\sigma^2,
$$

and, at a fixed test input $x$, define the average prediction across hypothetical training sets as $\bar f(x)=\mathbb E_{\mathcal D}[\hat f_{\mathcal D}(x)]$.

**Proposition 1.2.**
$$
\mathbb{E}_{\mathcal D,\varepsilon}\big[(y - \hat f_{\mathcal D}(x))^2\big] = \underbrace{\sigma^2}_{\text{noise}} + \underbrace{(f(x) - \bar f(x))^2}_{\text{bias}^2} + \underbrace{\mathbb{E}_{\mathcal D}\big[(\hat f_{\mathcal D}(x) - \bar f(x))^2\big]}_{\text{variance}} .
$$

*Proof.* Add and subtract $\bar f$:

$$
y-\hat f=\underbrace{\varepsilon}_{\text{observation noise}}
+\underbrace{(f-\bar f)}_{\text{systematic error}}
+\underbrace{(\bar f-\hat f)}_{\text{training-set fluctuation}}.
$$

Square this sum and take expectations. The three squared terms become the three terms in the proposition. Every cross term has expectation zero: $\varepsilon$ is independent of $\mathcal D$ and has mean zero; $f-\bar f$ is fixed at the chosen $x$; and $\mathbb E_{\mathcal D}[\bar f-\hat f]=0$ by the definition of $\bar f$. $\blacksquare$

The identity separates three ideas that are often blurred together:

- **Noise** is the irreducible floor: even knowing the true $f(x)$ cannot predict the random $\varepsilon$. It plays the role that entropy played for log-loss in Corollary 1.1.
- **Bias** is the distance between the truth and the *average* model produced by the learning procedure. A rigid family tends to repeat the same systematic mistake across training sets.
- **Variance** is how much the fitted model changes when the training set changes. A highly flexible procedure can chase the accidents of each small sample.

Smoothing in §1.2.7 makes the trade-off concrete. As $\alpha\to\infty$, the uniform model ignores the observed counts: its variance is zero, but its bias can be large. As $\alpha\to0$, the estimator can approach the truth with enough data, but on a small sample it reacts strongly to which transitions happened to appear. The classical picture therefore predicts a U-shaped test error as capacity grows: bias falls, variance rises, and their sum first decreases and then increases ([Geman et al. 1992](/references/#geman-1992); [Hastie et al. 2009](/references/#hastie-2009), §7.3).

## 1.5.3 No free lunch

The previous results may suggest that we only need to invent a sufficiently clever learning rule. Could one procedure be best on every possible prediction problem? No. The obstacle is not computation; it is the absence of information about unseen labels.

**Theorem 1.4 (no free lunch, finite version).** Let $\mathcal{X}$ be finite, labels binary, and fix any learner. Average over *all* $2^{|\mathcal X|}$ target functions $f:\mathcal X\to\{0,1\}$, with the training set labelled by $f$. Then on any point not in the training set, the learner's expected accuracy is exactly $1/2$.

*Proof.* Fix the training inputs $S$ and one test point $x\notin S$. After seeing the labels on $S$, the learner must make some prediction at $x$. But those training labels place no restriction on $f(x)$ when we average uniformly over *all* target functions.

For every target function consistent with the observed training labels and satisfying $f(x)=0$, there is a paired target that is identical everywhere except at $x$ and satisfies $f(x)=1$. Thus exactly half of the consistent targets assign each label at $x$. Whichever label the learner predicts, it is correct for one member of each pair and wrong for the other, giving expected accuracy $1/2$. $\blacksquare$

The theorem does **not** say that learning real problems is impossible. It says that learning from finite observations is impossible **without assumptions about which targets are likely** ([Wolpert 1996](/references/#wolpert-1996)). If arbitrary unseen labels are equally plausible, the data contain no basis for preferring one prediction over the other.

Every successful learner therefore wins on some problems by doing worse on others that it implicitly treats as unlikely. Assumptions such as smoothness, locality, sparsity, symmetry, or “similar inputs have similar outputs” are **inductive biases**. They are not defects to eliminate; they are the extra structure that makes generalization possible. Unit 02 studies how representations encode such biases, and Unit 04 derives one—attention's permutation equivariance—from a symmetry argument.

## 1.5.4 Where the classical theory stops

The classical results are valid, but a valid upper bound can still be too loose to explain what we observe. Apply Theorem 1.3's worst-case logic to a modern network. A network with millions of parameters can fit arbitrary labels on $n=50\,000$ points (§1.1), so its family has enough effective capacity to represent every labelling of that sample. A guarantee of the form “training error + worst-case capacity penalty” can then exceed 100% error and say only that the true error is at most a number larger than one. Such a bound is **vacuous**: correct, but uninformative.

The puzzle is that the same architecture and optimizer often generalize well on true labels. Three observations sharpen this mismatch:

- **Memorization of random labels** ([Zhang et al. 2017](/references/#zhang-2017)): standard architectures and optimizers fit random labels and random pixels; explicit regularization is neither necessary nor sufficient for generalization.
- **Double descent** ([Belkin et al. 2019](/references/#belkin-2019); [Nakkiran et al. 2020](/references/#nakkiran-2020)): as model size grows past the point where the model can just interpolate the training data, test error first rises (the classical U) and then *falls again*, often below the classical minimum. The same shape appears as a function of training time and of dataset size.
- **Benign overfitting** ([Bartlett et al. 2020](/references/#bartlett-2020)): in linear regression with many more parameters than examples, the minimum-norm interpolating solution fits noise exactly and still predicts nearly optimally — provided the data covariance has many small directions that absorb the noise harmlessly.

The common thread is that worst-case capacity asks only what a model family *can* fit, not what the training procedure is likely to find. In an overparameterized family there are many interpolating solutions, and they need not generalize equally well. Gradient descent from a small initialization has an *implicit bias* toward particular, often simpler solutions—minimum norm in linear models; Unit 03 §3.5 and Unit 06 return to this—and real data is structured so that useful simple solutions exist.

Generalization is therefore a joint property of the **data**, **architecture**, and **optimizer**, not of parameter count alone. A complete quantitative theory remains an open research area (§1.8). The unresolved theory does not weaken the practical rule on which this unit is built: **measure on held-out data drawn from the same source, and never use those measurements to make training or selection decisions.**

## 1.5.5 Put the results side by side

The results answer different questions and rely on different assumptions. None replaces the others.

| Result | Says | Assumes |
|---|---|---|
| Corollary 1.1 | log-loss = irreducible entropy + KL excess | — |
| Hoeffding (Thm 1.2) | held-out estimate within $O(1/\sqrt n)$ | i.i.d., bounded loss, model fixed in advance |
| Uniform convergence (Thm 1.3) | ERM's excess risk $O(\sqrt{\ln K / n})$ | i.i.d., bounded loss, finite or finite-VC class |
| Bias–variance (Prop 1.2) | error = noise + bias² + variance | squared loss |
| No free lunch (Thm 1.4) | no learner is best on all targets | uniform average over targets |
| Modern evidence (§1.5.4) | capacity bounds are vacuous for large networks; what the optimizer finds matters | empirical, partially theoretical |
