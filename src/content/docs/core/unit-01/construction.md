---
title: "1.3 Mathematical construction"
description: Population and empirical risk, cross-entropy = entropy + KL with Gibbs' inequality, the maximum-likelihood bigram by Lagrange multipliers, add-α smoothing as a Dirichlet posterior mean, Hoeffding's inequality, the cost of selection, and the standard error of a held-out loss.
sidebar: { label: "1.3 Construction", order: 3 }
---

Section 1.2 built the intuition with coins and character counts. This section writes the same argument precisely, without removing any steps. Keep two questions in view:

1. **What do we want to know?** The model's average loss on future data.
2. **What can we actually calculate?** Its average loss on a finite sample.

The mathematics explains when the second quantity tells us about the first. It then answers three follow-up questions: why maximum likelihood reduces to counting, why smoothing is a principled prediction rather than a patch, and how uncertain a held-out score is. Each subsection states a result, proves it, and returns to the bigram.

## 1.3.1 Setting: distributions, predictors and risk

Write one prediction problem as a pair $(x,y)$. The input $x$ is what the model sees, and $y$ is what it must predict. For a bigram, $x$ is the previous character and $y$ the next one. We imagine that these pairs are drawn from an unknown **data distribution** $p_{\text{data}}$.

A **predictor** $q$ maps each input $x$ to a probability distribution $q(\cdot\mid x)$ over the $V$ possible values of $y$. If the observed answer is $y$, the predictor's loss on this one example is $\ell(q;x,y)=-\ln q(y\mid x)$.

The quantity we ultimately care about is the expected loss on a *new* example. This is the **population risk**:

$$
L(q) = \mathbb{E}_{(x,y)\sim p_{\text{data}}}\big[-\ln q(y\mid x)\big].
$$

We cannot calculate $L(q)$ directly because $p_{\text{data}}$ is unknown. What we can calculate is the average loss on a finite sample $\mathcal{D}=\{(x_i,y_i)\}_{i=1}^n$, the **empirical risk**:

$$
\hat L(q) = \frac{1}{n}\sum_{i=1}^n -\ln q(y_i\mid x_i).
$$

The hat in $\hat L$ is a reminder that this is an estimate from a sample. If the examples are i.i.d. draws from $p_{\text{data}}$ and $q$ was chosen *without looking at* $\mathcal{D}$, then linearity of expectation gives

$$
\mathbb{E}[\hat L(q)] = L(q).
$$

Here the expectation imagines repeatedly drawing a fresh sample of size $n$. Across those repetitions, the sample average is centred on the future-data loss. This one line is why held-out evaluation works. It is also why training loss does not: if $q$ was chosen *using* $\mathcal{D}$, then $q$ and the sample are no longer independent, so this argument no longer applies (§1.3.6).

**Empirical risk minimization (ERM)** chooses, from a family $\mathcal{Q}$, the predictor with the smallest training value of $\hat L$. The central question of learning theory is whether this same predictor also has a small population risk $L$.

## 1.3.2 Cross-entropy, entropy and KL divergence

We now ask what the best possible probability prediction would be if the true distribution were known. Fix one context $x$ and suppress it in the notation. Let $p$ be the true distribution of the next character and $q$ the model's distribution. If outcomes really follow $p$, the model's expected log-loss is the **cross-entropy**

$$
H(p, q) = -\sum_{y} p(y)\ln q(y).
$$

The next identity separates uncertainty that belongs to the data from error that belongs to the model. Add and subtract $\sum_y p(y)\ln p(y)$:

$$
H(p, q) = \underbrace{-\sum_y p(y)\ln p(y)}_{H(p)\ \text{(entropy)}} \;+\; \underbrace{\sum_y p(y)\ln\frac{p(y)}{q(y)}}_{\mathrm{KL}(p\,\Vert\, q)\ \text{(Kullback–Leibler divergence)}} .
$$

The first term, $H(p)$, does not depend on the model: it is the uncertainty already present in the true distribution. The second term, $\mathrm{KL}(p\,\Vert\,q)$, compares the model with that truth. To use the decomposition, we need to know that this comparison can never be negative.

**Theorem 1.1 (Gibbs' inequality).** $\mathrm{KL}(p\,\Vert\,q)\ge 0$, with equality if and only if $q = p$.

*Proof.* The function $\ln$ is concave, so Jensen's inequality gives $\mathbb{E}_p[\ln Z]\le\ln\mathbb{E}_p[Z]$ for any positive random variable $Z$. Choose $Z=q(y)/p(y)$ and sum over outcomes with $p(y)>0$:

$$
-\mathrm{KL}(p\,\Vert\,q) = \sum_{y:\,p(y)>0} p(y)\ln\frac{q(y)}{p(y)} \le \ln\sum_{y:\,p(y)>0} p(y)\frac{q(y)}{p(y)} = \ln\sum_{y:\,p(y)>0} q(y) \le \ln 1 = 0 .
$$

Equality in Jensen requires $q(y)/p(y)$ to be constant where $p>0$, and equality in the last step requires $q$ to put no mass where $p = 0$; together, $q = p$. $\blacksquare$

**Corollary 1.1.** The expected log-loss $H(p,q)$ is minimized exactly at $q = p$, and its minimum value is the entropy $H(p)$. Every model's population risk decomposes as

$$
L(q) = \underbrace{\mathbb{E}_x\big[H(p(\cdot\mid x))\big]}_{\text{irreducible}} + \underbrace{\mathbb{E}_x\big[\mathrm{KL}\big(p(\cdot\mid x)\,\Vert\, q(\cdot\mid x)\big)\big]}_{\text{reducible}} .
$$

Two consequences will recur throughout the book:

1. **There is a floor.** No model can have population loss below the conditional entropy of the data, and that floor is usually not zero. Unit 06 calls it the *irreducible loss* $E$ in a scaling law.
2. **Loss above the floor is model error.** The **excess** loss is an expected KL divergence, which measures how far the model's probabilities are from the true ones.

Corollary 1.1 is also the precise reason that log-loss is a proper score: in expectation, it is minimized only by reporting the true probabilities (§1.2.3).

## 1.3.3 The maximum-likelihood bigram

We can now prove the counting rule used in §1.2. A bigram model is a table $q(b\mid a)$ with non-negative entries and rows that sum to one: $q(b\mid a)\ge0$ and $\sum_bq(b\mid a)=1$ for every preceding character $a$. On a training sequence $x_1,\dots,x_n$, its total log-loss is

$$
(n-1)\hat L(q) = -\sum_{t=2}^{n}\ln q(x_t\mid x_{t-1}) = -\sum_{a}\sum_{b} c(a\to b)\ln q(b\mid a),
$$

The second expression simply groups identical character pairs. Once grouped, each row $a$ can be optimized separately. A sequence of $n$ characters has $n-1$ transitions, which explains the factor $n-1$; prepending a beginning-of-sequence symbol would create $n$ transitions instead.

**Proposition 1.1.** For each $a$ with $c(a)>0$, the minimizer of $-\sum_b c(a\to b)\ln q(b\mid a)$ subject to $\sum_b q(b\mid a) = 1$ is $q(b\mid a) = c(a\to b)/c(a)$.

*Proof by Lagrange multiplier.* Fix one row and abbreviate $c_b=c(a\to b)$ and $q_b=q(b\mid a)$. We minimize its loss while enforcing that the probabilities sum to one:

$$
F(q,\lambda)=-\sum_b c_b\ln q_b+\lambda\Big(\sum_bq_b-1\Big).
$$

Setting $\partial F/\partial q_b=-c_b/q_b+\lambda=0$ gives $q_b=c_b/\lambda$. Enforcing $\sum_bq_b=1$ then gives $\lambda=\sum_bc_b=c(a)$, so $q_b=c_b/c(a)$. The objective is convex in $q$ because it is a non-negative weighted sum of $-\ln q_b$, so this stationary point is the global minimum. $\blacksquare$

*A second proof, without calculus.* Let $\hat p_b=c_b/c(a)$ be the observed frequency in this row. Then

$$
-\sum_b c_b\ln q_b=c(a)\,H(\hat p,q)=c(a)\,[H(\hat p)+\mathrm{KL}(\hat p\,\Vert\,q)].
$$

Gibbs' inequality says this is minimized at $q=\hat p$. In other words, **maximum likelihood minimizes the KL divergence from the empirical distribution to the model.** The same interpretation applies later to neural networks: cross-entropy training fits $q_\theta$ to the empirical distribution in KL.

This gives the claim used in §1.2.8 and in Unit 03's spine test: **on a fixed training text, no counted, smoothed or neural bigram can have lower training loss than the maximum-likelihood table.** Stage 03 verifies that a gradient-trained bigram approaches this value from above and never crosses below it.

## 1.3.4 Smoothing as Bayesian inference

Section 1.2 introduced smoothing as adding $\alpha$ imaginary counts to every possible pair. The Bayesian view makes the same operation precise: those extra counts encode what we believed before seeing the data.

Take one row $\theta=(\theta_1,\dots,\theta_V)$, where $\theta_b=q(b\mid a)$, and treat it as an unknown random vector. Give it a **Dirichlet prior** whose parameters all equal $\alpha$:

$$
\pi(\theta) \propto \prod_b \theta_b^{\alpha-1}, \qquad \theta_b\ge 0,\ \textstyle\sum_b\theta_b = 1 .
$$

The likelihood of counts $c_b$ is $\prod_b\theta_b^{c_b}$. Multiplying prior and likelihood gives the posterior. It is again Dirichlet, now with parameters $c_b+\alpha$:

$$
\pi(\theta\mid c) \propto \prod_b \theta_b^{c_b + \alpha - 1}.
$$

A Dirichlet distribution with parameters $\beta_b$ has mean $\beta_b/\sum_{b'}\beta_{b'}$. Therefore the **posterior mean** is

$$
\mathbb{E}[\theta_b \mid c] = \frac{c_b + \alpha}{c(a) + \alpha V},
$$

This is exactly the add-$\alpha$ formula. It is also the Bayes-optimal prediction under log-loss: after averaging over our remaining uncertainty about $\theta$, the probability that the *next* character is $b$ is $\mathbb{E}[\theta_b\mid c]$.

Smoothing is therefore not an arbitrary patch. It is the prediction made by a learner whose prior says that each row comes from a symmetric Dirichlet distribution. A small $\alpha$ favours concentrated rows, in which most pairs never occur; a large $\alpha$ favours rows closer to uniform. When $\alpha=1$ and $V=2$, the result is Laplace's rule of succession: after $s$ successes in $n$ trials, predict $(s+1)/(n+2)$.

Choosing a prior introduces an **inductive bias** — an assumption not supplied by the data. Every learner has such biases; smoothing makes this one explicit and tunable. Unit 02 introduces a much stronger one: similar characters should have similar rows.

## 1.3.5 How far can a measured loss be from the true one?

Even when the predictor $q$ is fixed, its score changes slightly from one held-out sample to another. We therefore need to ask not only “what loss did we measure?” but also “how much could that measurement vary?”

Let $Z_i=-\ln q(y_i\mid x_i)$ be the loss on held-out example $i$. For i.i.d. examples, $\hat L(q)$ is the average of $n$ independent $Z_i$ values, each with mean $L(q)$. Hoeffding's inequality gives a conservative answer to how far that average can lie from its mean.

**Theorem 1.2 (Hoeffding's inequality, [Hoeffding 1963](/references/#hoeffding-1963)).** If $Z_1,\dots,Z_n$ are independent with $Z_i\in[a,b]$, then for every $\varepsilon > 0$

$$
\Pr\Big(\Big|\tfrac{1}{n}\textstyle\sum_i Z_i - \mathbb{E}\big[\tfrac1n\sum_i Z_i\big]\Big| \ge \varepsilon\Big) \le 2\exp\!\Big(-\frac{2n\varepsilon^2}{(b-a)^2}\Big).
$$

(The proof appears in §1.5.1 ★.) Set the right-hand side to a chosen failure probability $\delta$ and solve for $\varepsilon$. With probability at least $1-\delta$,

$$
|\hat L(q) - L(q)| \le (b-a)\sqrt{\frac{\ln(2/\delta)}{2n}} .
$$

The width $(b-a)$ records how widely one example's loss can vary; $n$ is the number of independent examples; and $\delta$ controls how often the bound may fail. Log-loss is not bounded in general because assigning a probability near zero creates an arbitrarily large loss. A smoothed bigram *is* bounded: add-$\alpha$ smoothing assigns every outcome probability at least $\alpha/(c(a)+\alpha V)$, so its loss is at most $\ln((c(a)+\alpha V)/\alpha)$.

The most important feature is the **rate** $1/\sqrt n$. To cut this uncertainty in half, we need four times as many independent held-out examples. Worked Example 1.6 substitutes actual numbers.

## 1.3.6 Choosing among many models needs fresh data

Hoeffding's inequality describes a predictor fixed *before* looking at the evaluation data. Model selection is different: we evaluate many predictors on the same validation set and keep the one with the best observed score. Even if every estimate is unbiased on its own, the winner is unusually likely to be the model that benefited from favourable sampling noise.

Suppose we compare $K$ fixed predictors $q_1,\dots,q_K$. Give each predictor a failure probability of $\delta/K$. The **union bound**, $\Pr(\cup_kA_k)\le\sum_k\Pr(A_k)$, then says that with probability at least $1-\delta$, *all* $K$ estimates are within

$$
\varepsilon_K = (b-a)\sqrt{\frac{\ln K + \ln(2/\delta)}{2n}}
$$

of their true values. The price of choosing among $K$ candidates grows only as $\sqrt{\ln K}$, not linearly in $K$, but it is still a real price. If you try a thousand settings and keep the smallest validation loss, that loss is biased downward: it looks better than the model's true loss. A **test set used only once**, after all choices are finished, provides a fresh estimate. Applying the same idea to the training set, with every possible parameter setting as a “candidate”, leads to the uniform-convergence theory of §1.5.1.

Repeated *adaptive* reuse is more dangerous than this calculation suggests. If the 500th candidate was designed after seeing the first 499 scores, the candidate list was not fixed in advance. Feedback from the validation set has entered the design process, and a determined analyst can overfit it with far fewer than $e^n$ trials ([Dwork et al. 2015](/references/#dwork-2015)). This is one reason benchmark scores drift upward after years of reuse (§1.7.3; [Recht et al. 2019](/references/#recht-2019)).

## 1.3.7 The standard error of a held-out loss

Hoeffding gives a worst-case guarantee and can be loose. For an observed dataset, we usually describe sampling variation with a **standard error**. If the per-example losses $Z_i$ are independent and have variance $\sigma^2$, their mean has variance $\sigma^2/n$, so

$$
\mathrm{se}(\hat L) = \frac{\sigma}{\sqrt n},
$$

We estimate this quantity by replacing $\sigma$ with the sample standard deviation. The central limit theorem then gives the approximate 95% interval $\hat L\pm1.96\,\mathrm{se}$.

The word *independent* matters. Character losses within one story are related: on the spine corpus, consecutive character losses under the bigram have correlation $-0.32$. Treating every character as an independent sample would therefore claim more information than the data contain. The honest sampling unit is the **story**, because stories — not individual characters — were drawn independently. The simplest correct approach is to compute one average loss per story and take the standard error across stories (Worked Example 1.7). Unit 10 introduces the bootstrap for more general statistics.

**How many held-out examples do you need to tell two models apart?** Suppose model B improves on model A by $d$ nats per example on average, and the per-example *difference* in their losses has standard deviation $\sigma_d$. A two-sided test with 5% significance and 80% power needs approximately

$$
n \approx \Big(\frac{(1.96 + 0.84)\,\sigma_d}{d}\Big)^2
$$

independent examples. The formula uses $\sigma_d$, not the standard deviation of either model's losses. Evaluating both models on the *same* examples pairs their results: an unusually hard example tends to hurt both models, so subtracting the losses cancels some shared difficulty. This can make $\sigma_d$ much smaller (Unit 10, §10.3).

## 1.3.8 Units: nats, bits and perplexity

The literature uses three equivalent scales for the same average surprise:

| Quantity | Formula | Spine bigram (held-out) |
|---|---|---|
| loss in nats per character | $\hat L$ | 2.124 |
| bits per character | $\hat L/\ln 2$ | 3.065 |
| perplexity | $e^{\hat L}$ | 8.37 |

Perplexity translates log-loss back to an “effective number of choices”. A perplexity of 8.37 means that the model is, on average, as uncertain as a uniform choice among 8.37 possibilities. A uniform model over $V$ characters therefore has perplexity exactly $V$. Perplexity is often intuitive, but because it is exponential, modest differences in loss can look visually large. This book reports nats and uses perplexity as an interpretation aid.
