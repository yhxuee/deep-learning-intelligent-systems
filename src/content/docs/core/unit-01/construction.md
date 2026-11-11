---
title: "1.3 Mathematical construction"
description: Population risk and empirical risk, cross-entropy as entropy plus KL with Gibbs' inequality, deriving the MLE bigram with Lagrange multipliers, smoothing as a Dirichlet posterior mean, Hoeffding's inequality, the cost of selection, and standard errors of held-out loss.
sidebar: { label: "1.3 Construction", order: 3 }
---

Section 1.2 established intuitive foundations; this section formalizes them into the rigorous mathematical framework of modern statistical learning. Each subsection states a fundamental result, provides its complete step-by-step derivation, and analyzes its deep implications for language modeling and deep learning.

## 1.3.1 Setting: distributions, predictors and risk

In the physical world, data is not generated in a vacuum; it originates from an unknown **data-generating distribution** $p_{\text{data}}$.

We observe paired samples $(x, y)$ drawn i.i.d. from $p_{\text{data}}$:
- In language modeling, $x$ represents the observed context (for a bigram model, the single preceding character);
- $y$ represents the true target that follows in reality (the next character).

**Predictor / Model:** A model $q$ is a parameterized mapping that takes any context $x$ and outputs a valid conditional probability distribution $q(\cdot\mid x)$ over all candidate symbols in vocabulary $V$. For a single observation $(x, y)$, the loss incurred by the model is measured by the log-loss: $\ell(q; x, y) = -\ln q(y\mid x)$.

### Population risk: expected loss over unseen data
In empirical science, we care not about performance on the handful of training tokens at hand, but about the model's **expected performance across all unobserved future samples**. This is defined as the **population risk** (or true generalization risk):

$$
L(q) = \mathbb{E}_{(x,y)\sim p_{\text{data}}}\big[-\ln q(y\mid x)\big].
$$

However, population risk is an idealized theoretical quantity that cannot be computed directly in practice, because the underlying distribution $p_{\text{data}}$ is inaccessible in closed form.

### Empirical risk: average loss on finite samples
What we can compute is the average loss on a finite dataset of $n$ samples drawn from the distribution, $\mathcal{D} = \{(x_i, y_i)\}_{i=1}^n$. This is the **empirical risk**:

$$
\hat L(q) = \frac{1}{n}\sum_{i=1}^n -\ln q(y_i\mid x_i).
$$

A fundamental statistical symmetry governs this quantity:
- **If predictor $q$ was fixed before observing dataset $\mathcal{D}$ (e.g., an independent test set):** By linearity of expectation, the expectation of the empirical risk exactly equals the population risk: $\mathbb{E}[\hat L(q)] = L(q)$. Here, $\hat L(q)$ is an **unbiased estimator** of $L(q)$. This is the mathematical justification for held-out evaluation.
- **If predictor $q$ is chosen by minimizing $\hat L(q)$ itself (the training set):** This unbiasedness collapses. Because model parameters adapt dynamically to sample noise, $\hat L(q)$ is systematically optimistic relative to true risk $L(q)$.

**Empirical Risk Minimization (ERM):** Selecting the model from a hypothesis class $\mathcal{Q}$ that minimizes training loss: $\hat{q} = \arg\min_{q \in \mathcal{Q}} \hat{L}(q)$. Statistical learning theory asks: **Under what conditions does the ERM minimizer $\hat{q}$ guarantee a low population risk $L(\hat{q})$?**

## 1.3.2 Cross-entropy, entropy and KL divergence

To isolate the loss structure under a single context, fix context $x$ and suppress it in notation. Let $p$ denote the true objective target distribution, and let $q$ denote the model's predictive distribution.

The expected log-loss of model $q$ under true distribution $p$ is defined as the **cross-entropy**:

$$
H(p, q) = -\sum_{y} p(y)\ln q(y).
$$

Adding and subtracting the expected self-information of the true distribution, $\sum_y p(y)\ln p(y)$, yields the fundamental decomposition:

$$
H(p, q) = \underbrace{-\sum_y p(y)\ln p(y)}_{H(p)\ \text{(entropy)}} \;+\; \underbrace{\sum_y p(y)\ln\frac{p(y)}{q(y)}}_{\mathrm{KL}(p\,\Vert\, q)\ \text{(KL divergence)}} .
$$

This decomposition has a clear information-theoretic interpretation:
1. **Shannon Entropy $H(p)$:** Quantifies the irreducible intrinsic randomness of the data generator. Even an omniscient model cannot achieve expected loss below $H(p)$, because multiple valid characters can follow the same prefix in natural language.
2. **Kullback–Leibler Divergence $\mathrm{KL}(p\,\Vert\,q)$:** Quantifies the excess informational penalty incurred because our model beliefs $q$ deviate from nature's truth $p$.

**Theorem 1.1 (Gibbs' inequality).** For any two discrete probability distributions $p$ and $q$, the relative entropy is non-negative: $\mathrm{KL}(p\,\Vert\,q)\ge 0$, with equality if and only if $q = p$.

*Proof.* We leverage the strict concavity of the logarithm. Since $f(t) = \ln t$ has a strictly negative second derivative $f''(t) = -1/t^2 < 0$, Jensen's inequality guarantees that for any positive random variable $Z$, $\mathbb{E}_p[\ln Z] \le \ln \mathbb{E}_p[Z]$.

Define the auxiliary random variable $Z(y) = \frac{q(y)}{p(y)}$ restricted to the support $p(y) > 0$:

$$
-\mathrm{KL}(p\,\Vert\,q) = \sum_{y:\,p(y)>0} p(y)\ln\frac{q(y)}{p(y)} \le \ln\sum_{y:\,p(y)>0} p(y)\frac{q(y)}{p(y)} = \ln\sum_{y:\,p(y)>0} q(y) \le \ln 1 = 0 .
$$

Multiplying by $-1$ yields $\mathrm{KL}(p\,\Vert\,q) \ge 0$.
Equality in Jensen's inequality requires $Z = q(y)/p(y)$ to be almost surely constant, while equality in the final step $\sum_{p(y)>0} q(y) \le 1$ requires $q$ to place zero probability mass outside the support of $p$. Combining both conditions forces the constant to be 1, implying $q(y) = p(y)$ for all $y$. $\blacksquare$

**Corollary 1.1.** Expected cross-entropy loss $H(p, q)$ achieves its unique global minimum at $q = p$, with minimal value equal to entropy $H(p)$. Over the full data distribution, the population risk decomposes into:

$$
L(q) = \underbrace{\mathbb{E}_x\big[H(p(\cdot\mid x))\big]}_{\text{irreducible}} + \underbrace{\mathbb{E}_x\big[\mathrm{KL}\big(p(\cdot\mid x)\,\Vert\, q(\cdot\mid x)\big)\big]}_{\text{reducible}} .
$$

This establishes the fundamental lower bound across this entire book:
- **Irreducible Loss:** The inherent entropy of the conditional data source, identified as the asymptotic floor $E$ in the scaling laws of Unit 06;
- **Reducible Error:** The distance between our model representation and the ground truth. **Training neural networks with cross-entropy corresponds to minimizing this expected KL divergence across contexts.**

## 1.3.3 The maximum-likelihood bigram

The hypothesis class of bigram models consists of all $V \times V$ transition tables satisfying $q(b\mid a)\ge 0$ and the row-stochastic constraint $\sum_b q(b\mid a) = 1$.

Across a training sequence $x_1,\dots,x_n$, total training log-loss is grouped by transition counts:

$$
(n-1)\hat L(q) = -\sum_{t=2}^{n}\ln q(x_t\mid x_{t-1}) = -\sum_{a}\sum_{b} c(a\to b)\ln q(b\mid a),
$$

where $c(a\to b)$ counts transitions from $a$ to $b$. Notice that this summation decouples across distinct antecedent characters $a$, allowing us to solve $V$ independent optimization problems row by row.

**Proposition 1.1.** For each antecedent character $a$ with $c(a) > 0$, the unique conditional distribution that minimizes $-\sum_b c(a\to b)\ln q(b\mid a)$ subject to $\sum_b q(b\mid a) = 1$ is the empirical frequency $q(b\mid a) = c(a\to b)/c(a)$.

*Proof via Lagrange multipliers.* Let $c_b = c(a\to b)$ and $q_b = q(b\mid a)$. We construct the Lagrangian:
$\mathcal{L}(q, \lambda) = -\sum_b c_b \ln q_b + \lambda\Big(\sum_b q_b - 1\Big)$.

Differentiating with respect to $q_b$ and setting to zero:
$\frac{\partial \mathcal{L}}{\partial q_b} = -\frac{c_b}{q_b} + \lambda = 0 \quad\implies\quad q_b = \frac{c_b}{\lambda}$.

Substituting into the normalization constraint $\sum_b q_b = 1$:
$\sum_b \frac{c_b}{\lambda} = \frac{1}{\lambda}\sum_b c_b = 1 \quad\implies\quad \lambda = \sum_b c_b = c(a)$.

Substituting $\lambda$ back yields the unique stationary point: $q_b = \frac{c_b}{c(a)}$.
The Hessian of $\mathcal{L}$ with respect to $q$ has diagonal entries $\partial^2\mathcal{L}/\partial q_b^2 = c_b / q_b^2 > 0$ and zero off-diagonals. This positive-definite diagonal Hessian confirms strict convexity, ensuring the stationary point is the unique global minimum. $\blacksquare$

*Alternative proof via information geometry.* Let $\hat p_b = c_b/c(a)$ be the empirical distribution.
Then the empirical loss rewrites directly as: $-\sum_b c_b\ln q_b = c(a)\,H(\hat p, q) = c(a)\,[H(\hat p) + \mathrm{KL}(\hat p\,\Vert\,q)]$.
By Gibbs' inequality, $\mathrm{KL}(\hat p \Vert q) \ge 0$, minimized if and only if $q = \hat p$.
**Maximum likelihood estimation minimizes the KL divergence from empirical counts to model probabilities.**

This establishes an important benchmark: **No bigram model can achieve lower training loss on a given text than that text's empirical frequency table.** In Unit 03, the neural bigram trained with gradient descent will converge to this exact analytical limit from above.

## 1.3.4 Smoothing as Bayesian inference

Add-$\alpha$ smoothing is not merely an ad-hoc heuristic for handling zero counts; it has a rigorous Bayesian foundation: **smoothing is the posterior predictive distribution under a symmetric conjugate prior.**

### The probability simplex and Dirichlet prior
In frequentist MLE, conditional probabilities are treated as fixed, unknown parameters. In Bayesian statistics, each row $\theta = (\theta_1,\dots,\theta_V)$ of the bigram table (satisfying $\theta_b \ge 0$ and $\sum_b \theta_b = 1$) is treated as a random vector defined on the **probability simplex** $\Delta^{V-1}$.

Before observing data, a Bayesian agent places a prior density over $\theta$. For multinomial observations, the natural conjugate prior is the **symmetric Dirichlet distribution**:

$$
\pi(\theta) \propto \prod_b \theta_b^{\alpha-1}, \qquad \theta_b\ge 0,\ \textstyle\sum_b\theta_b = 1 .
$$

The hyperparameter $\alpha > 0$ reflects prior strength:
- For $\alpha = 1$, exponents vanish, yielding a uniform distribution over the simplex (total ignorance);
- For $\alpha > 1$, probability mass concentrates at the center ($\theta_b \approx 1/V$), favoring uniform distributions;
- For $\alpha < 1$, mass pushes toward the boundaries and vertices (sparse distributions).

### Conjugacy and belief updating
Given antecedent $a$, observed transition counts are $c = (c_1, \dots, c_V)$, with multinomial likelihood $\prod_b \theta_b^{c_b}$.

By Bayes' theorem, the posterior density is proportional to the product of prior and likelihood:
$\pi(\theta \mid c) \propto \pi(\theta) \cdot p(c \mid \theta) \propto \Big(\prod_b \theta_b^{\alpha-1}\Big) \cdot \Big(\prod_b \theta_b^{c_b}\Big)$.

Combining exponents yields the posterior distribution:

$$
\pi(\theta\mid c) \propto \prod_b \theta_b^{c_b + \alpha - 1}.
$$

The posterior belongs to the exact same Dirichlet family with updated parameters $\beta_b = c_b + \alpha$. This property is **conjugacy**. Under conjugate priors, Bayesian updating simplifies cleanly to adding prior pseudo-counts to observed counts.

### Posterior mean and Bayes-optimal prediction
Under log-loss, how should the full posterior $\pi(\theta \mid c)$ be converted into a point prediction?

Using the maximum a posteriori (MAP) point estimate under $\alpha = 1$ collapses back to raw MLE, leaving the zero-count issue unresolved.
The principled Bayesian decision is to integrate over all parameter vectors weighted by their posterior density—the **posterior predictive mean**:

$$
\mathbb{E}[\theta_b \mid c] = \frac{c_b + \alpha}{c(a) + \alpha V},
$$

which is **algebraically identical** to add-$\alpha$ smoothing.
Smoothing is the unique Bayes-optimal strategy that minimizes expected log-loss under a symmetric Dirichlet prior.

## 1.3.5 How far can empirical loss stray from true risk?

When evaluating a fixed model $q$, the empirical loss $\hat L(q)$ on a held-out set is the average of $n$ independent sample losses. With what statistical confidence can we guarantee that $\hat L(q)$ approximates true population risk $L(q)$?

**Theorem 1.2 (Hoeffding's inequality, [Hoeffding 1963](/references/#hoeffding-1963)).** Let $Z_1,\dots,Z_n$ be independent bounded random variables with $Z_i \in [a, b]$. For any tolerance $\varepsilon > 0$:

$$
\Pr\Big(\Big|\tfrac{1}{n}\textstyle\sum_i Z_i - \mathbb{E}\big[\tfrac1n\sum_i Z_i\big]\Big| \ge \varepsilon\Big) \le 2\exp\!\Big(-\frac{2n\varepsilon^2}{(b-a)^2}\Big).
$$

(Full proof in §1.5.1 ★.)
Setting the right-hand side to a significance level $\delta$ and solving for $\varepsilon$ guarantees that with probability at least $1-\delta$:

$$
|\hat L(q) - L(q)| \le (b-a)\sqrt{\frac{\ln(2/\delta)}{2n}} .
$$

This bound reveals a fundamental scaling law of statistical verification:
Uncertainty shrinks as $1/\sqrt{n}$. **To halve the width of an evaluation error bound, held-out data must be quadrupled.**

## 1.3.6 Model selection requires fresh data

Hoeffding's inequality applies exclusively to a single model frozen prior to evaluation.

Suppose you evaluate $K$ candidate models $q_1, \dots, q_K$ on the same validation set (e.g., searching across 1,000 hyperparameter configurations) and pick the one with the lowest empirical loss.
While the probability of any single model appearing artificially strong due to noise is small, **the union bound ($\Pr(\cup_k A_k)\le\sum_k\Pr(A_k)$)** implies the probability of at least one model getting lucky grows with $K$.
To ensure all $K$ estimates hold simultaneously with probability $1-\delta$, the bound incurs a logarithmic penalty:

$$
\varepsilon_K = (b-a)\sqrt{\frac{\ln K + \ln(2/\delta)}{2n}}
$$

This is the **cost of selection**. Every hyperparameter evaluated introduces optimistic bias. When thousands of trials are run, the best validation score largely fits validation noise. **Only an isolated test set evaluated strictly once provides an unbiased estimate of generalization.**

## 1.3.7 Standard error of held-out loss

Hoeffding's bound provides a distribution-free worst-case guarantee. In empirical practice, researchers report the tighter asymptotic **standard error (SE)**.

If individual sample losses $Z_i$ are independent with finite variance $\sigma^2$, the central limit theorem yields the standard error:

$$
\mathrm{se}(\hat L) = \frac{\sigma}{\sqrt n},
$$

with an asymptotic 95% confidence interval given by $\hat L \pm 1.96\,\mathrm{se}$.

**The independence trap:** Character losses within a single story are strongly correlated (empirical autocorrelation is $-0.32$). Valid statistical inference requires resampling over **independent complete stories** (Worked Example 1.7).

**Sample size planning:** To distinguish two competing models with true mean loss difference $d$ nats at a 5% significance level with 80% statistical power, the required number of paired test samples is approximately:

$$
n \approx \Big(\frac{(1.96 + 0.84)\,\sigma_d}{d}\Big)^2
$$

where $\sigma_d$ is the standard deviation of paired sample differences. Paired testing cancels common narrative difficulty across texts, making $\sigma_d$ substantially smaller than individual model variance.

## 1.3.8 Units: nats, bits and perplexity

In literature, language model performance is reported in three mutually equivalent units:

| Unit | Mathematical Definition | Spine Bigram Held-out | Practical Meaning |
|---|---|---|---|
| **Negative Log-Likelihood (nats)** | $\hat L$ | 2.124 nats/char | Mean surprisal in natural logarithms |
| **Bits per character** | $\hat L / \ln 2$ | 3.065 bits/char | Theoretical minimum code length per character |
| **Perplexity (PPL)** | $e^{\hat L}$ | 8.37 | Effective branching factor: equivalent to guessing uniformly among this many characters |

A perplexity of 8.37 indicates that on average, the model's uncertainty when predicting the next character is equivalent to guessing uniformly among 8.37 candidates. An omniscient model achieves a perplexity of 1; the uniform baseline has a perplexity equal to the vocabulary size $V = 50$.
