---
title: "1.3 Mathematical construction"
description: Population risk and empirical risk, cross-entropy as entropy plus KL with Gibbs' inequality, deriving the MLE bigram with Lagrange multipliers, smoothing as a Dirichlet posterior mean, Hoeffding's inequality, the cost of selection, and standard errors of held-out loss.
sidebar: { label: "1.3 Construction", order: 3 }
---

Section 1.2 established intuitive foundations; this section formalizes them into the rigorous mathematical architecture of modern statistical learning. Each subsection states a fundamental result, provides its complete step-by-step derivation, and analyzes its deep implications for language modeling and deep learning.

## 1.3.1 Setting: distributions, predictors and risk

In the physical world, data is not generated in a vacuum; it originates from an unknown **data-generating distribution** $p_{\text{data}}$.

We observe paired samples $(x, y)$ drawn i.i.d. from $p_{\text{data}}$:
- In language modeling, $x$ represents the observed context (for a bigram model, the single preceding character);
- $y$ represents the true target that follows in reality (the next character).

**Predictor / Model:** A model $q$ is a parameterized mapping that takes any context $x$ and outputs a valid conditional probability distribution $q(\cdot\mid x)$ over all candidate symbols in vocabulary $V$. For a single observation $(x, y)$, the loss incurred by the model is measured by the log-loss: $\ell(q; x, y) = -\ln q(y\mid x)$.

### Population risk: expected loss across the unseen ocean
In empirical science, we care not about performance on the handful of training tokens at hand, but about the model's **expected performance across all unobserved future samples**. This is defined as the **population risk** (or true generalization risk):

$$
L(q) = \mathbb{E}_{(x,y)\sim p_{\text{data}}}\big[-\ln q(y\mid x)\big].
$$

However, population risk is an idealized quantity that can never be computed directly in practice: the underlying distribution $p_{\text{data}}$ is an infinite ocean whose closed-form mathematical expression is forever inaccessible.

### Empirical risk: a single bucket drawn from the ocean
What we can compute is the average loss on a finite bucket of $n$ samples drawn from the ocean, $\mathcal{D} = \{(x_i, y_i)\}_{i=1}^n$. This is the **empirical risk**:

$$
\hat L(q) = \frac{1}{n}\sum_{i=1}^n -\ln q(y_i\mid x_i).
$$

A fundamental statistical symmetry governs this quantity:
- **If predictor $q$ was fixed before observing dataset $\mathcal{D}$ (e.g., an independent test set):** By linearity of expectation, the expectation of the empirical risk exactly equals the population risk: $\mathbb{E}[\hat L(q)] = L(q)$. Here, $\hat L(q)$ is an **unbiased estimator** of $L(q)$. This is the mathematical justification for held-out evaluation.
- **If predictor $q$ is chosen by minimizing $\hat L(q)$ itself (the training set):** This unbiasedness collapses! Because model parameters adapt dynamically to sample noise, $\hat L(q)$ is systematically optimistic relative to true risk $L(q)$.

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

This identity has profound physical and information-theoretic meaning:
1. **Shannon Entropy $H(p)$:** Quantifies the irreducible intrinsic randomness of the data generator. Even an omniscient intelligence cannot achieve expected loss below $H(p)$, because multiple valid characters can follow the same prefix in natural language.
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
- **Reducible Error:** The distance between our model representation and the ground truth. **Training neural networks with cross-entropy is literally the minimization of this KL divergence across the data manifold.**

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

Setting the partial derivative with respect to each $q_b$ to zero:
$\frac{\partial \mathcal{L}}{\partial q_b} = -\frac{c_b}{q_b} + \lambda = 0 \quad\implies\quad q_b = \frac{c_b}{\lambda}$.

Enforcing the normalization constraint $\sum_b q_b = 1$:
$\sum_b \frac{c_b}{\lambda} = \frac{1}{\lambda}\sum_b c_b = 1 \quad\implies\quad \lambda = \sum_b c_b = c(a)$.

Substituting $\lambda$ back yields the unique stationary point: $q_b = \frac{c_b}{c(a)}$.
The Hessian with respect to $q$ is diagonal with positive entries $\partial^2\mathcal{L}/\partial q_b^2 = c_b / q_b^2 > 0$. Being strictly positive-definite, the objective is strictly convex, confirming that the stationary point is the unique global minimum. $\blacksquare$

*Alternative proof via information geometry.* Let $\hat p_b = c_b/c(a)$ be the empirical distribution.
Then the empirical loss rewrites directly as: $-\sum_b c_b\ln q_b = c(a)\,H(\hat p, q) = c(a)\,[H(\hat p) + \mathrm{KL}(\hat p\,\Vert\,q)]$.
By Gibbs' inequality, this is minimized if and only if $q = \hat p$.
**Maximum likelihood estimation is the exact minimization of the KL divergence from the empirical distribution to the parametric model.**

This yields a universal benchmark: **no bigram model—whether count-based, smoothed, or parameterized as a deep neural network—can achieve a training loss lower than the maximum-likelihood table on that same training corpus.** In Unit 03, the gradient-trained neural bigram converges to this floor from above and never dips below it.

## 1.3.4 Smoothing as Bayesian inference

Why is add-$\alpha$ smoothing not an ad-hoc heuristic to avoid division by zero, but a principled inference method? The answer lies in Bayesian statistics: **smoothing is the exact Bayes-optimal predictive distribution under a symmetric conjugate prior.**

### The probability simplex and the Dirichlet prior
While maximum likelihood treats transition probabilities as fixed but unknown constants, Bayesian inference treats each row of the bigram table $\theta = (\theta_1,\dots,\theta_V)$ (where $\theta_b \ge 0$ and $\sum_b \theta_b = 1$) as a random vector over the **probability simplex** $\Delta^{V-1}$.

Prior to observing counts, an agent specifies an epistemic prior over $\theta$. For multinomial/categorical distributions, the natural conjugate prior is the **symmetric Dirichlet distribution**:

$$
\pi(\theta) \propto \prod_b \theta_b^{\alpha-1}, \qquad \theta_b\ge 0,\ \textstyle\sum_b\theta_b = 1 .
$$

The hyperparameter $\alpha > 0$ functions as the virtual pseudo-count strength:
- At $\alpha = 1$, the density is uniform across the simplex, representing an uninformative **uniform prior**;
- At $\alpha > 1$, density concentrates at the center ($\theta_b \approx 1/V$), expressing a strong preference for uniform distributions;
- At $\alpha < 1$, density pushes toward the simplex boundaries and corners, expressing a prior belief in sparsity.

### Conjugacy and belief updating
Given antecedent $a$, the likelihood of observing counts $c = (c_1, \dots, c_V)$ is proportional to the multinomial product $\prod_b \theta_b^{c_b}$.

By Bayes' theorem, the posterior density is proportional to the prior times the likelihood:
$\pi(\theta \mid c) \propto \pi(\theta) \cdot p(c \mid \theta) \propto \Big(\prod_b \theta_b^{\alpha-1}\Big) \cdot \Big(\prod_b \theta_b^{c_b}\Big)$.

Combining powers yields:

$$
\pi(\theta\mid c) \propto \prod_b \theta_b^{c_b + \alpha - 1}.
$$

Notice that **the posterior distribution belongs to the exact same functional family as the prior**, with parameters updated from $\alpha$ to $\beta_b = c_b + \alpha$. This algebraic property is called **conjugacy**. Under conjugate priors, statistical inference simplifies to basic addition of real counts to pseudo-counts.

### Posterior mean and Bayes-optimal prediction
Under the log-loss, what prediction should the agent make for the next token given posterior distribution $\pi(\theta \mid c)$?

Point-estimate MAP (maximum a posteriori) at $\alpha = 1$ collapses back to un-smoothed MLE, failing to rescue zero counts.
In contrast, Bayesian decision theory proves that the optimal decision under log-loss is the **posterior mean**:

$$
\mathbb{E}[\theta_b \mid c] = \frac{c_b + \alpha}{c(a) + \alpha V},
$$

which is **algebraically identical** to add-$\alpha$ smoothing!
Smoothing is the exact Bayes-optimal prediction for a rational learner with a symmetric Dirichlet prior under the log-loss.

## 1.3.5 How far can empirical loss deviate from true risk?

When evaluating a fixed model $q$, the held-out loss $\hat L(q)$ is the average of $n$ independent bounded loss variables. With what statistical confidence can we assert that $\hat L(q)$ is close to true population risk $L(q)$?

**Theorem 1.2 (Hoeffding's inequality, [Hoeffding 1963](/references/#hoeffding-1963)).** If $Z_1,\dots,Z_n$ are independent random variables strictly bounded such that $Z_i\in[a,b]$, then for every tolerance $\varepsilon > 0$:

$$
\Pr\Big(\Big|\tfrac{1}{n}\textstyle\sum_i Z_i - \mathbb{E}\big[\tfrac1n\sum_i Z_i\big]\Big| \ge \varepsilon\Big) \le 2\exp\!\Big(-\frac{2n\varepsilon^2}{(b-a)^2}\Big).
$$

(Proof in §1.5.1 ★.)
Setting the right-hand failure probability to significance level $\delta$ and solving for $\varepsilon$: with probability at least $1 - \delta$, the true population risk is bounded within the confidence interval:

$$
|\hat L(q) - L(q)| \le (b-a)\sqrt{\frac{\ln(2/\delta)}{2n}} .
$$

This bound establishes the **fundamental rate of statistical convergence**:
Uncertainty contracts at rate $O(1/\sqrt{n})$. **To halve evaluation uncertainty, an empirical benchmark must expand its test data fourfold.**

## 1.3.6 Model selection requires fresh data: the cost of search

Hoeffding's bound holds strictly for a single model fixed *prior* to seeing evaluation data.

Suppose you evaluate $K$ candidate models $q_1, \dots, q_K$ on the same validation set and pick the one with the lowest score.
While each individual model has only a small $\delta/K$ probability of deviating favorably by chance, the **union bound** ($\Pr(\cup_k A_k)\le\sum_k\Pr(A_k)$) shows that across $K$ candidates, the probability of selecting an artificially lucky model escalates.
To guarantee that all $K$ estimates remain simultaneously accurate with probability at least $1 - \delta$, the tolerance must widen:

$$
\varepsilon_K = (b-a)\sqrt{\frac{\ln K + \ln(2/\delta)}{2n}}
$$

This is the **cost of selection**. Every hyperparameter configuration tested introduces optimistic selection bias. When exploring thousands of architectures, the best validation score inevitably overfits validation noise. **Only an untouched, strictly quarantined test set evaluated once at the very end provides an honest, unbiased assessment of generalization.**

## 1.3.7 Standard error of held-out loss

Hoeffding's bound provides a conservative worst-case guarantee. In scientific practice, researchers report the more informative **standard error (SE)**.

If individual sample losses $Z_i$ are independent with variance $\sigma^2$, the Central Limit Theorem dictates that the sampling variance of the mean is $\sigma^2/n$:

$$
\mathrm{se}(\hat L) = \frac{\sigma}{\sqrt n},
$$

yielding an asymptotic 95% confidence interval of $\hat L \pm 1.96\,\mathrm{se}$.

**The Independence Trap:** Adjacent character losses within a story exhibit strong serial correlation (empirical autocorrelation $-0.32$). Standard errors must be computed by treating **complete stories** as the independent resampling units (Worked Example 1.7).

**Power and sample size planning:** To distinguish between two models whose mean loss differs by $d$ nats at a 5% significance level with 80% statistical power, the required paired sample size is approximately:

$$
n \approx \Big(\frac{(1.96 + 0.84)\,\sigma_d}{d}\Big)^2
$$

where $\sigma_d$ is the standard deviation of the paired difference in sample losses. Evaluating both models on identical samples subtracts out shared instance difficulty, dramatically reducing $\sigma_d$.

## 1.3.8 Units: nats, bits and perplexity

The quality of language models is reported in three equivalent metrics across the literature:

| Metric | Mathematical Definition | Spine Bigram (Held-Out) | Intuitive Physical Meaning |
|---|---|---|---|
| **Negative log-likelihood (nats)** | $\hat L$ | 2.124 nats/char | Mean surprisal in natural logarithms |
| **Bits per character (bpc)** | $\hat L / \ln 2$ | 3.065 bits/char | Shannon coding length under optimal binary compression |
| **Perplexity (PPL)** | $e^{\hat L}$ | 8.37 | Effective branching factor: equivalent to guessing uniformly among 8.37 options |

A perplexity of 8.37 means that when predicting the next token, the model is on average as uncertain as choosing uniformly from an 8.37-sided fair die.
