---
title: "D · Information theory"
description: Entropy, joint and conditional entropy and the chain rule; cross-entropy and the Kullback–Leibler divergence with Gibbs' inequality; mutual information and the data-processing inequality; differential entropy and the Gaussian; KL between Gaussians; coding and the meaning of a nat; forward versus reverse KL.
sidebar: { label: "D · Information theory", order: 4 }
---

The training loss of every language model in this book is a cross-entropy, its excess over the best achievable value is a KL divergence, the anchor that keeps post-training close to a pretrained model is a KL divergence, and the contrastive objective of Unit 07 bounds a mutual information. Unit 01 (§1.3.2, §1.3.8) introduces entropy, cross-entropy and KL for the bigram; this appendix gives the general definitions and the properties the later units use. All logarithms are natural and quantities are in **nats**; divide by $\ln 2$ for bits. Numbers in the examples are checked by `labs/test_appendices.py`.

| Section | Used in |
|---|---|
| D.1 Entropy, chain rule | 01 (loss floor), 07 (sequence models) |
| D.2 Cross-entropy, KL, Gibbs | 01, 03 (loss), 08 (ELBO), 09 (KL anchor) |
| D.3 Mutual information | 07 (InfoNCE), 11 (probes) |
| D.4–D.5 Differential entropy, Gaussian KL | 08 (VAEs, diffusion) |
| D.6 Coding | 01 (nats, bits, perplexity), 06 (compression) |
| D.7 Forward and reverse KL | 08 (sampling), 09 (post-training) |

## D.1 Entropy

The **entropy** of a discrete distribution $p$ is
$$
H(p) = -\sum_xp(x)\ln p(x) = \mathbb{E}_{x\sim p}[-\ln p(x)],
$$
the expected **surprisal** $-\ln p(x)$, with $0\ln0 = 0$.

**Proposition D.1.** $0\le H(p)\le\ln K$ for a distribution on $K$ outcomes, with $H = 0$ exactly for a point mass and $H = \ln K$ exactly for the uniform distribution.

*Proof.* Each term $-p\ln p\ge0$ for $p\in[0, 1]$, and all vanish only when one $p(x) = 1$. For the upper bound, $\ln K - H(p) = \mathrm{KL}(p\,\Vert\,u)\ge0$ with $u$ uniform (D.2). $\blacksquare$

**Example D.1.** $p = (0.5, 0.3, 0.15, 0.05)$ has $H = 1.142$ nats $= 1.648$ bits, below $\ln 4 = 1.386$ nats. It is the base distribution of Unit 08's temperature example: lowering the temperature lowers the entropy toward 0, raising it pushes the entropy toward $\ln 4$ (Unit 08, Worked Example 8.2).

**Joint and conditional entropy.** For a pair $(X, Y)$, $H(X, Y) = -\mathbb{E}\ln p(X, Y)$ and $H(Y\mid X) = -\mathbb{E}\ln p(Y\mid X) = \sum_xp(x)H(Y\mid X = x)$. Taking $-\mathbb{E}\ln$ of the product rule gives the **chain rule**
$$
H(X, Y) = H(X) + H(Y\mid X),\qquad H(X_1,\dots,X_T) = \sum_tH(X_t\mid X_{<t}).
$$
The second form is why a next-token model's per-token loss can be compared with an entropy *rate*: the best achievable average loss on a stationary source is $\lim_T\frac1TH(X_1,\dots,X_T)$. **Conditioning reduces entropy on average**, $H(Y\mid X)\le H(Y)$ (proof in D.3): more context can never make the optimal predictor worse in expectation, though it can make a *learned* one worse.

## D.2 Cross-entropy and the Kullback–Leibler divergence

The **cross-entropy** of a model $q$ relative to data $p$ is $H(p, q) = -\mathbb{E}_{x\sim p}\ln q(x)$ — the expected log-loss when the data follow $p$ and predictions follow $q$. The **Kullback–Leibler divergence** is the excess:
$$
\mathrm{KL}(p\,\Vert\,q) = \sum_xp(x)\ln\frac{p(x)}{q(x)} = H(p, q) - H(p).
$$
It is $+\infty$ if $q(x) = 0$ for some $x$ with $p(x) > 0$: a model that assigns zero probability to something that happens pays an infinite log-loss, which is why smoothing (Unit 01, §1.3.4) and softmax outputs with finite logits matter.

**Theorem D.2 (Gibbs' inequality).** $\mathrm{KL}(p\,\Vert\,q)\ge0$, with equality if and only if $p = q$.

*Proof.* Using $\ln t\le t - 1$, with equality only at $t = 1$, on the support of $p$:
$$
-\mathrm{KL}(p\,\Vert\,q) = \sum_xp(x)\ln\frac{q(x)}{p(x)}\le\sum_xp(x)\Big(\frac{q(x)}{p(x)} - 1\Big) = \sum_{x: p(x)>0}q(x) - 1\le0. \qquad\blacksquare
$$

So the cross-entropy is minimized, over all $q$, exactly at $q = p$, where it equals $H(p)$: **minimizing log-loss is maximum likelihood, and its floor is the data's entropy** (Unit 01, §1.3.2). Every reported training or held-out loss in the book is $H(p) + \mathrm{KL}(p\,\Vert\,q)$ estimated from a sample; only the second term can be reduced by a better model.

**KL is not a distance.** It is not symmetric and does not satisfy the triangle inequality.

**Example D.2.** For $p = (0.5, 0.5)$ and $q = (0.9, 0.1)$: $\mathrm{KL}(p\,\Vert\,q) = 0.5\ln\frac{0.5}{0.9} + 0.5\ln\frac{0.5}{0.1} = 0.511$, but $\mathrm{KL}(q\,\Vert\,p) = 0.9\ln1.8 + 0.1\ln0.2 = 0.368$. Underestimating an outcome that occurs half the time (assigning it 0.1) costs more than overestimating it.

**Convexity.** $\mathrm{KL}(p\,\Vert\,q)$ is jointly convex in $(p, q)$, and $H(p, q)$ is convex in $q$. For a softmax model $q = \mathrm{softmax}(z)$, the cross-entropy is convex in the logits $z$ (Appendix B, Proposition B.3), though not in the parameters of a deep network that produces them.

## D.3 Mutual information

The **mutual information** between $X$ and $Y$ is
$$
I(X; Y) = \mathrm{KL}\big(p(x, y)\,\Vert\,p(x)p(y)\big) = H(X) - H(X\mid Y) = H(Y) - H(Y\mid X).
$$
By Gibbs' inequality $I\ge0$, with equality exactly when $X$ and $Y$ are independent — which proves $H(X\mid Y)\le H(X)$.

**Example D.3.** Let $X$ be a fair bit and $Y$ a copy of it flipped with probability 0.2, so $p(x, y) = \begin{pmatrix}0.4 & 0.1\\0.1 & 0.4\end{pmatrix}$. Then $H(X) = \ln2 = 0.693$, $H(X\mid Y) = h(0.2) = -0.2\ln0.2 - 0.8\ln0.8 = 0.500$, and $I(X; Y) = 0.193$ nats: observing $Y$ removes 28% of the uncertainty about $X$.

**Theorem D.3 (data processing).** If $X\to Y\to Z$ is a Markov chain ($Z$ depends on $X$ only through $Y$), then $I(X; Z)\le I(X; Y)$.

*Proof.* By the chain rule for mutual information, $I(X; Y, Z) = I(X; Z) + I(X; Y\mid Z) = I(X; Y) + I(X; Z\mid Y)$. The Markov property gives $I(X; Z\mid Y) = 0$, and $I(X; Y\mid Z)\ge0$. $\blacksquare$

No processing of a representation can create information about the input that the representation did not contain. This is the information-theoretic form of Unit 02's observation that a representation can only discard (§2.2), and the reason a probe's accuracy bounds what a layer *contains* from below, never what the model *uses* (Unit 11, §11.3.6).

**Contrastive bounds.** Mutual information is hard to estimate in high dimensions. The InfoNCE objective with $K$ candidates gives the lower bound $I(X; Y)\ge\ln K - \mathcal L_{\text{InfoNCE}}$ (Unit 07, Proposition 7.1), which can never exceed $\ln K$ — so a contrastive estimate of a large mutual information needs a very large batch.

## D.4 Differential entropy and the Gaussian

For a density, the **differential entropy** is $h(p) = -\int p(x)\ln p(x)\,dx$. Unlike discrete entropy it can be negative and is not invariant under a change of variables: $h(aX) = h(X) + \ln|a|$. KL divergence and mutual information, by contrast, keep their meaning for densities and are invariant under invertible reparameterizations.

**Gaussian entropy.** For $N(\mu, \sigma^2)$, $h = \tfrac12\ln(2\pi e\sigma^2)$, which is $1.419$ nats for $\sigma = 1$; for $N(\mu, \Sigma)$ in $\mathbb{R}^n$, $h = \tfrac12\ln\big((2\pi e)^n\det\Sigma\big)$.

**Proposition D.4 (maximum entropy).** Among densities on $\mathbb{R}$ with variance $\sigma^2$, the Gaussian has the largest differential entropy.

*Proof.* For any such $p$ and the Gaussian $\phi$ with the same mean and variance, $0\le\mathrm{KL}(p\,\Vert\,\phi) = -h(p) - \int p\ln\phi$. Since $\ln\phi(x)$ is a quadratic in $x$, $\int p\ln\phi$ depends on $p$ only through its mean and variance, so it equals $\int\phi\ln\phi = -h(\phi)$. Hence $h(p)\le h(\phi)$. $\blacksquare$

The same argument with other constraints gives other maximum-entropy families; with constraints on expected features $\mathbb{E}[f(x)]$ it gives $p(x)\propto e^{\lambda^{\top}f(x)}$ — the exponential family, and the reason softmax outputs are natural (Appendix F.6).

## D.5 KL divergence between Gaussians

For one-dimensional Gaussians,
$$
\mathrm{KL}\big(N(\mu_1, \sigma_1^2)\,\Vert\,N(\mu_2, \sigma_2^2)\big) = \ln\frac{\sigma_2}{\sigma_1} + \frac{\sigma_1^2 + (\mu_1 - \mu_2)^2}{2\sigma_2^2} - \frac12 ,
$$
and for $n$-dimensional ones
$$
\mathrm{KL}\big(N(\mu_1, \Sigma_1)\,\Vert\,N(\mu_2, \Sigma_2)\big) = \tfrac12\Big[\mathrm{tr}(\Sigma_2^{-1}\Sigma_1) + (\mu_2 - \mu_1)^{\top}\Sigma_2^{-1}(\mu_2 - \mu_1) - n + \ln\frac{\det\Sigma_2}{\det\Sigma_1}\Big].
$$
*Derivation (1-D).* $\ln\frac{p_1}{p_2} = \ln\frac{\sigma_2}{\sigma_1} - \frac{(x - \mu_1)^2}{2\sigma_1^2} + \frac{(x - \mu_2)^2}{2\sigma_2^2}$; take the expectation under $p_1$, using $\mathbb{E}(x - \mu_1)^2 = \sigma_1^2$ and $\mathbb{E}(x - \mu_2)^2 = \sigma_1^2 + (\mu_1 - \mu_2)^2$. $\blacksquare$

**Example D.4.** $\mathrm{KL}(N(0, 1)\,\Vert\,N(1, 4)) = \ln2 + \frac{1 + 1}{8} - \frac12 = 0.443$, while $\mathrm{KL}(N(1, 4)\,\Vert\,N(0, 1)) = -\ln2 + \frac{4 + 1}{2} - \frac12 = 1.307$. A narrow distribution inside a broad one is cheap to describe with the broad one; the reverse is expensive.

The KL term of the variational autoencoder's ELBO, $\mathrm{KL}(N(\mu, \mathrm{diag}\,\sigma^2)\,\Vert\,N(0, I))$, is this formula with $\mu_2 = 0$, $\Sigma_2 = I$: $\tfrac12\sum_j(\sigma_j^2 + \mu_j^2 - 1 - \ln\sigma_j^2)$ (Unit 08, §8.3.3).

## D.6 Coding: what a nat means

A **prefix code** assigns codewords such that none is a prefix of another, so a stream of codewords can be decoded without separators. Codeword lengths $\ell(x)$ (in bits) of a binary prefix code satisfy the **Kraft inequality** $\sum_x2^{-\ell(x)}\le1$, and conversely any lengths satisfying it are achievable.

**Theorem D.5 (source coding, expected length).** Every prefix code for $X\sim p$ has expected length at least $H(p)/\ln2$ bits, and the lengths $\ell(x) = \lceil-\log_2p(x)\rceil$ achieve less than $H(p)/\ln2 + 1$.

*Proof of the lower bound.* Let $r(x) = 2^{-\ell(x)}/Z$ with $Z = \sum_x2^{-\ell(x)}\le1$. Then $\mathbb{E}[\ell]\ln 2 = \mathbb{E}[-\ln r(X)] - \ln Z\ge H(p, r)\ge H(p)$, by $\ln Z\le0$ and Gibbs' inequality. $\blacksquare$

**Example D.5.** For $p = (\tfrac12, \tfrac14, \tfrac18, \tfrac18)$ the code $0, 10, 110, 111$ has lengths $1, 2, 3, 3$, expected length 1.75 bits, equal to the entropy because every probability is a power of 2.

This is the operational meaning of a model's loss: with arithmetic coding, a model with cross-entropy $H(p, q)$ nats per token compresses the data to about $H(p, q)/\ln2$ bits per token. A language model is a compressor, and a better model is a better compressor. **Perplexity** $e^{H(p, q)}$ is the size of a uniform distribution with the same loss — "as uncertain as choosing among this many equally likely tokens" (Unit 01, §1.3.8).

## D.7 Forward and reverse KL

When a model $q$ from a restricted family is fitted to a target $p$, the direction of the divergence changes the answer — this is the precise sense of "mode covering" and "mode seeking".

- **Forward KL**, $\min_q\mathrm{KL}(p\,\Vert\,q)$: the integrand $p\ln(p/q)$ is huge wherever $p > 0$ and $q\approx0$, so $q$ must cover every region where $p$ has mass. Maximum-likelihood training minimizes forward KL from the data. Within the Gaussian family the minimizer matches the mean and variance of $p$ (the proof of Proposition D.4 shows that only these moments enter).
- **Reverse KL**, $\min_q\mathrm{KL}(q\,\Vert\,p)$: the integrand is weighted by $q$, so $q$ is penalized for putting mass where $p$ has little, and not penalized for ignoring regions of $p$. A restricted $q$ fits one mode well and drops the others.

**Example D.6.** Fit a single Gaussian $q$ to the mixture $p = \tfrac12N(-2, 0.5^2) + \tfrac12N(2, 0.5^2)$. Forward KL is minimized by moment matching, $q = N(0, 2.06^2)$, with $\mathrm{KL}(p\,\Vert\,q) = 0.72$; placing $q$ on one mode instead, $N(2, 0.5^2)$, gives $\mathrm{KL}(p\,\Vert\,q) = 15.3$. Reverse KL is the other way round: $\mathrm{KL}(q\,\Vert\,p) = 2.10$ for the moment-matched Gaussian, which puts mass between the modes where $p$ has almost none, and $0.69\approx\ln 2$ for the Gaussian on one mode.

**When the family is not restricted, both directions are minimized by $q = p$** and no mode is dropped. This is why the exact optimum of KL-regularized post-training keeps every output that the reference model allows; mode seeking enters only through an approximating policy family or incomplete optimization (Unit 09, §9.5.2).

## Exercises

**D.1** Compute the entropy of a fair six-sided die in nats and bits, and the perplexity of a model that predicts it uniformly.

**D.2** Show that $H(X, Y)\le H(X) + H(Y)$, with equality if and only if $X$ and $Y$ are independent.

**D.3** For $p = (0.7, 0.2, 0.1)$, find the cross-entropy of the uniform predictor and the KL divergence it incurs.

**D.4** Show that $\mathrm{KL}(p\,\Vert\,q)\ge\frac12\lVert p - q\rVert_1^2$ (Pinsker's inequality) in the two-point case $p = (a, 1-a)$, $q = (b, 1-b)$.

**D.5** Verify the 1-D Gaussian KL formula numerically for $\mu_1 = 0, \sigma_1 = 1, \mu_2 = 1, \sigma_2 = 2$.

**D.6** ★ Show that for a categorical model trained by maximum likelihood on $n$ i.i.d. samples, the expected training loss is below $H(p)$ and the expected held-out loss is above it, and estimate both gaps to first order in $1/n$.

**Answers.** **D.1** $\ln6 = 1.792$ nats $= 2.585$ bits; perplexity 6. **D.3** $H(p, u) = \ln3 = 1.099$; $H(p) = 0.802$; $\mathrm{KL} = 0.297$. **D.6** Both gaps are about $(K - 1)/(2n)$ nats for $K$ outcomes with all probabilities bounded away from 0.

## Further reading

[Cover & Thomas 2006](/references/#cover-2006) is the standard text; [MacKay 2003](/references/#mackay-2003) connects information theory, inference and learning; [Bishop & Bishop 2024](/references/#bishop-2024), §2.5, for the machine-learning view.
