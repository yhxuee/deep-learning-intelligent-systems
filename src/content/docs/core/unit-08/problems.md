---
title: "8.10 Problems and further study"
description: Twenty-two graded problems build from hand sampling to ELBO and score proofs, flow-matching continuity, numerical solvers, speculative decoding, guidance, and discrete diffusion, with staged hints and answer checks.
sidebar: { label: "8.10 Problems", order: 10 }
---

Difficulty: ★ one main step · ★★ combines ideas · ★★★ proof, implementation, or open-ended design. **[AI-off]** marks independent practice or assessment.

Use the page in three passes:

1. attempt a problem without scrolling below the reading list;
2. if blocked for ten minutes, read only its **first move** in the Hints section;
3. compare with the numerical or structural answer check only after completing a derivation.

A correct answer should name the distribution being manipulated, not only produce a number.

## Computation

**8.1 ★ [AI-off] Inverse CDF.** Use the half-open interval convention $[c_{k-1},c_k)$. For probabilities $(0.4,0.3,0.2,0.1)$ and $u=0.65$, which symbol is returned? For which interval of $u$ is the third symbol returned? What changes at exact boundaries if the implementation instead counts cumulative probabilities strictly less than $u$?

**8.2 ★ [AI-off] Temperature and units.** Compute the $\tau=0.5$ and $\tau=2$ distributions obtained from $(0.4,0.3,0.2,0.1)$. Compute each entropy in nats, then convert it to bits.

**8.3 ★ Nucleus.** For the same sorted distribution, find the smallest nucleus for $p=0.75$ and $p=0.95$. Renormalize the retained probabilities. Explain why retained mass need not equal the threshold.

**8.4 ★ Change of variables.** Let $X=2Z+3$ with $Z\sim N(0,1)$. Derive $p_X(x)$, evaluate it at $x=3$, and verify by substitution that it integrates to one. What value would the incorrect no-Jacobian density integrate to?

**8.5 ★★ Gaussian velocity.** For $\mu=(1,-2)$, $\sigma=0.5$, $t=0.5$, evaluate the exact field of §8.3.6 at $x=(1,-2)$. Then use the analytic trajectory to find the endpoint of a particle whose initial coordinate is $x_0=(0,0)$.

**8.6 ★★ Two Euler steps.** For the same Gaussian field, take two Euler steps of size $h=1/2$ starting from arbitrary $x_0$. Derive the final affine map $x_2=A x_0+b$ and hence the endpoint standard deviation when $x_0\sim N(0,I)$.

## Derivation

**8.7 ★ [AI-off] Exact ancestral sampling.** Prove Proposition 8.1 for a fixed length. Extend the statement to variable length by including EOS, and state one assumption needed for the sequence to terminate with probability one.

**8.8 ★★ [AI-off] ELBO two ways.** Derive the ELBO first with Jensen's inequality, then derive the exact decomposition

$$
\ln p_\theta(x)
=
\mathcal L_{\mathrm{ELBO}}(x)
+
\mathrm{KL}
\left(q_\phi(z\mid x)\,\Vert\,p_\theta(z\mid x)\right).
$$

Explain why the first derivation proves a bound while the second identifies its gap.

**8.9 ★★ Gaussian KL.** Starting from the definition of KL, derive

$$
\mathrm{KL}
\left(
N(\mu,\operatorname{diag}(\sigma^2))
\;\Vert\;
N(0,I)
\right)
=
\frac12\sum_i
\left(
\mu_i^2+\sigma_i^2-1-\ln\sigma_i^2
\right).
$$

Check that the expression is non-negative and identify every equality case.

**8.10 ★★ Exact Gaussian field.** Let $X_0\sim N(0,I)$ and $X_1\sim N(\mu,\sigma^2I)$ independently. Derive $\mathbb E[X_1-X_0\mid X_t=x]$ from the conditional-mean formula for jointly Gaussian vectors. Then verify directly that $x(t)=t\mu+s_t x_0$ solves the resulting ODE.

**8.11 ★★ Irreducible flow-matching loss.** Let $V=X_1-X_0$ and $M=\mathbb E[V\mid X_t,t]$. Prove

$$
\min_f
\mathbb E\|V-f(X_t,t)\|^2
=
\mathbb E\,
\operatorname{tr}
\operatorname{Cov}(V\mid X_t,t).
$$

State the exact condition for this minimum to be zero. Give one example in which geometric paths meet but the loss can still be zero, showing why “paths never cross” is only a sufficient intuition when crossings have different velocities, not the theorem.

## Proof and counterexample

**8.12 ★★★ Exact speculative sampling.** Let $q$ be a target categorical distribution and $r$ a draft. Prove that accepting $X\sim r$ with probability $\min(1,q(X)/r(X))$ and, on rejection, drawing from the normalized positive residual $(q-r)_+$ outputs exactly $q$. Handle symbols for which $r(x)=0$.

**8.13 ★★ Denoising score matching.** In one dimension, prove Proposition 8.3 in full. Write the constant explicitly and state the integrability conditions needed for the expectations to exist.

**8.14 ★★★ Continuity from random paths.** In one dimension, let

$$
X_t=(1-t)X_0+tX_1
$$

under an arbitrary coupling with a smooth density. Starting from a delta-function representation of $p_t(x)$, differentiate under the integral and prove

$$
\partial_t p_t(x)
+
\partial_x
\left(
p_t(x)\,
\mathbb E[X_1-X_0\mid X_t=x]
\right)
=0.
$$

State where regularity or distributional derivatives enter.

**8.15 ★★ Greedy is local.** Construct a two-step normalized autoregressive distribution for which greedy decoding chooses the first token $A$ but the globally most probable complete sequence begins with $B$. Compute all complete-sequence probabilities.

## Implementation and experimental design

**8.16 ★ [AI-off] Samplers from memory.** Implement temperature, top-$k$, and top-$p$ without looking at Stage 08. Specify tie behavior, protect normalization when probabilities underflow, and pass the sampler tests. Add a boundary test at $u=c_k$.

**8.17 ★★ Solver order per network evaluation.** Re-derive and implement explicit midpoint:

$$
k_1=v(x_j,t_j),\qquad
k_2=v\left(x_j+\frac h2k_1,t_j+\frac h2\right),\qquad
x_{j+1}=x_j+h k_2.
$$

On the analytic Gaussian field, compare Euler and midpoint by pathwise endpoint RMSE at equal **network-evaluation count**, not equal step count. Recover the verified spread values in Experiment 8.3.

**8.18 ★★★ Coupling and curvature.** For batches of eight noise and data points, enumerate assignments or use a linear-assignment solver to minimize $\sum_i\|x_{1,\pi(i)}-x_{0,i}\|^2$. Train independent-coupling and minibatch-coupling flow models with matched data, initialization, updates, and seeds. Define trajectory bending before looking at results, for example

$$
\sum_{j=1}^{m-1}
\|x_{j+1}-2x_j+x_{j-1}\|^2.
$$

Compare bending, endpoint metrics, and wall-clock cost. Explain why a minibatch optimum is not a global optimal-transport certificate.

**8.19 ★★★ Speculative decoding in the spine.** Use Stage 04 as draft and Stage 07 as target. First verify that their vocabularies and tokenization agree. On a fixed prefix, compare at least 100,000 one-token draws from speculative decoding with direct target draws using total-variation distance and multinomial uncertainty. Then compare complete short-sequence statistics and speed. Separate distributional exactness from speedup.

**8.20 ★★★ Quality, diversity, and memorization without a learned feature model.** Design an evaluation for spine stories using only declared symbolic or string-based measurements. It must include grammatical/format constraints, arithmetic accuracy, n-gram distribution similarity, exact and approximate duplicate rate against training data, and diversity across prompts and seeds. Predict how a model that copies training stories would score on every metric.

## Modern synthesis

**8.21 ★★ Guidance reweighting.** Suppose the unconditional and conditional scores are exact:

$$
s_u=\nabla\ln p_t(x),
\qquad
s_c=\nabla\ln p_t(x\mid c).
$$

Show that $s_u+w(s_c-s_u)$ is the score of a density proportional to
$p_t(x\mid c)^w p_t(x)^{1-w}$. Use Bayes' rule to rewrite it as
$p_t(x)[p_t(c\mid x)/p(c)]^w$. Explain why this emphasizes condition-compatible regions and why approximate time-indexed scores may still fail to define one consistent global generative model.

**8.22 ★★★ Parallel masked updates can break dependencies.** Let a length-two binary data distribution put probability $1/2$ on $00$ and $1/2$ on $11$. A forward corruption masks tokens. Derive the reverse posterior when (a) one bit is visible and the other masked, and (b) both are masked. Show that independently sampling both masked positions from their marginal posteriors generates invalid $01$ and $10$ with total probability $1/2$. Propose an iterative unmasking rule that preserves the dependency, and explain what this toy example teaches about discrete diffusion language models.

## Hints: first move only

- **8.1:** Write cumulative boundaries before looking at $u$.
- **8.2:** Use $p_\tau(k)\propto p(k)^{1/\tau}$; bits equal nats divided by $\ln2$.
- **8.3:** Stop at the first cumulative sum at least as large as the threshold.
- **8.4:** Begin with $z=(x-3)/2$ and $|d z/d x|=1/2$.
- **8.5:** Compute $s_t^2$ and $x-t\mu$ separately.
- **8.6:** After the first step, write $x_1-\tfrac12\mu$ as a multiple of $x_0$.
- **8.7:** Multiply conditional probabilities; for EOS consider the probability of never stopping.
- **8.8:** Insert $q_\phi(z\mid x)/q_\phi(z\mid x)$ inside the marginal integral.
- **8.9:** Expand $\mathbb E_q[\ln q-\ln p]$ coordinate by coordinate.
- **8.10:** Compute means, $\operatorname{Var}(X_t)$, and $\operatorname{Cov}(X_1-X_0,X_t)$.
- **8.11:** Add and subtract $M$ and show the cross term is zero.
- **8.12:** Accepted mass at $x$ is $\min(r(x),q(x))$.
- **8.13:** Condition the cross term on the noisy observation.
- **8.14:** Use $p_t(x)=\mathbb E[\delta(x-X_t)]$ and $\partial_t\delta(x-X_t)=-\dot X_t\,\partial_x\delta(x-X_t)$.
- **8.15:** Make $p(A)>p(B)$ but let the best continuation after $B$ be sufficiently more likely.
- **8.16:** Subtract the largest logit before exponentiation.
- **8.17:** One midpoint step costs two field evaluations.
- **8.18:** Freeze evaluation noise so solver and training comparisons remain paired.
- **8.19:** Test the one-token conditional before testing complete sequences.
- **8.20:** Exact copies can score well on grammar and n-gram similarity while scoring badly on novelty.
- **8.21:** Collect the coefficients of $\nabla\ln p_t(x)$ and $\nabla\ln p_t(x\mid c)$.
- **8.22:** When both bits are masked, each marginal is Bernoulli$(1/2)$ but the joint is not independent.

## Numerical and structural answer checks

- **8.1:** Cumulative boundaries are $(0.4,0.7,0.9,1.0)$. Under the stated half-open convention, $u=0.65$ returns symbol 2 and symbol 3 owns $[0.7,0.9)$.
- **8.2:** At $\tau=0.5$, $(0.5333,0.3000,0.1333,0.0333)$ with entropy $1.0785$ nats; at $\tau=2$, $(0.3254,0.2818,0.2301,0.1627)$ with entropy $1.3558$ nats.
- **8.3:** At $p=0.75$, retain three and renormalize to $(0.4444,0.3333,0.2222)$; at $p=0.95$, retain all four.
- **8.4:** $p_X(x)=\tfrac12\phi((x-3)/2)$ and $p_X(3)=1/(2\sqrt{2\pi})\approx0.1995$. Omitting the Jacobian integrates to $2$.
- **8.5:** $v^\star((1,-2),0.5)=(0.4,-0.8)$; the particle with $x_0=0$ ends at $\mu$.
- **8.6:** $x_2=\mu+0.2x_0$, so each endpoint coordinate has standard deviation $0.2$.
- **8.9:** Equality occurs exactly when every $\mu_i=0$ and every $\sigma_i^2=1$.
- **8.11:** The minimum is zero iff $V$ is a measurable function of $(X_t,t)$ almost surely, equivalently the conditional covariance vanishes almost surely.
- **8.12:** Accepted mass plus residual mass is $\min(r,q)+(q-r)_+=q$.
- **8.15:** One example is $p(A)=0.6$, $p(B)=0.4$, $\max p(\cdot\mid A)=0.6$, and $\max p(\cdot\mid B)=0.99$: greedy's best path has probability $0.36$, while the $B$ path has $0.396$.
- **8.22:** With both bits masked, independent marginals produce each of $00,01,10,11$ with probability $1/4$; half the mass violates the data dependency.

## Further reading: what to extract

**Decoding.** [Holtzman et al. 2020](/references/#holtzman-2020): identify the empirical symptom, the tail diagnosis, and what nucleus sampling changes. Do not read it as proof that one threshold is universal.

**Latent variables and invertible flows.** [Kingma & Welling 2014](/references/#kingma-2014): trace Jensen's inequality and reparameterization. [Dinh et al. 2017](/references/#dinh-2017): identify where the triangular Jacobian makes exact likelihood tractable.

**Diffusion and scores.** [Ho et al. 2020](/references/#ho-2020): map the discrete forward process to the marginal corruption notation used here. [Song et al. 2021](/references/#song-2021): separate reverse SDE from probability-flow ODE. [Vincent 2011](/references/#vincent-2011): follow the conditional-expectation identity behind denoising score matching.

**Flow matching and path design.** [Lipman et al. 2023](/references/#lipman-2023): locate the conditional-to-marginal argument. [Liu et al. 2023](/references/#liu-2023): distinguish straight conditional paths, reflow, and generated trajectories. [Albergo & Vanden-Eijnden 2023](/references/#albergo-2023): compare stochastic interpolants with the notation of §8.3. [Geng et al. 2025](/references/#geng-2025): contrast average velocity with instantaneous velocity.

**Discrete diffusion language modeling.** [Nie et al. 2025](/references/#nie-2025): identify the masking process, likelihood bound, reverse update schedule, and the exact empirical baselines used.

**Evaluation.** [Theis et al. 2016](/references/#theis-2016): write down one counterexample in each direction between likelihood and perceptual sample quality.
