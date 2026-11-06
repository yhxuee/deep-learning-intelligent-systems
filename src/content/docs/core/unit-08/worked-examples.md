---
title: "8.4 Worked examples"
description: Eight fully worked examples connect random numbers to categorical samples, decoding controls to entropy, Jacobians to density, and learned velocity fields to numerical generation.
sidebar: { label: "8.4 Worked examples", order: 4 }
---

These examples follow one routine: state what the procedure claims to sample, calculate what it actually does, and then identify the conclusion that the calculation supports. Exact numerical claims are recomputed by \`labs/test_worked_examples_u06_u08.py\`. Text samples come from \`spine/stage08_sampling.py\` with seed 0; changing the seed, checkpoint, prompt, or order of random draws changes them.

## Worked Example 8.1 — Turn one uniform number into one symbol

**Question.** How can a single draw $U\sim\operatorname{Uniform}[0,1)$ sample from

$$
p=(0.50,0.30,0.15,0.05).
$$

**Step 1: build the cumulative distribution.** Add probabilities from left to right:

$$
c=(0.50,0.80,0.95,1.00).
$$

These values partition a unit interval into pieces whose lengths are the desired probabilities:

| interval | length | symbol |
|---|---:|---:|
| $[0,0.50)$ | $0.50$ | 1 |
| $[0.50,0.80)$ | $0.30$ | 2 |
| $[0.80,0.95)$ | $0.15$ | 3 |
| $[0.95,1.00)$ | $0.05$ | 4 |

**Step 2: locate the draw.**

$$
\begin{array}{c|c|c}
u & \text{first cumulative value beyond }u & \text{output}\\
\hline
0.10 & 0.50 & 1\\
0.55 & 0.80 & 2\\
0.81 & 0.95 & 3\\
0.97 & 1.00 & 4
\end{array}
$$

Because the interval for symbol $k$ has length $p_k$, a uniform draw lands there with probability $p_k$. This is inverse-CDF sampling.

**Boundary detail.** The spine's \`draw\` counts how many cumulative values are *strictly less* than $u$. It therefore assigns $u=0.50$ to symbol 1, using intervals that are closed on the right rather than the half-open display above. Under an ideal continuous uniform distribution, any exact boundary has probability zero, so both conventions induce the same categorical distribution. In finite-precision code a convention is still necessary, and tests should pin it down. The implementation also clamps the index, protecting against round-off near the last cumulative value.

**Takeaway.** Sampling is not “choose a likely symbol somehow.” It is a deterministic map from a known random source to an outcome, designed so the lengths of preimage intervals equal the requested probabilities.

## Worked Example 8.2 — Temperature changes the whole distribution

**Question.** Starting from the same $p$, what does temperature do to uncertainty?

If the logits are $z_k=\ln p_k$, temperature $\tau$ gives

$$
p_\tau(k)
=
\frac{\exp(z_k/\tau)}
{\sum_j\exp(z_j/\tau)}
=
\frac{p_k^{1/\tau}}
{\sum_jp_j^{1/\tau}}.
$$

For $\tau=0.5$, square the probabilities:

$$
p^2=(0.25,0.09,0.0225,0.0025),
\qquad
\sum_kp_k^2=0.365.
$$

Renormalization gives

$$
p_{0.5}\approx(0.685,0.247,0.062,0.007).
$$

For $\tau=2$, take square roots and renormalize. The complete comparison is

| $\tau$ | transformed distribution | entropy (nats) |
|---:|---|---:|
| $0.5$ | $(0.685,0.247,0.062,0.007)$ | $0.810$ |
| $1.0$ | $(0.500,0.300,0.150,0.050)$ | $1.142$ |
| $2.0$ | $(0.379,0.294,0.208,0.120)$ | $1.308$ |

Entropy here is

$$
H(p_\tau)=-\sum_kp_\tau(k)\ln p_\tau(k).
$$

Because the logarithm is natural, the unit is the **nat**. A nat is simply an information unit: $1$ nat equals $1/\ln2\approx1.443$ bits. The unit does not change the ranking of uncertainties; using base-2 logarithms would multiply every entry by the same constant.

For this distribution, lowering temperature concentrates mass and lowers entropy; raising temperature flattens the distribution and raises entropy. With a unique maximum, the limiting entropy is $0$ as $\tau\to0^+$. As $\tau\to\infty$, four symbols become uniform and entropy approaches $\ln4\approx1.386$ nats.

**Takeaway.** Temperature does not directly command “creativity.” It applies a precise power transform to every local conditional. Diversity often changes as a consequence, along with error rate, calibration, and sequence-level behavior.

## Worked Example 8.3 — A nucleus can contain more mass than its threshold

**Question.** For the same sorted distribution, which symbols survive top-$p$ sampling?

At threshold $p=0.9$, inspect cumulative mass:

$$
0.50<0.90,\qquad
0.80<0.90,\qquad
0.95\ge0.90.
$$

The smallest qualifying set therefore contains the first three symbols. Its original mass is $0.95$, not exactly $0.90$. After renormalization,

$$
\widetilde p_{0.9}
=
\left(
\frac{0.50}{0.95},
\frac{0.30}{0.95},
\frac{0.15}{0.95},
0
\right)
\approx
(0.526,0.316,0.158,0).
$$

At threshold $p=0.6$, one symbol is insufficient because its mass is $0.50$; two have total mass $0.80$, so

$$
\widetilde p_{0.6}
=
\left(
\frac{0.50}{0.80},
\frac{0.30}{0.80},
0,0
\right)
=(0.625,0.375,0,0).
$$

The threshold is a minimum retained mass, not the exact mass after cutting. The set size adapts: a sharp conditional may need one token, whereas a flat conditional may need many.

The spine's \`nucleus_mask\` retains a symbol when the cumulative mass *before* that symbol is below the threshold. That detail ensures the token that crosses $p$ is included. Using “cumulative mass after this token is below $p$” would wrongly discard the crossing token and could retain less mass than requested.

**Takeaway.** Top-$p$ does two operations: choose a prefix of the probability-ranked vocabulary, then renormalize it. It samples exactly from that truncated conditional, not approximately from the original one.

## Worked Example 8.4 — Read four decoders as evidence, not anecdotes

**Question.** What changes when the same Stage 07 checkpoint continues the prompt \`Lily had a \` for 90 characters? The command draws three continuations per method.

**Greedy.** All three calls are identical:

> Lily had a good idea and they fixed it together. Lily had 2 plums. Then Lily found 4 more. Now Lily h…

This is expected: after tie-breaking, greedy decoding contains no randomness. Repeating the command cannot reveal alternative continuations.

**Temperature $0.7$.** Examples include:

> …Noah and Tom were proud. Noah gave Serun a hug…

> Lily had a fox called Lones. One sunny day Lily went to the town with Ruby…

The continuations vary while mostly preserving familiar sentence templates. The first excerpt introduces “Serun” without context; that is an observed entity-consistency error, not evidence that Serun is a pet.

**Temperature $1.5$.** Examples include:

> Lily had a mouse called Mufab. One warm day Lily went to the park with Ruby. Lily found a tried again…

> …A goat came to play with Lily and were…

The openings are more varied, but low-probability local choices produce malformed syntax. These samples show a quality–variety change for this checkpoint and prompt; three samples do not estimate the model's general error rate.

**Top-$p=0.9$.** Examples include:

> …Sam and Ruby were happy. Sharing makes play more fun.

> …Tom and Zoe were proud. Friends help each other.

The observed samples stay inside high-mass local sets and remain well formed, but the third sample changes from Lily to Anna. Truncating the tail does not enforce discourse-level consistency.

**What can we conclude?**

- Greedy is deterministic here and selects a high-frequency training-like template.
- Higher temperature exposes more of the learned tail, including both novel combinations and errors.
- Nucleus sampling removes some tail choices but cannot add a copying or state-tracking mechanism that the model has not learned.
- This run is a reproducible qualitative probe, not a benchmark. A comparison of methods would require many prompts, independent seeds, declared metrics, and uncertainty estimates.

## Worked Example 8.5 — The Jacobian prevents probability from doubling

**Question.** If $Z\sim N(0,1)$ and

$$
X=2Z+3,
$$

what is the density of $X$?

Invert the map:

$$
z=f(x)=\frac{x-3}{2},
\qquad
\left|f'(x)\right|=\frac12.
$$

Therefore

$$
p_X(x)
=
\frac12\phi\left(\frac{x-3}{2}\right).
$$

At the new mean $x=3$,

$$
p_X(3)
=
\frac12\phi(0)
\approx
\frac12(0.3989)
=0.1995.
$$

At $x=5$, the corresponding base point is $z=1$:

$$
p_X(5)
=
\frac12\phi(1)
\approx0.1210.
$$

Why must the factor be $1/2$? A short interval of width $d z$ becomes an interval of width $d x=2\,d z$. The probability inside is unchanged, so probability per unit length must halve. If we incorrectly used only $\phi((x-3)/2)$, substitution $x=2z+3$ would give

$$
\int_{-\infty}^{\infty}
\phi\left(\frac{x-3}{2}\right)d x
=
2\int_{-\infty}^{\infty}\phi(z)d z
=2,
$$

which cannot be a probability density.

**Takeaway.** The Jacobian is the factor that keeps total probability equal to one. In many dimensions, volume replaces length and $|\det J|$ replaces the one-dimensional derivative.

## Worked Example 8.6 — Evaluate the exact Gaussian velocity

**Question.** For the Gaussian flow-matching example, which way should a point at the origin move halfway through the path?

Use

$$
\mu=(1,-2),\qquad
\sigma=0.5,\qquad
t=0.5,\qquad
x=(0,0).
$$

First calculate the marginal variance:

$$
s_t^2
=(1-t)^2+t^2\sigma^2
=0.5^2+0.5^2(0.5^2)
=0.3125.
$$

The coefficient in the exact field is

$$
\frac{t\sigma^2-(1-t)}{s_t^2}
=
\frac{0.5(0.25)-0.5}{0.3125}
=-1.2.
$$

The point's displacement from the current marginal mean $t\mu=(0.5,-1)$ is

$$
x-t\mu=(-0.5,1).
$$

Substitute into the field:

$$
\begin{aligned}
v^\star(x,t)
&=
\mu-1.2(x-t\mu)\\
&=
(1,-2)-1.2(-0.5,1)\\
&=(1.6,-3.2).
\end{aligned}
$$

The marginal mean itself moves with velocity $\mu=(1,-2)$. The origin lies behind that moving mean, and the corrective term is $0.6\mu$, so its total velocity is $1.6\mu$.

**Diagnostic check.** The negative coefficient does not mean “move away from the data.” It multiplies the displacement from the *current* mean. Because the origin's displacement is opposite $\mu$, the product points toward the moving cloud.

## Worked Example 8.7 — An exact field plus an inexact solver

**Question.** Does evaluating an exact velocity field once produce an exact endpoint sample?

Start with 20,000 draws from $N(0,I)$ and target

$$
N\!\left((1,-2),0.25I\right),
$$

whose per-coordinate standard deviation is $0.5$. Integrate the exact field from Worked Example 8.6 by Euler's method. Seed 0 gives:

| Euler steps | 1 | 2 | 5 | 20 | 200 |
|---:|---:|---:|---:|---:|---:|
| mean, rounded | $(1.000,-2.000)$ | $(1.001,-2.000)$ | $(1.002,-2.000)$ | $(1.003,-2.000)$ | $(1.003,-2.001)$ |
| mean coordinate std | $0.000$ | $0.200$ | $0.370$ | $0.465$ | $0.497$ |
| target std | $0.500$ | $0.500$ | $0.500$ | $0.500$ | $0.500$ |

The one-step result follows symbolically. At $t=0$,

$$
v^\star(x,0)=\mu-x.
$$

With step size $h=1$,

$$
x_1=x_0+h(\mu-x_0)=\mu.
$$

Every sample becomes the same point, so the standard deviation is exactly zero. The exact continuous solution would instead be $\mu+\sigma x_0$.

Why is the mean already nearly correct while the spread converges slowly? The field is affine. Euler updates the center along the linear path $t\mu$, but it multiplies deviations from that center by an approximate scale factor. With $n$ steps,

$$
r_{j+1}
=
\left[
1+h\,
\frac{t_j\sigma^2-(1-t_j)}
{(1-t_j)^2+t_j^2\sigma^2}
\right]r_j,
$$

where $r_j=x_j-t_j\mu$. The product of these factors approaches the correct endpoint scale $\sigma=0.5$ as $h$ shrinks. For one step, its first factor is zero and all deviations disappear.

**Takeaway.** Field error and solver error are different. This field is exact, yet a coarse solver produces the wrong distribution. In this example the error appears as too little spread; that direction is not universal for every field or solver.

## Worked Example 8.8 — A low training loss is not the sampling metric

**Question.** Can a network learn a flow from Gaussian noise to eight separated modes even though its straight training paths cross?

**Setup.**

- Data: eight isotropic Gaussians with standard deviation $0.15$, centered evenly on a circle of radius $2$.
- Model: a three-layer tanh MLP $v_\theta(x,t)$ with $64$ hidden units.
- Training: $3{,}000$ Adam steps, $512$ independently paired noise–data examples per step.
- Sampling: $2{,}000$ initial noise points and $100$ Euler steps.

With seed 0, the executable experiment reports:

| quantity | result |
|---|---:|
| mean loss, first 100 steps | $5.272$ |
| mean loss, last 100 steps | $3.807$ |
| samples within $0.5$ of some mode | $96\%$ |
| modes receiving at least one nearby sample | $8/8$ |

The positive final regression loss has at least three possible components:

1. **Irreducible conditional variance.** Independently paired straight paths cross, so $X_1-X_0$ is not determined by $(X_t,t)$.
2. **Function approximation error.** A finite MLP may not represent the exact conditional mean field.
3. **Optimization and finite-sample error.** Three thousand stochastic steps need not reach the best representable field.

Proposition 8.2 proves that the population minimizer is the conditional mean; it does **not** prove that the observed value $3.807$ is the irreducible minimum.

The sample diagnostics answer different questions from the regression loss. “$96\%$ near a mode” checks concentration near the eight centers, and “$8/8$ modes” catches total mode omission. Neither checks whether modes have equal probability, whether each cluster has standard deviation $0.15$, or whether the density between modes is correct. Good-looking values on these two metrics are evidence of a successful toy transport, not proof that the learned distribution equals the data distribution.

**Takeaway.** A training objective, an ODE solver, and sample-quality diagnostics observe different failure modes. A convincing generative experiment reports all three rather than treating one number as a substitute for the others.
