---
title: "12.3 Mathematical construction"
description: Estimands, paired and factorial designs, variance decomposition, intervals, meaningful-effect and equivalence decisions, power, multiple comparisons, sequential stopping, replication, and tamper-evident registration.
sidebar: { label: "12.3 Construction", order: 3 }
---

The mathematics in this section has one purpose: connect the scientific question to the randomness that actually generated the data. A sophisticated interval around the wrong experimental unit is still the wrong interval.

## 12.3.1 Define the random variables and estimand

Let condition $A$ use learned positional embeddings and condition $B$ omit them. For training seed $s$ and evaluation item $i$, write the item loss as

$$
\ell_{csi},
\qquad c\in\{A,B\}.
$$

The checkpoint-level held-out mean is

$$
Y_{cs}=\frac1m\sum_{i=1}^{m}\ell_{csi}.
$$

The capstone estimand is the mean training-procedure contrast

$$
\delta=\mathbb E_s[Y_{Bs}-Y_{As}].
$$

This is not the same as the difference between two particular checkpoints. A useful hierarchical model is

$$
\ell_{csi}=\mu_c+u_{cs}+v_i+\varepsilon_{csi},
$$

where $u_{cs}$ is training-run variation, $v_i$ item difficulty, and $\varepsilon_{csi}$ residual variation. More evaluation items reduce uncertainty from $v_i$ and $\varepsilon_{csi}$; only more independent training runs reveal the distribution of $u_{cs}$.

## 12.3.2 Paired design and the covariance term

For genuinely matched seed $s$, define

$$
d_s=Y_{Bs}-Y_{As},
\qquad
\hat\delta=\bar d=\frac1n\sum_{s=1}^n d_s.
$$

The sample standard deviation and standard error are

$$
s_d=\sqrt{\frac1{n-1}\sum_s(d_s-\bar d)^2},
\qquad
\operatorname{SE}(\bar d)=\frac{s_d}{\sqrt n}.
$$

Why pair? If $Y_A$ and $Y_B$ have standard deviations $\sigma_A,\sigma_B$ and correlation $\rho$ across matched blocks,

$$
\operatorname{Var}(Y_B-Y_A)
=\sigma_B^2+\sigma_A^2-2\rho\sigma_A\sigma_B.
$$

Positive $\rho$ removes shared nuisance variation. If $ho=0$, pairing gains nothing; if $ho<0$, it increases difference variance.

### Broken pairing in the capstone

The implementation uses one pseudorandom stream. In condition A it draws a positional table before later weights; condition B skips that draw. Therefore subsequent tensors differ even with the same integer seed. The observed original-run correlation is negative, but with five pairs that estimate is itself unstable. The decisive fact is structural: shared tensors were not initialized identically.

After repairing the random streams, verify

$$
\theta^{A}_{s,k}(0)=\theta^{B}_{s,k}(0)
$$

for every shared parameter tensor $k$. Pair labels are not enough.

## 12.3.3 Paired intervals: t and bootstrap

If paired differences are approximately independent and not extremely non-normal, a small-sample $t$ interval is

$$
\bar d\pm t_{n-1,,1-\alpha/2}\frac{s_d}{\sqrt n}.
$$

For $n=5$, $t_{4,0.975}=2.776$, substantially larger than the normal value $1.96$.

A paired percentile bootstrap resamples the $n$ pairs with replacement:

1. draw indices $s_1^*,\ldots,s_n^*$ from $\{1,\ldots,n\}$;
2. compute $\bar d^*=n^{-1}\sum_j d_{s_j^*}$;
3. repeat $B$ times;
4. take empirical $\alpha/2$ and $1-\alpha/2$ quantiles.

Resample **pairs**, not the two condition arrays independently. Independent resampling destroys the covariance the design was meant to use.

Bootstrap does not create information. With five observations, there are few distinct resamples, tail quantiles are coarse, and the empirical distribution may omit plausible unseen seeds. Report the raw $d_s$, a $t$ interval as a sensitivity analysis, and the bootstrap interval registered by the capstone.

## 12.3.4 Three different questions about a threshold

Let $	au>0$ be the smallest effect of interest and $I=[\ell,u]$ an interval for $delta$.

### Does the effect exceed a positive threshold?

The capstone registers

$$
R=\begin{cases}
\text{needed} & \ell>\tau,\\
\text{not needed by this threshold rule} & u<\tau,\\
\text{inconclusive} & \ell\le\tau\le u.
\end{cases}
$$

This classifies the claim $\delta>\tau$. The second branch does **not** mean $\delta=0$; a large negative effect also satisfies $u<\tau$.

### Is the effect practically equivalent to zero?

Equivalence requires a symmetric negligible region $[-\tau,+\tau]$. Evidence for equivalence requires

$$
\ell>-\tau
\quad\text{and}\quad
u<+\tau,
$$

the interval form of two one-sided tests. Merely crossing zero or failing a difference test is not equivalence.

### Is there any directional difference?

A conventional two-sided difference test concerns $\delta=0$, a third question. Scientific design should choose among these questions before selecting the test.

## 12.3.5 Power and seed count

Under a normal approximation, paired differences have standard deviation $\sigma_d$. To detect effect $\delta_1$ with two-sided type-I error $alpha$ and power $1-\beta$,

$$
n\approx
\left[
\frac{(z_{1-\alpha/2}+z_{1-\beta})\sigma_d}
{\lvert\delta_1\rvert}
\right]^2.
$$

At $alpha=0.05$ and power $0.8$, $1.96+0.84=2.80$. With capstone pilot $s_d=0.098$ and target $0.02$,

$$
n\approx\left(\frac{2.8\times0.098}{0.02}\right)^2
\approx188.2,
$$

so round **up** to 189 pairs.

Inverting the formula, five pairs have 80% power for $0.02$ only if

$$
\sigma_d\lesssim
\frac{0.02\sqrt5}{2.8}
\approx0.016.
$$

The observed $0.098$ is about six times larger.

### What the formula assumes

- independent seed pairs drawn from the target randomness distribution;
- a stable variance and approximately normal mean;
- fixed sample size and analysis;
- no unaccounted multiplicity;
- the pilot variance is representative.

With a noisy two- or three-seed pilot, the estimated $n$ is uncertain because $n$ scales as $\sigma_d^2$. Plan conservatively, use an upper variance estimate, or run an internal pilot whose use is specified in advance.

## 12.3.6 Variance reduction is often cheaper than more runs

Because $n\propto\sigma_d^2$, halving $sigma_d$ quarters the required seed count. Possible reductions include

- correct paired initialization;
- common data order and augmentation randomness;
- a more stable endpoint, such as average loss over a registered late window;
- longer training if early optimization noise is not the target;
- more evaluation data to reduce item-level noise;
- covariate adjustment specified before outcome inspection.

Each changes the estimand or protocol and must be documented. For example, longer training answers a different question from “performance after 400 steps.”

## 12.3.7 Factorial designs estimate interactions

Suppose depth $A$ and context length $B$ may interact. A $2\times2$ factorial model is

$$
Y_{s}=\beta_0+\beta_AA+\beta_BB+\beta_{AB}AB+u_s+\varepsilon_s.
$$

$\beta_{AB}$ asks whether the effect of longer context changes with depth. One-factor-at-a-time comparisons cannot estimate that interaction.

For four binary factors, the full design has $2^4=16$ cells. It may be expensive, but it separates main effects and interactions that a single before/after comparison confounds. Fractional factorial designs trade some interactions for fewer cells; the aliasing assumptions must be explicit.

## 12.3.8 Multiple comparisons and false discoveries

If $m$ null hypotheses are each tested at level $0.05$, the expected number of false positives is $0.05m$ under valid tests. The probability of at least one false positive under independence is

$$
1-(1-0.05)^m.
$$

For $m=20$, this is about $0.642$.

Controls depend on the goal:

- **family-wise error rate:** Bonferroni tests each at $\alpha/m$;
- **false discovery rate:** Benjamini–Hochberg sorts $p_{(1)}\le\cdots\le p_{(m)}$ and keeps the largest $k$ with

$$
p_{(k)}\le\frac{k}{m}q;
$$

- **pre-specified primary comparison:** reserve confirmatory status for a small set and label the rest exploratory.

Multiplicity includes metrics, subgroups, checkpoints, exclusion rules, and model variants—not only the tests printed in the final table.

## 12.3.9 Optional stopping changes error rates

Running five seeds, checking the result, then adding seeds until significance is reached is not a fixed-$n$ test. Repeated looks create more chances to stop on a favourable fluctuation.

Valid options include

- register a fixed $n$;
- use an alpha-spending or group-sequential design;
- use confidence sequences valid at every stopping time;
- treat the first stage as a pilot and run a fresh confirmatory stage.

“We stopped when compute ran out” can be honest, but the resulting analysis must match that stopping process.

## 12.3.10 Replication comparisons

Let independent original and replication estimates be $\hat\delta_1,\hat\delta_2$ with standard errors $s_1,s_2$.

### Do they disagree?

Under independence,

$$
Z_{\mathrm{diff}}=
\frac{\hat\delta_2-\hat\delta_1}
{\sqrt{s_1^2+s_2^2}}.
$$

A small magnitude means no evidence of disagreement; it does not prove the effects are identical.

### What is the pooled estimate?

Under a common fixed-effect assumption,

$$
\hat\delta_{\mathrm{pool}}
=\frac{w_1\hat\delta_1+w_2\hat\delta_2}{w_1+w_2},
\qquad
w_j=\frac1{s_j^2}.
$$

Pooling should be registered or clearly labelled after the fact. If settings differ or heterogeneity is plausible, a single common effect may be the wrong model.

“Same sign,” “both significant,” “original inside replication interval,” and “no significant difference” answer different questions. Report the estimates and intervals, not a single replicated/not-replicated badge.

## 12.3.11 Tamper-evident registration

Let $r$ be a canonical serialization of the registration. Store

$$
h=\operatorname{SHA256}(r).
$$

At analysis time, recompute $h'$ and verify $h'=h$. This checks record integrity under the assumed hash function. It does not prove authorship, trusted timing, good design, correct execution, or truthful reporting.

The capstone record fixes the architecture, metric, threshold, seed lists, steps, paired bootstrap, and `decide` rule. For trusted timing, publish the digest or commit it to a remote repository before running.

## 12.3.12 From equation to claim

| Calculation | Direct conclusion | Stronger claim requiring more evidence |
|---|---|---|
| narrow item-bootstrap interval | fixed checkpoints differ precisely on sampled items | training procedures differ reliably |
| small paired SE | matching removed shared variation | pairing is causally valid |
| $u<\tau$ | interval lies below positive threshold | practical equivalence to zero |
| power $=0.8$ under assumptions | 80% long-run detection probability at specified effect | 80% probability this hypothesis is true |
| original and replication overlap | estimates are statistically compatible | mechanism generalizes to new domains |
| matching hash | record unchanged since hash creation | record existed before data inspection |

Mathematics narrows uncertainty only after the design has connected the random variables to the scientific question.
