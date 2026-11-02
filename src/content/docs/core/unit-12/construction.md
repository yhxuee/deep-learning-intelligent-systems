---
title: "12.3 Mathematical construction"
description: Paired designs blocked by seed, effect sizes and intervals, power and the number of seeds, three-way decision rules, controlling the false discovery rate across many ablations, and what agreement between an original and a replication can show.
sidebar: { label: "12.3 Construction", order: 3 }
---

## 12.3.1 The paired design

Let $y^{A}_s$ and $y^{B}_s$ be the metric (held-out NLL) of conditions A and B trained with seed $s$, for $s = 1,\dots,n$. The **paired differences** $d_s = y^B_s - y^A_s$ estimate the effect $\delta = \mathbb{E}[y^B - y^A]$ by $\bar d$, with standard error $\mathrm{sd}(d)/\sqrt n$. Pairing helps exactly when the conditions are positively correlated across seeds (Unit 10, §10.3.5): a seed that happens to give a good initialization helps both. The spine capstone intends the seed to fix both the initialization and the data order in the two conditions, so that the positional embedding is the only difference. It only half succeeds: removing the positional table shifts the random stream, so the initializations differ — Worked Example 12.3 shows the cost.

**Interval.** With small $n$, the normal approximation is poor; the capstone uses a percentile bootstrap over the $n$ paired differences (Unit 10). With five pairs a bootstrap interval is itself rough — another reason to compute power beforehand.

## 12.3.2 Power and the number of seeds

Suppose the differences have standard deviation $\sigma_d$ and the effect of interest is $\delta$. For a two-sided test at level $\alpha$ to detect $\delta$ with probability (power) $1 - \beta$, the number of paired seeds must satisfy

$$
n \gtrsim \Big(\frac{(z_{1-\alpha/2} + z_{1-\beta})\,\sigma_d}{\delta}\Big)^2 ,
$$

with $z_{0.975} = 1.96$ and $z_{0.8} = 0.84$ for the conventional 5% level and 80% power — Unit 01's formula (§1.3.7) with seeds in place of stories. The formula needs $\sigma_d$ before the experiment: estimate it from a small **pilot** (two or three seeds), register the resulting $n$, then run. Underpowered experiments are the main source of inconclusive results, and of published false positives: when power is low, the effects that happen to reach significance are disproportionately overestimates ([Ioannidis 2005](/references/#ioannidis-2005)).

## 12.3.3 A three-way decision rule

With a threshold $\tau$ (the smallest effect that matters) and a 95% interval $[\ell, u]$ for $\delta$:

$$
\text{decision} = \begin{cases}\text{"effect exceeds }\tau\text{"} & \ell > \tau,\\ \text{"effect below }\tau\text{"} & u < \tau,\\ \text{"inconclusive"} & \text{otherwise.}\end{cases}
$$

The rule treats "no meaningful effect" as a conclusion that must also be earned — by an interval that lies entirely below $\tau$ — rather than as the default when significance is not reached. It is written into the registration (`decide` in `stage12_capstone.py`) and applied mechanically; the spine test checks all three branches.

## 12.3.4 Many ablations

A study that ablates $m$ components and reports those with a "significant" effect faces the many-comparisons problem of Unit 10 (§10.3.6). The Benjamini–Hochberg procedure controls the expected fraction of false discoveries among those reported (Unit 10, §10.5.4). The simplest protection is to register *which* comparisons are confirmatory before running, and to label the rest exploratory.

## 12.3.5 What a replication can show

Let the original and replication estimates be $\hat\delta_1$ and $\hat\delta_2$ with standard errors $s_1, s_2$. Three summaries are common and answer different questions:

- **Same sign, both significant** — the strictest; it fails often even when the effect is real, if power is modest.
- **Original estimate inside the replication's interval** — consistency; satisfied easily by imprecise replications.
- **Difference** $\hat\delta_2 - \hat\delta_1$ relative to $\sqrt{s_1^2 + s_2^2}$ — whether the two disagree more than chance allows; the most informative when both are precise.

The capstone reports the first two. With both experiments inconclusive and the signs opposite, the honest conclusion is that neither run could detect an effect of 0.02 nats — not that the effect is absent, and not that the replication "failed".

## 12.3.6 Registering the analysis, not only the hypothesis

A registration that states the hypothesis but leaves the analysis open still permits forking paths. The capstone's record fixes: the architecture and training budget; the metric (held-out NLL of the fixed split, nats per character); the seeds for original and replication; the analysis (paired percentile bootstrap, 95%); the threshold (0.02) and the three-way rule. Its digest is printed before any training run, and recomputed at the end:

```json
{"hypothesis": "Learned positional embeddings are needed by the spine model.",
 "prediction": "Removing them raises held-out NLL by more than 0.02 nats/char.",
 "threshold": 0.02, "seeds": [0,1,2,3,4], "replication_seeds": [10,11,12,13,14], "steps": 400, ...}
```
