---
title: "1.4 Worked examples"
description: Eight fully computed examples — the coin's generalization gap, a bigram by hand, the spine baselines, learning curves, choosing α, Hoeffding in numbers, the standard error of a held-out loss, and memorization.
sidebar: { label: "1.4 Worked examples", order: 4 }
---

Every number on this page is recomputed by `labs/test_worked_examples_u01_u02.py` (run in CI). Work each example yourself before reading the solution.

## Worked Example 1.1 — The coin's generalization gap

**Setting.** Ten tosses, seven heads. The maximum-likelihood prediction is $q = 0.7$.

**Training loss.** The average log-loss on the ten observed tosses is the entropy of the empirical distribution:

$$
\hat L = -(0.7\ln 0.7 + 0.3\ln 0.3) = 0.7\times0.3567 + 0.3\times1.2040 = 0.6109 \text{ nats}.
$$

**Expected loss on new tosses** depends on the true $p$, by Corollary 1.1: $L(q) = H(p) + \mathrm{KL}(p\,\Vert\,q)$.

- If $p = 0.5$ (a fair coin that happened to give 7 heads): $H(0.5) = \ln 2 = 0.6931$ and $\mathrm{KL}(0.5\,\Vert\,0.7) = 0.5\ln\frac{0.5}{0.7} + 0.5\ln\frac{0.5}{0.3} = -0.1682 + 0.2554 = 0.0872$. So $L = 0.7803$.
- If $p = 0.6$: $H(0.6) = 0.6730$ and $\mathrm{KL}(0.6\,\Vert\,0.7) = 0.0226$, so $L = 0.6956$.

**Reading.** In both cases the loss on new data exceeds the training loss (0.6109). The training loss is optimistic for two separate reasons: the model was fitted to these tosses (the KL term), and the entropy of the *observed* frequencies (0.6109) underestimates the entropy of the true coin. The gap is not bad luck; it is built into fitting.

## Worked Example 1.2 — A bigram by hand

**Setting.** Alphabet $\{\texttt a, \texttt b, \texttt \_\}$ ($V = 3$), training text `ab_ab_ba` (seven pairs), held-out text `ba_aa` (four pairs).

**Counts.** Pairs: `a→b`, `b→_`, `_→a`, `a→b`, `b→_`, `_→b`, `b→a`.

| from \ to | a | b | _ | total |
|---|---|---|---|---|
| a | 0 | 2 | 0 | 2 |
| b | 1 | 0 | 2 | 3 |
| _ | 1 | 1 | 0 | 2 |

**Maximum likelihood.** $q(\texttt b\mid\texttt a) = 1$; $q(\texttt a\mid\texttt b) = 1/3$, $q(\texttt\_\mid\texttt b) = 2/3$; $q(\texttt a\mid\texttt\_) = q(\texttt b\mid\texttt\_) = 1/2$; all others 0.

Training loss: two `a→b` cost 0 each; two `b→_` cost $-\ln\frac23 = 0.4055$ each; `_→a` and `_→b` cost $\ln 2 = 0.6931$ each; `b→a` costs $\ln 3 = 1.0986$. Total $3.2958$, mean **0.4708** nats per pair — far below $\ln 3 = 1.0986$.

Held-out loss: `a→_` and `a→a` have probability 0. The held-out loss is **infinite**.

**Add-one smoothing** ($\alpha = 1$): each row's denominator grows by $\alpha V = 3$.

| from \ to | a | b | _ |
|---|---|---|---|
| a | 1/5 | 3/5 | 1/5 |
| b | 2/6 | 1/6 | 3/6 |
| _ | 2/5 | 2/5 | 1/5 |

Training loss: $2(0.5108) + 2(0.6931) + 0.9163 + 0.9163 + 1.0986 = 5.3391$, mean **0.7627**. Held-out: `b→a` $1.0986$, `a→_` $1.6094$, `_→a` $0.9163$, `a→a` $1.6094$; total $5.2338$, mean **1.3084**.

**Reading.** Smoothing made the training loss worse (0.4708 → 0.7627) and the held-out loss finite. Note that the smoothed held-out loss (1.3084) is *worse than uniform* (1.0986): seven training pairs say almost nothing about a held-out text that happens to contain `aa`, which never occurred. Beating $\ln V$ on held-out data is a real achievement, not a formality.

## Worked Example 1.3 — The spine corpus baselines

**Setting.** The spine corpus (`spine/data/tales.txt`): 1 400 short stories, $V = 50$ distinct characters; the last 140 stories are held out (385 274 training and 42 881 held-out characters).

| Model | Parameters | Train NLL | Held-out NLL (nats/char) |
|---|---|---|---|
| uniform | 0 | 3.912 | **3.912** $= \ln 50$ |
| unigram (α = 0.1) | 49 | 3.060 | **3.060** |
| bigram (α = 0.1) | 2 450 | 2.123 | **2.124** |
| bigram (α → ∞) | — | 3.912 | **3.912** |

**Reading.** Each step uses more context and gains on held-out data: 0.85 nats from character frequencies, another 0.94 from the previous character. The last row is the spine's verification target for Stage 01: infinite smoothing recovers the uniform model *exactly*, which tests the code's normalization. (The bigram's held-out and training losses differ by only 0.002 nats; Worked Example 1.7 shows that a difference this small is within the noise of the estimate.)

## Worked Example 1.4 — The learning curve

**Setting.** Fit the bigram (α = 0.1) on the first $n$ training characters; evaluate on the full held-out set.

| $n$ | Train NLL | Held-out NLL | Gap |
|---|---|---|---|
| 500 | 1.925 | 2.763 | 0.838 |
| 2 000 | 2.057 | 2.371 | 0.314 |
| 8 000 | 2.098 | 2.198 | 0.100 |
| 32 000 | 2.126 | 2.148 | 0.022 |
| 128 000 | 2.120 | 2.127 | 0.006 |
| 385 274 | 2.123 | 2.124 | 0.002 |

**Reading.** This answers the prediction prompt of §1.1: with 2 000 characters the held-out loss is 0.31 nats *higher* than the training loss; with all 385 000, the two agree to within 0.002. Two curves approach each other from opposite sides: the training loss *rises* with $n$, apart from small fluctuations (a fixed-size table can no longer fit every accident of a larger sample), the held-out loss *falls* (the counts estimate the true probabilities better). Both converge to the bigram's population risk, about 2.12 nats — the best a one-character context can do on this source. More data cannot push a bigram below that floor; only more context can (Units 02–05).

## Worked Example 1.5 — Choosing α on held-out data

**Setting.** Bigram fitted on the first 2 000 characters; held-out loss as a function of α.

| α | 0.001 | 0.01 | 0.03 | 0.1 | 0.3 | 1 | 3 |
|---|---|---|---|---|---|---|---|
| train NLL | 1.977 | 1.986 | 2.004 | 2.057 | 2.170 | 2.413 | 2.759 |
| held-out NLL | 2.523 | 2.415 | 2.378 | **2.371** | 2.428 | 2.607 | 2.892 |

**Reading.** The training loss increases monotonically with α (smoothing moves the table away from the training optimum of Proposition 1.1). The held-out loss is U-shaped with a flat minimum between α = 0.03 and 0.1. If α had been chosen by the training loss, the answer would be α → 0 and the held-out loss would grow without bound as α → 0 (5.4% of held-out pairs are unseen). A choice made on this held-out set must now be confirmed on a *different* test set (§1.3.6).

## Worked Example 1.6 — Hoeffding's inequality in numbers

**Setting.** A loss bounded in $[0, 1]$ (for example, 0–1 error), $n = 10\,000$ held-out examples, confidence $1-\delta = 0.95$.

**One fixed model.** $\varepsilon = \sqrt{\ln(2/0.05)/(2\times10\,000)} = \sqrt{3.689/20\,000} = 0.0136$. With 95% probability the measured error is within 1.4 percentage points of the true error.

**Best of 1 000 models.** $\varepsilon = \sqrt{(\ln 1000 + 3.689)/20\,000} = \sqrt{10.597/20\,000} = 0.0230$.

**Reading.** Selecting among a thousand candidates costs 70% more uncertainty, not a thousandfold more — the $\sqrt{\ln K}$ of §1.3.6. But the reported best score is biased upward by an amount of this order, which is why it must be re-measured on untouched data.

## Worked Example 1.7 — The standard error of a held-out loss

**Setting.** The bigram of Worked Example 1.3 on the 140 held-out stories.

**Per-story.** The 140 per-story mean losses have mean 2.1204 and standard deviation 0.0617. Standard error $0.0617/\sqrt{140} = 0.0052$; an approximate 95% interval is $2.120 \pm 0.010$.

**Per-character (naive).** The 42 741 per-character losses have standard deviation 1.244; treating them as independent gives a standard error $1.244/\sqrt{42\,741} = 0.0060$.

**Reading.** The two answers differ because characters within a story are dependent (consecutive losses have correlation $-0.32$ here: a surprising character is often followed by a predictable one). The independent unit is the story, so the per-story standard error is the defensible one. The difference between training and held-out loss in Worked Example 1.3 ($0.002$) is well inside $\pm0.010$: it is noise, not evidence that held-out data is "easier".

**How many stories to detect a 0.02-nat improvement?** With a per-example standard deviation of the paired difference of $\sigma_d = 0.5$ (a typical character-level value), $n \approx (2.80\times0.5/0.02)^2 = 4\,900$ independent examples. Per-story differences vary far less than per-character ones, which is why Unit 10 compares models story by story.

## Worked Example 1.8 — Memorization is cheap

**Setting.** The random-label experiment of §1.1: 50 000 images, 10 classes, labels assigned uniformly at random.

**A lookup table** that stores each training image's label reaches 100% training accuracy. On a new image it has no stored entry and can do no better than guessing: expected accuracy $1/10$. Storing the labels takes $50\,000\times\log_2 10 = 166\,096$ bits — about 20 kilobytes, a tiny fraction of a network with millions of 32-bit parameters.

**Reading.** A model with enough capacity to store its training labels can reach zero training error on *any* labelling, true or random. So zero training error says nothing about generalization, and neither does a capacity count larger than the information content of the labels. What distinguishes the real-label run is that the true labels are a *simple function* of the images that the network's training procedure finds first — a property of data, architecture and optimizer together (§1.5.4).
