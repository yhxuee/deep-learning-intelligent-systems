---
title: "1.4 Worked examples"
description: Eight fully computed examples — the coin's generalization gap, a bigram by hand, the spine baselines, learning curves, choosing α, Hoeffding in numbers, the standard error of a held-out loss, and memorization.
sidebar: { label: "1.4 Worked examples", order: 4 }
---

The previous sections developed the ideas in symbols. This page turns those symbols into numbers. The eight examples follow one continuous question: how can a model fit its training data well yet predict new data poorly, and how can we measure that difference honestly?

Examples 1.1–1.3 build the basic intuition with a coin and character models. Examples 1.4–1.5 show what changes when we add data or tune a hyperparameter. Examples 1.6–1.7 quantify uncertainty. Example 1.8 returns to the central warning: fitting is not the same as learning a reusable rule.

Every number on this page is recomputed by `labs/test_worked_examples_u01_u02.py` (run in CI). For each example, first cover the solution and ask what you expect to happen. Then calculate it yourself and compare your reasoning with the interpretation.

## Worked Example 1.1 — The coin's generalization gap

**Question.** If a coin produces seven heads in ten tosses, how much better will the fitted probability look on those ten tosses than on future tosses?

**Setting.** Ten tosses, seven heads. Maximum likelihood copies the observed fraction, so the fitted model predicts heads with probability $q = 0.7$.

**Training loss.** The average log-loss on the ten observed tosses is the entropy of the empirical distribution:

$$
\hat L = -(0.7\ln 0.7 + 0.3\ln 0.3) = 0.7\times0.3567 + 0.3\times1.2040 = 0.6109 \text{ nats}.
$$

**Expected loss on new tosses.** The training data tell us $q$, but they do not reveal the true coin probability $p$. We therefore examine two possible values of $p$. By Corollary 1.1, $L(q) = H(p) + \mathrm{KL}(p\,\Vert\,q)$.

- If $p = 0.5$ (a fair coin that happened to give 7 heads): $H(0.5) = \ln 2 = 0.6931$ and $\mathrm{KL}(0.5\,\Vert\,0.7) = 0.5\ln\frac{0.5}{0.7} + 0.5\ln\frac{0.5}{0.3} = -0.1682 + 0.2554 = 0.0872$. So $L = 0.7803$.
- If $p = 0.6$: $H(0.6) = 0.6730$ and $\mathrm{KL}(0.6\,\Vert\,0.7) = 0.0226$, so $L = 0.6956$.

**What the result means.** In both cases, the loss on new data exceeds the training loss of 0.6109. There are two reasons.

1. We chose $q$ using these same ten tosses, so it is especially well matched to them. On future data, any mismatch between $q$ and the true $p$ contributes the KL term.
2. The observed split, 7 heads to 3 tails, is less balanced than either possible true coin considered here. Its empirical entropy, 0.6109, is therefore smaller than the true entropy.

The generalization gap is not merely bad luck after training. Reusing the same observations both to choose and to judge a model makes the training loss systematically optimistic.

## Worked Example 1.2 — A bigram by hand

**Question.** How can a model obtain a very small training loss and yet have infinite loss on a new text?

**Setting.** The alphabet is $\{\texttt a, \texttt b, \texttt \_\}$, so $V = 3$. The training text is `ab_ab_ba` (seven adjacent pairs), and the held-out text is `ba_aa` (four pairs). A bigram model predicts the next character from the current one.

**Step 1: count transitions.** Read `ab_ab_ba` from left to right. Its seven transitions are `a→b`, `b→_`, `_→a`, `a→b`, `b→_`, `_→b`, `b→a`. Each row below fixes the current character; each column records what came next.

| from \ to | a | b | _ | total |
|---|---|---|---|---|
| a | 0 | 2 | 0 | 2 |
| b | 1 | 0 | 2 | 3 |
| _ | 1 | 1 | 0 | 2 |

**Step 2: turn each row into probabilities.** Maximum likelihood divides every count by its row total. Thus $q(\texttt b\mid\texttt a) = 1$; $q(\texttt a\mid\texttt b) = 1/3$, $q(\texttt\_\mid\texttt b) = 2/3$; $q(\texttt a\mid\texttt\_) = q(\texttt b\mid\texttt\_) = 1/2$; all others are 0.

Training loss: two `a→b` cost 0 each; two `b→_` cost $-\ln\frac23 = 0.4055$ each; `_→a` and `_→b` cost $\ln 2 = 0.6931$ each; `b→a` costs $\ln 3 = 1.0986$. Total $3.2958$, mean **0.4708** nats per pair — far below $\ln 3 = 1.0986$.

**Step 3: evaluate new pairs.** In `ba_aa`, the transitions `a→_` and `a→a` were never seen after `a`, so maximum likelihood assigns each probability 0. Because $-\ln 0 = \infty$, even one zero-probability event makes the mean held-out loss **infinite**.

**Step 4: remove zero probabilities.** With add-one smoothing ($\alpha = 1$), we add one imaginary count to every cell. Each row's denominator therefore grows by $\alpha V = 3$.

| from \ to | a | b | _ |
|---|---|---|---|
| a | 1/5 | 3/5 | 1/5 |
| b | 2/6 | 1/6 | 3/6 |
| _ | 2/5 | 2/5 | 1/5 |

Training loss: $2(0.5108) + 2(0.6931) + 0.9163 + 0.9163 + 1.0986 = 5.3391$, mean **0.7627**. Held-out: `b→a` $1.0986$, `a→_` $1.6094$, `_→a` $0.9163$, `a→a` $1.6094$; total $5.2338$, mean **1.3084**.

**What the result means.** Smoothing deliberately makes the fit to the training text worse (0.4708 → 0.7627), but it prevents the model from declaring an unseen event impossible, so the held-out loss becomes finite. This is protection, not a guarantee of good prediction: the smoothed held-out loss, 1.3084, is still *worse than uniform*, 1.0986. Seven training pairs provide almost no evidence about a held-out text containing `aa`, which never occurred. Beating $\ln V$ on held-out data is therefore a real achievement, not a formality.

## Worked Example 1.3 — The spine corpus baselines

**Question.** How much does each additional piece of context improve prediction on a realistic corpus?

**Setting.** The spine corpus (`spine/data/tales.txt`) contains 1 400 short stories and $V = 50$ distinct characters. The last 140 stories are held out, leaving 385 274 training and 42 881 held-out characters. The models form a ladder: uniform uses no observed frequencies, unigram uses overall character frequencies, and bigram also uses the previous character.

| Model | Parameters | Train NLL | Held-out NLL (nats/char) |
|---|---|---|---|
| uniform | 0 | 3.912 | **3.912** $= \ln 50$ |
| unigram (α = 0.1) | 49 | 3.060 | **3.060** |
| bigram (α = 0.1) | 2 450 | 2.123 | **2.124** |
| bigram (α → ∞) | — | 3.912 | **3.912** |

**What the result means.** The uniform model spends 3.912 nats because it treats all 50 next characters equally. Learning their unequal frequencies saves 0.85 nats on held-out data. Conditioning on the previous character saves another 0.94 nats. The improvement is measured on stories that were not used for fitting, so it reflects reusable predictive information rather than a better memory of the training stories.

The last row is also a software check for Stage 01 of the spine. As $\alpha\to\infty$, the imaginary counts dominate the observed counts, so every bigram row must become exactly uniform. Recovering 3.912 verifies the implementation's normalization. The bigram's held-out and training losses differ by only 0.002 nats; Worked Example 1.7 will show that a difference this small is within the estimate's noise.

## Worked Example 1.4 — The learning curve

**Question.** As the training set grows, why can training loss rise while held-out loss falls?

**Setting.** Fit the same bigram ($\alpha = 0.1$) on the first $n$ training characters and always evaluate it on the full held-out set. Only the amount of training evidence changes.

| $n$ | Train NLL | Held-out NLL | Gap |
|---|---|---|---|
| 500 | 1.925 | 2.763 | 0.838 |
| 2 000 | 2.057 | 2.371 | 0.314 |
| 8 000 | 2.098 | 2.198 | 0.100 |
| 32 000 | 2.126 | 2.148 | 0.022 |
| 128 000 | 2.120 | 2.127 | 0.006 |
| 385 274 | 2.123 | 2.124 | 0.002 |

**What the result means.** With 2 000 characters, the held-out loss is 0.31 nats *higher* than the training loss. With all 385 000 characters, the difference has shrunk to 0.002. The two curves approach the same value from opposite directions:

- Training loss generally *rises*. A small sample contains accidental patterns that a fixed-size table can fit unusually well. As more varied examples arrive, those accidents no longer dominate.
- Held-out loss *falls*. More counts make each estimated transition probability less sensitive to chance.

Both approach the bigram's population risk, about 2.12 nats. This is the prediction error of the model family, not just of one fitted table: on this source, one character of context cannot do better. More data can reduce estimation error, but it cannot remove this model limitation. Doing better requires more context, as Units 02–05 develop.

## Worked Example 1.5 — Choosing α on held-out data

**Question.** Why should a smoothing strength be chosen on held-out data rather than on the training data it smooths?

**Setting.** Fit the bigram on the first 2 000 characters and vary $\alpha$. Small $\alpha$ stays close to the raw counts; large $\alpha$ pulls every row toward uniform.

| α | 0.001 | 0.01 | 0.03 | 0.1 | 0.3 | 1 | 3 |
|---|---|---|---|---|---|---|---|
| train NLL | 1.977 | 1.986 | 2.004 | 2.057 | 2.170 | 2.413 | 2.759 |
| held-out NLL | 2.523 | 2.415 | 2.378 | **2.371** | 2.428 | 2.607 | 2.892 |

**What the result means.** Training loss increases monotonically with $\alpha$ because smoothing moves the table away from the maximum-likelihood training optimum of Proposition 1.1. If we used training loss to choose $\alpha$, we would always choose $\alpha\to0$.

Held-out loss tells a different story. It is U-shaped, with a broad, nearly flat minimum between $\alpha=0.03$ and $0.1$. Too little smoothing leaves 5.4% of held-out pairs unseen and drives their loss without bound as $\alpha\to0$; too much smoothing discards useful counts and approaches uniform prediction. The middle balances those two errors. Because we used this held-out set to make the choice, it has now become validation data; the selected $\alpha$ must be evaluated once more on a *different*, untouched test set (§1.3.6).

## Worked Example 1.6 — Hoeffding's inequality in numbers

**Question.** How much does trying many models weaken the certainty of a held-out result?

**Setting.** The loss is bounded in $[0,1]$, as with 0–1 error. We have $n=10\,000$ held-out examples and want confidence $1-\delta=0.95$.

**One fixed model.** $\varepsilon = \sqrt{\ln(2/0.05)/(2\times10\,000)} = \sqrt{3.689/20\,000} = 0.0136$. With 95% probability the measured error is within 1.4 percentage points of the true error.

**Best of 1 000 models.** $\varepsilon = \sqrt{(\ln 1000 + 3.689)/20\,000} = \sqrt{10.597/20\,000} = 0.0230$.

**What the result means.** Evaluating one preselected model gives a 1.36-percentage-point radius. Searching among 1 000 candidates increases it to 2.30 points: 70% more uncertainty, not 1 000 times more, because the penalty grows as $\sqrt{\ln K}$ (§1.3.6).

The direction of the selection bias matters. If “best” means the *smallest error*, random measurement noise helps the winner look too good, so the reported best error is biased **downward**. (For a score where larger is better, the best score would be biased upward.) In either convention, the chosen model must be measured again on untouched data.

## Worked Example 1.7 — The standard error of a held-out loss

**Question.** What should count as one independent observation when characters are grouped into stories?

**Setting.** Evaluate the bigram from Worked Example 1.3 on the 140 held-out stories. We will estimate uncertainty two ways and see why the unit of analysis matters.

**Per-story.** The 140 per-story mean losses have mean 2.1204 and standard deviation 0.0617. Standard error $0.0617/\sqrt{140} = 0.0052$; an approximate 95% interval is $2.120 \pm 0.010$.

**Per-character (naive).** The 42 881 held-out characters yield 42 741 scored transitions: the first character of each of the 140 stories has no preceding character and therefore no bigram loss. These 42 741 per-character losses have standard deviation 1.244; treating them as independent gives a standard error $1.244/\sqrt{42\,741} = 0.0060$.

**What the result means.** A formula such as $s/\sqrt n$ only works when $n$ represents suitably independent observations. Characters within one story share topic, style, and local context; here, consecutive losses even have correlation $-0.32$, because a surprising character is often followed by a predictable one. Counting 42 741 characters as fully independent therefore gives a misleading justification, even though its numerical answer happens to be similar in this example.

The sampling unit is the story, so the per-story standard error is the defensible estimate. The 0.002-nat difference between training and held-out loss in Worked Example 1.3 lies well inside $\pm0.010$. It is noise, not evidence that the held-out stories are “easier.”

**How many independent units are needed to detect a 0.02-nat improvement?** With a per-example standard deviation of the paired difference of $\sigma_d = 0.5$ (a typical character-level value), $n \approx (2.80\times0.5/0.02)^2 = 4\,900$ independent examples. This calculation does **not** say that 4 900 stories are required: to calculate a story count, we must estimate the standard deviation of the per-story paired differences. Pairing matters because both models face the same easy and difficult stories; much of that shared variation cancels when we compare their losses story by story. Unit 10 therefore uses paired, per-story comparisons.

## Worked Example 1.8 — Memorization is cheap

**Question.** Does zero training error prove that a model has discovered a meaningful rule?

**Setting.** Return to the random-label experiment of §1.1: 50 000 images, 10 classes, and labels assigned uniformly at random.

**A lookup table** that stores each training image's label reaches 100% training accuracy. On a new image it has no stored entry and can do no better than guessing: expected accuracy $1/10$. Storing the labels takes $50\,000\times\log_2 10 = 166\,096$ bits — about 20 kilobytes, a tiny fraction of a network with millions of 32-bit parameters.

**What the result means.** The 20-kilobyte calculation is the key scale comparison. A network with millions of 32-bit parameters has vastly more raw storage than the random labels require. It can therefore reach zero training error by acting like a lookup table, without discovering any relationship between image and label.

Zero training error alone says nothing about generalization, and neither does a parameter count larger than the labels' information content. The real-label run differs because the true labels are a *simple function* of the images and the training procedure tends to find that function before a memorized exception table. Generalization is therefore a joint property of the data, architecture, and optimizer (§1.5.4), not a consequence of training accuracy alone.
