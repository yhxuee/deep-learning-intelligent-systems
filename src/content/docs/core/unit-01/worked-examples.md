---
title: "1.4 Worked examples"
description: Eight fully computed examples — the coin's generalization gap, a bigram by hand, the spine baselines, learning curves, choosing α, Hoeffding in numbers, the standard error of a held-out loss, and memorization.
sidebar: { label: "1.4 Worked examples", order: 4 }
---

The previous sections developed theoretical foundations in formal symbols; this page translates those abstractions into step-by-step numerical calculations. The eight examples pursue a single recurring question: **how can a model achieve a near-perfect fit on training data yet falter on unobserved future data, and how do we quantify that gap with scientific honesty?**

- **Examples 1.1–1.3:** Build intuition from a coin toss to character language models, dissecting the generalization gap, the zero-count catastrophe, and baseline ladders.
- **Examples 1.4–1.5:** Empirically trace learning curves and hyperparameter sweeps ($\alpha$), balancing sample size against regularization strength.
- **Examples 1.6–1.7:** Convert theoretical concentration bounds and the Central Limit Theorem into concrete confidence intervals.
- **Example 1.8:** Contrast raw parametric storage capacity against information-theoretic bounds to expose the chasm between brute-force memorization and true generalization.

Every number quoted on this page is recomputed and verified in continuous integration by `labs/test_worked_examples_u01_u02.py`. We recommend covering the solution first, predicting the direction of change, and then computing each step yourself before reading the physical interpretation.

## Worked Example 1.1 — The coin's generalization gap

**Question.** If a coin produces seven heads in ten tosses, how much better will the fitted probability look on those ten tosses than on unobserved future tosses?

**Setting.** Ten observed tosses with seven heads. By maximum likelihood estimation (MLE), the model adopts the observed relative frequency: $q = 0.7$.

**Step 1: Compute training loss.**
On the ten observed tosses, seven heads incur $-\ln 0.7 \approx 0.3567$ nats each, and three tails incur $-\ln 0.3 \approx 1.2040$ nats each. The average log-loss on the training sample is exactly the Shannon entropy of the empirical sample distribution:

$$
\hat L = -(0.7\ln 0.7 + 0.3\ln 0.3) = 0.7\times0.3567 + 0.3\times1.2040 = 0.6109 \text{ nats}.
$$

**Step 2: Compute expected loss on future tosses.**
The training sample reveals empirical frequency $q = 0.7$, but does not reveal nature's true probability $p$. By Corollary 1.1, the expected loss on future tosses is governed by:
$L(q) = H(p) + \mathrm{KL}(p\,\Vert\,q) = -p\ln p - (1-p)\ln(1-p) + p\ln\frac{p}{q} + (1-p)\ln\frac{1-p}{1-q}$.

Consider two representative possibilities for nature's parameter $p$:
- **Case A: The coin is fair ($p = 0.5$) and simply produced 7 heads by chance.**
  True entropy: $H(0.5) = \ln 2 \approx 0.6931$ nats.
  KL divergence: $\mathrm{KL}(0.5\,\Vert\,0.7) = 0.5\ln\frac{0.5}{0.7} + 0.5\ln\frac{0.5}{0.3} = 0.5(-0.3365) + 0.5(0.5108) = 0.0872$ nats.
  Future expected loss: $L = 0.6931 + 0.0872 = 0.7803$ nats.
- **Case B: The coin is genuinely biased toward heads ($p = 0.6$).**
  True entropy: $H(0.6) = -(0.6\ln 0.6 + 0.4\ln 0.4) \approx 0.6730$ nats.
  KL divergence: $\mathrm{KL}(0.6\,\Vert\,0.7) = 0.6\ln\frac{0.6}{0.7} + 0.4\ln\frac{0.4}{0.3} = 0.6(-0.1542) + 0.4(0.2877) = 0.0226$ nats.
  Future expected loss: $L = 0.6730 + 0.0226 = 0.6956$ nats.

**What the result means (anatomy of the generalization gap).**
In both cases, the expected loss on future data (0.7803 and 0.6956) is noticeably higher than the training loss (0.6109). This **generalization gap** arises from two distinct mechanisms:
1. **Model over-adaptation (the KL term):** Because $q = 0.7$ was chosen specifically to match these ten tosses, any discrepancy between $0.7$ and nature's true $p$ acts as an inevitable cognitive penalty on new data.
2. **Sample luck (the entropy term):** The observed 7:3 split is more skewed than either Case A (5:5) or Case B (6:4). A more concentrated distribution has lower empirical entropy (0.6109 < 0.6730 < 0.6931).

The generalization gap is not merely bad luck after training; **reusing the exact same data points both to parameterize and evaluate a model guarantees systematic empirical optimism.**

## Worked Example 1.2 — A bigram by hand

**Question.** How can a model achieve a very small training loss and yet incur an infinite loss on fresh text?

**Setting.** A toy alphabet $\{\texttt a, \texttt b, \texttt \_\}$ with vocabulary size $V = 3$.
- Training sequence: `ab_ab_ba` (8 characters, 7 adjacent transitions);
- Held-out test sequence: `ba_aa` (5 characters, 4 adjacent transitions).

**Step 1: Count transitions in training data.**
Scanning `ab_ab_ba` from left to right yields 7 transitions:
`a→b`, `b→_`, `_→a`, `a→b`, `b→_`, `_→b`, `b→a`.
Construct the $3 \times 3$ transition count matrix:

| from \ to | a | b | _ | total $c(a)$ |
|---|---|---|---|---|
| **a** | 0 | 2 | 0 | 2 |
| **b** | 1 | 0 | 2 | 3 |
| **_** | 1 | 1 | 0 | 2 |

**Step 2: Construct the MLE probability table.**
Normalizing each row by its row total:
- Row `a`: $q(\texttt b\mid\texttt a) = 2/2 = 1.0$; $q(\texttt a\mid\texttt a) = 0$; $q(\texttt\_\mid\texttt a) = 0$;
- Row `b`: $q(\texttt a\mid\texttt b) = 1/3$; $q(\texttt\_\mid\texttt b) = 2/3$; $q(\texttt b\mid\texttt b) = 0$;
- Row `_`: $q(\texttt a\mid\texttt\_) = 1/2$; $q(\texttt b\mid\texttt\_) = 1/2$; $q(\texttt\_\mid\texttt\_) = 0$.

Training loss:
- Two `a→b`: cost $-\ln 1.0 = 0$ each;
- Two `b→_`: cost $-\ln(2/3) \approx 0.4055$ each;
- One `_→a` and one `_→b`: cost $-\ln(1/2) \approx 0.6931$ each;
- One `b→a`: cost $-\ln(1/3) \approx 1.0986$.
Total training loss: $2(0) + 2(0.4055) + 2(0.6931) + 1.0986 = 3.2958$ nats.
**Mean training NLL:** $3.2958 / 7 \approx \mathbf{0.4708}$ nats/char—far lower than the uniform baseline $\ln 3 \approx 1.0986$!

**Step 3: Evaluate on held-out text.**
The transitions in `ba_aa` are `b→a`, `a→_`, `_→a`, and `a→a`.
Checking model predictions:
- `b→a`: $q = 1/3$, cost $-\ln(1/3) \approx 1.0986$;
- `a→_`: In training, `_` never followed `a`, so $q(\texttt\_\mid\texttt a) = 0$. Cost: $-\ln 0 = +\infty$!
- `a→a`: Unseen in training, $q = 0$, cost $-\ln 0 = +\infty$!
**Mean held-out loss: $+\infty$.** The model suffers a catastrophic breakdown.

**Step 4: Apply add-one Laplace smoothing ($\alpha = 1$).**
Add 1 virtual observation to every cell; row denominators increase by $\alpha V = 3$:

| from \ to | a | b | _ | total $c(a) + 3$ |
|---|---|---|---|---|
| **a** | 1/5 | 3/5 | 1/5 | 5 |
| **b** | 2/6 | 1/6 | 3/6 | 6 |
| **_** | 2/5 | 2/5 | 1/5 | 5 |

Recomputing losses:
- **Training set:**
  $2(-\ln 0.6) + 2(-\ln 0.5) + (-\ln 0.4) + (-\ln 0.4) + (-\ln(1/3)) = 2(0.5108) + 2(0.6931) + 0.9163 + 0.9163 + 1.0986 = 5.3391$ nats.
  Mean training loss rises from 0.4708 to **0.7627** nats/char.
- **Held-out set:**
  `b→a`: cost $-\ln(2/6) \approx 1.0986$;
  `a→_`: cost $-\ln(1/5) \approx 1.6094$;
  `_→a`: cost $-\ln(2/5) \approx 0.9163$;
  `a→a`: cost $-\ln(1/5) \approx 1.6094$.
  Total test loss: $1.0986 + 1.6094 + 0.9163 + 1.6094 = 5.2338$ nats.
  **Mean held-out loss:** $5.2338 / 4 \approx \mathbf{1.3084}$ nats/char. **The infinity is successfully averted!**

**What the result means.**
Smoothing deliberately degrades training fit (0.4708 $\to$ 0.7627) to rescue the model from the cliff of zero counts.
However, note that the smoothed held-out loss of 1.3084 nats is **still worse than the uniform baseline $\ln 3 \approx 1.0986$ nats**! Seven training pairs provide insufficient evidence to beat blind guessing on unseen pairs like `aa`. Beating $\ln V$ on test data is a genuine achievement that demands adequate data scale.

## Worked Example 1.3 — The spine corpus baselines

**Question.** Exactly how much predictive gain is achieved as we ladder up context complexity on a real corpus?

**Setting.** The spine corpus (`spine/data/tales.txt`) comprises 1,400 short stories across $V = 50$ distinct characters. The final 140 stories are strictly reserved as held-out test data, leaving 385,274 training characters and 42,881 held-out characters. Three models form a baseline ladder:

| Model | Parameters | Training NLL (nats/char) | Held-Out NLL (nats/char) | Interpretation |
|---|---|---|---|---|
| **Uniform Baseline** | 0 | 3.912 | **3.912** $= \ln 50$ | Equal-probability guessing |
| **Unigram ($\alpha=0.1$)** | 49 | 3.060 | **3.060** | Marginal character frequencies |
| **Bigram ($\alpha=0.1$)** | 2,450 | 2.123 | **2.124** | 1-character conditional context |
| **Bigram ($\alpha\to\infty$)** | — | 3.912 | **3.912** | Prior dominates data (uniform limit) |

**What the result means.**
1. **Marginal frequency value:** Moving from uniform guessing (3.912) to the unigram model (3.060) saves 0.852 nats on held-out text, reflecting the non-uniform frequency of letters in natural language.
2. **Context value:** Moving from unigram (3.060) to bigram (2.124) saves an additional 0.936 nats purely by conditioning on the immediate predecessor.
3. **Software verification:** Setting $\alpha \to \infty$ ($10^{12}$) washes away all empirical counts, recovering $\ln 50 = 3.9120$ to six decimal places, confirming numerical normalization in Stage 01.
4. **Generalization parity:** With nearly 400,000 characters, the bigram's training loss (2.123) and test loss (2.124) differ by only 0.001 nats, virtually eliminating the generalization gap.

## Worked Example 1.4 — Learning curves

**Question.** As training data scales from hundreds of characters to hundreds of thousands, why does training loss rise while held-out loss falls?

**Setting.** Train the bigram model ($\alpha = 0.1$) on prefixes of length $n$, evaluating on the full held-out test set of 42,881 characters:

| Training Tokens $n$ | Training NLL (nats/char) | Held-Out NLL (nats/char) | Generalization Gap |
|---|---|---|---|
| **500** | 1.925 | 2.763 | 0.838 |
| **2,000** | 2.057 | 2.371 | 0.314 |
| **8,000** | 2.098 | 2.198 | 0.100 |
| **32,000** | 2.126 | 2.148 | 0.022 |
| **128,000** | 2.120 | 2.127 | 0.007 |
| **385,274** | 2.123 | 2.124 | 0.001 |

**What the result means (asymptotic convergence).**
Training and held-out curves converge toward approximately 2.124 nats from opposite directions:
- **Training loss rises:** At small $n$ (500 tokens), sparse samples allow 2,500 parameters to memorize random noise (loss 1.925); as diverse data enters, the model can no longer overfit idiosyncrasies.
- **Held-out loss falls:** More observations drive empirical frequencies toward their true expectation under the Law of Large Numbers.
- **The model ceiling:** Even with infinite data, bigram loss cannot drop below 2.12 nats. This is the **inductive bias ceiling**: a single-character context cannot capture multicharacter syntax or semantics. Surpassing 2.12 nats requires larger context representations (Units 02–05).

## Worked Example 1.5 — Choosing α on held-out data

**Question.** Why must hyperparameter $\alpha$ be chosen using an independent validation set rather than the training set?

**Setting.** Fit the bigram on the first 2,000 characters and sweep smoothing parameter $\alpha$:

| Smoothing α | 0.001 | 0.01 | 0.03 | 0.1 | 0.3 | 1.0 | 3.0 |
|---|---|---|---|---|---|---|---|
| **Training NLL** | 1.977 | 1.986 | 2.004 | 2.057 | 2.170 | 2.413 | 2.759 |
| **Held-Out NLL** | 2.523 | 2.415 | 2.378 | **2.371** | 2.428 | 2.607 | 2.892 |

**What the result means.**
- **The training trap:** Training loss increases strictly monotonically with $\alpha$ because $\alpha = 0$ is optimal on training data by Proposition 1.1. Optimizing hyperparameters on training data inevitably picks $\alpha \to 0$.
- **The validation U-curve:** Held-out loss traces a distinct U-shaped curve:
  - At $\alpha \le 0.001$, insufficient smoothing leaves unseen transitions vulnerable, degrading loss to 2.523 nats.
  - At $\alpha \ge 1.0$, excessive smoothing flattens learned transitions toward uniform ignorance (2.607 and 2.892 nats).
  - Optimal balance occurs at $\alpha = 0.1$ (2.371 nats).

## Worked Example 1.6 — Hoeffding in numbers

**Question.** How much does the uncertainty bound widen when selecting the best of 1,000 models on the same validation set?

**Setting.** Loss bounded in $[0, 1]$ (e.g., error rate), with $n = 10\,000$ held-out samples and confidence $1 - \delta = 0.95$ ($\delta = 0.05$).

**Case 1: Single fixed model.**
$\varepsilon = \sqrt{\frac{\ln(2/\delta)}{2n}} = \sqrt{\frac{\ln 40}{20\,000}} = \sqrt{\frac{3.6889}{20\,000}} \approx \mathbf{0.0136}$.
With 95% confidence, the observed error rate is within **1.36 percentage points** of true risk.

**Case 2: Best of 1,000 candidate models.**
Applying the union bound over $K = 1\,000$ models:
$\varepsilon_K = \sqrt{\frac{\ln 1000 + 3.6889}{20\,000}} = \sqrt{\frac{6.9078 + 3.6889}{20\,000}} = \sqrt{\frac{10.5967}{20\,000}} \approx \mathbf{0.0230}$.

**What the result means.**
Searching over 1,000 models widens the uncertainty interval from 1.36% to 2.30%—an increase of approximately **70%**.
The bound does not scale by 1,000 because the union bound cost grows logarithmically as $\sqrt{\ln K}$. Nonetheless, this 0.94 percentage point shift represents genuine **optimism bias**: the winning model was partly boosted by lucky validation noise. Only an untouched test set provides an honest evaluation.

## Worked Example 1.7 — Standard error of a held-out loss

**Question.** When text is grouped into stories, what constitutes an independent observation?

**Setting.** Evaluate the bigram model on the 140 held-out stories (42,881 characters) from Worked Example 1.3.

**Approach A: Complete stories as units (statistically sound).**
Compute mean NLL per story across 140 stories: sample mean $2.1204$, sample standard deviation $s = 0.0617$.
Standard error:
$\mathrm{se} = \frac{0.0617}{\sqrt{140}} \approx \mathbf{0.0052}$ nats,
yielding an asymptotic 95% confidence interval of $2.1204 \pm 1.96(0.0052) \approx \mathbf{2.120 \pm 0.010}$ nats.

**Approach B: Individual characters as units (naive and flawed).**
42,881 characters yield 42,741 transitions with sample standard deviation $s_{\text{char}} = 1.244$.
Assuming independent characters yields:
$\mathrm{se}_{\text{naive}} = \frac{1.244}{\sqrt{42\,741}} \approx \mathbf{0.0060}$ nats.

**What the result means.**
The formula $\sigma / \sqrt{n}$ requires genuinely independent observations. Adjacent characters within a story exhibit strong temporal correlation (autocorrelation **$-0.32$**). Treating individual characters as independent draws violates basic statistical assumptions.

**Sample size planning:**
To confirm a performance difference of $d = 0.02$ nats with 95% confidence and 80% power (assuming paired sample standard deviation $\sigma_d = 0.5$):
$n \approx \Big(\frac{(1.96 + 0.84) \times 0.5}{0.02}\Big)^2 = \Big(\frac{1.40}{0.02}\Big)^2 = \mathbf{4\,900}$ paired samples.
Paired testing on identical test prompts subtracts out shared instance difficulty, dramatically lowering variance.

## Worked Example 1.8 — Memorization is cheap

**Question.** Does zero training error prove that a model has discovered meaningful generalizing rules?

**Setting.** Revisit §1.1's benchmark: 50,000 images with random labels uniformly assigned across 10 classes.

**Information capacity of a lookup table:**
A Python dictionary mapping each image hash to its random label achieves **100% training accuracy**, but drops to **10%** (chance) on test images.
Storing all labels for the 50,000 training images requires only $50{,}000 \times \log_2 10 \approx 166{,}096 \text{ bits} \approx \mathbf{20.27 \text{ KB}}$.

**What the result means.**
20 KB is trivial—smaller than a compressed thumbnail.
A standard ResNet-50 contains over 25 million 32-bit parameters, consuming roughly **100 MB** of memory.

**Modern neural networks possess raw storage capacity thousands of times greater than required to memorize training labels.**
Zero training error can be achieved by brute-force memorization alone. Generalization does not stem from "fewer parameters than data points," but from the inductive bias of network architectures and gradient-based optimization dynamics (continuous geometry in Unit 02 and implicit regularization in Unit 03).
