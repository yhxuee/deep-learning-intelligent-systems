---
title: "1.2 First principles"
description: Learning as estimating a distribution — the coin, the log-loss, the uniform baseline, maximum likelihood as counting, the bigram model, zero counts and smoothing, and why held-out data is needed.
sidebar: { label: "1.2 First principles", order: 2 }
---

We construct the foundations of learning from the smallest mathematical example that encapsulates the concept, expanding it step by step until it transforms into a language model.

## 1.2.1 The smallest learner: a coin

Imagine an unbalanced physical coin whose true probability of landing heads is an unknown constant $p \in [0, 1]$.

You observe ten consecutive tosses: **seven heads and three tails**. You must now output a predicted probability $q \in [0, 1]$ that the *eleventh* toss—an unobserved future event—will land heads.

The intuitively obvious answer is $q = 0.7$, the empirical relative frequency. Before formalizing this intuition, pause to observe what has fundamentally transpired: you used **past empirical records** (ten historical observations) to make a **reasoned prediction about an unobserved future outcome** (the eleventh toss).

This is machine learning and statistical inference in its purest, most elemental form. For this inductive leap to be valid, three foundational pillars are indispensable:

1. **A Model:** A mathematical representation of belief—here, a single parameter $q$, quantifying the agent's expectation of observing heads.
2. **A Scoring Rule:** An objective, mathematically principled metric to evaluate the prediction once reality is revealed (whether the coin actually lands heads or tails).
3. **An Inductive Principle:** A justifiable reason to believe that the past carries genuine, reliable information about the unobserved future.

The remainder of this section establishes the mathematical and philosophical foundations of the second and third pillars.

## 1.2.2 Why the past can inform the future: induction and the i.i.d. assumption

Suppose the eleventh toss were made with a *different* coin, deliberately chosen by an adversary after observing your prediction $q$. In that scenario, no prediction could ever be justified: if you predict $q = 0.7$, the adversary simply selects a double-tailed coin ($p = 0$) to maximize your error.

This is the famous **Problem of Induction**, formulated by philosopher David Hume in the 18th century: why should past observations ("the sun has risen every morning in recorded history") logically guarantee future events ("the sun will rise tomorrow morning")? Pure empirical observation cannot self-justify future regularity without an underlying assumption.

In modern statistical learning theory, this bridge is formalized as the **independent and identically distributed (i.i.d.) assumption**:
- **Identically Distributed:** Historical training data points and unobserved future test data points are drawn from the exact same underlying data distribution $p_{\text{data}}$.
- **Independent:** Each individual observation occurs without influencing or being influenced by other draws, conditional on the generating distribution.

Under the i.i.d. assumption, the **Law of Large Numbers** guarantees that as the sample size $n \to \infty$, the observed frequency $s/n$ converges with probability 1 to the true parameter $p$. The past is informative precisely because past and future are samples drawn from the same underlying reality.

Keep this central principle in mind: **Generalization is never an isolated property of a neural network or learning algorithm. It is a joint property of the model coupled with the structural relationship between training and deployment distributions.**

In the random-label experiment of §1.1, researchers shuffled the training labels of 50,000 images. The structural coupling between input features and target labels was severed. Although a deep network achieved 100% training accuracy through brute-force memorization, it scored exactly 10% on test data (random chance). Without an invariant data distribution, empirical optimization degenerates into useless noise memorization.

## 1.2.3 Scoring a probabilistic prediction: the log-loss and strict propriety

Suppose you predict $q = 0.7$ for heads. The eleventh toss lands tails. Exactly how severe was your error?

A naive approach is **classification accuracy**: since $q = 0.7 > 0.5$, the model predicts "heads"; since the outcome was tails, record a mistake (loss 1 for error, 0 for correct).

However, accuracy is deeply flawed for probabilistic systems: **it completely discards confidence and uncertainty.**
Consider two weather forecasters:
- Forecaster A predicts a $51\%$ probability of rain tomorrow.
- Forecaster B predicts a $99.9\%$ probability of rain tomorrow.

If tomorrow is completely sunny, zero-one accuracy penalizes both forecasters identically with an error of 1. But this is absurd: Forecaster A expressed mild uncertainty, leaving generous room for doubt; Forecaster B asserted dogmatic certainty and was catastrophically wrong. Dogmatic overconfidence that contradicts reality demands a much harsher penalty.

To properly penalize errors, this book uses the **log-loss** throughout, also known as the **negative log-likelihood (NLL)** or **surprisal**:

$$
\ell(q; x) = -\ln q(x),
$$

where $x$ is the outcome that actually occurred, and $q(x)$ is the probability assigned to $x$ by your model prior to the event.

Observe how the log-loss scales continuously and penalizes overconfidence:
- If the coin lands tails and you predicted $q(\text{tails}) = 0.3$, your loss is $-\ln 0.3 \approx 1.204$ nats.
- If you predicted $q(\text{tails}) = 0.001$, your loss leaps to $-\ln 0.001 \approx 6.908$ nats.
- If you dogmatically declared $q(\text{tails}) = 0$ (asserting that tails is physically impossible), your loss explodes to $-\ln 0 = +\infty$!

Three fundamental mathematical properties establish the log-loss as the default objective across machine learning:

### 1. Rewards Honesty (Strict Propriety)
Suppose nature's true probability of heads is $p$. If a forecaster reports probability $q$, the forecaster's **expected log-loss** is:

$$
\mathbb{E}_{X\sim p}[\ell(q; X)] = -p\ln q - (1-p)\ln(1-q).
$$

To find the report $q$ that minimizes expected penalty, differentiate with respect to $q$:

$$
\frac{d}{dq}\mathbb{E}[\ell(q; X)] = -\frac{p}{q} + \frac{1-p}{1-q} = 0 \quad\implies\quad \frac{p}{q} = \frac{1-p}{1-q} \quad\implies\quad q = p.
$$

The second derivative is $\frac{p}{q^2} + \frac{1-p}{(1-q)^2} > 0$ for all $q \in (0, 1)$, confirming that $q = p$ is the unique global strict minimum.

This proves that **under the log-loss, an agent's optimal mathematical strategy is to report its true internal belief with complete honesty.** Scoring rules with this incentive-compatible property are called **strictly proper scoring rules** ([Gneiting & Raftery 2007](/references/#gneiting-2007)). Accuracy is not proper (it is indifferent between 51% and 99%), whereas log-loss actively drives models toward calibrated probability distributions.

### 2. Additivity Over Independent Events
For independent sequential events, joint probability is the product $\prod_i q(x_i)$. Taking the negative logarithm converts the product into an additive sum of losses: $\sum_i -\ln q(x_i)$. Averaging log-loss across a dataset is mathematically consistent and numerically stable.

### 3. Optimal Compression and Information Theory
In base 2, $-\log_2 q(x)$ is precisely the theoretical minimum number of bits required to encode outcome $x$ using an optimal code (such as Huffman or arithmetic coding) constructed from $q$ (§1.6.1). A language model that achieves lower log-loss on held-out text is, in a literal mathematical sense, a **superior data compressor**.

We measure log-loss in **nats** (natural logarithm). Dividing by $\ln 2 \approx 0.69315$ converts nats to bits ($1 \text{ nat} \approx 1.4427 \text{ bits}$).

## 1.2.4 The first baseline: the cost of uniform ignorance $\ln V$

Before training any neural network or tuning hyperparameters, a rigorous researcher establishes a reference point: **what loss does an agent incur by knowing nothing?**

Suppose a task has $V$ discrete possible outcomes (such as $V = 2$ for a coin, or $V$ tokens in a vocabulary). In complete ignorance, the only unbiased prediction is the uniform distribution $q(x) = 1/V$. When any outcome occurs, the log-loss is identically:

$$
-\ln\frac{1}{V} = \ln V \text{ nats}.
$$

This is the **uniform baseline**:
- For a coin ($V = 2$), the uniform baseline is $\ln 2 \approx 0.6931$ nats.
- For our spine corpus of $V = 50$ distinct characters, the baseline is $\ln 50 \approx 3.9120$ nats per character.

This baseline serves two crucial functions:
1. **The Minimal Threshold of Generalization:** Any model that fails to achieve a held-out loss below $\ln V$ has learned nothing useful from data—its predictions are inferior to a blind random guess.
2. **Initialization Sanity Check:** At step 0 of training a Transformer or deep network, weights are randomly initialized, so output logits should be approximately uniform. If initial loss deviates significantly from $\ln V$, it immediately flags an implementation bug (e.g., incorrect normalization or a transposed loss axis).

## 1.2.5 Learning by counting: maximum likelihood estimation

Returning to our coin toss: among all candidate values $q \in [0, 1]$, which one maximizes the probability of observing seven heads and three tails?

The joint probability of the data (the likelihood function) is $L(q) = q^7(1-q)^3$. Taking the natural logarithm yields the log-likelihood: $\ln L(q) = 7\ln q + 3\ln(1-q)$.
Setting the derivative to zero:

$$
\frac{7}{q} - \frac{3}{1-q} = 0 \quad\Longrightarrow\quad q = \frac{7}{10}.
$$

The result matches the observed relative frequency $7/10$. This is the **maximum-likelihood estimate (MLE)**.

Notice the duality: **maximizing the log-likelihood of data is mathematically identical to minimizing total log-loss on the training set.**
Training a deep neural network on trillions of tokens with cross-entropy is the exact multi-parameter extension of this fundamental counting principle.

The same derivation extends from two outcomes to a vocabulary of $V$ symbols: if symbol $v$ appears $c_v$ times in $n$ observations, the MLE is $q(v) = c_v / n$ (proven in §1.3.3). A context-free model that simply predicts each character by its empirical frequency is the **unigram model**. On the spine corpus, its held-out loss is **3.060 nats per character**—0.852 nats better than the uniform baseline (3.912 nats), because characters such as space, `e`, and `a` occur far more frequently than `Z` or `9`.

## 1.2.6 Exploiting context: the bigram model and the combinatorial cliff

Human language is structured sequentially: `q` is almost invariably followed by `u`; a period is almost always followed by a space. Conditioning on the previous character yields substantial predictive gains.

The **bigram model** maintains a separate probability distribution for every preceding character: $q(b \mid a)$ is the probability that character $b$ follows character $a$. By the same MLE principle, the optimal estimate is the normalized transition count:

$$
q(b \mid a) = \frac{c(a \to b)}{c(a)},
$$

where $c(a\to b)$ counts occurrences of the pair "$a$ followed by $b$" in training data, and $c(a) = \sum_b c(a\to b)$ is the total occurrences of $a$.

Mathematically, the bigram model is a $V \times V$ matrix of transition probabilities. It is our **anchor exemplar**: simple enough to calculate by hand, yet rich enough to serve as Stage 01 of the spine language model.

On the spine corpus, the bigram drops held-out loss to **2.124 nats per character**!
Notice the progression:
- **Uniform ignorance:** 3.912 nats per character;
- **Unigram (no context):** 3.060 nats per character (down 0.852 nats);
- **Bigram (context length 1):** 2.124 nats per character (down another 0.936 nats).

### The Combinatorial Cliff
If conditioning on 1 character yields such dramatic gains, why not condition on the preceding 10 or 100 characters?
This immediately encounters the **Curse of Dimensionality**:
- Context length 1 (Bigram): $V^2 = 50^2 = 2\,500$ parameters;
- Context length 2 (Trigram): $V^3 = 50^3 = 125\,000$ parameters;
- Context length 5: $V^6 = 50^6 \approx 1.56 \times 10^{10}$ parameters (15.6 billion);
- Context length 10: $V^{11} \approx 4.88 \times 10^{18}$ parameters!

For modern word-level tokenizers with $V \approx 100\,000$, a 4-token context demands $100\,000^5 = 10^{25}$ parameters—exceeding the number of sand grains on Earth.

**Discrete counting cannot scale to long contexts.** The physical universe cannot provide enough text to populate even a microscopic fraction of this table. This fundamental bottleneck compelled the deep learning revolution in [Unit 02](/core/unit-02/): replacing discrete lookup tables with continuous, dense vector representations (Embeddings) that share statistical strength across semantically related contexts.

## 1.2.7 The zero-count crisis and smoothing

Even in small bigram tables ($V = 50$), discrete counting faces an immediate failure mode on finite data.

When trained on the first 2,000 characters of the spine corpus, the text covers only 269 transitions out of $2\,500$ possible pairs; **89.2% of the table remains zero**.
When evaluating on held-out text, unseen transitions inevitably appear (5.4% of held-out pairs were never seen in training).

For these unobserved transitions, the MLE table assigns $q(b \mid a) = 0$. The log-loss penalty triggers instantly:

$$
-\ln q(b \mid a) = -\ln 0 = +\infty.
$$

**A single unseen transition renders the entire held-out loss infinite.**

This reveals a fatal flaw of empirical frequency: **an unobserved event in a finite sample is not an impossible event in reality.** Zero counts reflect sampling sparsity rather than physical impossibility.

To prevent catastrophic collapse, we must eliminate absolute zero probabilities. The classic solution is **add-$\alpha$ smoothing**: pretend that prior to observing real data, every possible transition was pre-observed $\alpha > 0$ times:

$$
q_\alpha(b\mid a) = \frac{c(a\to b) + \alpha}{c(a) + \alpha V}.
$$

### Laplace's Rule of Succession: History and Intuition
When $\alpha = 1$ and outcomes are binary ($V = 2$, such as coin tosses), this formula reduces to Pierre-Simon Laplace's celebrated **rule of succession** (1814).

Laplace formulated the rule to resolve the "Sunrise Problem": **if the sun has risen every day for $n$ days in human history ($n$ successes, 0 failures), what is the probability that it will rise again tomorrow?**
- Under MLE, the frequency is $n/n = 1.0$. The model declares sunrise tomorrow 100% certain, assigning probability 0 to failure. But if an unprecedented celestial anomaly occurred, the penalty would be infinite.
- Laplace adopted a Bayesian stance: assuming a uniform prior $\mathrm{Uniform}(0, 1)$ over the true probability, observing $s$ successes in $n$ trials yields a posterior expectation of:

$$
q(X_{n+1}=1 \mid s, n) = \frac{s + 1}{n + 2}.
$$

The intuition is transparent: adding 1 to the numerator and 2 to the denominator introduces two "virtual observations"—one prior success and one prior failure:
- Before any data ($n = 0, s = 0$), the formula yields $(0+1)/(0+2) = 1/2$ (unbiased symmetry under total ignorance).
- After one toss that lands heads ($n = 1, s = 1$), MLE declares heads 100% certain, while Laplace offers a measured $(1+1)/(1+2) = 2/3$.
- After millions of days, the virtual observations vanish asymptotically, approaching 1 without ever dogmatically claiming impossibility.

Add-$\alpha$ smoothing generalizes Laplace's principle to $V$ symbols: each token receives $\alpha$ pseudo-counts, and the denominator expands by $\alpha V$ to maintain normalization. §1.3.4 proves this is the exact Bayes-optimal prediction under a symmetric Dirichlet prior.

### The Tradeoff: Bias vs. Variance
Smoothing deliberately sacrifices training fit to guarantee robustness on unseen data:
- As $\alpha \to 0$, the model collapses to pure MLE (perfect training loss, but brittle and vulnerable to $-\ln 0 = \infty$).
- As $\alpha \to \infty$, the prior overwhelms the data, forcing all probabilities to $1/V$ (loss collapses to $\ln V$).

Between these extremes lies an optimal $\alpha^*$. Worked Example 1.5 plots this **U-shaped validation curve**: on 2,000 training characters, held-out loss starts at 2.523 nats at $\alpha = 0.001$, reaches a minimum of 2.371 nats at $\alpha = 0.1$, and degrades to 2.607 nats when over-smoothed at $\alpha = 1.0$.

This U-curve illustrates the core tension in machine learning: **a model flexible enough to fit training data is flexible enough to overfit empirical noise. Selecting hyperparameters balances empirical fidelity against prior regularization.**

## 1.2.8 The discipline of data splitting: train, validation, and test

How did we discover the optimal smoothing parameter $\alpha = 0.1$? We evaluated loss on an independent held-out dataset that was excluded from count estimation.

If we had chosen $\alpha$ by minimizing loss on the training text itself, the answer would always be $\alpha = 0$: the raw MLE table is mathematically optimal on its own training sample (§1.3.3).

**Training loss is systematically optimistic.** Because the model was chosen specifically to minimize training error, evaluating performance on training data is like allowing a student to grade their own exam.

This establishes the **threefold data discipline** of modern AI:

| Data Split | Primary Purpose | Strict Prohibition |
|---|---|---|
| **Training Set** | Fit internal parameters (counts $c(a\to b)$ or neural weights $W, b$) | Never use to evaluate generalization |
| **Validation Set** | Select hyperparameters ($\alpha$, learning rates, architectures) | Never report as final unbiased benchmark |
| **Test Set** | One-time honest evaluation of final frozen model | Never use to guide any tuning or decisions |

Why can a validation set not serve as a test set? Because **repeatedly selecting the best hyperparameter configuration leaks validation information into your design decisions.** If you test 1,000 configurations, the best validation score reflects lucky sample noise as well as true performance (§1.3.6 quantifies this cost of selection).

### The Principle of Independent Units
A crucial engineering rule: **split data by independent high-level units, never by arbitrary internal chunks.**

In our corpus, text is organized into distinct stories. Characters within a story are tightly coupled, while separate stories are approximately independent.
Therefore, the held-out split must comprise the **final 10% of complete stories**, not a random 10% of characters. Randomly splitting characters places fragments of the same narrative into both training and test sets, allowing the model to achieve artificial accuracy by memorizing plot fragments rather than learning reusable linguistic rules.

## 1.2.9 Summary of concepts

| Concept | In Coin Tossing | In Bigram Language Modeling |
|---|---|---|
| **Parameters** | Single probability $q \in [0, 1]$ | $V \times V$ transition matrix $q(b \mid a)$ |
| **Fitting** | Relative frequency $s/n$ | Normalized counts $c(a\to b)/c(a)$ |
| **Scoring** | Single log-loss $-\ln q(x)$ | Mean negative log-likelihood in nats/char |
| **Baseline** | $\ln 2 \approx 0.6931$ nats | $\ln V = \ln 50 \approx 3.9120$ nats/char |
| **Assumption** | Independent coin tosses | Stories drawn i.i.d. from language source |
| **Failure Mode** | Overconfidence from small samples | Zero counts cause infinite loss; noise overfitting |

:::tip[Claim check template]
Every significant claim in empirical science must state its verification conditions. The template used throughout this book:

- **Claim.** The bigram model's held-out NLL accurately estimates expected loss on future stories from the same generator.
- **Assumptions.** Held-out stories and training stories are sampled i.i.d. from the same source, and held-out data was strictly quarantined during model fitting and selection (§1.2.8).
- **Test.** Compute standard error across independent stories (Worked Example 1.7) and evaluate on a fresh set of held-out stories: estimates must agree within error bars.
- **Limitation.** The estimate provides no guarantee under distribution shift (§1.7.1), and repeated model selection on the held-out set induces optimistic bias (§1.7.3).
:::
