---
title: "1.2 First principles"
description: Learning as estimating a distribution — the coin, the log-loss, the uniform baseline, maximum likelihood as counting, the bigram model, zero counts and smoothing, and why held-out data is needed.
sidebar: { label: "1.2 First principles", order: 2 }
---

We begin with the smallest setting in which learning is possible, then enlarge it one step at a time until it becomes a language model.

## 1.2.1 The smallest learner: a coin

A coin lands heads with some unknown probability $p$. You see ten tosses, seven of them heads. You must now state a probability $q$ for "heads" on the *next* toss — a toss you have not seen.

A natural first estimate is $q = 0.7$, the observed frequency. Before justifying it, notice what has happened: you used **past cases** (ten tosses) to make a **prediction about a future case** (the eleventh). This is learning in its most stripped-down form, and it needs three ingredients:

1. A **model**: here a single number $q$, the predicted probability of heads.
2. A way to **score** a prediction once the outcome is known.
3. A reason to believe the **past tells us something about the future**.

The rest of this section supplies the second and third ingredients.

## 1.2.2 Why the past can inform the future

Suppose the eleventh toss used a *different* coin, chosen by an adversary after seeing your prediction. No prediction would then be defensible: whatever you say, the adversary could choose a coin that makes you wrong. Learning from experience therefore needs an assumption that connects the cases you have seen with the case you have not.

The standard assumption is that all tosses — past and future — are **independent and identically distributed (i.i.d.)**. *Identically distributed* means that every toss comes from the same distribution. *Independent* means that, beyond what a toss reveals about that shared distribution, its outcome tells you nothing extra about another toss. Under this assumption, the **law of large numbers** says that the frequency of heads in $n$ tosses converges to $p$ as $n$ grows. The past is informative because it is a sample from the same source as the future.

Everything in this unit rests on this assumption, and much of [Unit 10](/core/unit-10/) examines what happens when it fails. Keep the point in view: **a model generalizes only relative to an assumption about how its training data are related to future data.** In the real dataset of §1.1, image content connects the training labels to future labels. After the labels are shuffled, that connection disappears, so there is no pattern to carry from training photographs to new ones.

## 1.2.3 Scoring a probabilistic prediction: the log-loss

You predicted $q = 0.7$ for heads. The toss comes up tails. How bad was the prediction?

A tempting score is *accuracy*: predict the more likely outcome and count a hit or miss. But accuracy throws away the probability. A forecaster who says "70% heads" and one who says "99.9% heads" both predicted heads; if tails comes up, the second should be penalized far more — they claimed near-certainty and were wrong.

The score used throughout this book is the **log-loss** (also called the negative log-likelihood or surprisal):

$$
\ell(q; x) = -\ln q(x),
$$

where $q(x)$ is the probability your prediction assigned to the outcome $x$ that actually occurred. Because the formula uses the natural logarithm $\ln$, the loss is measured in **nats** (short for *natural units of information*). Predicting $q(\text{tails}) = 0.3$ and seeing tails costs $-\ln 0.3 = 1.204$ nats; predicting $q(\text{tails}) = 0.001$ and seeing tails costs $6.91$ nats; predicting $q(\text{tails}) = 0$ costs $+\infty$ — a forecaster who rules out something that then happens is infinitely wrong.

Three properties make the log-loss the right default:

- **It rewards honesty.** If the true probability of heads is $p$, the *expected* log-loss of predicting $q$ is $-p\ln q - (1-p)\ln(1-q)$. Setting the derivative with respect to $q$ to zero gives $q = p$: you minimize your expected loss by stating your actual belief. A score with this property is called **proper** ([Gneiting & Raftery 2007](/references/#gneiting-2007)); accuracy is not (it is indifferent between 51% and 99%).
- **It adds up over independent events.** The probability of a sequence of independent outcomes is a product, so its log-loss is the *sum* of the individual log-losses. Averages over many predictions are therefore meaningful.
- **It measures information.** In base 2, $-\log_2 q(x)$ is the number of bits an optimal code built from $q$ uses to transmit $x$ (§1.6.1). A model with lower log-loss is literally a better compressor of the data.

The scale is easy to read: assigning the outcome probability $0.5$ costs $-\ln 0.5 = 0.693$ nats, while assigning it probability $0.1$ costs $2.303$ nats. Lower is better. Divide a value in nats by $\ln 2 \approx 0.693$ to convert it to bits.

## 1.2.4 The first baseline: predicting uniformly

Before fitting anything, ask what loss you get by knowing nothing. If there are $V$ possible outcomes and you assign each probability $1/V$, every outcome costs exactly

$$
-\ln\frac{1}{V} = \ln V \text{ nats}.
$$

This is the **uniform baseline**. For a coin, $\ln 2 = 0.693$. For the 50 distinct characters of the spine corpus, $\ln 50 = 3.912$ nats per character. **A model must beat $\ln V$ on unseen data before we can claim that it has learned something useful for this prediction task.** The baseline also serves as a check on code: a freshly initialized language model, whose predictions are close to uniform, should produce a loss close to $\ln V$ — Units 03 and 04 use exactly this check.

## 1.2.5 Learning by counting: maximum likelihood

Return to the coin. Among all values of $q$, which one makes the observed data — seven heads, three tails — most probable? The probability of the data is $q^7(1-q)^3$, and its logarithm is $7\ln q + 3\ln(1-q)$. Setting the derivative to zero,

$$
\frac{7}{q} - \frac{3}{1-q} = 0 \quad\Longrightarrow\quad q = \frac{7}{10}.
$$

The observed frequency is the **maximum-likelihood estimate (MLE)**. Maximizing the probability of the data is the same as *minimizing the total log-loss on the training data*, because log-loss is the negative log-probability. Thus “fit by maximum likelihood” and “minimize the training log-loss” are two ways of giving the same instruction — the basis of likelihood training throughout this book.

The same reasoning extends from two outcomes to $V$: if outcome $v$ occurred $c_v$ times out of $n$, the MLE is $q(v) = c_v/n$ (derived in §1.3.3). Learning a distribution over characters, *ignoring context*, is counting how often each character occurs. This is the **unigram** model. On the spine corpus its held-out loss is 3.06 nats per character — already 0.85 nats better than the uniform baseline, because some characters (space, `e`, `a`) are far more common than others (`Z`, `9`).

## 1.2.6 Using context: the bigram model

Characters are not independent: after `q` comes `u`; after a full stop comes a space. A model that conditions on the previous character can do better. The **bigram model** keeps one distribution *per preceding character*: $q(b \mid a)$ is the probability that $b$ follows $a$. By the same argument as before, its maximum-likelihood estimate is a normalized count:

$$
q(b \mid a) = \frac{c(a \to b)}{c(a)},
$$

where $c(a\to b)$ counts occurrences of the pair "$a$ then $b$" in the training text and $c(a) = \sum_b c(a\to b)$. The whole model is a $V\times V$ table of counts, normalized row by row. It is our **anchor exemplar**: simple enough to compute by hand, yet substantial enough to serve as the first stage of the spine model.

On the spine corpus, the bigram reaches a held-out loss of **2.124 nats per character**, a full 1.79 nats below the uniform baseline and 0.94 below the unigram. The progression $\ln V \to$ unigram $\to$ bigram introduces a pattern repeated throughout the book: evaluate each model against its predecessor on held-out data.

## 1.2.7 The zero-count problem and smoothing

Now fit the bigram on only the first 2 000 characters of the corpus. Of the $50^2 = 2\,500$ possible pairs, only 269 occur. A held-out text inevitably contains some pairs that never appeared in training — on the spine corpus, 5.4% of held-out pairs are unseen. The maximum-likelihood table assigns them probability zero, and a single zero makes the held-out log-loss **infinite**.

The fix is to pretend we saw every pair a small number $\alpha$ of extra times — **add-$\alpha$ smoothing**:

$$
q_\alpha(b\mid a) = \frac{c(a\to b) + \alpha}{c(a) + \alpha V}.
$$

With $\alpha = 1$ this is Laplace's "rule of succession" from 1814; §1.3.4 shows it is the Bayesian answer under a particular prior. Smoothing **trades fit for robustness**: as $\alpha \to 0$ we recover the maximum-likelihood table (best on the training data, fragile on new data); as $\alpha\to\infty$ every row becomes uniform and the loss becomes exactly $\ln V$. Somewhere in between is the best held-out performance. Worked Example 1.5 finds it: on 2 000 training characters the held-out loss is 2.523 at $\alpha = 0.001$, 2.371 at $\alpha = 0.1$ and 2.607 at $\alpha = 1$.

This U-shape is the simplest instance of the central tension of learning: **a model flexible enough to fit the training data exactly is also flexible enough to fit accidental patterns in those data.** Choosing $\alpha$ is choosing how much to trust the data versus a prior belief.

## 1.2.8 Why you must hold data out

How did we choose $\alpha$? By measuring the loss on text the counts did not see. If we had measured on the training text instead, the answer would always be $\alpha = 0$: the unsmoothed table is, by construction, the best possible bigram *on its own training data* (§1.3.3). The training loss is a biased, optimistic estimate of performance on new data — the model has been chosen to make it small.

This gives the three data splits used throughout the rest of the book:

| Split | Used for | Never used for |
|---|---|---|
| **Training set** | fitting parameters (the counts) | judging the model |
| **Validation set** | choosing among models and settings ($\alpha$, architecture, training length) | final claims |
| **Test set** | one final, honest estimate of performance on new data | any decision |

The reason the validation set cannot double as the test set is the same reason the training set cannot: once you *choose* something because it scored well on a set, that set's score is no longer an unbiased estimate. §1.3.6 makes this quantitative; [Unit 10](/core/unit-10/) and [Unit 12](/core/unit-12/) return to it at scale.

One more rule: **split by the unit that is independent.** The spine corpus is a collection of stories. Characters within a story are strongly dependent; stories are (approximately) independent. So the held-out set is the last 10% of *stories*, not a random 10% of characters — a random character split would put half of a story in training and the other half in "held-out" data, and the held-out loss would measure memory of the story, not generalization to new ones.

## 1.2.9 What we have so far

| Idea | In the coin example | In the bigram |
|---|---|---|
| model | one number $q$ | a $V\times V$ table of conditional probabilities |
| fitting | frequency of heads | normalized pair counts |
| score | $-\ln q(x)$ | mean $-\ln q(x_t\mid x_{t-1})$ in nats/char |
| baseline | $\ln 2$ | $\ln V = 3.912$ |
| assumption | i.i.d. tosses | stories drawn from one source |
| risk | overconfidence on 10 tosses | zero counts; fitting accidents of a small corpus |

The next section makes each row precise.

:::tip[Claim check — a template used throughout the book]
Every major claim in this book can be stated in four lines. Applied to this section's central claim:

- **Claim.** The bigram's held-out NLL estimates its loss on new stories from the same source.
- **Assumptions.** Held-out stories are drawn independently from the same source as the training stories, and nothing was chosen by looking at them (§1.2.8).
- **Test.** Compute the standard error over stories (Worked Example 1.7), then score a fresh set of held-out stories: the two estimates should agree within that error.
- **Limitation.** It says nothing about text from another source (§1.7.1), and it becomes optimistic once the held-out set has been used to make choices (§1.7.3).

Units 03, 04, 06, 07, 09 and 10 each fill in the template once for a claim of their own. When you meet a claim anywhere else, fill it in yourself; a claim whose test you cannot state is not yet a finding (Unit 12).
:::

