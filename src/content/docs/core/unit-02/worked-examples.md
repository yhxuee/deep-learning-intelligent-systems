---
title: "2.4 Worked examples"
description: Eight examples — XOR by hand, a one-hot read-out as a table, parameter accounting for long contexts, the factorized bigram's width, the geometry of a learned table, count versus neural 4-grams, the lookup gradient, and Cover's count for XOR.
sidebar: { label: "2.4 Worked examples", order: 4 }
---

Exact numbers are recomputed by `labs/test_worked_examples_u01_u02.py`; trained-model results (2.4–2.6) by `spine/stage02_embeddings.py` and checked in `spine/test_spine.py`, and quoted to two decimals.

## Worked Example 2.1 — XOR, impossible and then trivial

**Setting.** The four XOR points of §2.2.2.

**No line works.** The four conditions $b\le0$, $w_2 + b > 0$, $w_1 + b > 0$, $w_1 + w_2 + b\le 0$ are contradictory: the middle two sum to $w_1 + w_2 + 2b > 0$, i.e. $w_1 + w_2 + b > -b\ge 0$.

**One extra feature.** With $\phi = (x_1, x_2, x_1x_2)$, $w = (1, 1, -2)$, $b = -0.5$:

| $x$ | $\phi(x)$ | score $w^{\top}\phi + b$ | prediction | label |
|---|---|---|---|---|
| (0,0) | (0,0,0) | −0.5 | 0 | 0 |
| (0,1) | (0,1,0) | +0.5 | 1 | 1 |
| (1,0) | (1,0,0) | +0.5 | 1 | 1 |
| (1,1) | (1,1,1) | −0.5 | 0 | 0 |

**Reading.** The margin is the same 0.5 on every point. A two-layer network can *compute* a feature like $x_1x_2$ — for instance $\mathrm{ReLU}(x_1 + x_2 - 1)$ equals $x_1x_2$ on binary inputs — which is how hidden layers turn non-linear problems into linear ones.

## Worked Example 2.2 — A one-hot read-out is a table

**Setting.** The three-symbol alphabet of Worked Example 1.2 and its add-one smoothed bigram.

Put $W[a, b] = \ln q_1(b\mid a)$. For input `b` (one-hot $(0, 1, 0)$), the scores are $W^{\top}(0,1,0)^{\top} = W[\texttt b] = (\ln\tfrac26, \ln\tfrac16, \ln\tfrac36) = (-1.0986, -1.7918, -0.6931)$. Their softmax is $(\tfrac26, \tfrac16, \tfrac36)$ — exactly row `b` of the table, because the row already sums to one. Adding any constant to a row of $W$ changes nothing, so the linear model has $V(V-1)$ effective parameters: exactly the table's.

**Reading.** A linear model on one-hot inputs has no parameters that two inputs share. This is why Unit 01's "counting cannot share" is really "one-hot cannot share".

## Worked Example 2.3 — Parameter accounting for longer contexts

**Setting.** $V = 50$. Count models over the previous $k$ characters versus a neural n-gram with $d = 16$, $H = 128$.

| $k$ | possible contexts $V^k$ | contexts seen in the 385 274 training characters | neural n-gram parameters |
|---|---|---|---|
| 1 | 50 | 50 | 9 426 |
| 2 | 2 500 | 401 | 11 474 |
| 3 | 125 000 | 2 078 | 13 522 |
| 4 | 6 250 000 | 4 502 | 15 570 |
| 6 | 15 625 000 000 | 12 903 | 19 666 |

The neural count is $Vd + kdH + H + HV + V = 7\,378 + 2\,048k$.

**Reading.** The count model's parameter count is set by the contexts it happens to see; anything it has not seen gets a uniform guess. The neural model's size grows by 2 048 per extra character of context, and it can give a sensible prediction for a context it has never seen, provided the context's characters resemble characters it has seen in that position. (The synthetic corpus is highly repetitive, which is why so few of the possible contexts occur; on natural text the coverage problem is far worse.)

## Worked Example 2.4 — How wide must the embedding be?

**Setting.** The factorized bigram $z_{ab} = e_a^{\top}Me_b + c_b$ on the spine corpus, trained with Adam (1 500 steps of 4 096 pairs), for several widths $d$.

| $d$ | 2 | 4 | 8 | 16 | 32 | 50 | count bigram |
|---|---|---|---|---|---|---|---|
| parameters | 154 | 266 | 514 | 1 106 | 2 674 | 5 050 | 2 450 |
| held-out NLL | 2.65 | 2.37 | 2.21 | **2.15** | 2.15 | 2.15 | 2.12 |

**Reading.** Width 16 comes within 0.03 nats of the full count table with less than half its parameters; wider tables add nothing at this training budget (the last 0.03 nats would need longer training — the optimization question of Unit 03). With $d = 2$ the model must squeeze 50 characters onto a plane and loses 0.53 nats. The rank of the representation is a capacity knob, and most of the bigram's structure lives in a 16-dimensional subspace.

## Worked Example 2.5 — What the table learned

**Setting.** The $d = 16$ model of Worked Example 2.4. Nearest neighbours by cosine similarity of embeddings.

| query | nearest neighbours (cosine) |
|---|---|
| `3` | `4` 0.98, `2` 0.98, `1` 0.97, `5` 0.97 |
| `T` | `Z` 0.86, `N` 0.85, `S` 0.83, `L` 0.76 |
| `a` | `u` 0.45, `e` 0.39, `A` 0.23 |
| `.` | `,` 0.59 |

**Reading.** The model was told nothing about digits, capitals or vowels; it was only asked to predict the next character. Digits are almost interchangeable as predictors in this corpus (each is followed by a space and preceded by a space or `had`), so their embeddings nearly coincide. Capitals that start names cluster together. Vowels are weakly similar — they share some contexts but not most. Similarity in the learned geometry measures *interchangeability for the task*, nothing more (§2.7.4).

## Worked Example 2.6 — When does sharing pay?

**Setting.** Predict from the previous four characters. Count 4-gram with the best α from $\{0.003, 0.01, 0.03, 0.1, 0.3\}$ versus a neural 4-gram ($d = 16$, $H = 128$, 2 500 Adam steps of 512 windows, weight decay $10^{-4}$), trained on the first $n$ characters.

| training characters $n$ | 5 000 | 20 000 | 80 000 | 385 274 |
|---|---|---|---|---|
| count 4-gram, held-out NLL | 1.11 | 0.69 | 0.56 | 0.52 |
| neural 4-gram, held-out NLL | **0.98** | **0.63** | 0.56 | 0.54 |

**Reading.** With 5 000 characters the representation is worth 0.13 nats; with 80 000 the two are tied; with everything, the count model is slightly better (the neural model is small and briefly trained). Sharing is a way to *spend fewer examples per situation*. When examples are plentiful for every situation that matters, a model that shares nothing can do as well — but for any model with long contexts, that regime never arrives (Worked Example 2.3).

## Worked Example 2.7 — The gradient of a lookup

**Setting.** $V = 3$, $d = 2$, indices $x = (0, 2, 0)$, upstream gradient $G = \begin{pmatrix}1&2\\3&4\\5&6\end{pmatrix}$ (one row per looked-up position).

**Scatter-add** (Proposition 2.1): row 0 receives rows 1 and 3 of $G$, row 2 receives row 2:

$$
\frac{\partial L}{\partial E} = \begin{pmatrix}1+5 & 2+6\\ 0 & 0\\ 3 & 4\end{pmatrix} = \begin{pmatrix}6&8\\0&0\\3&4\end{pmatrix} = U^{\top}G,\quad U = \begin{pmatrix}1&0&0\\0&0&1\\1&0&0\end{pmatrix}.
$$

**The bug.** Writing `dE[x] = G` instead of `np.add.at(dE, x, G)` leaves row 0 equal to $(5, 6)$ — the last write wins — and silently halves the learning signal for every character that occurs twice in a batch. A [gradient check](/atlas/nodes/method-gradient-check/) catches it; a loss curve usually does not.

## Worked Example 2.8 — Counting what a line can do

**Setting.** Four points in general position in the plane, and affine classifiers $\mathrm{sign}(w^{\top}x + b)$.

**Cover's count** (Theorem 2.3, §2.5.2): $N$ points in general position in $\mathbb{R}^p$ admit $C(N, p+1) = 2\sum_{k=0}^{p}\binom{N-1}{k}$ affinely separable labellings. With $N = 4$, $p = 2$: $C = 2\big(\binom30 + \binom31 + \binom32\big) = 2(1 + 3 + 3) = 14$ of the $2^4 = 16$ possible labellings.

**Reading.** For the four XOR points, the two missing labellings are exactly XOR and its complement. Lifting to $\phi = (x_1, x_2, x_1x_2)$ puts the points in $\mathbb{R}^3$, where $C(4, 4) = 2(1 + 3 + 3 + 1) = 16$: every labelling becomes separable. Adding dimensions makes more labellings linearly reachable — the reason random features and wide hidden layers help, and the reason high-dimensional representations can also fit noise (Unit 01, §1.5.4).
