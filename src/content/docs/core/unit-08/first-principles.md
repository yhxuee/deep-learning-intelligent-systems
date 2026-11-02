---
title: "8.2 First principles"
description: Sampling one categorical by inverting its cumulative distribution, ancestral sampling of sequences, temperature and truncation, and generating continuous data by moving noise along a velocity field.
sidebar: { label: "8.2 First principles", order: 2 }
---

## 8.2.1 Drawing one sample from a categorical distribution

A model outputs probabilities $p = (p_1,\dots,p_V)$ for the next symbol. To draw a sample, form the cumulative sums $c_k = p_1 + \dots + p_k$, draw $u$ uniformly from $[0, 1)$, and return the first $k$ with $c_k > u$. Each symbol $k$ owns an interval of length $p_k$ in $[0, 1)$, so it is returned with probability exactly $p_k$. This **inverse-CDF** method is all the randomness a language model needs; `spine/stage08_sampling.py` implements it in one line (`draw`), and the spine test checks that 20 000 draws reproduce the probabilities to within 0.012.

## 8.2.2 Sampling a sequence: one conditional at a time

The chain rule writes a sequence distribution as a product of conditionals (Unit 07, §7.2.2). **Ancestral sampling** follows it literally: draw $x_1\sim q(x_1)$, then $x_2\sim q(x_2\mid x_1)$, and so on, appending each sample to the context. The probability of producing a particular sequence is the product of the probabilities of its steps — exactly $q(x_{1:T})$ (Proposition 8.1). **Ancestral sampling draws exactly from the model**; nothing is approximated. Each step costs one forward pass over the new token, which is why the KV cache of Unit 04 (§4.3.10) and the memory-bound arithmetic of Unit 06 (§6.3.4) govern generation speed.

Exactness is a double-edged property. If the model is a perfect copy of the data distribution, its samples are as good as the data — including its errors. If the model has put a little probability on bad continuations everywhere, sampling will eventually pick some of them, and in a long sequence "eventually" arrives.

## 8.2.3 Temperature: sharpening and flattening

Divide the scores by a **temperature** $\tau > 0$ before the softmax:

$$
p_\tau(k) = \frac{e^{z_k/\tau}}{\sum_j e^{z_j/\tau}} .
$$

$\tau = 1$ is the model; $\tau\to0$ puts all probability on the largest score (**greedy** decoding); $\tau\to\infty$ approaches uniform. For probabilities $(0.5, 0.3, 0.15, 0.05)$, temperature 0.5 gives $(0.685, 0.247, 0.062, 0.007)$ and temperature 2 gives $(0.379, 0.294, 0.208, 0.120)$; the entropy moves from 1.142 nats to 0.810 and 1.308 (Worked Example 8.2). Unit 04 met the same operation as the inverse temperature $\beta$ of attention.

Low temperature trades diversity for reliability: the model's most confident guesses are usually its best. But the limit, greedy decoding, is not "the most likely sequence" — choosing the best next token at every step does not maximize the product of probabilities — and it is prone to **loops**: once a phrase has been produced, the context makes the same continuation even more likely. The spine model's greedy output repeats "had a good idea and they fixed it together" for this reason.

## 8.2.4 Truncation: cut the unreliable tail

A model trained on a large corpus assigns small probabilities to many continuations that are individually unlikely and collectively frequent. **Truncated sampling** removes them: **top-$k$** keeps the $k$ most likely symbols; **nucleus** (top-$p$) sampling keeps the smallest set of most-likely symbols whose total probability reaches $p$, and renormalizes ([Holtzman et al. 2020](/references/#holtzman-2020)). Nucleus sampling adapts to the model's confidence: after "Now Ben has " the nucleus may be one digit; after "One windy day Ben went to " it contains several places. Both are modifications of the model's distribution, made at sampling time, that the training objective never saw.

## 8.2.5 Continuous data: transform noise

For a continuous variable there is no table of probabilities to invert — but the inverse-CDF idea still works in one dimension: if $u$ is uniform and $F$ is the cumulative distribution of the target, $F^{-1}(u)$ has the target distribution. More generally, any sample can be made by drawing **noise** from a simple distribution (uniform, Gaussian) and applying a **transformation**. A shift and scale $x = \mu + \sigma z$ turns a standard Gaussian into any Gaussian. A learned, flexible transformation can turn a Gaussian into an image distribution.

There are three ways to learn such a transformation, and this unit develops all three:

1. **In one step, invertibly** — a *normalizing flow* computes the exact likelihood of data through the change-of-variables formula (§8.3.2).
2. **Through a latent variable** — a *variational autoencoder* maps noise to data with a decoder and is trained through a lower bound on the likelihood (§8.3.3).
3. **In many small steps** — move a sample from noise toward data along a learned direction field, as a particle moves in a flow. *Diffusion* and *flow matching* learn the field (§8.3.4–§8.3.6). This is now the dominant approach for images, audio and video.

## 8.2.6 Transport along straight lines: the idea of flow matching

Pair each noise sample $x_0$ with a data sample $x_1$ and connect them by a straight line, $x_t = (1-t)x_0 + tx_1$ for $t$ from 0 to 1. A particle on that line moves with constant velocity $x_1 - x_0$. Now look at all such lines at once: at a point $x$ and time $t$, many lines pass, with different velocities. Their *average* velocity defines a field $v(x, t)$. **Flow matching** trains a network $v_\theta(x, t)$ to predict $x_1 - x_0$ from $(x_t, t)$ by least squares; the least-squares optimum is exactly that average ([Lipman et al. 2023](/references/#lipman-2023); [Liu et al. 2023](/references/#liu-2023)). To generate, start at noise and follow the field: solve $\dot x = v_\theta(x, t)$ from $t = 0$ to $t = 1$ with an ODE solver.

Two features make this attractive. Training is ordinary regression — no adversarial game, no likelihood to compute. And the ODE is the same object as the continuous-time memory of Unit 05: the number of solver steps trades accuracy for cost, and too few steps distort the samples in predictable ways (Worked Example 8.7). For Gaussian data the average velocity can be computed in closed form, which gives the spine a ground truth to test against.
