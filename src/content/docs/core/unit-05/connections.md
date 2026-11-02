---
title: "5.6 Connections"
description: How the unit's memory mechanisms connect to signal processing, optimal estimation and the Kalman filter, numerical ODEs, hidden Markov models, neuroscience, retrieval systems, and other units of the book.
sidebar: { label: "5.6 Connections", order: 6 }
---

Deep-learning memory did not appear from nowhere. Each mechanism in this unit is a well-studied object in another field, and knowing the connection lets you borrow decades of theory.

## 5.6.1 Signal processing: recurrences are IIR filters, windows are FIR filters

A causal filter that computes $y_t$ from a **finite** window of past inputs, $y_t = \sum_{j=0}^{W-1}K_jx_{t-j}$, is a *finite impulse response* (FIR) filter — a convolution, as in Unit 04's convolutional layers. A recurrence produces an impulse response that never exactly reaches zero (§5.3.2): an *infinite impulse response* (IIR) filter. The EMA is the first-order IIR low-pass filter of every signal-processing course; §5.5.1 derived its frequency response. The engineering trade-offs are classical: IIR filters are cheap (constant state) but can be unstable and have phase delay; FIR filters are always stable but cost grows with the window ([Oppenheim & Schafer 2010](/references/#oppenheim-2010)). The same trade-off reappears in the sequence-model table of §5.3.13.

## 5.6.2 Optimal estimation: the EMA is a Kalman filter

Section 5.2 chose $\alpha$ by trading noise against lag. Estimation theory gives a principled choice. Model the true temperature as a random walk observed in noise:

$$
\mu_t = \mu_{t-1} + w_t,\quad \mathrm{Var}(w_t) = q; \qquad x_t = \mu_t + \varepsilon_t,\quad \mathrm{Var}(\varepsilon_t) = r .
$$

The **Kalman filter** ([Kalman 1960](/references/#kalman-1960)) gives the minimum mean-squared-error estimate of $\mu_t$ under linear-Gaussian assumptions. For this model it reads

$$
m_t = m_{t-1} + K_t\,(x_t - m_{t-1}),
$$

with a gain $K_t$ computed from the variances. Once the gain settles to its steady-state value $K$, this *is* an EMA with $\alpha = 1 - K$. The steady state follows from the variance recursion: if $P$ is the predicted variance, then after an update the variance is $Pr/(P+r)$, and the next prediction adds $q$:

$$
P = \frac{Pr}{P+r} + q \;\Longrightarrow\; P^2 = qP + qr \;\Longrightarrow\; P = \frac{q + \sqrt{q^2+4qr}}{2},\qquad K = \frac{P}{P+r}.
$$

For $q = 0.01$, $r = 1$ (the true temperature drifts slowly compared with sensor noise): $P = 0.1051$, $K = 0.0951$, so $\alpha = 0.905$ — within the range found in Worked Example 5.1. The EMA's decay factor is determined by the **signal-to-noise ratio** $q/r$.

Two further lessons. Before reaching steady state, $K_t$ varies with time: the classical filter already has a *time-dependent gate*. And in adaptive or nonlinear variants, the gain depends on the observations themselves — the classical ancestor of input-dependent gating and selectivity.

## 5.6.3 Numerical ODEs: discretization and stability

Section 5.2.8 compared exact (zero-order hold) and forward-Euler discretizations of $\tau\dot h = x - h$. A third classical choice, **backward (implicit) Euler**, evaluates the right-hand side at the *new* state:

$$
\tau\frac{h_t - h_{t-1}}{\Delta t} = x_t - h_t \;\Longrightarrow\; h_t = \frac{1}{1+\Delta t/\tau}\,h_{t-1} + \frac{\Delta t/\tau}{1+\Delta t/\tau}\,x_t ,
$$

giving $\alpha = 1/(1+\Delta t/\tau) \in (0,1)$ for **every** step size — stable, like the exact solution, though less accurate for large steps. The general theory of which discretizations preserve stability (A-stability, stiffness) belongs to numerical analysis; for deep learning the practical consequences are that state-space models use exact or implicit-style discretizations, and that the **learning rate** of Unit 03 is the step size of forward Euler applied to gradient flow, with the same $2/\lambda$ limit.

## 5.6.4 Probabilistic models: the state as a sufficient statistic

In a hidden Markov model, the **forward algorithm** maintains the filtering distribution $\alpha_t(i) = P(z_t = i \mid x_{1:t})$ over hidden states and updates it with a recurrence: propagate through the transition matrix, multiply by the likelihood of the new observation, renormalize ([Rabiner 1989](/references/#rabiner-1989)). The filtering distribution is a **sufficient statistic** — exactly the notion of state in §5.2.1 — and the update is a linear recurrence followed by a data-dependent rescaling. Many neural recurrent models can be read as learned, approximate versions of such filters.

## 5.6.5 Neuroscience: leaky integrators and fast weights

The membrane potential of a leaky integrate-and-fire neuron obeys $\tau\dot V = -V + RI(t)$ — the cooling law again, with the input current in place of the room temperature ([Dayan & Abbott 2001](/references/#dayan-2001)). Single neurons are therefore EMAs of their input, with time constants of milliseconds; longer-lasting memory requires recurrent circuits or changes in synapses. The associative memory of §5.3.12 — store by adding an outer product, read by multiplying — is the Hebbian learning rule ("cells that fire together wire together") used as a *fast* memory, an idea revived as "fast weight programmers" for linear Transformers ([Schlag et al. 2021](/references/#schlag-2021)). Classical Hopfield networks are its recurrent cousin ([Hopfield 1982](/references/#hopfield-1982)).

## 5.6.6 Retrieval systems: the KV cache as an index

Attention's KV cache is a key–value store queried by similarity, the core operation of information retrieval. Retrieval-augmented generation extends the same idea beyond the context window: an external index of documents is searched with a learned query, and the results are placed into the model's context ([Lewis et al. 2020](/references/#lewis-2020)). In the vocabulary of this unit, it is a third design point beyond "compress" and "store everything in context": **store everything outside, and fetch on demand**.

## 5.6.7 Links within this book

| Unit | Connection |
|---|---|
| [03 · Optimization](/core/unit-03/) | backpropagation through time is reverse-mode differentiation on the unrolled graph; the learning-rate limit $\eta < 2/\lambda$ is the Euler stability limit of §5.2.8 |
| [04 · Interaction](/core/unit-04/) | attention is the "store everything, read by content" memory; linear attention is its recurrent relative |
| [06 · Compute](/core/unit-06/) | KV-cache memory and the $O(T^2)$ vs $O(T)$ costs of §5.3.13 determine inference economics |
| [10 · Failure](/core/unit-10/) | recall stress tests are evaluation design: what a benchmark measures vs. what a memory claim needs |
| [11 · Interpretation](/core/unit-11/) | induction heads implement content-based recall from the KV cache — a mechanism you can locate in the spine model |
