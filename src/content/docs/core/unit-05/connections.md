---
title: "5.6 Connections"
description: How the unit's memory mechanisms connect to signal processing, optimal estimation and the Kalman filter, numerical ODEs, hidden Markov models, neuroscience, retrieval systems, and other units of the book.
sidebar: { label: "5.6 Connections", order: 6 }
---

Deep-learning memory did not appear from nowhere. Each mechanism in this unit is a well-studied object in another field, and knowing the connection lets you borrow decades of theory.

## 5.6.1 Signal processing: recurrences are usually IIR; finite windows are FIR

A causal filter that computes $y_t$ from a **finite** window, $y_t=\sum_{j=0}^{W-1}K_jx_{t-j}$, is a *finite impulse response* (FIR) filter—a convolution, as in Unit 04. A recurrence whose reachable and observable modes keep feeding the output generally has a nonzero tail: an *infinite impulse response* (IIR) filter. “Recurrence” does not logically imply IIR: a nilpotent transition, or cancellation of every long-lived mode, can give a finite response. The EMA is the standard first-order IIR low-pass filter; §5.5.1 derived its frequency response.

The classical engineering trade-off is now more precise. A low-order IIR can realize a long response with little state, but its feedback can be unstable and its phase response may be inconvenient. A finite-coefficient FIR is BIBO-stable and can be designed with linear phase, but a long response needs a long window ([Oppenheim & Schafer 2010](/references/#oppenheim-2010)). These are design tendencies, not “IIR has delay, FIR has none”—both can delay or distort a signal. The same state--window trade-off reappears in §5.3.13.

## 5.6.2 Optimal estimation: a steady-state Kalman filter becomes an EMA

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

For $q=0.01$, $r=1$ (the true temperature drifts slowly compared with sensor noise), $P=0.1051$, $K=0.0951$, so $\alpha=0.905$—within the range found in Worked Example 5.1. More precisely, the decay is determined here by the ratio of **process-noise variance to measurement-noise variance**, $q/r$; calling it a generic signal-to-noise ratio can hide what the two noises mean.

Two further lessons. Before steady state, $K_t$ varies with time, so even the classical filter has a *time-dependent gain*. In extended, ensemble or adaptive variants, the gain can also change with the predicted state, estimated uncertainty and incoming data. This is a structural analogy to learned gates and selectivity, not a claim that those neural architectures were historically derived from the Kalman filter.

## 5.6.3 Numerical ODEs: discretization and stability

Section 5.2.8 compared exact (zero-order hold) and forward-Euler discretizations of $\tau\dot h = x - h$. A third classical choice, **backward (implicit) Euler**, evaluates the right-hand side at the *new* state:

$$
\tau\frac{h_t - h_{t-1}}{\Delta t} = x_t - h_t \;\Longrightarrow\; h_t = \frac{1}{1+\Delta t/\tau}\,h_{t-1} + \frac{\Delta t/\tau}{1+\Delta t/\tau}\,x_t ,
$$

giving $\alpha=1/(1+\Delta t/\tau)\in(0,1)$ for **every** positive step size—stable for this decaying scalar system, like the exact solution, though less accurate for large steps. The general theory of A-stability and stiff equations belongs to numerical analysis. In deep learning, many state-space models use zero-order-hold or bilinear discretizations for related stability reasons. Gradient descent is likewise forward Euler applied to gradient flow: on a quadratic, stability requires $0<\eta<2/\lambda_{\max}$, where $\lambda_{\max}$ is the largest Hessian eigenvalue—not one universal $2/\lambda$ for every objective.

## 5.6.4 Probabilistic models: the state as a sufficient statistic

In a hidden Markov model, the **forward algorithm** maintains the filtering distribution $\alpha_t(i) = P(z_t = i \mid x_{1:t})$ over hidden states and updates it with a recurrence: propagate through the transition matrix, multiply by the likelihood of the new observation, renormalize ([Rabiner 1989](/references/#rabiner-1989)). The filtering distribution is a **sufficient statistic** — exactly the notion of state in §5.2.1 — and the update is a linear recurrence followed by a data-dependent rescaling. Many neural recurrent models can be read as learned, approximate versions of such filters.

## 5.6.5 Neuroscience: leaky integrators and fast weights

Below threshold and before a spike reset, the simplified leaky integrate-and-fire membrane obeys $\tau\dot V=-V+RI(t)$—the cooling law with input current in place of room temperature ([Dayan & Abbott 2001](/references/#dayan-2001)). That subthreshold voltage is a continuous-time leaky integrator; spiking, reset, refractory dynamics and active ion channels make a real neuron more than an EMA. Longer biological timescales can come from recurrent circuits, synaptic changes or slower cellular processes. The associative memory of §5.3.12—store by adding an outer product, read by multiplying—resembles a Hebbian fast-weight rule, revived as “fast weight programmers” for linear Transformers ([Schlag et al. 2021](/references/#schlag-2021)). Classical Hopfield networks are a recurrent relative ([Hopfield 1982](/references/#hopfield-1982)).

## 5.6.6 Retrieval systems: the KV cache as an index

Attention's KV cache is a temporary key--value store queried by learned similarity. An information-retrieval system also maps a query to scored records, but the records are externally curated documents or chunks rather than one cache slot per current-context token. Retrieval-augmented generation searches such an index and places selected results into the model's context ([Lewis et al. 2020](/references/#lewis-2020)). In this unit's vocabulary, it is a third design point beyond “compress the stream” and “retain a record per context token”: **keep a selected external corpus and fetch a small candidate set on demand**. Neither similarity search nor softmax guarantees exact lookup; indexing, chunking and recall errors become part of the memory system.

## 5.6.7 Links within this book

| Unit | Connection |
|---|---|
| [03 · Optimization](/core/unit-03/) | backpropagation through time is reverse-mode differentiation on the unrolled graph; for a quadratic mode, the learning-rate limit $0<\eta < 2/\lambda$ is the Euler stability limit of §5.2.8 |
| [04 · Interaction](/core/unit-04/) | attention retains one learned record per cached token and reads a weighted blend; linear attention replaces the growing cache with a recurrent matrix state |
| [06 · Compute](/core/unit-06/) | KV-cache memory and the sequence-length scaling of attention and structured recurrences shape inference economics |
| [10 · Failure](/core/unit-10/) | recall stress tests are evaluation design: what a benchmark measures vs. what a memory claim needs |
| [11 · Interpretation](/core/unit-11/) | induction heads are a testable mechanism for content-based copying; the unit first checks the behaviour, then asks whether the trained spine model actually contains that mechanism |
