---
title: "G · Linear dynamical systems and signals"
description: Linear recurrences and their impulse responses; stability for non-diagonalizable and non-normal systems; controllability, observability and minimal realizations; internal versus input–output stability; the z-transform, transfer functions and frequency response; convolution, Toeplitz matrices and the FFT; discretizing continuous-time systems; the cost of applying a recurrence.
sidebar: { label: "G · Linear systems", order: 7 }
---

Unit 05 builds memory from the exponential moving average up to state-space models and linear attention; every one of those layers is, or contains, a linear dynamical system. This appendix gives the system-theoretic facts in general form, including the ones the unit uses informally — what the poles of a transfer function can and cannot tell you, and what it costs to run a recurrence. Numbers in the examples are checked by `labs/test_appendices.py`.

| Section | Used in |
|---|---|
| G.1 Recurrences and impulse responses | 05 (§5.3.2), 04 (attention as a lower-triangular map) |
| G.2 Stability | 03 (gradient descent as a linear recurrence), 05 (§5.3.4–§5.3.5) |
| G.3 Controllability, observability, realizations | 05 (§5.5.1), 11 (what a probe can read) |
| G.4 Transfer functions and frequency response | 05 (§5.5.1) |
| G.5 Convolution and the FFT | 04 (convolutions), 05 (§5.5.4) |
| G.6 Discretization | 05 (§5.2.7–§5.2.8, §5.3.9), 08 (ODE samplers) |
| G.7 Cost of a recurrence | 05 (§5.5.4), 06 (accounting) |

## G.1 Linear recurrences and impulse responses

A discrete-time **linear time-invariant (LTI) system** with state $h_t\in\mathbb{R}^N$, input $x_t\in\mathbb{R}^m$ and output $y_t\in\mathbb{R}^p$ is
$$
h_t = Ah_{t-1} + Bx_t,\qquad y_t = Ch_t .
$$
Unrolling from $h_0$,
$$
h_t = A^th_0 + \sum_{s=1}^{t}A^{t-s}Bx_s,\qquad y_t = CA^th_0 + \sum_{s=1}^{t}\underbrace{CA^{t-s}B}_{g_{t-s}}\,x_s .
$$
The sequence $g_k = CA^kB$ is the **impulse response**: the output at lag $k$ after a unit input. From a zero initial state, the system is the convolution $y = g * x$, and its matrix from inputs to outputs is lower-triangular Toeplitz — the "structure of $M$" of Unit 05, §5.5.4. The first term, $CA^th_0$, is the **free response**: what the state does with no input.

**Scalar case.** With $h_t = ah_{t-1} + (1 - a)x_t$ — the exponential moving average — the impulse response is $(1 - a)a^k$, a geometric decay whose **half-life** is $\ln2/(-\ln a)$ steps. For $a = 0.9$ that is 6.58 steps (Unit 05, §5.2.5).

## G.2 Stability

**Internal (asymptotic) stability**: the free response tends to zero from every initial state, that is, $A^t\to0$. By Appendix A, Theorem A.6, this holds if and only if the spectral radius $\rho(A) < 1$, for every matrix $A$ — diagonalizable or not. Unit 05's Theorem 5.1 proves the diagonalizable case.

Two refinements matter in practice.

1. **Rate.** $\lVert A^t\rVert$ decays like $\rho(A)^t$ only eventually. For a non-diagonalizable $A$ the decay carries polynomial factors $t^{m-1}$ from Jordan blocks of size $m$, and for a non-normal $A$ it can be preceded by **transient growth** (Appendix A, Example A.5; Unit 05, §5.3.5).
2. **A certificate without eigenvalues.** $\rho(A) < 1$ if and only if the **Lyapunov equation** $A^{\top}PA - P = -Q$ has a positive-definite solution $P$ for some (equivalently, every) $Q\succ0$. Then $V(h) = h^{\top}Ph$ decreases along every free trajectory. Lyapunov functions are how stability is proved for nonlinear and time-varying systems, where eigenvalues say little.

**Time-varying systems.** For $h_t = A_th_{t-1}$ — a gated RNN or a selective SSM (Unit 05, §5.3.8, §5.3.10) — the eigenvalues of each $A_t$ do *not* determine stability: a product of matrices each with spectral radius below one can grow. A sufficient condition is $\lVert A_t\rVert\le1 - \epsilon$ in some fixed norm for all $t$. Diagonal transitions with entries in $(0, 1)$, as in practical SSMs, satisfy it trivially — one reason they are used.

## G.3 Controllability, observability and minimal realizations

The input–output behaviour of a system is determined by its impulse response, but many triples $(A, B, C)$ produce the same one. Which parts of the state actually matter?

- The system is **controllable** if every state can be reached from zero by some input sequence. Equivalently, the **controllability matrix** $\mathcal C = [B, AB, \dots, A^{N-1}B]$ has rank $N$.
- It is **observable** if the initial state can be recovered from the outputs of the free response. Equivalently, the **observability matrix** $\mathcal O = [C; CA; \dots; CA^{N-1}]$ has rank $N$.

(The rank tests stop at $A^{N-1}$ because, by the Cayley–Hamilton theorem, $A^N$ is a linear combination of lower powers.)

A realization is **minimal** if no realization of the same impulse response has a smaller state, and **a realization is minimal if and only if it is both controllable and observable** ([Kailath 1980](/references/#kailath-1980)). Modes that are uncontrollable never appear in the state when starting from zero; modes that are unobservable never appear in the output. Either way, they are invisible from input to output.

**Example G.1.** $A = \mathrm{diag}(0.5, 1.2)$, $B = (1, 0)^{\top}$, $C = (1, 0)$. Then $\mathcal C = \begin{pmatrix}1 & 0.5\\0 & 0\end{pmatrix}$ and $\mathcal O = \begin{pmatrix}1 & 0\\0.5 & 0\end{pmatrix}$ both have rank 1: the second mode, with eigenvalue 1.2, can be neither reached nor seen. The impulse response is $g_k = 0.5^k$, exactly that of the one-dimensional system $a = 0.5$, which is a minimal realization of it.

## G.4 Transfer functions, poles and frequency response

The **z-transform** of a sequence is $X(z) = \sum_tx_tz^{-t}$; a delay multiplies by $z^{-1}$. From a zero initial state, $Y(z) = G(z)X(z)$ with the **transfer function**
$$
G(z) = \sum_{k\ge0}g_kz^{-k} = C(I - Az^{-1})^{-1}B = C\,\frac{z\,\mathrm{adj}(zI - A)}{\det(zI - A)}\,B .
$$

**Poles and eigenvalues.** Every pole of $G$ is an eigenvalue of $A$, because the only denominator is $\det(zI - A)$. The converse fails in general: factors of $\det(zI - A)$ can cancel against the numerator. In Example G.1, $\det(zI - A) = (z - 0.5)(z - 1.2)$ but $G(z) = z/(z - 0.5)$. **For a minimal realization, the poles of $G$ are exactly the eigenvalues of $A$** (with multiplicities suitably counted).

**Two kinds of stability.** A system is **bounded-input bounded-output (BIBO) stable** if every bounded input produces a bounded output from a zero initial state. For LTI systems this holds if and only if $\sum_k|g_k| < \infty$, which for a rational $G$ means every *pole of $G$* lies strictly inside the unit circle. Internal stability ($\rho(A) < 1$) implies BIBO stability; the converse holds for minimal realizations but not in general — Example G.1 is BIBO stable and internally unstable. A recurrent model that carries its state indefinitely, and whose state can be perturbed by rounding or by a nonzero initialization, needs internal stability.

**Frequency response.** For a BIBO-stable system and input $x_t = e^{i\omega t}$, the output settles to $G(e^{i\omega})e^{i\omega t}$: the gain at frequency $\omega\in[0, \pi]$ is $|G(e^{i\omega})|$.

**Example G.2.** The EMA $G(z) = (1 - a)/(1 - az^{-1})$ with $a = 0.9$ has gain 1 at $\omega = 0$ (constants pass unchanged), $0.074$ at $\omega = \pi/2$, and $(1 - a)/(1 + a) = 0.053$ at $\omega = \pi$ (alternating signals are suppressed 19-fold). It is a low-pass filter: memory, in signal-processing terms, is the suppression of fast variation (Unit 05, §5.5.1).

## G.5 Convolution, Toeplitz matrices and the FFT

A causal convolution of a length-$T$ input with a kernel $g_0,\dots,g_{T-1}$ is multiplication by a lower-triangular Toeplitz matrix — $O(T^2)$ operations if done directly. The **discrete Fourier transform** diagonalizes *circular* convolution: if $\hat x = Fx$ with $F_{jk} = e^{-2\pi ijk/n}$, then the transform of the circular convolution $g\circledast x$ is the elementwise product $\hat g\odot\hat x$. The **fast Fourier transform** computes $F$ in $O(n\log n)$ operations, so a circular convolution costs $O(n\log n)$.

A causal (linear) convolution is not circular: computed naively with a length-$T$ FFT, late inputs wrap around and contaminate early outputs. **Zero-padding** both sequences to length $n\ge2T - 1$ makes the circular convolution equal the linear one, at the same $O(T\log T)$ cost. Convolution layers and long-convolution sequence models use exactly this, and it is the $O(T\log T)$ route for time-invariant SSMs — once their kernel has been computed (G.7).

## G.6 Discretizing continuous-time systems

A continuous-time system $\dot h(\tau) = A_ch(\tau) + B_cx(\tau)$ sampled every $\Delta$ with the input held constant between samples (**zero-order hold**) is exactly
$$
h_t = \bar Ah_{t-1} + \bar Bx_t,\qquad\bar A = e^{\Delta A_c},\qquad\bar B = A_c^{-1}(e^{\Delta A_c} - I)B_c ,
$$
for invertible $A_c$. The eigenvalues of $\bar A$ are $e^{\Delta\lambda}$ for the eigenvalues $\lambda$ of $A_c$, so a stable continuous system ($\mathrm{Re}\,\lambda < 0$) always gives a stable discrete one ($|e^{\Delta\lambda}| < 1$) — for every step size. This is the discretization of the structured SSMs in Unit 05, §5.3.9.

Approximate schemes do not inherit stability for free:

| Scheme | $\bar A$ (scalar $\lambda$) | Stable for $\lambda < 0$ when |
|---|---|---|
| exact (zero-order hold) | $e^{\Delta\lambda}$ | always |
| forward Euler | $1 + \Delta\lambda$ | $\Delta < 2/|\lambda|$ |
| backward Euler | $1/(1 - \Delta\lambda)$ | always |
| bilinear (Tustin) | $(1 + \Delta\lambda/2)/(1 - \Delta\lambda/2)$ | always |

**Example G.3.** For $\lambda = -10$, forward Euler is stable only for $\Delta < 0.2$; at $\Delta = 0.3$, $\bar A = -2$ and the state oscillates and explodes, although the continuous system decays (Unit 05, §5.2.8). With $\Delta = 0.1$, the exact factor is $e^{-1} = 0.368$ while forward Euler gives $0$ — stable, but a poor approximation. Gradient descent is forward Euler on the gradient flow $\dot\theta = -\nabla L(\theta)$, and the condition $\Delta < 2/|\lambda|$ is the step-size limit $\eta < 2/\lambda_{\max}$ of Unit 03 (§3.6.1). The ODE samplers of Unit 08 face the same trade-off between step size and accuracy.

## G.7 What it costs to apply a recurrence

Counting multiply–adds for a sequence of length $T$, state size $N$, one input and one output channel:

| Transition | Per step | Whole sequence | Notes |
|---|---|---|---|
| dense $A$ | $O(N^2)$ | $O(TN^2)$ | generic recurrence or RNN |
| diagonal (or diagonalized) $A$ | $O(N)$ | $O(TN)$ | S4D-, Mamba-style SSMs; time-varying diagonal also $O(TN)$ |
| time-invariant, via convolution | — | $O(TN)$ to form the kernel $g_k = CA^kB$ (diagonal $A$), then $O(T\log T)$ by FFT | requires time invariance |
| low-rank-plus-diagonal (DPLR) | $O(N)$ | $O(TN)$, or kernel via Cauchy-kernel algorithms | S4 |

A parallel **scan** computes the same recurrence in $O(\log T)$ sequential depth on parallel hardware, with the same total work up to a constant, because composing affine maps $h\mapsto a_th + b_t$ is associative. The $O(TN)$ figures assume a structured transition; quoting them for a dense one understates the cost by a factor of $N$ (Unit 05, §5.5.4).

## Exercises

**G.1** Show that a system with $\rho(A) < 1$ has $\sum_k\lVert CA^kB\rVert < \infty$, hence is BIBO stable.

**G.2** Compute the controllability and observability matrices of $A = \begin{pmatrix}0.9 & 1\\0 & 0.9\end{pmatrix}$, $B = (0, 1)^{\top}$, $C = (1, 0)$. Is the realization minimal? What is its impulse response?

**G.3** Find the half-life of an EMA with $a = 0.99$, and the frequency at which its gain falls to $1/\sqrt2$ of its value at $\omega = 0$.

**G.4** Show that zero-padding to length $2T - 1$ makes circular convolution equal to linear convolution for two length-$T$ sequences.

**G.5** For the bilinear discretization, show that $|\bar A| < 1$ whenever $\mathrm{Re}\,\lambda < 0$.

**G.6** ★ Give two $2\times2$ matrices, each with spectral radius below 1, whose product has spectral radius above 1. What does this imply for gated recurrences?

**Answers.** **G.2** $\mathcal C = \begin{pmatrix}0 & 1\\1 & 0.9\end{pmatrix}$ and $\mathcal O = \begin{pmatrix}1 & 0\\0.9 & 1\end{pmatrix}$, both of rank 2: minimal. $g_k = CA^kB = k\,0.9^{k-1}$ — a Jordan block's polynomial factor, peaking near $k = 9$. **G.3** $\ln2/(-\ln0.99) = 69.0$ steps. **G.6** For example $\begin{pmatrix}0 & 2\\0 & 0\end{pmatrix}$ and $\begin{pmatrix}0 & 0\\2 & 0\end{pmatrix}$ (both nilpotent) have product $\mathrm{diag}(4, 0)$.

## Further reading

[Oppenheim & Schafer 2010](/references/#oppenheim-2010) for discrete-time signals; [Kailath 1980](/references/#kailath-1980) for state-space theory and realizations; [Antoulas 2005](/references/#antoulas-2005) for model reduction; [Trefethen & Embree 2005](/references/#trefethen-2005) for non-normal dynamics.
