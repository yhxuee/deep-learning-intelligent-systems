---
title: "5.4 Worked examples"
description: Nine fully computed examples — designing a filter, a change detector, a kernel via diagonalization, oscillating and non-normal memories, backpropagation through time by hand, associative-memory cross-talk, and a parallel scan.
sidebar: { label: "5.4 Worked examples", order: 4 }
---

Each example is computed completely, with every intermediate number. Work it on paper before reading the solution. All numbers are reproduced by `labs/unit-05-memory` or by the short scripts noted.

## Worked Example 5.1 — Designing the sensor filter

**Problem.** The sensor of §5.1 is sampled once per second with independent noise of standard deviation $\sigma$. The specification is: (a) the filtered output's noise standard deviation must be at most $\sigma/4$; (b) after a genuine step change, the output must be within 5% of the new value within 60 seconds. Which decay factors $\alpha$ satisfy both? What time constant does a valid choice correspond to?

**Solution.** From §5.2.6 the noise standard deviation factor is $\sqrt{(1-\alpha)/(1+\alpha)}$.

(a) $\sqrt{\tfrac{1-\alpha}{1+\alpha}} \le \tfrac14 \iff \tfrac{1-\alpha}{1+\alpha}\le\tfrac1{16} \iff 16 - 16\alpha \le 1+\alpha \iff \alpha \ge \tfrac{15}{17} \approx 0.8824$.

(b) The residual error after $t$ steps is $\alpha^t$, so we need $\alpha^{60}\le 0.05 \iff 60\ln\alpha \le \ln 0.05 \iff \alpha \le e^{\ln(0.05)/60} = e^{-0.04993} \approx 0.9513$.

Any $\alpha \in [0.8824,\ 0.9513]$ works; (a) pushes $\alpha$ up, (b) pushes it down — the noise–lag trade-off as two inequalities. Take $\alpha = 0.9$. By §5.2.7, $\alpha = e^{-\Delta t/\tau}$, so $\tau = -\Delta t/\ln\alpha = 1/0.10536 \approx 9.49$ s.

**Check.** At $\alpha = 0.9$: noise factor $\sqrt{0.1/1.9} = 0.229 < 0.25$ ✓; 95% response time $28.4 < 60$ s ✓.

## Worked Example 5.2 — A change detector from two memories

**Problem.** Use the bank of §5.3.1, $A = \mathrm{diag}(0.5, 0.95)$, $B = (0.5,\ 0.05)^{\top}$, $C = (1,\ -1)$, $h_0 = 0$. The input is 0 before $t = 1$ and 1 from $t = 1$ on. Compute $y_t$ and explain its shape.

**Solution.** Each component is an EMA with $B_i = 1-\alpha_i$, so by §5.2.6 (with a jump from 0 to 1) the components are $1-0.5^t$ and $1-0.95^t$. Hence

$$
y_t = (1-0.5^t) - (1-0.95^t) = 0.95^t - 0.5^t .
$$

| $t$ | 1 | 2 | 3 | 4 | 5 | 20 | 50 |
|---|---|---|---|---|---|---|---|
| $y_t$ | 0.450 | 0.653 | 0.732 | **0.752** | 0.743 | 0.358 | 0.077 |

The output rises quickly (the fast memory catches up), peaks at $t = 4$, and then decays as the slow memory catches up too. It is large *only shortly after a change* and near zero when the input is steady: a change detector. No single EMA can produce this response — it needs two timescales and a readout that compares them.

## Worked Example 5.3 — A kernel by diagonalization

**Problem.** For $A = \begin{pmatrix}0.5 & 0.5\\ 0 & 0.8\end{pmatrix}$, $B = \begin{pmatrix}0\\1\end{pmatrix}$, $C = (1,\ 0)$, find a closed formula for the impulse response $K_j = CA^jB$ and check it against direct multiplication for $j = 1, 2$.

**Solution.** *Eigenvalues.* $A$ is upper triangular, so $\lambda_1 = 0.5$, $\lambda_2 = 0.8$.

*Eigenvectors.* For $\lambda_1 = 0.5$: $(A - 0.5I)v = \begin{pmatrix}0&0.5\\0&0.3\end{pmatrix}v = 0$ gives $v_1 = (1,0)^{\top}$. For $\lambda_2 = 0.8$: $\begin{pmatrix}-0.3&0.5\\0&0\end{pmatrix}v = 0$ gives $-0.3v^{(1)} + 0.5v^{(2)} = 0$, e.g. $v_2 = (5, 3)^{\top}$.

*Diagonalize.* $V = \begin{pmatrix}1&5\\0&3\end{pmatrix}$, $V^{-1} = \frac13\begin{pmatrix}3&-5\\0&1\end{pmatrix}$. Then

$$
V^{-1}B = \tfrac13\begin{pmatrix}-5\\1\end{pmatrix}, \qquad CV = (1,\ 5),
$$

$$
K_j = CV\Lambda^jV^{-1}B = 1\cdot 0.5^j\cdot\bigl(-\tfrac53\bigr) + 5\cdot 0.8^j\cdot\tfrac13 = \tfrac53\bigl(0.8^j - 0.5^j\bigr).
$$

*Check by multiplication.* $AB = (0.5,\ 0.8)^{\top}$, so $K_1 = 0.5$; formula: $\tfrac53(0.3) = 0.5$ ✓. $A^2B = A(0.5,0.8)^{\top} = (0.25+0.4,\ 0.64)^{\top}$, so $K_2 = 0.65$; formula: $\tfrac53(0.64-0.25) = 0.65$ ✓.

**Interpretation.** The input enters the slow mode, which then feeds the fast one. The response is the *difference* of two decaying modes: it rises, peaks (at $j = 2$: $0.65$), and then decays at the slow rate $0.8^j$. The long-run memory is set by the largest eigenvalue; the shape at short lags by both.

## Worked Example 5.4 — A memory that rings

**Problem.** $A = 0.9\,R(\pi/6)$ where $R(\theta)$ is the rotation by $\theta$. Starting from $h_0 = (1, 0)^{\top}$ with no input, compute the first component of $h_t$ and the eigenvalues of $A$.

**Solution.** Rotation by $\theta$ composed $t$ times is rotation by $t\theta$, and scalars commute with matrices, so $A^t = 0.9^tR(t\pi/6)$ and the first component of $A^th_0$ is $0.9^t\cos(t\pi/6)$:

| $t$ | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 9 | 12 |
|---|---|---|---|---|---|---|---|---|---|
| $0.9^t\cos(t\pi/6)$ | 1.000 | 0.779 | 0.405 | 0.000 | −0.328 | −0.511 | −0.531 | 0.000 | 0.282 |

The characteristic polynomial is $\det(A - \lambda I) = (0.9\cos\theta - \lambda)^2 + (0.9\sin\theta)^2$, with roots $\lambda = 0.9(\cos\theta \pm i\sin\theta) = 0.9e^{\pm i\pi/6} \approx 0.779 \pm 0.450i$. Both have modulus $0.9 = \rho(A)$, so the envelope decays like $0.9^t$; the argument $\pi/6$ gives a period of $2\pi/(\pi/6) = 12$ steps, visible in the table: the response returns to a positive peak at $t = 12$, scaled by $0.9^{12} = 0.282$.

## Worked Example 5.5 — Transient growth, computed exactly

**Problem.** For $A = \begin{pmatrix}0.9 & 5\\ 0 & 0.9\end{pmatrix}$ compute $A^{10}$, and find the input that is amplified most at $t = 10$.

**Solution.** From §5.3.5, $A^t = \begin{pmatrix}0.9^t & 5t\,0.9^{t-1}\\0&0.9^t\end{pmatrix}$, so

$$
A^{10} = \begin{pmatrix}0.3487 & 19.371\\0 & 0.3487\end{pmatrix}.
$$

A unit vector $u$ is amplified by $\|A^{10}u\|$; the maximum over unit vectors is the spectral norm $\|A^{10}\|_2 \approx 19.38$, attained near $u = (0, 1)^{\top}$ (exactly: the top right singular vector). An input that enters the *second* component appears almost twentyfold larger in the *first* component ten steps later — although both eigenvalues are 0.9 and the "half-life" suggested by the eigenvalues is 6.6 steps. The lab test `test_non_normal_transient_growth` checks the peak.

## Worked Example 5.6 — Backpropagation through time by hand

**Problem.** A scalar RNN $h_t = \tanh(w\,h_{t-1} + x_t)$ with $w = 0.5$, $h_0 = 0$, inputs $x = (1,\ -0.5,\ 0.25)$, and loss $L = \tfrac12(h_3 - 0.5)^2$ on the final state only. Compute $\partial L/\partial w$ by BPTT, and $\partial L/\partial x_1$ and $\partial L/\partial x_3$.

**Forward pass** (pre-activation $a_t = wh_{t-1} + x_t$):

| $t$ | $a_t$ | $h_t = \tanh a_t$ | $1-h_t^2$ |
|---|---|---|---|
| 1 | $0.5\cdot0 + 1 = 1.000000$ | 0.761594 | 0.419974 |
| 2 | $0.5\cdot0.761594 - 0.5 = -0.119203$ | −0.118642 | 0.985924 |
| 3 | $0.5\cdot(-0.118642) + 0.25 = 0.190679$ | 0.188401 | 0.964505 |

$L = \tfrac12(0.188401 - 0.5)^2 = 0.048547$.

**Backward pass.** Seed $\bar h_3 = h_3 - 0.5 = -0.311599$. At each step: $\bar a_t = \bar h_t(1-h_t^2)$; the contribution to $\bar w$ is $\bar a_t h_{t-1}$; the adjoint passed back is $\bar h_{t-1} = \bar a_t\,w$ (there is no direct loss on $h_1, h_2$).

| $t$ | $\bar h_t$ | $\bar a_t$ | contribution $\bar a_t h_{t-1}$ |
|---|---|---|---|
| 3 | −0.311599 | −0.300538 | $(-0.300538)(-0.118642) = +0.035656$ |
| 2 | $-0.300538\cdot0.5 = -0.150269$ | −0.148154 | $(-0.148154)(0.761594) = -0.112833$ |
| 1 | $-0.148154\cdot0.5 = -0.074077$ | −0.031110 | $(-0.031110)(0) = 0$ |

$$
\frac{\partial L}{\partial w} = 0.035656 - 0.112833 + 0 = -0.077177 .
$$

A central finite difference with $h = 10^{-6}$ gives $-0.0771769$ ✓.

**Inputs.** Since $a_t = wh_{t-1}+x_t$, $\partial L/\partial x_t = \bar a_t$: $\partial L/\partial x_3 = -0.3005$ but $\partial L/\partial x_1 = -0.0311$ (finite difference: $-0.031110$ ✓). The earliest input's gradient is ten times smaller after just two steps back, because each step multiplies by $w(1-h^2) \le 0.5$. Three observations to remember:

1. The gradient for the *shared* parameter $w$ is a **sum over time** (fan-out, Unit 03).
2. The step-1 contribution is exactly 0 because $h_0 = 0$ — the parameter $w$ does nothing at the first step.
3. Signals from the loss shrink by a factor $|w(1-h_t^2)|$ per step backward: vanishing gradients, in miniature.

## Worked Example 5.7 — Associative memory and cross-talk

**Problem.** In $\mathbb{R}^2$, store $(k_1, v_1) = ((1,0), (2,0))$ and $(k_2, v_2) = ((0,1), (0,3))$ in the linear-attention memory $S = \sum_j v_jk_j^{\top}$. Read with each key. Then also store $(k_3, v_3) = ((1,1)/\sqrt2,\ (1,1))$ and read again.

**Solution.** Two orthonormal keys:

$$
S = \begin{pmatrix}2\\0\end{pmatrix}(1,\ 0) + \begin{pmatrix}0\\3\end{pmatrix}(0,\ 1) = \begin{pmatrix}2&0\\0&3\end{pmatrix},
\qquad Sk_1 = (2,0)^{\top} = v_1,\quad Sk_2 = (0,3)^{\top} = v_2 .
$$

Exact recall — the memory is "full" at $d_k = 2$ orthonormal keys. Now add the third pair: $v_3k_3^{\top} = \tfrac1{\sqrt2}\begin{pmatrix}1&1\\1&1\end{pmatrix}$, so

$$
S' = \begin{pmatrix}2.7071 & 0.7071\\ 0.7071 & 3.7071\end{pmatrix}.
$$

| read with | result $S'k$ | stored value | relative error $\|S'k - v\|/\|v\|$ |
|---|---|---|---|
| $k_1$ | $(2.707,\ 0.707)$ | $(2, 0)$ | 0.50 |
| $k_2$ | $(0.707,\ 3.707)$ | $(0, 3)$ | 0.33 |
| $k_3$ | $(2.414,\ 3.121)$ | $(1, 1)$ | 1.80 |

Every read is now contaminated by the values stored under overlapping keys: reading $k_1$ adds $(k_3^{\top}k_1)v_3 = 0.707\,v_3$; reading $k_3$ adds $0.707v_1 + 0.707v_2$, which swamps $v_3$. A third pair cannot fit into a $2\times2$ memory without interference.

## Worked Example 5.8 — A parallel scan by hand

**Problem.** Compute $h_1,\dots,h_4$ for $h_t = a_th_{t-1} + b_t$, $h_0 = 0$, with $(a_t, b_t) = (0.5, 1), (0.8, 2), (0.9, -1), (0.5, 4)$, first sequentially and then with the associative operator $(a_1,b_1)\bullet(a_2,b_2) = (a_2a_1,\ a_2b_1+b_2)$ of §5.3.10.

**Sequential.** $h_1 = 1$; $h_2 = 0.8(1)+2 = 2.8$; $h_3 = 0.9(2.8)-1 = 1.52$; $h_4 = 0.5(1.52)+4 = 4.76$.

**Parallel (up-sweep).** Combine neighbours in pairs, independently:

$$
(0.5,1)\bullet(0.8,2) = (0.4,\ 0.8\cdot1+2) = (0.4,\ 2.8), \qquad
(0.9,-1)\bullet(0.5,4) = (0.45,\ 0.5\cdot(-1)+4) = (0.45,\ 3.5).
$$

Then combine the two results: $(0.4, 2.8)\bullet(0.45, 3.5) = (0.18,\ 0.45\cdot2.8 + 3.5) = (0.18,\ 4.76)$. Applied to $h_0 = 0$ it gives $h_4 = 4.76$ ✓, in two parallel levels instead of four sequential steps.

**Down-sweep (the missing prefixes).** $h_2 = 2.8$ is the first level-1 result. $h_3$ is $(0.4, 2.8)\bullet(0.9, -1) = (0.36,\ 0.9\cdot2.8 - 1) = (0.36,\ 1.52)$ ✓. With $T$ elements both sweeps take $O(\log T)$ levels.

## Worked Example 5.9 — Which memory for which task?

**Problem.** For each task, choose the cheapest design from the table in §5.3.13 that can solve it, and justify with a number.

1. Smooth a sensor whose noise must drop 4× with a 60 s response (Example 5.1).
2. Detect that a reading has changed within the last ~5 s.
3. Answer "what was the reading exactly 3600 s ago?" with 0.1 °C precision.
4. Given a sequence of 2 000 key–value pairs and then a key, return its value.

**Solution.**

1. **One EMA**, $\alpha \in [0.88, 0.95]$ — 1 number of state (Example 5.1).
2. **Two EMAs and a difference** (Example 5.2), 2 numbers; the peak at 4 steps matches the 5 s window. A selective or gated model would also work but is unnecessary.
3. **A window of 3 600 values** (or attention over them). No decaying memory can do it: the influence of a 3600-step-old input in an EMA with $\alpha = 0.999$ is $0.999^{3600} \approx 0.027$, mixed with 3 599 other readings; and §5.5.2 proves any exact solution needs state proportional to the window.
4. **Softmax attention with a KV cache** of 2 000 entries; or a linear-attention memory with $d_k \gtrsim 2000$ (Example 5.7 and §5.5.3 show interference otherwise). A fixed-size recurrent state of a few hundred numbers cannot store 2 000 arbitrary associations exactly (§5.5.2).

The general lesson: **compute the memory requirement of the task first**, then pick the design whose capacity matches.
