---
title: "5.3 Mathematical construction: from linear recurrences to attention"
description: Vector states, eigen-modes, stability, transient growth, recurrent networks, backpropagation through time, gating, state-space models, selectivity, the KV cache and linear attention — each derived from the previous step.
sidebar: { label: "5.3 Construction", order: 3 }
---

Section 5.2 built a memory with one timescale. This section generalizes it step by step. Each subsection fixes a limitation of the previous one; together they lead from the EMA to the architectures used in current sequence models.

## 5.3.1 Many timescales: a vector of memories

Language, audio and sensor data carry information on several timescales at once. One EMA has one half-life. The obvious generalization is to run several EMAs side by side, each with its own decay, and let a readout combine them:

$$
h_t = A\,h_{t-1} + B\,x_t, \qquad y_t = C\,h_t,
$$

with state $h_t \in \mathbb{R}^N$, input $x_t \in \mathbb{R}^m$, output $y_t\in\mathbb{R}^p$, and matrices $A \in \mathbb{R}^{N\times N}$, $B \in \mathbb{R}^{N\times m}$, $C \in \mathbb{R}^{p\times N}$. This is the **linear recurrence** (in control theory, a *discrete-time linear time-invariant state-space system*). The EMA is the case $N = m = p = 1$, $A = \alpha$, $B = 1-\alpha$, $C = 1$.

**Example: a bank of two memories.** Take a scalar input and

$$
A = \begin{pmatrix} 0.5 & 0 \\ 0 & 0.95 \end{pmatrix}, \qquad
B = \begin{pmatrix} 0.5 \\ 0.05 \end{pmatrix}.
$$

The first component is a fast EMA (half-life 1 step), the second a slow one (half-life 13.5 steps). A readout can use them separately or together. For instance $C = (1, -1)$ outputs *fast minus slow* — near zero when the signal is steady, large just after a change. With one line of linear algebra, the device now detects change, which no single EMA can do.

When $A$ is diagonal the components do not interact. The general case allows $A$ to mix them. The next subsections show that, under a mild condition, every linear recurrence is secretly a bank of independent memories — in the right coordinates.

## 5.3.2 Closed form, impulse response and convolution

Repeat the unrolling of §5.2.4 with matrices. Substituting $h_{t-1} = Ah_{t-2} + Bx_{t-1}$ and continuing,

$$
\boxed{\,h_t = A^{t} h_0 + \sum_{k=1}^{t} A^{\,t-k} B\, x_k.\,}
$$

The proof by induction is word-for-word that of §5.2.4 with $\alpha$ replaced by $A$; the only care needed is that matrix products are written in the right order ($A$ always multiplies from the left).

Three consequences:

1. **Influence.** The input at time $k$ reaches the state at time $t$ through the matrix $A^{t-k}B$. Memory length is the rate at which the powers $A^j$ shrink.
2. **Impulse response.** Start from $h_0 = 0$ and feed a single unit input at time 1 ($x_1 = 1$, all others 0). Then $y_t = CA^{t-1}B$. The sequence
   $$K_j = C A^{j} B, \qquad j = 0,1,2,\dots$$
   is the **impulse response** — the system's complete "memory profile".
3. **Convolution.** With $h_0 = 0$, multiplying the closed form by $C$ gives
   $$y_t = \sum_{k=1}^{t} K_{t-k}\, x_k .$$
   The output is the input **convolved** with the kernel $K$. A linear time-invariant recurrence and a causal convolution with kernel $K_j = CA^jB$ are two ways of computing the same function.

The convolution view matters for computation. The recurrence produces outputs one at a time, which is ideal for streaming inference. The convolution can be computed for all $t$ at once — in $O(T\log T)$ time with the fast Fourier transform — which is ideal for training on a whole sequence in parallel. Structured state-space models (§5.3.9) exploit exactly this duality.

## 5.3.3 Eigenvalues are memory modes

How fast do the powers $A^j$ shrink? For a scalar, $\alpha^j$ shrinks iff $|\alpha|<1$. For a matrix we need a change of coordinates.

**Review: eigenvalues.** A nonzero vector $v$ is an **eigenvector** of $A$ with **eigenvalue** $\lambda$ if $Av = \lambda v$: along $v$, the matrix acts as multiplication by the number $\lambda$. Eigenvalues are the roots of $\det(A - \lambda I) = 0$ and may be complex even when $A$ is real. If $A$ has $N$ linearly independent eigenvectors $v_1,\dots,v_N$, collect them as the columns of $V$ and the eigenvalues on the diagonal of $\Lambda$; then $AV = V\Lambda$, i.e.

$$
A = V\Lambda V^{-1}, \qquad A^{j} = V\Lambda^{j}V^{-1}.
$$

(The second identity follows because the inner $V^{-1}V$ factors cancel: $A^2 = V\Lambda V^{-1}V\Lambda V^{-1} = V\Lambda^2V^{-1}$.) Such an $A$ is called **diagonalizable**.

**Decoupling.** Define new coordinates $z_t = V^{-1}h_t$. Multiplying the recurrence by $V^{-1}$:

$$
z_t = V^{-1}AV\,z_{t-1} + V^{-1}B\,x_t = \Lambda\,z_{t-1} + \tilde B\,x_t .
$$

Because $\Lambda$ is diagonal, the $N$ components evolve **independently**:

$$
z_t^{(i)} = \lambda_i\, z_{t-1}^{(i)} + \bigl(\tilde B x_t\bigr)_i .
$$

Each component is a scalar recurrence with decay factor $\lambda_i$ — the EMA of §5.2, up to the input scaling. So **a diagonalizable linear recurrence is a bank of independent memories in disguise**, one per eigenvalue. We call them *memory modes*.

**Real eigenvalues** behave as in §5.2: $0<\lambda<1$ decays smoothly with half-life $\ln 2/(-\ln \lambda)$; $-1<\lambda<0$ decays while flipping sign each step.

**Complex eigenvalues** come in conjugate pairs $\lambda = re^{\pm i\theta}$ for a real matrix. Then $\lambda^t = r^t e^{\pm it\theta}$: the magnitude decays like $r^t$ while the phase rotates by $\theta$ per step. In real coordinates such a pair is a $2\times2$ block

$$
A = r\begin{pmatrix}\cos\theta & -\sin\theta \\ \sin\theta & \cos\theta\end{pmatrix},
$$

a rotation by $\theta$ followed by shrinking by $r$. Its impulse response oscillates with period $2\pi/\theta$ steps inside a decaying envelope — a memory that "rings". For $r = 0.9$, $\theta = \pi/6$: the eigenvalues are $0.9e^{\pm i\pi/6} \approx 0.779 \pm 0.450i$, both of modulus 0.9, and the response completes one oscillation every 12 steps (Worked Example 5.4).

![Impulse responses of three memory modes over 40 steps: λ = 0.5 decays within a few steps, λ = 0.95 decays slowly, and a complex pair 0.9·e^{±iπ/6} oscillates with period 12 inside a decaying envelope.](/figures/u05-impulse-responses.svg)

*Figure 5.3 — Each eigenvalue is a memory mode with its own timescale; complex pairs oscillate.*

Oscillating modes are not a curiosity. They let a linear memory represent *periodic* structure (rhythm in audio, line breaks in code) and, with many frequencies, approximate the history by a sum of sinusoids — the idea behind the frequency-domain view in §5.5.1.

## 5.3.4 Stability

**Definition.** The spectral radius of $A$ is $\rho(A) = \max_i |\lambda_i|$, the largest eigenvalue modulus.

**Theorem 5.1.** Let $A$ be diagonalizable. Then $A^t \to 0$ as $t\to\infty$ if and only if $\rho(A) < 1$. Moreover, for any submultiplicative matrix norm,

$$
\|A^t\| \;\le\; \kappa(V)\,\rho(A)^t, \qquad \kappa(V) = \|V\|\,\|V^{-1}\|.
$$

*Proof.* $A^t = V\Lambda^tV^{-1}$, and $\Lambda^t$ is diagonal with entries $\lambda_i^t$, so in the spectral norm $\|\Lambda^t\|_2 = \rho(A)^t$. Hence $\|A^t\| \le \|V\|\,\rho(A)^t\,\|V^{-1}\|$, which tends to 0 if $\rho < 1$. Conversely, if some $|\lambda_i| \ge 1$, then $A^tv_i = \lambda_i^t v_i$ does not tend to 0, so neither does $A^t$. $\blacksquare$

**Bounded input, bounded state.** If $\rho(A)<1$ and all inputs satisfy $\|x_k\| \le X$, the closed form gives

$$
\|h_t\| \le \|A^t\|\,\|h_0\| + \sum_{j=0}^{t-1}\|A^j\|\,\|B\|\,X \le \kappa(V)\Bigl(\|h_0\| + \frac{\|B\|X}{1-\rho}\Bigr),
$$

using the geometric series. The state stays bounded forever. If $\rho \ge 1$, some bounded input makes the state grow without bound (feed the input that excites the mode with $|\lambda|\ge1$).

So for a linear memory there are three regimes:

| $\rho(A)$ | behaviour | as a memory |
|---|---|---|
| $<1$ | every mode decays | forgets gradually; stable |
| $=1$ | some mode neither decays nor grows (if not defective) | perfect but fragile memory of that mode; noise accumulates |
| $>1$ | some mode grows geometrically | unstable; numbers overflow |

Useful memories live just inside the unit circle: eigenvalues of modulus slightly below 1 give long, stable half-lives.

**★ Advanced: the non-diagonalizable case.** Not every matrix is diagonalizable, but every square matrix has a Jordan form, and Gelfand's formula $\lim_{t\to\infty}\|A^t\|^{1/t} = \rho(A)$ holds for every matrix and every norm. Hence $A^t\to0 \iff \rho(A)<1$ in general; only the constant in front of $\rho^t$ changes (it may grow polynomially in $t$, as the next subsection shows).

## 5.3.5 ★ When eigenvalues mislead: transient growth

Theorem 5.1 describes the *long run*. Its constant $\kappa(V)$ can be enormous, and then the short run looks nothing like $\rho^t$. Consider

$$
A = \begin{pmatrix} 0.9 & 5 \\ 0 & 0.9 \end{pmatrix}.
$$

Both eigenvalues equal $0.9$ (the matrix is triangular), so $\rho(A) = 0.9$ and Theorem 5.1's long-run conclusion holds. But this matrix is not diagonalizable: it has only one eigenvector direction. Compute its powers exactly. Write $A = 0.9I + N$ with $N = \begin{pmatrix}0&5\\0&0\end{pmatrix}$. Since $N^2 = 0$ and $I$ commutes with $N$, the binomial theorem stops after two terms:

$$
A^t = (0.9I + N)^t = 0.9^t I + t\,0.9^{t-1}N = \begin{pmatrix} 0.9^t & 5t\,0.9^{t-1} \\ 0 & 0.9^t \end{pmatrix}.
$$

The off-diagonal entry $5t\,0.9^{t-1}$ first **grows** linearly with $t$ before the geometric factor wins. Treating $t$ as continuous, its derivative vanishes at $t^* = -1/\ln 0.9 \approx 9.49$; at the integers $t = 9$ and $t = 10$ it equals $19.37$. An input that enters the second component is **amplified almost twentyfold** before it decays, even though every eigenvalue is 0.9.

![Spectral norm of A^t for the non-normal matrix [[0.9, 5], [0, 0.9]] rises to about 19.4 near t = 9–10 before decaying, while the normal matrix diag(0.9, 0.9) with the same eigenvalues decays monotonically.](/figures/u05-nonnormal.svg)

*Figure 5.4 — Same eigenvalues, very different transients. Eigenvalues govern the long run; for the tens of steps that matter in training, the whole matrix matters.*

The phenomenon is called **non-normal transient growth**: it occurs whenever $A$ does not commute with its transpose, and is strongest when eigenvectors are nearly parallel (large $\kappa(V)$). It is a standard topic in numerical analysis ([Trefethen & Embree 2005](/references/#trefethen-2005)). For memory design the lesson is concrete: *"$\rho(A) < 1$" guarantees eventual forgetting, not gentle forgetting.* This is one reason state-space models (§5.3.9) restrict $A$ to diagonal or other well-conditioned forms.

## 5.3.6 Nonlinear memory: the recurrent neural network

A linear recurrence can only remember **linear** functions of its history: by the closed form, $y_t$ is a weighted sum of past inputs. Many useful summaries are not linear. Consider the question "is the number of 1s seen so far odd?" (the parity of a bit stream). Its answer flips with every 1, no matter how long ago the previous 1 occurred. A linear $y_t = \sum_k K_{t-k}x_k$ with a decaying kernel cannot do this, and even a non-decaying kernel can only count, not compute "odd or even" of the count with a linear readout. Nonlinear updates can: the two-state rule "flip the state on a 1" is trivially nonlinear.

The **recurrent neural network** (RNN; [Elman 1990](/references/#elman-1990)) inserts a nonlinearity into the linear recurrence:

$$
h_t = \tanh\bigl(W h_{t-1} + U x_t + b\bigr), \qquad y_t = C h_t ,
$$

with learned $W$, $U$, $b$, $C$, and $\tanh$ applied elementwise. Near $h = 0$, $\tanh(z)\approx z$ and the RNN behaves like the linear recurrence with $A = W$; away from 0 the saturation of $\tanh$ bends the dynamics.

The local behaviour is governed by the **Jacobian** of one step (Unit 03, §3.3.5). With pre-activation $a_t = Wh_{t-1}+Ux_t+b$ and $\tanh'(a) = 1-\tanh^2(a)$,

$$
J_t \equiv \frac{\partial h_t}{\partial h_{t-1}} = \mathrm{diag}\bigl(1 - h_t^{2}\bigr)\, W .
$$

The diagonal factor has entries in $(0,1]$: saturation can only *shrink* the influence of the past relative to the linear case. Hold on to this formula; it controls learning in the next subsection.

## 5.3.7 Learning through time: backpropagation through time

To train $W$, $U$, $b$, $C$ with gradient descent (Unit 03), we need the gradient of a loss summed over time, $L = \sum_{t=1}^{T} L_t(y_t)$. The trick is the one used for any program: **unroll** the recurrence into a computation graph with one copy of the update per time step. The copies **share** the same parameters. Reverse-mode differentiation on this unrolled graph is called **backpropagation through time** (BPTT; [Werbos 1990](/references/#werbos-1990)).

**The adjoint recursion.** Let $\bar h_t = \partial L/\partial h_t$, the total derivative including all future losses. The state $h_t$ influences $L$ in two ways: directly through $L_t$, and through the next state $h_{t+1}$. The chain rule (Unit 03, §3.2.6) adds the two paths:

$$
\bar h_t = \frac{\partial L_t}{\partial h_t} + J_{t+1}^{\top}\,\bar h_{t+1}, \qquad \bar h_{T} = \frac{\partial L_T}{\partial h_T}.
$$

It runs **backward in time**, from $t = T$ down to $t = 1$.

**Gradients of the shared parameters.** Let $\bar a_t = \bar h_t \odot (1 - h_t^2)$ be the adjoint of the pre-activation. Because $W$ is used at every step, its gradient is a sum of one contribution per step — the fan-out rule of Unit 03:

$$
\frac{\partial L}{\partial W} = \sum_{t=1}^{T} \bar a_t\, h_{t-1}^{\top}, \qquad
\frac{\partial L}{\partial U} = \sum_{t=1}^{T} \bar a_t\, x_{t}^{\top}, \qquad
\frac{\partial L}{\partial b} = \sum_{t=1}^{T} \bar a_t .
$$

Worked Example 5.6 carries out this computation by hand for a three-step scalar RNN and checks it against finite differences.

**Why long-range learning is hard.** Unroll the adjoint recursion. The contribution of the loss at time $t$ to the adjoint at an earlier time $k$ is

$$
\frac{\partial L_t}{\partial h_k} = \frac{\partial L_t}{\partial h_t}\; J_t J_{t-1}\cdots J_{k+1}
\quad\text{(as a row vector)}.
$$

A product of $t-k$ Jacobians. Using $\|J_j\| \le \|\mathrm{diag}(1-h_j^2)\|\,\|W\| \le \|W\|$ (spectral norms; the diagonal factor has entries at most 1),

$$
\Bigl\|\frac{\partial L_t}{\partial h_k}\Bigr\| \le \Bigl\|\frac{\partial L_t}{\partial h_t}\Bigr\|\;\|W\|^{\,t-k}.
$$

If the largest singular value of $W$ is below 1, the signal from a loss $t-k$ steps later **vanishes geometrically** — the learning signal that would teach the network to keep information over long spans is exponentially small ([Bengio et al. 1994](/references/#bengio-1994); [Pascanu et al. 2013](/references/#pascanu-2013)). If $\rho(W) > 1$, products can instead **explode**, and a single update can throw the parameters far away.

For the *linear* recurrence the connection is exact: $\partial h_t/\partial h_k = A^{t-k}$, the very matrix that measures how much $h_k$ is remembered in $h_t$ (§5.3.2). **Memory decay and gradient decay are the same object.** A network can only learn to use a dependency that its current dynamics already preserve long enough for the gradient to arrive.

![Log of the norm of the product of Jacobians ∂h_t/∂h_k for a tanh RNN with 64 units, against the time lag, for recurrent matrices scaled to spectral radius 0.5, 1.0 and 1.5. All three decline; ρ = 0.5 fastest, reaching about 10^-29 at lag 60.](/figures/u05-bptt-gradient-decay.svg)

*Figure 5.5 — Gradients through time in a tanh RNN. Saturation of $\tanh$ shrinks the products even when $\rho(W) \ge 1$, so vanishing is the common failure; explosion appears mainly in the near-linear regime.*

**Two standard remedies, and what they do not fix.**

- **Gradient clipping**: if $\|g\| > c$, replace $g$ by $c\,g/\|g\|$ ([Pascanu et al. 2013](/references/#pascanu-2013)). It prevents explosive updates; it does nothing for vanishing gradients.
- **Truncated BPTT**: run the backward recursion for at most $k$ steps. It bounds memory and compute, but by construction it cannot learn dependencies longer than $k$.

To learn long dependencies, the dynamics themselves must preserve information. That is the job of gates.

## 5.3.8 Gates: an input-dependent decay

Return to the EMA and to question 3 of §5.1. A fixed $\alpha$ forced a compromise between forgetting noise and keeping signal. Let the decay be **computed from the current input and state**:

$$
f_t = \sigma\bigl(W_f[h_{t-1}; x_t] + b_f\bigr) \in (0,1)^N,
\qquad
h_t = f_t \odot h_{t-1} + (1 - f_t)\odot \tilde h_t ,
$$

where $\sigma(z) = 1/(1+e^{-z})$ is the logistic function, $[\,\cdot\,;\,\cdot\,]$ concatenates vectors, $\odot$ is elementwise multiplication, and $\tilde h_t$ is a proposed new content (for example $\tanh(W_h[h_{t-1};x_t] + b_h)$). Component by component, this is an EMA whose $\alpha$ is chosen anew at every step: $f_t \approx 1$ means *hold*, $f_t \approx 0$ means *overwrite*. The instantaneous half-life of component $i$ is $\ln 2/(-\ln f_{t,i})$, now under the network's control.

**Why gates help learning.** Ignoring the (smaller) paths through $f_t$ and $\tilde h_t$, the Jacobian of the direct path is

$$
\frac{\partial h_t}{\partial h_{t-1}} \approx \mathrm{diag}(f_t).
$$

If the network sets $f_t \approx 1$ on a component, both the memory *and* its gradient pass through almost unchanged over many steps. Unlike the RNN of §5.3.6, there is no repeated multiplication by $W$ and no $\tanh'$ factor on this path.

**The LSTM.** The long short-term memory cell ([Hochreiter & Schmidhuber 1997](/references/#hochreiter-1997)) with a forget gate ([Gers et al. 2000](/references/#gers-2000)) separates the gated memory, the **cell state** $c_t$, from the exposed output $h_t$, and gives writing and forgetting separate gates:

$$
\begin{aligned}
f_t &= \sigma(W_f[h_{t-1};x_t]+b_f) && \text{forget gate (decay)}\\
i_t &= \sigma(W_i[h_{t-1};x_t]+b_i) && \text{input gate (write strength)}\\
o_t &= \sigma(W_o[h_{t-1};x_t]+b_o) && \text{output gate (read strength)}\\
g_t &= \tanh(W_g[h_{t-1};x_t]+b_g) && \text{candidate content}\\
c_t &= f_t\odot c_{t-1} + i_t\odot g_t && \text{gated memory update}\\
h_t &= o_t\odot\tanh(c_t) && \text{exposed state}
\end{aligned}
$$

The cell update is our gated EMA with the constraint "weights sum to one" removed: forgetting ($f_t$) and writing ($i_t$) are decided independently. A common practical detail is to initialize $b_f$ to a positive value (e.g. 1) so that cells start out remembering; this was found to help in a large empirical comparison of recurrent architectures ([Jozefowicz et al. 2015](/references/#jozefowicz-2015)). The **GRU** ([Cho et al. 2014](/references/#cho-2014)) keeps the convex-combination form with an *update gate* and adds a *reset gate* that controls how much of $h_{t-1}$ enters the candidate; sources differ on whether the update gate weights the old or the new state, so check conventions when comparing implementations.

:::note[Pedagogical reconstruction, not history]
We arrived at gates by asking the EMA to choose its own $\alpha$. That is a reconstruction for teaching. Historically, the 1997 LSTM was motivated by keeping error flow constant through a self-connection of weight 1 (the "constant error carousel") and had **no forget gate**; the forget gate was added in 2000 because cells otherwise could not reset their memory on long continual streams ([Gers et al. 2000](/references/#gers-2000)).
:::

## 5.3.9 State-space models: exact discretization of continuous-time memory

Gates give flexible memory but bring back the sequential bottleneck: $h_t$ depends nonlinearly on $h_{t-1}$, so training must step through time one position at a time. A different line of work returns to the **linear** recurrence of §5.3.1 and asks how to make it a good long-range memory while keeping the parallel convolution form of §5.3.2.

**Start from continuous time**, as in §5.2.7:

$$
\frac{dh}{dt} = A\,h(t) + B\,x(t), \qquad y(t) = C\,h(t),
$$

with $A\in\mathbb{R}^{N\times N}$. Sample with step $\Delta$ under the zero-order hold. On one interval $x$ is constant, and the equation can be solved exactly with an integrating factor: multiply by $e^{-At}$ (the matrix exponential), note that $\tfrac{d}{dt}\bigl(e^{-At}h\bigr) = e^{-At}(\dot h - Ah) = e^{-At}Bx$, and integrate from $t_0$ to $t_0+\Delta$:

$$
h(t_0+\Delta) = e^{A\Delta}h(t_0) + \Bigl(\int_0^{\Delta} e^{As}\,ds\Bigr)B\,x
              = \underbrace{e^{A\Delta}}_{\bar A}\,h(t_0) + \underbrace{A^{-1}\bigl(e^{A\Delta} - I\bigr)B}_{\bar B}\,x ,
$$

where the integral was evaluated assuming $A$ is invertible. The discrete system $h_t = \bar Ah_{t-1} + \bar Bx_t$ is the exact sampled version.

**Check: this is the EMA.** Take $N = 1$, $A = -1/\tau$, $B = 1/\tau$ (the cooling law $\tau\dot h = x - h$). Then $\bar A = e^{-\Delta/\tau}$ and $\bar B = (-\tau)\bigl(e^{-\Delta/\tau}-1\bigr)\tfrac1\tau = 1 - e^{-\Delta/\tau}$: exactly $\alpha$ and $1-\alpha$ from §5.2.7. *The EMA is a one-dimensional state-space model.*

**What makes these models work** is the choice of $A$ and the dual computation:

- **Stable, well-conditioned $A$.** If the eigenvalues of $A$ have negative real parts, those of $\bar A = e^{A\Delta}$ have modulus $e^{\Delta\,\mathrm{Re}\lambda} < 1$ for every $\Delta > 0$: exact discretization never produces the instability of Euler (§5.2.8). Diagonal or diagonal-plus-low-rank $A$ avoids the transient growth of §5.3.5 and makes $\bar A$ cheap to compute. A diagonal $A$ with negative real parts is exactly the bank of continuous-time EMAs of §5.3.1, each with its own time constant ([Gu et al. 2022b](/references/#gu-2022b)).
- **Designed long memory.** HiPPO ([Gu et al. 2020](/references/#gu-2020)) derives specific matrices $A$ for which the state $h(t)$ holds the coefficients of the best polynomial approximation of the input history — a principled answer to "what should a fixed-size state store?".
- **Two computations of one function.** Because the system is linear and time-invariant, training computes all outputs at once as a convolution with kernel $K_j = C\bar A^{j}\bar B$; inference runs the recurrence with constant memory per step. The S4 model made this practical for very long sequences ([Gu et al. 2022a](/references/#gu-2022)).
- **Learned timescales.** $\Delta$ is a learned parameter (one per channel): the network learns how fast each memory should forget.

## 5.3.10 Selectivity: letting the input choose what to remember

A time-invariant system cannot choose. Its kernel $K$ is the same whatever the input, so it filters a glitch exactly as it filters a genuine change — the noise–lag trade-off of §5.2.6 in many dimensions. Gates (§5.3.8) solved this for RNNs by making the decay input-dependent. **Selective state-space models** do the same for SSMs: the step size and input/output maps become functions of the current input,

$$
\Delta_t = \mathrm{softplus}(w_\Delta^{\top} x_t + b_\Delta), \qquad
\bar A_t = e^{\Delta_t A}, \qquad \bar B_t \approx \Delta_t B(x_t), \qquad y_t = C(x_t)\,h_t ,
$$

and $h_t = \bar A_t h_{t-1} + \bar B_t x_t$ ([Gu & Dao 2024](/references/#gu-2024); the simplified $\bar B_t$ is the one used there). For a single real channel with $A = -1$: $\bar A_t = e^{-\Delta_t}$.

- $\Delta_t \to 0$: $\bar A_t \to 1$ and $\bar B_t \to 0$ — **hold** the state and ignore this input.
- $\Delta_t$ large: $\bar A_t \to 0$ — **reset** and write the current input.

This is a gate, $f_t = e^{-\Delta_t}$, derived from the continuous-time view rather than added by hand. The price is that the system is no longer time-invariant, so the convolution form of §5.3.2 is lost.

**Recovering parallelism: the scan.** The recurrence $h_t = a_th_{t-1} + b_t$ (with $a_t$ diagonal, here written as scalars) can still be computed in parallel. Represent one step by the pair $(a_t, b_t)$, meaning the map $h \mapsto a_th + b_t$. Applying step 1 and then step 2 gives $h \mapsto a_2(a_1h + b_1) + b_2$, so define the composition

$$
(a_1, b_1)\bullet(a_2, b_2) = (a_2a_1,\; a_2b_1 + b_2).
$$

**Claim: $\bullet$ is associative.** Both $\bigl((a_1,b_1)\bullet(a_2,b_2)\bigr)\bullet(a_3,b_3)$ and $(a_1,b_1)\bullet\bigl((a_2,b_2)\bullet(a_3,b_3)\bigr)$ equal $(a_3a_2a_1,\; a_3a_2b_1 + a_3b_2 + b_3)$ — expand both to check. (More simply: composition of functions is always associative.) $\blacksquare$

Because the operator is associative, all prefix compositions $h_1,\dots,h_T$ can be computed by a **parallel prefix scan** in $O(\log T)$ parallel steps and $O(T)$ total work ([Blelloch 1990](/references/#blelloch-1990)). Worked Example 5.8 does a four-step scan by hand. Selective SSMs combine this scan with careful use of fast on-chip memory to train efficiently.

## 5.3.11 The opposite strategy: store everything

Every memory so far compresses: the state has a fixed size $N$ however long the stream. Unit 04's **attention** takes the opposite route. At step $t$ it keeps the key and value of *every* previous token — the **KV cache** — and reads by content:

$$
y_t = \sum_{j\le t} \frac{\exp(q_t^{\top}k_j/\sqrt{d})}{\sum_{j'\le t}\exp(q_t^{\top}k_{j'}/\sqrt{d})}\; v_j .
$$

In the language of §5.2.1, the "state" is the whole cache $\{(k_j, v_j)\}_{j\le t}$, the shift register of the fourth row of the table, extended to the full history. Consequences:

- **Exact retrieval at any distance.** A token 10 000 steps back is as accessible as the previous one, provided the query matches its key.
- **No decision at write time.** Nothing has to be forgotten when a token arrives; the choice of what matters is made at **read** time, by the query. This sidesteps the dilemma of §5.2.6 completely.
- **Cost grows with the history.** Memory and per-step compute are $O(t\,d)$; over a sequence of length $T$, total attention compute is $O(T^2 d)$ (Unit 04, §4.5.6). The memory of the KV cache is computed exactly in the [KV cache Atlas node](/atlas/nodes/mech-kv-cache/).

## 5.3.12 In between: linear attention is a recurrence with a matrix state

Is there a design that reads by content, like attention, but keeps a fixed-size state, like a recurrence? Remove the softmax and use any feature map $\phi$ with non-negative outputs ([Katharopoulos et al. 2020](/references/#katharopoulos-2020)):

$$
y_t = \frac{\sum_{j\le t} v_j\,\phi(k_j)^{\top}\phi(q_t)}{\sum_{j\le t}\phi(k_j)^{\top}\phi(q_t)} .
$$

The sums over $j$ do not involve $q_t$, so they can be accumulated as a recurrence:

$$
S_t = S_{t-1} + v_t\,\phi(k_t)^{\top} \in \mathbb{R}^{d_v\times d_k}, \qquad
z_t = z_{t-1} + \phi(k_t), \qquad
y_t = \frac{S_t\,\phi(q_t)}{z_t^{\top}\phi(q_t)} .
$$

This is a linear recurrence whose state is a **matrix**: writing stores the outer product $v_t\phi(k_t)^\top$; reading multiplies by a query. It is an **associative memory**: a key retrieves its value by matrix–vector multiplication.

How much can it hold? Drop $\phi$ and the normalizer to see the mechanism: $S = \sum_j v_jk_j^{\top}$, read with $Sq$.

- **Orthonormal keys: exact recall up to $d_k$ pairs.** If $k_1,\dots,k_n$ are orthonormal ($n\le d_k$), then $Sk_i = \sum_j v_j(k_j^\top k_i) = v_i$, because $k_j^\top k_i$ is 1 for $j = i$ and 0 otherwise.
- **More pairs than dimensions: interference.** At most $d_k$ vectors can be mutually orthogonal in $\mathbb{R}^{d_k}$. Beyond that, $Sk_i = v_i + \sum_{j\ne i}(k_j^{\top}k_i)\,v_j$: every other stored value leaks into the answer. §5.5.3 computes the size of this *cross-talk*.

So linear attention trades the KV cache's exact, growing memory for a fixed $d_v\times d_k$ memory with bounded capacity. Recent variants add a decay factor ($S_t = \gamma S_{t-1} + \dots$, as in retention-style models; [Sun et al. 2023](/references/#sun-2023)) or **overwrite** instead of accumulate with the *delta rule* $S_t = S_{t-1} + \beta_t\,(v_t - S_{t-1}k_t)\,k_t^{\top}$ ([Schlag et al. 2021](/references/#schlag-2021); [Yang et al. 2024](/references/#yang-2024)), which replaces the value currently stored under $k_t$ rather than adding a second copy.

## 5.3.13 The landscape, in one table

| Design | State at step $t$ | Per-step cost | Write decision | Exact recall at distance $T$? |
|---|---|---|---|---|
| Fixed window $W$ | last $W$ inputs | $O(Wd)$ | none (FIFO) | only if $T\le W$ |
| EMA / LTI recurrence / LTI SSM | $N$ numbers | $O(N)$ | fixed decay | no — geometric decay |
| Gated RNN (LSTM, GRU) | $N$ numbers | $O(N^2)$ | input-dependent gates | only what the gates chose to keep |
| Selective SSM | $N$ per channel | $O(N)$ per channel | input-dependent $\Delta_t$ | only what $\Delta_t$ chose to keep |
| Linear attention | $d_v\times d_k$ matrix | $O(d_vd_k)$ | additive (or delta rule) | up to $\approx d_k$ items, then interference |
| Softmax attention + KV cache | $2td$ numbers | $O(td)$ | none (store all) | yes, within the context window |

Every row answers §5.1's question differently. Moving down the table buys recall with memory and compute; moving up buys efficiency with forgetting. §5.5.4 shows that all of these layers can be written in one common matrix form, which makes the comparison precise.

**Next:** [§5.4 Worked examples](/core/unit-05/worked-examples/) computes each mechanism by hand.
