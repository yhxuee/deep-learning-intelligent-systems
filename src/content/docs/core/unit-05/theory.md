---
title: "5.5 General theory: filters, capacity and the matrix view"
description: Transfer functions and frequency response of linear memories; counting and rank bounds on what a finite state can recall; associative-memory capacity; one matrix form for attention, SSMs and linear attention.
sidebar: { label: "5.5 General theory", order: 5 }
---

Sections 5.2–5.4 built memories one mechanism at a time. This section steps back and asks questions that apply to all of them: *what does a linear memory do to different frequencies? what can any fixed-size state remember, in principle? and is there one language in which every design of §5.3.13 is a special case?*

## 5.5.1 Linear memories are filters

**The z-transform.** For a sequence $x_1, x_2, \dots$ define $X(z) = \sum_{t\ge1}x_tz^{-t}$, a power series in $z^{-1}$. Its key property is that a one-step delay multiplies by $z^{-1}$: the transform of $(x_{t-1})_t$ is $z^{-1}X(z)$ (when the sequence starts from zero).

**Transfer function.** Take the transform of $h_t = Ah_{t-1} + Bx_t$ with $h_0 = 0$:

$$
H(z) = Az^{-1}H(z) + BX(z) \;\Longrightarrow\; (I - Az^{-1})H(z) = BX(z) \;\Longrightarrow\; H(z) = (I - Az^{-1})^{-1}BX(z).
$$

With $y_t = Ch_t$, the output is $Y(z) = G(z)X(z)$ where

$$
G(z) = C\,(I - Az^{-1})^{-1}B = C\,z\,(zI - A)^{-1}B .
$$

$G$ is the **transfer function**. It is a ratio of polynomials in $z$, and since $(zI - A)^{-1} = \mathrm{adj}(zI - A)/\det(zI - A)$, every **pole** of $G$ is an eigenvalue of $A$. The converse holds only for a **minimal** realization — one in which every mode can be excited by the input (controllable) and seen in the output (observable). Otherwise factors of $\det(zI - A)$ cancel and some eigenvalues are invisible from input to output ([Kailath 1980](/references/#kailath-1980)).

*Counterexample.* $A = \mathrm{diag}(0.5, 1.2)$, $B = (1, 0)^{\top}$, $C = (1, 0)$. Then $G(z) = 1/(1 - 0.5z^{-1})$, with the single pole 0.5. The eigenvalue 1.2 belongs to a mode the input never reaches and the output never reads: started from $h_0 = 0$, the system is input–output stable, yet any nonzero initial value of the second state component grows as $1.2^t$.

So two notions of stability must be kept apart. **Internal (asymptotic) stability**, $\rho(A) < 1$ — the condition of Theorem 5.1 — says every mode decays, whatever the initial state. **Input–output (BIBO) stability** says bounded inputs give bounded outputs from a zero initial state; it requires only that the *poles of $G$* lie inside the unit circle. The two coincide for minimal realizations, which is the classical criterion of signal processing ([Oppenheim & Schafer 2010](/references/#oppenheim-2010)). For a learned memory, internal stability is the one that matters: the state is carried across arbitrarily many steps, and a hidden unstable mode excited by rounding error or a nonzero initial state will eventually dominate.

**Frequency response.** Feed a complex sinusoid $x_t = e^{i\omega t}$ to a stable system. After transients die out, the output is $G(e^{i\omega})\,e^{i\omega t}$: the same frequency, scaled by $|G(e^{i\omega})|$ and phase-shifted. So $|G(e^{i\omega})|$ says how strongly the memory passes each frequency $\omega \in [0, \pi]$ (in radians per step; $\omega = \pi$ is the fastest possible alternation $+1, -1, +1, \dots$).

**The EMA is a low-pass filter.** With $A = \alpha$, $B = 1-\alpha$, $C = 1$:

$$
G(z) = \frac{1-\alpha}{1-\alpha z^{-1}}, \qquad
|G(e^{i\omega})| = \frac{1-\alpha}{\sqrt{1 - 2\alpha\cos\omega + \alpha^2}} ,
$$

using $|1-\alpha e^{-i\omega}|^2 = (1-\alpha\cos\omega)^2 + \alpha^2\sin^2\omega$. For $\alpha = 0.9$:

| frequency $\omega$ | 0 (constant) | $\pi/8$ | $\pi/2$ | $\pi$ (alternating) |
|---|---|---|---|---|
| gain $\lvert G\rvert$ | 1.000 | 0.261 | 0.074 | 0.053 |

Slow changes pass; fast fluctuations are suppressed — which is why the EMA removes sensor noise (mostly high-frequency) and also why it delays genuine changes (a step contains all frequencies). The noise–lag trade-off of §5.2.6 is one statement about this curve: a lower cutoff removes more noise *and* more of the step's sharp edge.

Oscillating modes (§5.3.3) produce gains that peak near their own frequency $\theta$: they are *band-pass* memories. A bank of modes at many frequencies is a filter bank — the frequency-domain counterpart of "many timescales".

## 5.5.2 What a finite state cannot remember

Section 5.2.1 claimed that answering "what was the input $W$ steps ago?" needs a state at least as large as the window. Here is the precise statement.

**Theorem 5.2 (counting bound).** Consider a deterministic system with state update $s_t = F(s_{t-1}, x_t)$ and output $y_t = G(s_t)$, reading symbols from an alphabet of size $V$. Suppose that after reading *any* window $w = (x_1,\dots,x_W)$, the system can, during the next $W$ steps, output $x_1, x_2, \dots, x_W$ in order (whatever filler symbols it receives meanwhile). Then the number of distinct states reachable after step $W$ is at least $V^W$. If the state is stored in $b$ bits, $b \ge W\log_2V$.

*Proof.* Suppose two different windows $w \ne w'$ led to the same state $s_W$. By the definition of a state, the system's future outputs depend on the past only through $s_W$; given the same filler inputs, the outputs during the recall phase would be identical. But $w$ and $w'$ differ in some position $i$, so the outputs at recall step $i$ must differ. Contradiction. Hence the map from windows to states is injective, and there are at least $V^W$ states. $\blacksquare$

**What the bound does and does not say.** For a GPT-2-sized vocabulary ($V = 50\,257$) and $W = 1000$ tokens, $W\log_2V \approx 15\,617$ bits — about 2 KB. A recurrent state of 1 024 numbers in 16-bit precision holds 16 384 bits. So the counting bound alone does **not** rule out such a state recalling 1 000 tokens; it says the state cannot be much smaller. What makes recall hard in practice is not the bit count but whether *learned, smooth, decaying* dynamics can store information that densely and read it back robustly — a question the bound cannot answer, and which the empirical studies of §5.7 address.

:::caution[A counterexample to a careless capacity argument]
Theorem 5.2 counts *distinct states*. With infinitely precise real numbers, a single real state has infinitely many values and can store anything: the update $h_t = (h_{t-1} + x_t)/V$ with $x_t \in \{0,\dots,V-1\}$ writes the history into the base-$V$ digits of $h_t$. Statements such as "a $d$-dimensional state can remember only $O(d)$ items" are therefore claims about **finite precision**, noise, or the restricted dynamics of a model class — never about real arithmetic. Always ask which of these a capacity claim assumes.
:::

## 5.5.3 Capacity of an associative memory

Linear attention (§5.3.12) stores pairs in a matrix $S\in\mathbb{R}^{d_v\times d_k}$ and reads with $Sk$. Two questions: how many pairs can be stored *exactly*, and how fast does recall degrade beyond that?

**Theorem 5.3 (exact capacity).** Let $K = [k_1,\dots,k_n] \in \mathbb{R}^{d_k\times n}$ and $V = [v_1,\dots,v_n]$.

1. With **outer-product (Hebbian) storage** $S = \sum_j v_jk_j^{\top} = VK^{\top}$, recall is exact for every choice of values iff $K^{\top}K = I$, i.e. the keys are orthonormal. In particular $n \le d_k$.
2. With **any linear memory** $S$ (for example the one the delta rule converges to), exact recall $SK = V$ for every $V$ is possible iff the keys are linearly independent. Again $n \le d_k$.

*Proof.* (1) $SK = VK^{\top}K$, which equals $V$ for all $V$ iff $K^{\top}K = I$. (2) $SK = V$ has a solution $S$ for every $V$ iff the linear map $S\mapsto SK$ is onto, i.e. iff $K$ has full column rank $n$; $n$ independent vectors in $\mathbb{R}^{d_k}$ require $n\le d_k$. $\blacksquare$

The delta rule improves on Hebbian storage exactly as part 2 suggests: it only needs independent keys, not orthogonal ones.

**Graceful degradation with random keys.** Draw keys independently and uniformly on the unit sphere of $\mathbb{R}^{d_k}$, and values independently with mean zero and $\mathbb{E}\|v\|^2 = s^2$. Reading key $k_i$ from the Hebbian memory returns

$$
Sk_i = v_i + \underbrace{\sum_{j\ne i}(k_j^{\top}k_i)\,v_j}_{\text{cross-talk } e_i}.
$$

By symmetry each coordinate of a uniform unit vector has $\mathbb{E}[k_m^2] = 1/d_k$ and distinct coordinates are uncorrelated, so $\mathbb{E}[(k_j^{\top}k_i)^2] = \sum_m\mathbb{E}[k_{jm}^2]\,\mathbb{E}[k_{im}^2] = d_k\cdot\tfrac{1}{d_k^2} = \tfrac1{d_k}$. The cross-talk terms are uncorrelated with mean zero, so

$$
\mathbb{E}\|e_i\|^2 = \frac{n-1}{d_k}\,s^2,
\qquad\text{relative RMS error} \approx \sqrt{\frac{n-1}{d_k}} .
$$

At $n = d_k$ the noise is already as large as the signal. Figure 5.6 compares this first-order estimate with simulation; the measured mean relative error sits slightly above the estimate at high load, because averaging a ratio of norms is not the same as the ratio of mean squares.

![Relative retrieval error of a linear-attention memory with 16-dimensional random unit keys, as the number of stored pairs grows to 8 times the key dimension; measured points track the first-order estimate square root of (n−1)/d_k, passing 1 at n = d_k.](/figures/u05-linear-attention-capacity.svg)

*Figure 5.6 — Interference grows like $\sqrt{n/d_k}$: a fixed matrix state degrades gracefully, not abruptly.*

<div class="dlis-widget not-content" data-widget="linear-attention-capacity">Interactive figure — requires JavaScript. The static figure and tables above contain the same information.</div>

Classical Hopfield networks, a different associative memory with binary states and a recurrent recall dynamics, store about $0.15N$ patterns in $N$ units before recall breaks down ([Hopfield 1982](/references/#hopfield-1982)). The same moral holds across associative memories: capacity is linear in the dimension, and exceeding it causes interference.

## 5.5.4 One matrix to describe every sequence layer

Every layer in §5.3.13 maps an input sequence to an output sequence causally. For one input channel and one output channel, any such linear-in-$x$ map can be written

$$
y_t = \sum_{s\le t} M_{ts}\,x_s ,\qquad\text{i.e.}\qquad y = Mx\ \text{with } M \text{ lower-triangular}.
$$

The designs differ only in the **structure of $M$** ([Dao & Gu 2024](/references/#dao-2024)):

| Layer | $M_{ts}$ | Structure of $M$ | Cost to apply |
|---|---|---|---|
| LTI recurrence / SSM, dense $\bar A$ | $C\bar A^{\,t-s}\bar B$ | Toeplitz (depends only on $t-s$) | $O(TN^2)$ by recurrence |
| LTI SSM, diagonal (or diagonalized) $\bar A$ | same | Toeplitz | $O(TN)$ by recurrence; or form the length-$T$ kernel, then $O(T\log T)$ by FFT |
| Selective SSM, diagonal $\bar A_t$ | $C_t\bigl(\prod_{r=s+1}^{t}\bar A_r\bigr)\bar B_s$ | semiseparable, data-dependent | $O(TN)$ by scan |
| Linear attention (unnormalized) | $q_t^{\top}k_s$ | lower triangle of a rank-$d_k$ matrix | $O(Td_kd_v)$ by recurrence |
| Softmax attention | $\mathrm{softmax}_s(q_t^{\top}k_s/\sqrt d)$ | dense, data-dependent | $O(T^2d)$ |

Costs are for one input and one output channel with the state size $N$ and head sizes held fixed. A dense $N\times N$ transition costs $O(N^2)$ per step to apply; the $O(TN)$ entries assume a diagonal or otherwise structured transition, which is why practical SSMs use them. The FFT route requires first computing the $T$ kernel values $C\bar A^{k}\bar B$ (itself $O(TN)$ for diagonal $\bar A$) and applies only when $M$ is Toeplitz, that is, for time-invariant layers. (For attention, $M$ depends on the inputs through $q$ and $k$; it is "linear in $x$" only in its action on the values.)

**The rank bound — why a fixed state limits recall, in linear-algebra form.** Pick a cut time $c$ and look at the block of $M$ that maps *past* inputs ($s\le c$) to *future* outputs ($t > c$). For any recurrence with an $N$-dimensional state, everything the future knows about the past passes through $h_c\in\mathbb{R}^N$:

$$
M_{ts} = \underbrace{C_t\bar A_t\cdots\bar A_{c+1}}_{\text{row } t\text{ of a } (T-c)\times N \text{ matrix}}\;\underbrace{\bar A_c\cdots\bar A_{s+1}\bar B_s}_{\text{column } s\text{ of an } N\times c \text{ matrix}} .
$$

So the past-to-future block is a product of an $(T-c)\times N$ and an $N\times c$ matrix and has **rank at most $N$**, for every cut. Softmax attention has no such constraint: its past-to-future block can have rank up to $\min(c, T-c)$. Recalling $W$ independent items from the past requires a past-to-future map of rank at least $W$ (one independent direction per item), so a state of size $N < W$ cannot do it with any linear readout — the linear-algebra counterpart of Theorem 5.2.

This one picture organizes the whole unit: **memory capacity is the rank of the past-to-future block; computational cost is the structure that makes $M$ cheap to apply.** Attention buys full rank with quadratic cost; recurrences buy linear cost with rank at most $N$; selective and gated models spend their limited rank on the inputs they choose.
