---
title: "5.5 General theory: filters, capacity and the matrix view"
description: Transfer functions and frequency response; finite-precision counting bounds; associative-memory capacity; and the conditional mixing-matrix and cut-rank views of sequence layers.
sidebar: { label: "5.5 General theory", order: 5 }
---

Sections 5.2–5.4 built memories one mechanism at a time. This section steps back and asks three questions: *what does a linear memory do to different frequencies? what can a finite-precision state remember in principle? and which sequence layers can be compared through one matrix language?* The qualifiers matter. A frequency response belongs to a linear time-invariant system; a counting bound needs a precision model; and one mixing matrix describes nonlinear layers only after their input-dependent coefficients have been fixed.

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

So two notions of stability must be kept apart. **Internal (asymptotic) stability**, $\rho(A)<1$—the condition of Theorem 5.1—says every mode decays, whatever the initial state. **Input–output (BIBO) stability** says bounded inputs give bounded outputs from a zero initial state; it requires only that the *poles of $G$* lie inside the unit circle. The two coincide for minimal realizations, which is the classical criterion of signal processing ([Oppenheim & Schafer 2010](/references/#oppenheim-2010)). Which notion matters depends on the claim. BIBO stability is enough for the ideal input--output map; internal stability is safer when the state itself must remain numerically controlled. A mathematically unobservable unstable mode does not affect $y_t$ in the exact model, but it can overflow or leak into the output after quantization, parameter drift or implementation error.

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

Slow changes pass; fast fluctuations are suppressed—which is why the EMA reduces sensor noise and also why it delays genuine changes. Independent white noise is **not mostly high-frequency**: its power is flat across frequency. A low-pass filter reduces its variance by rejecting most of that frequency band, while retaining the low-frequency part. A step contains all frequencies, so the same filter also removes its sharp edge. The noise--lag trade-off of §5.2.6 is one statement about this curve.

When the input and readout actually couple to them, oscillating modes (§5.3.3) can produce gains that peak near their frequency $\theta$: they act as *band-pass* memories. A bank of suitably coupled modes at many frequencies forms a filter bank—the frequency-domain counterpart of “many timescales.”

## 5.5.2 What a finite state cannot remember

Section 5.2.1 claimed that an exact $W$-step delay needs storage that grows with $W$. The phrase “state as large as the window” is imprecise until we choose a unit. The next theorem counts **bits**, not vector coordinates. It models a streaming delay: after a window has entered, continuing to clock the system must reveal all $W$ stored symbols in order.

**Theorem 5.2 (counting bound).** Consider a deterministic system with state update $s_t = F(s_{t-1}, x_t)$ and output $y_t = G(s_t)$, reading symbols from an alphabet of size $V$. Suppose that after reading *any* window $w = (x_1,\dots,x_W)$, the system can, during the next $W$ steps, output $x_1, x_2, \dots, x_W$ in order (whatever filler symbols it receives meanwhile). Then the number of distinct states reachable after step $W$ is at least $V^W$. If the state is stored in $b$ bits, $b \ge W\log_2V$.

*Proof.* Suppose two different windows $w \ne w'$ led to the same state $s_W$. By the definition of a state, the system's future outputs depend on the past only through $s_W$; given the same filler inputs, the outputs during the recall phase would be identical. But $w$ and $w'$ differ in some position $i$, so the outputs at recall step $i$ must differ. Contradiction. Hence the map from windows to states is injective, and there are at least $V^W$ states. $\blacksquare$

**What the lower bound does and does not say.** For a GPT-2-sized vocabulary ($V=50\,257$) and $W=1000$ tokens, $W\log_2V\approx15\,617$ bits—about 2 KB. A recurrent state of 1,024 independently usable 16-bit numbers would contain 16,384 raw bits. The counting bound alone therefore does **not** rule out that state recalling 1,000 tokens; it says only that a lossless code cannot use fewer than about 15,617 state bits. It does not show that all floating-point bit patterns are reachable, distinguishable under noise, learnable by gradient descent, or decodable by the model. Those robustness and learnability questions are what make dense storage hard in practice, and §5.7 tests them empirically.

:::caution[A counterexample to a careless capacity argument]
Theorem 5.2 counts *distinct states*. With infinitely precise real numbers, a single real state has infinitely many values and can store anything: the update $h_t=(h_{t-1}+x_t)/V$ with $x_t\in\{0,\dots,V-1\}$ writes the history into the base-$V$ digits of $h_t$. Statements such as “a $d$-dimensional state can remember only $O(d)$ items” are therefore claims about **finite precision**, noise, a regularity requirement, or the restricted dynamics of a model class—never about unconstrained real arithmetic. The digit encoding is also catastrophically non-robust: recovering an old digit requires resolving exponentially small differences. Always ask what precision and robustness a capacity claim assumes.
:::

## 5.5.3 Capacity of an associative memory

Linear attention (§5.3.12) stores pairs in a matrix $S\in\mathbb{R}^{d_v\times d_k}$ and reads with $Sk$. Two questions: how many pairs can be stored *exactly*, and how fast does recall degrade beyond that?

**Theorem 5.3 (exact capacity).** Let $K = [k_1,\dots,k_n] \in \mathbb{R}^{d_k\times n}$ and $V = [v_1,\dots,v_n]$.

1. With **outer-product (Hebbian) storage** $S = \sum_j v_jk_j^{\top} = VK^{\top}$, recall is exact for every choice of values iff $K^{\top}K = I$, i.e. the keys are orthonormal. In particular $n \le d_k$.
2. With **any linear memory** $S$ (for example the one the delta rule converges to), exact recall $SK = V$ for every $V$ is possible iff the keys are linearly independent. Again $n \le d_k$.

*Proof.* (1) $SK = VK^{\top}K$, which equals $V$ for all $V$ iff $K^{\top}K = I$. (2) $SK = V$ has a solution $S$ for every $V$ iff the linear map $S\mapsto SK$ is onto, i.e. iff $K$ has full column rank $n$; $n$ independent vectors in $\mathbb{R}^{d_k}$ require $n\le d_k$. $\blacksquare$

Part 2 describes what an ideal linear solver could represent. A delta-rule update can approach such a solution under suitable normalization, step sizes and repeated presentation of the pairs; one pass is not guaranteed to solve $SK=V$, and ill-conditioned keys make the solution sensitive to noise. Thus “independent” is an algebraic existence condition, not a training guarantee.

**Graceful degradation with random keys.** Draw keys independently and uniformly on the unit sphere of $\mathbb{R}^{d_k}$, and values independently with mean zero and $\mathbb{E}\|v\|^2 = s^2$. Reading key $k_i$ from the Hebbian memory returns

$$
Sk_i = v_i + \underbrace{\sum_{j\ne i}(k_j^{\top}k_i)\,v_j}_{\text{cross-talk } e_i}.
$$

By symmetry each coordinate of a uniform unit vector has $\mathbb{E}[k_m^2] = 1/d_k$ and distinct coordinates are uncorrelated, so $\mathbb{E}[(k_j^{\top}k_i)^2] = \sum_m\mathbb{E}[k_{jm}^2]\,\mathbb{E}[k_{im}^2] = d_k\cdot\tfrac{1}{d_k^2} = \tfrac1{d_k}$. The cross-talk terms are uncorrelated with mean zero, so

$$
\mathbb{E}\|e_i\|^2 = \frac{n-1}{d_k}\,s^2,
\qquad\frac{\sqrt{\mathbb E\|e_i\|^2}}{\sqrt{\mathbb E\|v_i\|^2}}
= \sqrt{\frac{n-1}{d_k}} .
$$

At $n=d_k+1$ this RMS cross-talk equals the signal RMS (at $n=d_k$ it is $\sqrt{1-1/d_k}$ times as large). Figure 5.6 compares the RMS-scale prediction with a simulation that reports a mean ratio of norms; those are different statistics, so they need not coincide exactly.

![Relative retrieval error of a linear-attention memory with 16-dimensional random unit keys as the number of stored pairs grows to eight times the key dimension; measured points track the RMS-scale estimate square root of (n−1)/d_k and become order one near n = d_k.](/figures/u05-linear-attention-capacity.svg)

*Figure 5.6 — Interference grows like $\sqrt{n/d_k}$: a fixed matrix state degrades gracefully, not abruptly.*

<div class="dlis-widget not-content" data-widget="linear-attention-capacity">Interactive figure — requires JavaScript. The static figure and tables above contain the same information.</div>

For comparison, the classical dense Hopfield model with random unbiased binary patterns, Hebbian weights and its standard recurrent retrieval dynamics has a critical load near $0.14N$ patterns under the usual asymptotic analysis ([Hopfield 1982](/references/#hopfield-1982); [Amit et al. 1985](/references/#amit-1985)). That number belongs to that model and success criterion; sparse coding, different learning rules and modern continuous Hopfield variants have different scaling. The reusable lesson is narrower: always define the storage rule, query distribution, error tolerance and success probability before quoting a capacity.

## 5.5.4 A conditional mixing-matrix view

There is a useful common language, but it does **not** turn every sequence layer into the same linear function. Freeze the coefficients produced for one input sequence. An LTI recurrence is already linear; for a selective SSM, freeze $\bar A_t,\bar B_t,C_t$; for attention, freeze the queries and keys and ask only how the value vectors are mixed. For one scalar value channel, the resulting conditional map is

$$
y_t=\sum_{s\le t}M_{ts}x_s,\qquad\text{or}\qquad y=Mx,
$$

where causality makes $M$ lower triangular. The full selective-SSM or attention layer is still nonlinear because $M$ itself depends on the input. With that warning in place, its structure reveals both cost and a precise kind of cross-time communication ([Dao & Gu 2024](/references/#dao-2024)):

| Layer | Conditional coefficient $M_{ts}$ | Structure relevant across a time cut | Cost to apply across length $T$ |
|---|---|---|---|
| LTI recurrence / SSM, dense $\bar A$ | $C\bar A^{t-s}\bar B$ | Toeplitz; cut blocks have rank $\le N$ | $O(TN^2)$ by recurrence |
| LTI SSM, diagonal $\bar A$ | same | Toeplitz; cut blocks have rank $\le N$ | $O(TN)$ by recurrence, or kernel plus FFT |
| Selective SSM, diagonal $\bar A_t$ | $C_t(\prod_{r=s+1}^{t}\bar A_r)\bar B_s$ | data-dependent semiseparable; cut blocks have rank $\le N$ | $O(TN)$ work by scan |
| Linear attention, unnormalized feature width $r$ | $\phi(q_t)^\top\phi(k_s)$ | unmasked kernel and every past-to-future block have rank $\le r$ | $O(Trd_v)$ by matrix-state recurrence |
| Softmax attention | row-normalized $\exp(q_t^\top k_s/\sqrt{d_k})$ | dense and data-dependent; a cut block can be full rank | $O(T^2d)$ at the usual head-width scale |

The table suppresses batch, channel and projection costs. A dense $N\times N$ transition costs $O(N^2)$ per step; the $O(TN)$ SSM entry needs diagonal or comparable structure. The FFT route first forms $T$ kernel values—$O(TN)$ for a diagonal SSM—and works only for the time-invariant Toeplitz case. For linear attention, the causal lower triangle of a rank-$r$ unmasked kernel need **not itself have rank $r$**: masking can raise the rank of the whole triangular matrix. The low-rank statement that survives masking is about a block lying completely across a past--future cut.

**The cut-rank bound.** Consider a conditionally linear recurrence

$$
h_t=A_th_{t-1}+B_tx_t,\qquad y_t=C_th_t,
$$

with an $N$-dimensional state and fixed coefficient matrices. Choose a cut $c$. For $s\le c<t$,

$$
M_{ts}
=\underbrace{C_tA_t\cdots A_{c+1}}_{\text{row }t\text{ of }P\in\mathbb R^{(T-c)\times N}}
\underbrace{A_c\cdots A_{s+1}B_s}_{\text{column }s\text{ of }Q\in\mathbb R^{N\times c}}.
$$

Therefore the block carrying past inputs to future outputs factors as $PQ$ and has **rank at most $N$**. By contrast, a softmax-attention cut block can generically reach rank $\min(c,T-c)$ because different future queries can assign independent weight patterns to the cached past. If a task asks a *linear* map to transmit $W$ independent real-valued items into $W$ independent future output directions, that block needs rank at least $W$; an $N<W$ linear state cannot do it.

This is complementary to Theorem 5.2, not a replacement for it. The counting theorem concerns discrete information in a finite-precision state. Cut rank concerns independent directions transmitted by a conditional linear map. An arbitrary nonlinear recurrence is not globally described by one $M$; its Jacobian across a cut still factors through the state dimension locally, but nonlinear coding can store information in ways that a global linear-rank statement does not count.

The safe summary is therefore two-dimensional: **structure controls how cheaply a conditional mixing matrix can be applied; cut rank measures one form of linear communication across time.** Full cut rank does not guarantee exact retrieval, and low cut rank is not the total memory capacity of every nonlinear model. Selective and gated models use input-dependent dynamics to decide which information should occupy their limited state.
