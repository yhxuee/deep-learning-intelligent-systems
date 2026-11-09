---
title: "5.5 General theory: filters, capacity and the matrix view"
description: Transfer functions and frequency response; finite-precision counting bounds; associative-memory capacity; and the conditional mixing-matrix and cut-rank views of sequence layers.
sidebar: { label: "5.5 General theory", order: 5 }
---

Sections 5.2–5.4 systematically built sequence memory from simple exponential moving averages to modern selective state-space models (SSMs) and associative memories. This section steps back to evaluate sequence architectures from the foundational perspectives of control theory, information theory, and linear algebra, addressing three core theoretical questions:
1. **Spectral Geometry**: What filtering transformation does a linear memory intrinsically impose on different frequency components of a sequence?
2. **Capacity Bounds**: Under the finite-precision constraints of physical computing hardware, how much historical information can a fixed-dimensional state space mathematically retain?
3. **Architectural Unification**: How do recurrent neural networks, state-space models, and attention mechanisms compare within a unified framework of conditional mixing matrices and cross-cut rank bounds?

These theoretical qualifiers establish crucial boundaries: frequency response analysis strictly governs linear time-invariant (LTI) systems; state capacity bounds rely on rigorous information-theoretic counting; and highly nonlinear modern architectures (such as selective SSMs and Softmax attention) reveal their fundamental costs only when evaluated as conditional linear maps conditioned on specific input trajectories.

## 5.5.1 Linear memories are filters

In the time domain, linear recurrence intertwines past inputs through convolution sums $\sum_k K_{t-k} x_k$. The most illuminating lens for understanding this dynamic is the complex frequency domain.

**The z-Transform and Physical Delay.** For a discrete-time sequence $x_1, x_2, \dots$, its $z$-transform is defined as $X(z) = \sum_{t\ge1}x_tz^{-t}$, a power series in the complex variable $z^{-1}$. In discrete-time systems, $z^{-1}$ functions as the **unit delay operator**: for a system initialized at zero, the $z$-transform of the time-shifted sequence $(x_{t-1})_t$ is exactly $z^{-1}X(z)$.

**Transfer Functions and Eigenmode Poles.** Applying the $z$-transform to both sides of the discrete LTI system $h_t = Ah_{t-1} + Bx_t$ (with zero initial state $h_0 = 0$):

$$
H(z) = Az^{-1}H(z) + BX(z) \;\Longrightarrow\; (I - Az^{-1})H(z) = BX(z) \;\Longrightarrow\; H(z) = (I - Az^{-1})^{-1}BX(z).
$$

Coupling this with the linear readout $y_t = Ch_t$, the system's output in the $z$-domain satisfies $Y(z) = G(z)X(z)$, where:

$$
G(z) = C\,(I - Az^{-1})^{-1}B = C\,z\,(zI - A)^{-1}B .
$$

The matrix rational function $G(z)$ is the **transfer function** of the system. Expanding the matrix inverse via the adjugate matrix, $(zI - A)^{-1} = \mathrm{adj}(zI - A)/\det(zI - A)$, shows that the roots of the denominator polynomial—the **poles** of $G(z)$—correspond geometrically to the eigenvalues of the recurrence transition matrix $A$.

This correspondence between poles and eigenvalues holds strictly if and only if the realization is **minimal**—meaning that every internal eigenmode can be excited by the input (**controllability**) and observed in the output (**observability**). Without minimality, pole-zero cancellations occur, rendering certain internal dynamical modes completely hidden from input to output ([Kailath 1980](/references/#kailath-1980)).

*Counterexample.* Consider $A = \mathrm{diag}(0.5, 1.2)$, $B = (1, 0)^{\top}$, and $C = (1, 0)$. Direct calculation reveals $G(z) = 1/(1 - 0.5z^{-1})$, possessing a single pole at $0.5$. The explosive eigenvalue $1.2 > 1$ belongs to an internal mode that input $B$ never excites and readout $C$ never queries. Initiated at $h_0 = 0$, the system appears perfectly bounded and stable in its input–output behavior. However, any nonzero initial condition or numerical rounding perturbation in the second state coordinate will trigger exponential divergence $O(1.2^t)$, eventually causing catastrophic numerical overflow.

Thus, two distinct notions of stability must be distinguished in sequence architecture design:
- **Internal (Asymptotic) Stability**: The spectral radius satisfies $\rho(A) < 1$ (Theorem 5.1). This guarantees that regardless of initial perturbations or internal disturbances, all eigenmodes decay asymptotically to zero.
- **Input–Output (BIBO) Stability**: Starting from zero initial conditions, every bounded input sequence produces a bounded output sequence. This requires that all poles of $G(z)$ lie strictly inside the open unit disk $\{z \in \mathbb{C} : |z| < 1\}$.

For minimal realizations, the two definitions coincide ([Oppenheim & Schafer 2010](/references/#oppenheim-2010)). In physical deep learning systems subject to finite numerical precision, internal stability $\rho(A) < 1$ is the essential safeguard against hidden numerical divergence.

**Frequency Response Geometry.** Driving a stable system with a complex sinusoid $x_t = e^{i\omega t}$ (where angular frequency $\omega \in [0, \pi]$) yields, after transients decay, the steady-state output $G(e^{i\omega})\,e^{i\omega t}$. The output maintains frequency $\omega$, scaled in amplitude by $|G(e^{i\omega})|$ with an associated phase shift. The magnitude curve $|G(e^{i\omega})|$ describes how the memory selectively transmits or attenuates different frequency components.

**The EMA as a Canonical Low-Pass Filter.** Consider the scalar recurrence $A = \alpha$, $B = 1-\alpha$, $C = 1$ with $0 < \alpha < 1$. Evaluating $G(z)$ along the unit circle $z = e^{i\omega}$:

$$
G(z) = \frac{1-\alpha}{1-\alpha z^{-1}}, \qquad
|G(e^{i\omega})| = \frac{1-\alpha}{\sqrt{1 - 2\alpha\cos\omega + \alpha^2}} ,
$$

utilizing the identity $|1-\alpha e^{-i\omega}|^2 = (1-\alpha\cos\omega)^2 + \alpha^2\sin^2\omega = 1 - 2\alpha\cos\omega + \alpha^2$. For a typical coefficient $\alpha = 0.9$:

| Frequency $\omega$ | 0 (DC constant) | $\pi/8$ (Low frequency) | $\pi/2$ (Mid frequency) | $\pi$ (Nyquist alternation $+1, -1, \dots$) |
|---|---|---|---|---|
| Gain $\lvert G\rvert$ | 1.000 | 0.261 | 0.074 | 0.053 |

At DC ($\omega = 0$, $\cos 0 = 1$), the denominator is $\sqrt{(1-\alpha)^2} = 1-\alpha$, giving unit gain $1.000$. At the highest alternating frequency ($\omega = \pi$, $\cos \pi = -1$), the denominator expands to $1+\alpha = 1.9$, reducing gain to $(1-\alpha)/(1+\alpha) = 0.1/1.9 \approx 0.053$—a 94.7% suppression of high-frequency fluctuations! This provides the exact frequency-domain explanation of why exponential moving averages smooth sensor noise while inherently introducing lag behind sharp transitions (§5.2.6).

When the state dimension increases and complex conjugate eigenvalue pairs are introduced (§5.3.3), the system can form resonant peaks around natural frequencies $\theta$, functioning as a **bandpass filter**. Combining multiple modes across varying decay rates and frequencies yields a multiscale **filter bank**.

## 5.5.2 What a finite state cannot remember

Intuitively, a fixed-size vector state cannot store infinite history. In continuous real arithmetic, a single real number could theoretically store infinite discrete tokens in its decimal expansion ($h_t = \sum_k x_k 10^{-k}$). In physical computing systems, however, such infinite encodings are physically impossible: resolving tokens stored thousands of steps ago would require measuring perturbations on the scale of $10^{-1000}$, which are instantly obliterated by thermal noise and floating-point quantization.

Rigorous capacity limits must therefore be established on **physically realizable finite state spaces**. The following theorem applies the information-theoretic pigeonhole principle to derive the minimum bit capacity required for lossless streaming recall.

**Theorem 5.2 (counting bound).** Consider a deterministic system with state update $s_t = F(s_{t-1}, x_t)$ and output $y_t = G(s_t)$, processing symbol sequences from an alphabet of size $V$. Suppose that after reading *any* arbitrary window $w = (x_1,\dots,x_W)$, the system can, over the subsequent $W$ steps, output $x_1, x_2, \dots, x_W$ in exact sequential order (regardless of what fixed filler symbols it receives during the recall phase). Then the number of distinct internal states reachable at step $W$ is at least $V^W$. If the state is physically stored in a register of $b$ binary bits, then:
$$
b \ge W\log_2V .
$$

*Proof.*Suppose by contradiction that two distinct input sequences $w = (x_1,\dots,x_W) \ne w' = (x'_1,\dots,x'_W)$ produce the identical internal state at step $W$, so $s_W = s'_W$.
By definition of a deterministic system, all future outputs for $t > W$ under identical filler inputs depend solely on $s_W$. Because $s_W = s'_W$, the system must emit identical output sequences during the recall phase for both histories.
However, $w$ and $w'$ differ at some coordinate $i \in \{1,\dots,W\}$ ($x_i \ne x'_i$). Lossless recall requires the system to emit $x_i$ and $x'_i$ respectively at recall step $W+i$, which are distinct. This contradicts the fact that the emitted outputs must be identical.
Therefore, the map from the $V^W$ input histories to the state space $w \mapsto s_W$ must be **injective**. The system must possess at least $V^W$ distinguishable physical states. Storing $V^W$ states in $b$ bits requires $2^b \ge V^W$, which implies $b \ge W \log_2 V$. $\blacksquare$

**Physical Implications.** For a standard language model vocabulary (e.g. GPT-2 with $V = 50\,257$) and context window $W = 1000$ tokens, Theorem 5.2 establishes a theoretical minimum requirement of $1000 \times \log_2(50257) \approx 15\,617\text{ bits}$ (approximately 1.95 KB). A recurrent hidden state consisting of 1,024 FP16 numbers contains $1024 \times 16 = 16\,384\text{ bits}$.
Pure bit counting does not rule out compressing 1,000 tokens into this state. However, in practice, floating-point mantissa bits cannot be packed with arbitrary density due to numerical noise, optimization geometry, and the inability of gradient descent to discover highly non-convex fractal encodings (§5.7).

:::caution[A counterexample to a careless capacity argument]
Theorem 5.2 counts *distinct states*. With infinitely precise real numbers, a single real state has infinitely many values and can store anything: the update $h_t=(h_{t-1}+x_t)/V$ with $x_t\in\{0,\dots,V-1\}$ writes the history into the base-$V$ digits of $h_t$. Statements such as “a $d$-dimensional state can remember only $O(d)$ items” are therefore claims about **finite precision**, noise, a regularity requirement, or the restricted dynamics of a model class—never about unconstrained real arithmetic. The digit encoding is also catastrophically non-robust: recovering an old digit requires resolving exponentially small differences. Always ask what precision and robustness a capacity claim assumes.
:::

## 5.5.3 Capacity of an associative memory

Linear attention (§5.3.12) replaces expanding KV caches by accumulating outer-product associations into a fixed matrix state $S \in \mathbb{R}^{d_v\times d_k}$, retrieved via $y = Sk$. This prompts two fundamental theoretical questions:
1. **Exact Capacity**: How many key-value associations can a fixed $d_v \times d_k$ matrix state store with zero distortion?
2. **Degradation Characteristics**: When the number of stored pairs exceeds algebraic limits, how does retrieval quality degrade?

**Theorem 5.3 (exact capacity).** Let $K = [k_1,\dots,k_n] \in \mathbb{R}^{d_k\times n}$ denote the key matrix and $V = [v_1,\dots,v_n] \in \mathbb{R}^{d_v\times n}$ denote the value matrix.

1. Under **outer-product (Hebbian) storage** $S = \sum_{j=1}^n v_jk_j^{\top} = VK^{\top}$, exact retrieval ($Sk_i = v_i$ for all $i$) holds for **all** value configurations $V$ if and only if the keys form an orthonormal set, i.e., $K^{\top}K = I_n$. Consequently, the maximum exact capacity is bounded by key dimension: $n \le d_k$.
2. Under **any general linear memory matrix** $S$ (such as the least-squares solution reached via delta-rule updates), exact retrieval $SK = V$ is possible for all $V$ if and only if the keys are linearly independent. Again, $n \le d_k$.

*Proof.*
For (1): Substituting $S = VK^{\top}$ into the retrieval query yields $SK = (VK^{\top})K = V(K^{\top}K)$. This equals $V$ for arbitrary $V$ if and only if the Gram matrix satisfies $K^{\top}K = I_n$, meaning $k_i^\top k_j = \delta_{ij}$. In $\mathbb{R}^{d_k}$, at most $d_k$ non-zero vectors can be pairwise orthogonal, establishing $n \le d_k$.
For (2): The linear system $SK = V$ possesses a valid solution $S$ for every $V \in \mathbb{R}^{d_v \times n}$ if and only if the linear operator $S \mapsto SK$ is surjective, which requires $K$ to have full column rank $n$. In $d_k$-dimensional Euclidean space, a linearly independent set has at most $d_k$ vectors, enforcing $n \le d_k$. $\blacksquare$

**Graceful Degradation and RMS Cross-Talk.** When the number of associations exceeds the key dimension ($n > d_k$), how does associative recall behave? Rather than suffering catastrophic failure, continuous matrix memory exhibits **graceful degradation**.

Let keys be drawn independently and uniformly from the unit sphere in $\mathbb{R}^{d_k}$, and values be independent with zero mean and variance $\mathbb{E}[\|v\|^2] = s^2$. Querying the Hebbian memory with a previously stored key $k_i$ yields:

$$
Sk_i = v_i + \underbrace{\sum_{j\ne i}(k_j^{\top}k_i)\,v_j}_{\text{cross-talk } e_i}.
$$

The first term is the target value $v_i$, while the second term $e_i$ represents the **cross-talk noise** from the remaining $n-1$ stored pairs.
By hyperspherical symmetry, each coordinate of a random unit vector satisfies $\mathbb{E}[k_{im}^2] = 1/d_k$. Because $k_j$ and $k_i$ ($j \ne i$) are independent, the expected squared inner product is:
$$
\mathbb{E}[(k_j^{\top}k_i)^2] = \sum_{m=1}^{d_k}\mathbb{E}[k_{jm}^2]\,\mathbb{E}[k_{im}^2] = d_k \cdot \left(\frac{1}{d_k}\right)^2 = \frac{1}{d_k}.
$$
Because the value vectors $v_j$ have zero mean and are uncorrelated with the keys, accumulating the variances across the $n-1$ interfering terms yields:

$$
\mathbb{E}\|e_i\|^2 = \frac{n-1}{d_k}\,s^2,
\qquad\frac{\sqrt{\mathbb E\|e_i\|^2}}{\sqrt{\mathbb E\|v_i\|^2}}
= \sqrt{\frac{n-1}{d_k}} .
$$

This concise relationship reveals that the root-mean-square (RMS) cross-talk noise relative to signal amplitude scales gracefully as $\sqrt{(n-1)/d_k}$. At $n = d_k + 1$, the noise magnitude matches the signal magnitude; as $n$ grows further, the memory degrades smoothly rather than abruptly crashing. Figure 5.6 illustrates this theoretical $\sqrt{n/d_k}$ scaling in empirical simulations.

![Relative retrieval error of a linear-attention memory with 16-dimensional random unit keys as the number of stored pairs grows to eight times the key dimension; measured points track the RMS-scale estimate square root of (n−1)/d_k and become order one near n = d_k.](/figures/u05-linear-attention-capacity.svg)

*Figure 5.6 — Interference grows like $\sqrt{n/d_k}$: a fixed matrix state degrades gracefully, not abruptly.*

<div class="dlis-widget not-content" data-widget="linear-attention-capacity">Interactive figure — requires JavaScript. The static figure and tables above contain the same information.</div>

For comparison, classical discrete Hopfield networks ([Hopfield 1982](/references/#hopfield-1982); [Amit et al. 1985](/references/#amit-1985)) exhibit a sharp critical capacity around $C_{\text{crit}} \approx 0.14 N$ patterns, beyond which spurious minima induce catastrophic memory collapse.

## 5.5.4 A conditional mixing-matrix view

To place recurrent neural networks, state-space models, and attention mechanisms on a common mathematical footing, we examine the **conditional mixing matrix** perspective.

Conditioned on an input sequence (freezing input-dependent dynamic weights), any causal sequence transformation acting on a scalar channel can be represented as an affine operation:

$$
y_t=\sum_{s\le t}M_{ts}x_s,\qquad\text{or}\qquad y=Mx,
$$

where causality strictly constrains $M \in \mathbb{R}^{T\times T}$ to be a **lower triangular matrix**. For selective SSMs and Softmax attention, $M$ is a nonlinear function of input $x$; once conditioned, the algebraic structure and rank properties of $M$ govern both compute efficiency and cross-time communication capacity ([Dao & Gu 2024](/references/#dao-2024)):

| Layer | Conditional coefficient $M_{ts}$ | Structural properties | Length-$T$ application cost |
|---|---|---|---|
| LTI recurrence / SSM (dense $\bar A$) | $C\bar A^{t-s}\bar B$ | Toeplitz lower triangular; cut blocks have rank $\le N$ | $O(TN^2)$ by recurrence |
| LTI SSM (diagonal $\bar A$) | same | Toeplitz lower triangular; cut blocks have rank $\le N$ | $O(TN)$ by recurrence, or $O(T\log T)$ via FFT |
| Selective SSM (e.g. Mamba, diagonal $\bar A_t$) | $C_t(\prod_{r=s+1}^{t}\bar A_r)\bar B_s$ | Data-dependent semiseparable; cut blocks have rank $\le N$ | $O(TN)$ via parallel associative scan |
| Linear attention (feature width $r$) | $\phi(q_t)^\top\phi(k_s)$ | Causal lower triangle; cut blocks have rank $\le r$ | $O(Trd_v)$ via matrix-state recurrence |
| Softmax attention | row-normalized $\exp(q_t^\top k_s/\sqrt{d_k})$ | Dense, data-dependent; cut blocks are **full rank** | $O(T^2d)$ dense matrix multiplication |

**The Cut-Rank Bound and the Memory Bottleneck.** Consider partitioning a sequence of length $T$ at an arbitrary time boundary $c$, separating past indices $s \le c$ from future indices $t > c$. Inspect the sub-block $M_{\text{past}\to\text{future}} = [M_{ts}]_{t > c, s \le c}$ responsible for transmitting historical information across the cut.

For any conditionally linear recurrence with state dimension $N$:

$$
h_t=A_th_{t-1}+B_tx_t,\qquad y_t=C_th_t,
$$

unrolling the recurrence across cutoff point $c$ decomposes the transmission coefficient as:

$$
M_{ts}
=\underbrace{C_tA_t\cdots A_{c+1}}_{\text{row }t\text{ of }P\in\mathbb R^{(T-c)\times N}}
\underbrace{A_c\cdots A_{s+1}B_s}_{\text{column }s\text{ of }Q\in\mathbb R^{N\times c}}.
$$

**Core Insight: The sub-block factors as $M_{\text{past}\to\text{future}} = P Q$. Because the intermediate state bottleneck has dimension $N$, Sylvester's rank inequality strictly bounds the rank of this cross-time transmission matrix:**
$$
\operatorname{rank}(M_{\text{past}\to\text{future}}) \le N.
$$

By contrast, the cross-cut sub-block in standard Softmax attention avoids any state compression bottleneck, achieving full rank $\min(c, T-c)$ generically across arbitrary sequence lengths.

This comparison reveals the foundational **architectural trade-off** of modern sequence modeling:
- **Recurrent and State-Space Models (SSM)**: By compressing sequence history into a fixed-size state, information transmission across time is constrained by a rank-$N$ bottleneck. In exchange, they achieve $O(1)$ constant-time autoregressive inference latency and $O(T)$ linear training complexity.
- **Softmax Attention (Transformer)**: Avoids temporal compression by maintaining an expanding KV cache. The cross-time mixing block achieves full rank, granting future queries unhindered selective recall of past details, but at the cost of $O(T)$ memory growth and $O(T^2)$ computational complexity.

Frontier architectures—including selective state-space mechanisms like Mamba and hybrid attention-SSM networks (such as Jamba)—navigate this fundamental frontier between information capacity, cross-cut rank, and physical compute constraints.
