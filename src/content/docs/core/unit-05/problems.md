---
title: "5.10 Problems and further study"
description: Twenty graded problems — computations, derivations, proofs, implementations and design questions — with hints, plus an annotated reading list.
sidebar: { label: "5.10 Problems", order: 10 }
---

Difficulty: ★ routine application · ★★ requires combining ideas · ★★★ proof or open-ended design. Problems marked **[AI-off]** are intended for unaided practice and assessment; the rest may be explored with the [Mentor](/mentor/) after a first attempt. Short answers to selected problems follow the list.

## Computation

**5.1 ★ [AI-off]** Apply the EMA with $\alpha = 0.7$ and $h_0 = 10$ to the inputs $(10, 12, 10, 30, 10)$. Report $h_1,\dots,h_5$ and the fraction of the spike that remains in $h_5$.

**5.2 ★** For $\alpha = 0.95$ compute the exact half-life and the approximation $\ln2/(1-\alpha)$. Show that the relative error of the approximation is below $1-\alpha$ for $\alpha\in[0.5,1)$. *Hint: compare $-\ln\alpha$ with $1-\alpha$ using $-\ln(1-\varepsilon) = \varepsilon + \varepsilon^2/2 + \dots$*

**5.3 ★** Show that the mean age of the information in an EMA, $\sum_{j\ge0}j(1-\alpha)\alpha^j$, equals $\alpha/(1-\alpha)$. *Hint: differentiate $\sum_j\alpha^j = 1/(1-\alpha)$ with respect to $\alpha$.*

**5.4 ★ [AI-off]** A sensor is sampled at 10 Hz and should have a memory time constant of 2 s. Give the exact $\alpha$, the forward-Euler $\alpha$, and the backward-Euler $\alpha$ (§5.6.3). Which is closest to exact?

**5.5 ★** Find the eigenvalues, the spectral radius, and the oscillation period of $A = \begin{pmatrix}0.8&-0.6\\0.6&0.8\end{pmatrix}$. Is the recurrence stable? What happens to the impulse response?

**5.6 ★★** Compute $K_j = CA^jB$ for $A = \begin{pmatrix}0.6&0.2\\0&0.9\end{pmatrix}$, $B = (0, 1)^\top$, $C = (1, 0)$ in closed form by diagonalization, and find the lag $j$ at which $K_j$ peaks.

## Derivation

**5.7 ★★ [AI-off]** Prove the matrix closed form $h_t = A^th_0 + \sum_{k=1}^tA^{t-k}Bx_k$ by induction, stating where the order of matrix multiplication matters.

**5.8 ★★** Derive the zero-order-hold discretization $\bar A = e^{A\Delta}$, $\bar B = A^{-1}(e^{A\Delta}-I)B$ for a diagonal $A$ with negative entries, component by component, and verify that each component reduces to the EMA with $\alpha_i = e^{a_i\Delta}$ when $B_i = -a_i$.

**5.9 ★★** Let $A = \lambda I + N$ with $N^3 = 0$ but $N^2\ne0$. Derive $A^t$ and find the step at which the coefficient of $N^2$ peaks for $\lambda = 0.9$. Compare with the $N^2 = 0$ case of §5.3.5.

**5.10 ★★ [AI-off]** For a vector RNN $h_t = \tanh(Wh_{t-1}+Ux_t+b)$ with loss $L = \sum_t L_t(h_t)$, derive the BPTT recursion for $\bar h_t$ and the expression for $\partial L/\partial W$. Then implement it in NumPy and check it with central differences (Unit 03's [gradient check](/atlas/nodes/method-gradient-check/)).

**5.11 ★★** For the LSTM of §5.3.8, derive $\partial c_t/\partial c_{t-1}$ including the paths through $h_{t-1} = o_{t-1}\odot\tanh(c_{t-1})$ into the gates at time $t$. Under what conditions is the direct term $\mathrm{diag}(f_t)$ dominant?

**5.12 ★★** Show that the composition operator $(a_1,b_1)\bullet(a_2,b_2) = (a_2a_1, a_2b_1+b_2)$ is associative but not commutative. Write the down-sweep of the parallel scan for $T = 8$ and count the number of $\bullet$ operations in each sweep.

**5.13 ★★** For the system of Worked Example 5.3, derive the transfer function $G(z) = Cz(zI-A)^{-1}B$, identify its poles, and evaluate $|G(e^{i\omega})|$ at $\omega = 0$ and $\omega = \pi$.

**5.14 ★★** In the random-walk-plus-noise model of §5.6.2, express the steady-state $\alpha$ as a function of the ratio $\rho = q/r$ only. Find $\alpha$ for $\rho = 10^{-4}, 10^{-2}, 1$, and explain the limits $\rho\to0$ and $\rho\to\infty$.

## Proof

**5.15 ★★ [AI-off]** Prove Theorem 5.1 (stability of diagonalizable recurrences), including the bound $\|A^t\|\le\kappa(V)\rho(A)^t$.

**5.16 ★★** Show that one delta-rule update $S \leftarrow S + (v - Sk)k^\top$ with $\|k\| = 1$ makes $Sk = v$ exactly, and that it leaves $Sk'$ unchanged for every $k'\perp k$. Conclude that the delta rule stores orthonormal keys with zero interference in a single pass.

**5.17 ★★★** Extend Theorem 5.2 to recall that is allowed to fail with probability $\varepsilon$ on uniformly random windows. *Hint: Fano's inequality gives a lower bound on the entropy of the state.*

**5.18 ★★★** Prove that for any cut $c$ the past-to-future block of the matrix $M$ of a selective SSM with state size $N$ has rank at most $N$ (§5.5.4). Then construct inputs for which a single softmax attention layer produces a past-to-future block of rank $\min(c, T-c)$.

## Design and implementation

**5.19 ★★★** Design a bank of $N = 8$ EMAs whose half-lives are geometrically spaced from 1 to $10^4$ steps. Give the $\alpha_i$. Propose a readout that estimates "the mean of the last $W$ inputs" for arbitrary $W$ in that range, and analyse its error. What does this suggest about initializing the time constants of diagonal SSMs?

**5.20 ★★ (Spine)** Complete Experiment 5.8: KV-cache decoding for the spine model with the equivalence test, plus the recall stress test with its control. Report one page: prediction, result, discrepancy, explanation.

## Selected answers

- **5.1** $h = (10, 10.6, 10.42, 16.294, 14.406)$. The spike enters $h_5$ with weight $(1-\alpha)\alpha = 0.21$; its excess of 20 °C over the baseline contributes $0.21\cdot 20 = 4.2$ to $h_5$ (it contributed 6 to $h_4$, so 70% of its effect survives one more step).
- **5.4** $\Delta t/\tau = 0.05$: exact $e^{-0.05} = 0.951229$; forward Euler $0.95$ (error 0.00123); backward Euler $1/1.05 = 0.952381$ (error 0.00115). Both are first-order accurate and bracket the exact value; backward Euler is slightly closer here, and unlike forward Euler it stays stable for any step.
- **5.5** $\lambda = 0.8\pm0.6i$, $|\lambda| = 1$: the matrix is a pure rotation by $\arctan(0.6/0.8)\approx0.6435$ rad, period $\approx 9.76$ steps. Not asymptotically stable: the impulse response oscillates forever without decaying (marginal stability, §5.3.4).
- **5.14** $\alpha = 1 - K$ with $K = P/(P+1)$ and $P = (\rho + \sqrt{\rho^2+4\rho})/2$ (in units of $r$). As $\rho\to0$, $\alpha\to1$ (average forever); as $\rho\to\infty$, $\alpha\to0$ (trust only the newest reading).

## Further study

**Foundations.**
[Oppenheim & Schafer 2010](/references/#oppenheim-2010), chapters on the z-transform and IIR filters — the classical theory behind §5.5.1.
[Trefethen & Embree 2005](/references/#trefethen-2005) — why eigenvalues are not enough; the definitive treatment of transient growth.
[Kalman 1960](/references/#kalman-1960) — the original state-space estimation paper; read for the idea of a recursively updated sufficient statistic.

**Recurrent networks.**
[Bengio et al. 1994](/references/#bengio-1994) and [Pascanu et al. 2013](/references/#pascanu-2013) — the vanishing/exploding gradient analysis and gradient clipping.
[Hochreiter & Schmidhuber 1997](/references/#hochreiter-1997); [Gers et al. 2000](/references/#gers-2000) — the LSTM and the forget gate, in their original framing.

**State-space models and linear attention.**
[Gu et al. 2020](/references/#gu-2020) (HiPPO) → [Gu et al. 2022a](/references/#gu-2022) (S4) → [Gu & Dao 2024](/references/#gu-2024) (Mamba) → [Dao & Gu 2024](/references/#dao-2024) (state-space duality): read in this order to see the ideas of §5.3.9–§5.3.10 and §5.5.4 develop.
[Katharopoulos et al. 2020](/references/#katharopoulos-2020); [Schlag et al. 2021](/references/#schlag-2021) — linear attention as recurrence and as a fast-weight memory.

**Limits.**
[Arora et al. 2024](/references/#arora-2024); [Jelassi et al. 2024](/references/#jelassi-2024) — the evidence behind §5.7.1, and a model of how to test a memory claim.
