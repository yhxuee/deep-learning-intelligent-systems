---
title: "3.4 Worked examples"
description: Nine fully computed examples — gradient descent on a line fit, adjoint accumulation, a two-layer network by hand, softmax cross-entropy, ill-conditioning and momentum, choosing a finite-difference step, Adam's first step, forward versus reverse cost, and initialization by variance.
sidebar: { label: "3.4 Worked examples", order: 4 }
---

Work each example on paper before reading the solution. Every number is reproduced by the code noted.

## Worked Example 3.1 — Gradient descent on a line fit

**Problem.** For the data of §3.2.1, $L(w) = \tfrac13(14w^2 - 56.2w + 56.42)$. Run gradient descent with $\eta = 0.1$ from $w_0 = 0$ for five steps. Predict the error-reduction factor per step before computing.

**Solution.** $L'(w) = \tfrac13(28w - 56.2)$ and $L''(w) = 28/3 \approx 9.333$. On a parabola the error is multiplied by $1-\eta L'' = 1 - 0.9333 = 0.0667$ each step (§3.2.3), so each step should remove about 93% of the remaining error.

| $k$ | $w_k$ | $L(w_k)$ | $L'(w_k)$ | $w_{k+1} = w_k - 0.1L'(w_k)$ |
|---|---|---|---|---|
| 0 | 0.000000 | 18.806667 | −18.733333 | 1.873333 |
| 1 | 1.873333 | 0.089985 | −1.248889 | 1.998222 |
| 2 | 1.998222 | 0.006800 | −0.083259 | 2.006548 |
| 3 | 2.006548 | 0.006430 | −0.005551 | 2.007103 |
| 4 | 2.007103 | 0.006429 | −0.000370 | 2.007140 |

The error $w^* - w_k$ goes $2.00714 \to 0.13381 \to 0.00892 \to 0.00059 \to 0.00004$: each ratio is $0.0667$, exactly as predicted. The minimum loss $0.006429$ is not zero because no line through the origin passes through all three points.

## Worked Example 3.2 — Accumulation at a shared input

**Problem.** For $f(a,b) = (a+b)\cdot a + \tanh a$ at $a = 0.5$, $b = -1$, compute $\partial f/\partial a$ and $\partial f/\partial b$ by reverse mode, showing each contribution to $\bar a$. Check against the analytic derivative.

**Solution.** Trace: $v_1 = a + b = -0.5$; $v_2 = v_1\cdot a = -0.25$; $v_3 = \tanh a = 0.462117$; $f = v_2 + v_3 = 0.212117$.

The input $a$ has **three children** ($v_1$, $v_2$, $v_3$). Backward, in reverse order:

| step | rule | contribution | running $\bar a$ |
|---|---|---|---|
| $\bar f = 1$ | seed | | 0 |
| $f = v_2 + v_3$ | $\bar v_2 \mathrel{+}= 1$, $\bar v_3 \mathrel{+}= 1$ | | 0 |
| $v_3 = \tanh a$ | $\bar a \mathrel{+}= \bar v_3(1-v_3^2)$ | $+0.786448$ | 0.786448 |
| $v_2 = v_1\cdot a$ | $\bar v_1 \mathrel{+}= \bar v_2\,a$; $\ \bar a \mathrel{+}= \bar v_2\,v_1$ | $\bar v_1 = 0.5$; $-0.5$ | 0.286448 |
| $v_1 = a + b$ | $\bar a \mathrel{+}= \bar v_1$; $\ \bar b \mathrel{+}= \bar v_1$ | $+0.5$; $\bar b = 0.5$ | **0.786448** |

**Check.** $f = a^2 + ab + \tanh a$, so $\partial f/\partial a = 2a + b + (1-\tanh^2a) = 1 - 1 + 0.786448 = 0.786448$ ✓ and $\partial f/\partial b = a = 0.5$ ✓. Had the implementation *assigned* instead of adding, $\bar a$ would be $0.5$ (the last contribution only) — wrong by 36%, and the lab's fan-out tests would fail.

## Worked Example 3.3 — A two-layer network by hand

**Problem.** Input $x = (1, 2)$; hidden layer $z_1 = W_1x + b_1$, $h_1 = \tanh z_1$ with
$W_1 = \begin{pmatrix}0.5&-0.25\\0.3&0.1\end{pmatrix}$, $b_1 = (0.1, -0.2)$; output $\hat y = w_2^{\top}h_1 + b_2$ with $w_2 = (1, -2)$, $b_2 = 0.05$; loss $L = \tfrac12(\hat y - 1)^2$. Compute every gradient with the algorithm of §3.3.6, and verify $\partial L/\partial (W_1)_{12}$ by central differences.

**Forward.**
$z_1 = (0.5 - 0.5 + 0.1,\ 0.3 + 0.2 - 0.2) = (0.1,\ 0.3)$;
$h_1 = (\tanh 0.1,\ \tanh 0.3) = (0.099668,\ 0.291313)$;
$\hat y = 0.099668 - 2(0.291313) + 0.05 = -0.432957$;
$L = \tfrac12(-1.432957)^2 = 1.026683$.

**Backward.**

| quantity | rule | value |
|---|---|---|
| $\bar{\hat y}$ | $\hat y - 1$ | −1.432957 |
| $\bar w_2$ | $\bar{\hat y}\,h_1$ | (−0.142820, −0.417439) |
| $\bar b_2$ | $\bar{\hat y}$ | −1.432957 |
| $\bar h_1$ | $w_2\,\bar{\hat y}$ | (−1.432957, 2.865914) |
| $1-h_1^2$ | | (0.990066, 0.915137) |
| $\bar z_1$ | $\bar h_1\odot(1-h_1^2)$ | (−1.418723, 2.622704) |
| $\bar W_1$ | $\bar z_1\,x^{\top}$ | $\begin{pmatrix}-1.418723&-2.837445\\2.622704&5.245409\end{pmatrix}$ |
| $\bar b_1$ | $\bar z_1$ | (−1.418723, 2.622704) |

**Check.** Perturbing $(W_1)_{12}$ by $\pm10^{-6}$ and taking the central difference gives $-2.8374453$ ✓.

**Reading the result.** The gradient for the second hidden unit's weights is larger and of opposite sign to the first's: the output weight $-2$ both amplifies and flips the error signal on its way back — the transposed weight matrix in $\bar h_{\ell-1} = W_\ell^\top\bar z_\ell$ at work.

## Worked Example 3.4 — Softmax cross-entropy

**Problem.** Scores $s = (2.0, 1.0, 0.1)$, target class 1 (the first). Compute $p$, the loss, and $\bar s$.

**Solution.** $e^{s} = (7.389, 2.718, 1.105)$, sum $11.212$, so $p = (0.659, 0.242, 0.099)$ (to three decimals: $0.659001, 0.242433, 0.098566$). Loss $= -\ln 0.659001 = 0.417030$. Gradient $\bar s = p - e_1 = (-0.341, 0.242, 0.099)$.

The gradient raises the correct score and lowers the others in proportion to how much probability they took. Its entries sum to zero (softmax is unchanged by adding a constant to all scores, so the loss cannot change in the all-ones direction).

## Worked Example 3.5 — Ill-conditioning and momentum

**Problem.** $L(\theta) = \tfrac12(\theta_1^2 + 10\theta_2^2)$, starting at $(1, 1)$. (a) Find the best fixed step for gradient descent and its convergence factor. (b) How many steps to reduce the error by $10^3$? (c) Repeat for heavy-ball momentum with the optimal parameters of §3.5.7.

**Solution.** (a) The curvatures are $\lambda_1 = 1$, $\lambda_2 = 10$; each coordinate's error is multiplied by $1-\eta\lambda_i$. The slowest factor is minimized when $|1-\eta\cdot1| = |1-\eta\cdot10|$, i.e. $\eta = 2/(1+10) = 2/11 \approx 0.182$, giving factor $(10-1)/(10+1) = 0.818$ in both coordinates (§3.5.4).

(b) $0.818^k = 10^{-3} \Rightarrow k = \ln10^{-3}/\ln0.818 \approx 34.4$. Simulation: 35 steps.

(c) With $\kappa = 10$: $\eta = 4/(\sqrt{10}+1)^2 = 0.231$, $\beta = \bigl((\sqrt{10}-1)/(\sqrt{10}+1)\bigr)^2 = 0.270$, asymptotic factor $(\sqrt{10}-1)/(\sqrt{10}+1) = 0.519$, predicting about 10.5 steps for a factor $10^3$. Simulation: 15 steps (the asymptotic rate takes a few steps to take hold), measured asymptotic factor $0.52$. Figure 3.2 shows both paths.

## Worked Example 3.6 — Choosing a finite-difference step

**Problem.** Estimate $\frac{d}{dx}\sin x$ at $x = 1$ by central differences in double precision. Which step $h$ minimizes the error, and how small can the error get?

**Solution.** Two errors compete. *Truncation:* Taylor expansion gives $\frac{f(x+h)-f(x-h)}{2h} = f'(x) + \frac{h^2}{6}f'''(x) + O(h^4)$, an error of about $\frac{h^2}{6}|f'''|$. *Rounding:* each computed value of $f$ carries a relative error up to the unit roundoff $u = 2^{-53}\approx1.11\times10^{-16}$; the difference of two nearly equal numbers keeps this absolute error while dividing by $2h$ amplifies it, giving about $u|f|/h$. Minimize $E(h) = \frac{h^2}{6}|f'''| + \frac{u|f|}{h}$ by setting $E'(h) = 0$:

$$
h^* = \Bigl(\frac{3u|f|}{|f'''|}\Bigr)^{1/3} = \Bigl(\frac{3\cdot1.11\times10^{-16}\cdot0.8415}{0.5403}\Bigr)^{1/3}\approx 8.0\times10^{-6},\qquad E(h^*)\approx1.7\times10^{-11}.
$$

Measured errors: $9.0\times10^{-10}$ at $h = 10^{-4}$, $1.1\times10^{-11}$ at $h = 10^{-5}$, $2.8\times10^{-11}$ at $h=10^{-6}$, $2.6\times10^{-9}$ at $h = 10^{-8}$. Smaller steps are *worse*. Forward differences ($O(h)$ truncation) have their best step near $\sqrt u\approx10^{-8}$ and best error only about $3\times10^{-9}$. Figure 3.4 (§3.7.5) shows the full V-shaped curves.

## Worked Example 3.7 — Adam's first step

**Problem.** A model has three parameters with first gradients $g_1 = (0.001,\ -50,\ 3)$. With $\beta_1 = 0.9$, $\beta_2 = 0.999$, $\epsilon = 10^{-8}$, compute Adam's first update direction $\hat m_1/(\sqrt{\hat v_1}+\epsilon)$.

**Solution.** $m_1 = 0.1g_1$, $v_1 = 0.001g_1^2$; bias correction gives $\hat m_1 = g_1$ and $\hat v_1 = g_1^2$. So the direction is $g_1/(|g_1| + \epsilon) = (0.99999,\ -1,\ 1)$. Every coordinate moves by (almost exactly) $\eta$, regardless of a 50 000-fold difference in gradient magnitude. Without bias correction the first step would be $\frac{0.1g}{\sqrt{0.001}|g|} \approx 3.16$ times larger in every coordinate — this is why implementations that omit it behave strangely early in training.

## Worked Example 3.8 — Forward or reverse?

**Problem.** Count passes needed for the full derivative of (a) a loss $L:\mathbb{R}^{1000}\to\mathbb{R}$; (b) a trajectory $\gamma:\mathbb{R}\to\mathbb{R}^{1000}$; (c) a map $F:\mathbb{R}^{1000}\to\mathbb{R}^{1000}$.

**Solution.** Forward mode gives one Jacobian *column* per pass; reverse mode one *row*. (a) Jacobian $1\times1000$: forward 1000 passes, reverse **1**. (b) $1000\times1$: forward **1**, reverse 1000. (c) $1000\times1000$: 1000 passes either way — unless only a product $J^{\top}u$ (one reverse pass) or $Jv$ (one forward pass) is needed, which is the common case in practice.

## Worked Example 3.9 — Initialization by variance

**Problem.** A layer computes $z = \sum_{j=1}^{n}w_jx_j$ with $n = 256$ independent inputs of mean 0 and variance 1, and independent weights of mean 0 and variance $\sigma_w^2$. Choose $\sigma_w$ so that $\mathrm{Var}(z) = 1$. How should it change if the inputs are outputs of a ReLU applied to unit-variance pre-activations of the previous layer?

**Solution.** For independent zero-mean factors, $\mathrm{Var}(w_jx_j) = \mathrm{Var}(w_j)\mathrm{Var}(x_j)$, and variances of independent terms add: $\mathrm{Var}(z) = n\sigma_w^2$. So $\sigma_w^2 = 1/n$, $\sigma_w = 0.0625$.

If $x_j = \mathrm{ReLU}(u_j)$ with $u_j$ symmetric around zero and $\mathrm{Var}(u_j) = 1$, then $x_j$ has nonzero mean, so use the second moment instead: $\mathbb E[x_j^2] = \tfrac12\mathbb E[u_j^2] = \tfrac12$ (half the mass is zeroed), and $\mathrm{Var}(z) = n\sigma_w^2\,\mathbb E[x_j^2] = n\sigma_w^2/2$. Keeping $\mathrm{Var}(z) = 1$ requires $\sigma_w^2 = 2/n$, $\sigma_w = 0.0884$ — the rule of [He et al. (2015)](/references/#he-2015). §3.8.1 derives the backward-pass version and Figure 3.3 shows what happens when the rule is violated.
