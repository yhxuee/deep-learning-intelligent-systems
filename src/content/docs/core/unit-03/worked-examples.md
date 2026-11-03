---
title: "3.4 Worked examples"
description: Nine fully computed examples — gradient descent on a line fit, adjoint accumulation, a two-layer network by hand, softmax cross-entropy, ill-conditioning and momentum, choosing a finite-difference step, Adam's first step, forward versus reverse cost, and initialization by variance.
sidebar: { label: "3.4 Worked examples", order: 4 }
---

These examples are not a second list of formulas to memorize. They are rehearsals for the decisions that make a derivation reliable: What is the quantity being differentiated? What is its shape? Which direction does the update move? What assumptions make the conclusion true?

For each example, first cover the solution and write down the next line you expect. Then compare signs, shapes, and orders of magnitude—not just the final decimal. Every quoted number is reproduced by `labs/test_worked_examples.py` or by the code cited in the text.

## Worked Example 3.1 — Gradient descent on a line fit

**Problem.** For the data of §3.2.1, $L(w) = \tfrac13(14w^2 - 56.2w + 56.42)$. Run gradient descent with $\eta = 0.1$ from $w_0 = 0$ for five steps. Predict the error-reduction factor per step before computing.

**Before calculating.** This is a one-dimensional quadratic, so its curvature is constant:

$$
L'(w) = \tfrac13(28w - 56.2),\qquad L''(w) = 28/3 \approx 9.333.
$$

Let the **parameter error** be $e_k=w_k-w^*$. For a quadratic, gradient descent gives

$$
e_{k+1}=(1-\eta L'')e_k=(1-0.1\times9.333)e_k\approx0.0667e_k.
$$

So each step should remove about 93% of the remaining *parameter* error. Because the loss gap of a quadratic is proportional to $e_k^2$, $L(w_k)-L(w^*)$ should shrink by about $0.0667^2\approx0.00444$ per step. Keeping those two meanings of “error” separate prevents a common misreading of convergence plots.

**Now calculate.**

| $k$ | $w_k$ | $L(w_k)$ | $L'(w_k)$ | $w_{k+1} = w_k - 0.1L'(w_k)$ |
|---|---|---|---|---|
| 0 | 0.000000 | 18.806667 | −18.733333 | 1.873333 |
| 1 | 1.873333 | 0.089985 | −1.248889 | 1.998222 |
| 2 | 1.998222 | 0.006800 | −0.083259 | 2.006548 |
| 3 | 2.006548 | 0.006430 | −0.005551 | 2.007103 |
| 4 | 2.007103 | 0.006429 | −0.000370 | 2.007140 |

The parameter error $w^* - w_k$ goes $2.00714 \to 0.13381 \to 0.00892 \to 0.00059 \to 0.00004$: each ratio is $0.0667$, exactly as predicted. The loss does not converge to zero; it converges to $L(w^*)=0.006429$, because no line through the origin passes through all three data points. Optimization can find the best member of the chosen model class, but it cannot remove error caused by that class being too restrictive.

## Worked Example 3.2 — Accumulation at a shared input

**Problem.** For $f(a,b) = (a+b)\cdot a + \tanh a$ at $a = 0.5$, $b = -1$, compute $\partial f/\partial a$ and $\partial f/\partial b$ by reverse mode, showing each contribution to $\bar a$. Check against the analytic derivative.

**Forward trace.** Break the expression into primitive operations:

$$
v_1=a+b=-0.5,\qquad v_2=v_1a=-0.25,\qquad
v_3=\tanh a=0.462117,\qquad f=v_2+v_3=0.212117.
$$

The input $a$ is used on three outgoing edges: once in $v_1$, once as the second input of $v_2$, and once in $v_3$. Reverse mode must therefore **add three contributions** into the same adjoint $\bar a$. Walk backward in the reverse of the forward order:

| step | rule | contribution | running $\bar a$ |
|---|---|---|---|
| $\bar f = 1$ | seed | | 0 |
| $f = v_2 + v_3$ | $\bar v_2 \mathrel{+}= 1$, $\bar v_3 \mathrel{+}= 1$ | | 0 |
| $v_3 = \tanh a$ | $\bar a \mathrel{+}= \bar v_3(1-v_3^2)$ | $+0.786448$ | 0.786448 |
| $v_2 = v_1\cdot a$ | $\bar v_1 \mathrel{+}= \bar v_2\,a$; $\ \bar a \mathrel{+}= \bar v_2\,v_1$ | $\bar v_1 = 0.5$; $-0.5$ | 0.286448 |
| $v_1 = a + b$ | $\bar a \mathrel{+}= \bar v_1$; $\ \bar b \mathrel{+}= \bar v_1$ | $+0.5$; $\bar b = 0.5$ | **0.786448** |

**Independent check.** Expanding first gives $f=a^2+ab+\tanh a$, hence

$$
\frac{\partial f}{\partial a}=2a+b+(1-\tanh^2a)=1-1+0.786448=0.786448,\qquad
\frac{\partial f}{\partial b}=a=0.5.
$$

Both agree with reverse mode. If an implementation *assigned* to $\bar a$ instead of accumulating with `+=`, only the last contribution would survive and it would report $0.5$—wrong by 36%. This tiny graph exposes the same fan-out bug that can silently corrupt a large network.

## Worked Example 3.3 — A two-layer network by hand

**Problem.** Input $x = (1, 2)$; hidden layer $z_1 = W_1x + b_1$, $h_1 = \tanh z_1$ with
$W_1 = \begin{pmatrix}0.5&-0.25\\0.3&0.1\end{pmatrix}$, $b_1 = (0.1, -0.2)$; output $\hat y = w_2^{\top}h_1 + b_2$ with $w_2 = (1, -2)$, $b_2 = 0.05$; loss $L = \tfrac12(\hat y - 1)^2$. Compute every parameter gradient with the algorithm of §3.3.6, and verify $\partial L/\partial (W_1)_{12}$ by central differences.

Here “two-layer” counts the two trainable affine maps: $x\mapsto z_1$ and $h_1\mapsto\hat y$. The task is to compute all **parameter** gradients; the input $x$ is data, not a trainable parameter.

**Forward.**
$z_1 = (0.5 - 0.5 + 0.1,\ 0.3 + 0.2 - 0.2) = (0.1,\ 0.3)$;
$h_1 = (\tanh 0.1,\ \tanh 0.3) = (0.099668,\ 0.291313)$;
$\hat y = 0.099668 - 2(0.291313) + 0.05 = -0.432957$;
$L = \tfrac12(-1.432957)^2 = 1.026683$.

**Backward.** Start from the scalar loss and move backward through the graph, following the table from top to bottom. The shape column is a quick guard against accidentally using an outer product where a dot product is required.

| quantity | shape | rule | value |
|---|---:|---|---|
| $\bar{\hat y}$ | scalar | $\hat y - 1$ | −1.432957 |
| $\bar w_2$ | 2-vector | $\bar{\hat y}\,h_1$ | (−0.142820, −0.417439) |
| $\bar b_2$ | scalar | $\bar{\hat y}$ | −1.432957 |
| $\bar h_1$ | 2-vector | $w_2\,\bar{\hat y}$ | (−1.432957, 2.865914) |
| $1-h_1^2$ | 2-vector | derivative of $\tanh z_1$ | (0.990066, 0.915137) |
| $\bar z_1$ | 2-vector | $\bar h_1\odot(1-h_1^2)$ | (−1.418723, 2.622704) |
| $\bar W_1$ | $2\times2$ | $\bar z_1\,x^{\top}$ | $\begin{pmatrix}-1.418723&-2.837445\\2.622704&5.245409\end{pmatrix}$ |
| $\bar b_1$ | 2-vector | $\bar z_1$ | (−1.418723, 2.622704) |

**Numerical check.** Perturb only $(W_1)_{12}$ while holding everything else fixed:

$$
\frac{L((W_1)_{12}+10^{-6})-L((W_1)_{12}-10^{-6})}{2\times10^{-6}}
=-2.8374453,
$$

which agrees with backpropagation. A finite-difference check tests a selected derivative; it does not replace the efficient computation of the whole gradient.

**Reading the result.** The gradient for the second hidden unit's weights is larger and has the opposite sign to the first's. The output weight $-2$ both amplifies and reverses that unit's error signal on the way back. This is the concrete meaning of sending an adjoint through a transposed weight matrix. If an input gradient were needed, the same chain would continue one more step as $\bar x=W_1^\top\bar z_1$.

## Worked Example 3.4 — Softmax cross-entropy

**Problem.** Scores $s = (2.0, 1.0, 0.1)$, with the first of the three classes as the target. We number the classes 1, 2, 3 here, so its one-hot target is $e_1=(1,0,0)$. Compute $p$, the loss, and $\bar s$.

**Solution.** Mathematically, softmax is unchanged if the same constant is subtracted from every score. Computationally, subtracting the maximum prevents exponentials from overflowing. Here

$$
s-\max(s)=(0,-1,-1.9),\qquad
e^{s-\max(s)}=(1,0.367879,0.149569).
$$

Their sum is $1.517448$, so

$$
p=(0.659001,0.242433,0.098566),\qquad
L=-\ln p_1=0.417030,\qquad
\bar s=p-e_1=(-0.340999,0.242433,0.098566).
$$

The vector above is the direction of **increasing** loss. Gradient descent subtracts it, so the update raises the correct score and lowers the other scores in proportion to the probability they received. Its entries sum to zero. That is not an accident: adding the same constant to all scores leaves softmax unchanged, so the loss has zero directional derivative along the all-ones vector.

## Worked Example 3.5 — Ill-conditioning and momentum

**Problem.** $L(\theta) = \tfrac12(\theta_1^2 + 10\theta_2^2)$, starting at $(1, 1)$. (a) Find the fixed step that minimizes the worst-case per-step contraction over the two eigendirections. (b) How many steps reduce the parameter-error norm by a factor of $10^3$? (c) Repeat for heavy-ball momentum with the optimal quadratic parameters of §3.5.7.

**Solution.** (a) The minimum is at $\theta^*=(0,0)$, and the two curvatures are $\lambda_1=1$ and $\lambda_2=10$. After one gradient step, eigendirection $i$ is multiplied by $1-\eta\lambda_i$. We therefore minimize

$$
\rho(\eta)=\max\{|1-\eta|,|1-10\eta|\}.
$$

The minimax choice balances the two magnitudes: $|1-\eta|=|1-10\eta|$. Thus

$$
\eta=\frac{2}{1+10}=\frac2{11}\approx0.182,\qquad
\rho=\frac{10-1}{10+1}=\frac9{11}\approx0.818.
$$

The low-curvature coordinate keeps its sign while the high-curvature coordinate alternates sign, but both magnitudes contract by $0.818$ each step. “Best” here means best worst-case spectral contraction for a constant step—not best after every finite number of steps under every possible criterion.

(b) Because both coordinate magnitudes have the same factor, the Euclidean parameter-error norm also contracts by $\rho$:

$$
0.818^k=10^{-3}\quad\Longrightarrow\quad
k=\frac{\ln10^{-3}}{\ln0.818}\approx34.4.
$$

An integer run therefore needs 35 steps. The objective gap is quadratic in the parameter error and contracts by $\rho^2$ here, so one must always say which “error” is being counted.

(c) The condition number is $\kappa=10$. The optimal heavy-ball parameters for this quadratic interval are

$$
\eta=\frac4{(\sqrt{10}+1)^2}=0.231,\qquad
\beta=\left(\frac{\sqrt{10}-1}{\sqrt{10}+1}\right)^2=0.270,
$$

with asymptotic factor

$$
\rho_{\mathrm{HB}}=\frac{\sqrt{10}-1}{\sqrt{10}+1}=0.519.
$$

Using only that asymptotic factor suggests $\ln(10^{-3})/\ln(0.519)\approx10.5$ steps. The stated simulation takes 15 because its early iterates have not yet settled into the asymptotic regime; the later measured factor is $0.52$. Momentum improves this quadratic, but the formula is not a promise that an arbitrary nonconvex neural-network loss will attain the same rate. Figure 3.2 shows both paths.

## Worked Example 3.6 — Choosing a finite-difference step

**Problem.** Estimate $\frac{d}{dx}\sin x$ at $x = 1$ by central differences in double precision. Which step $h$ minimizes the error, and how small can the error get?

**Solution.** Making $h$ smaller helps only until floating-point cancellation takes over. A simple error model exposes the tradeoff.

- **Truncation:** Taylor expansion gives $\frac{f(x+h)-f(x-h)}{2h}=f'(x)+\frac{h^2}{6}f'''(x)+O(h^4)$, so the leading error has magnitude about $\frac{h^2}{6}|f'''|$.
- **Rounding and cancellation:** each computed function value has rounding error on the scale of unit roundoff $u=2^{-53}\approx1.11\times10^{-16}$. Subtracting nearby values and dividing by $2h$ amplifies that error, giving the rough scale $u|f|/h$.

Treating the omitted constants as one gives the model

$$
E(h)=\frac{h^2}{6}|f'''|+\frac{u|f|}{h}.
$$

Setting $E'(h)=0$ predicts

$$
h^* = \Bigl(\frac{3u|f|}{|f'''|}\Bigr)^{1/3} = \Bigl(\frac{3\cdot1.11\times10^{-16}\cdot0.8415}{0.5403}\Bigr)^{1/3}\approx 8.0\times10^{-6},\qquad E(h^*)\approx1.7\times10^{-11}.
$$

On the reference run, the measured errors are $9.0\times10^{-10}$ at $h=10^{-4}$, $1.1\times10^{-11}$ at $h=10^{-5}$, $2.8\times10^{-11}$ at $h=10^{-6}$, and $2.6\times10^{-9}$ at $h=10^{-8}$. Once cancellation dominates, smaller steps are *worse*.

The formula predicts a scale, not a universal exact optimum: hardware, the math library, and the local values of the function change the rounding constants. On the same reference configuration, forward differences—with $O(h)$ rather than $O(h^2)$ truncation—work best near $\sqrt u\approx10^{-8}$ and reach only about $3\times10^{-9}$ error. Figure 3.4 (§3.7.5) shows the full V-shaped curves.

## Worked Example 3.7 — Adam's first step

**Problem.** A model has three parameters with first gradients $g_1 = (0.001,\ -50,\ 3)$. With $\beta_1 = 0.9$, $\beta_2 = 0.999$, $\epsilon = 10^{-8}$, compute Adam's first update direction $\hat m_1/(\sqrt{\hat v_1}+\epsilon)$.

**Solution.** The moment accumulators start at zero, so after the first gradient

$$
m_1=(1-\beta_1)g_1=0.1g_1,\qquad
v_1=(1-\beta_2)g_1^2=0.001g_1^2.
$$

Bias correction divides by the missing accumulated mass:

$$
\hat m_1=\frac{m_1}{1-0.9}=g_1,\qquad
\hat v_1=\frac{v_1}{1-0.999}=g_1^2.
$$

Therefore

$$
\frac{\hat m_1}{\sqrt{\hat v_1}+\epsilon}
=\frac{g_1}{|g_1|+\epsilon}
=(0.99999,-1,1).
$$

The parameter update is the **negative** of this direction times $\eta$. Thus, when $|g_i|\gg\epsilon$, each coordinate's first-step magnitude is almost $\eta$, even though these gradient magnitudes span a factor of 50,000. This sign-like first step is a property of Adam's normalization, not evidence that gradient magnitude never matters: later moments retain history, and $\epsilon$ matters when gradients are tiny.

If bias correction were omitted, the normalized first direction would be approximately

$$
\frac{0.1g}{\sqrt{0.001}|g|}=3.16\,\mathrm{sign}(g),
$$

for these coordinates, where $|g_i|\gg\epsilon$. It is therefore about 3.16 times the corrected sign-like direction, not exactly so for arbitrary gradients once $\epsilon$ is included. Bias correction makes the intended early-step scale explicit.

## Worked Example 3.8 — Forward or reverse?

**Problem.** Count passes needed for the full derivative of (a) a loss $L:\mathbb{R}^{1000}\to\mathbb{R}$; (b) a trajectory $\gamma:\mathbb{R}\to\mathbb{R}^{1000}$; (c) a map $F:\mathbb{R}^{1000}\to\mathbb{R}^{1000}$.

**Solution.** For a function $f:\mathbb R^n\to\mathbb R^m$, its Jacobian has shape $m\times n$. A forward-mode sweep with one seed direction returns one Jacobian *column*; a reverse-mode sweep with one output seed returns one *row*.

| map | Jacobian shape | forward sweeps for full $J$ | reverse sweeps for full $J$ | preferable mode |
|---|---:|---:|---:|---|
| (a) $L:\mathbb R^{1000}\to\mathbb R$ | $1\times1000$ | 1000 | **1** | reverse |
| (b) $\gamma:\mathbb R\to\mathbb R^{1000}$ | $1000\times1$ | **1** | 1000 | forward |
| (c) $F:\mathbb R^{1000}\to\mathbb R^{1000}$ | $1000\times1000$ | 1000 | 1000 | neither wins by count alone |

This counts coordinate-seeded sweeps and treats their costs as comparable, not necessarily identical. More importantly, many algorithms never form the full Jacobian. They ask for $Jv$, obtainable in one forward sweep, or $J^\top u$, obtainable in one reverse sweep. That is why a seemingly impossible $1000\times1000$ matrix may never need to be stored or computed explicitly.

## Worked Example 3.9 — Initialization by variance

**Problem.** A layer computes $z = \sum_{j=1}^{n}w_jx_j$ with $n = 256$ independent inputs of mean 0 and variance 1, and independent weights of mean 0 and variance $\sigma_w^2$. Choose $\sigma_w$ so that $\mathrm{Var}(z) = 1$. How should it change if the inputs are outputs of a ReLU applied to unit-variance pre-activations of the previous layer?

**Solution.** Because the weights have zero mean and are independent of the inputs, $\mathbb E[w_jx_j]=0$. Under the stated independence assumptions, cross-covariances vanish and

$$
\mathrm{Var}(z)
=\sum_{j=1}^n\mathrm{Var}(w_jx_j)
=\sum_{j=1}^n\mathbb E[w_j^2]\mathbb E[x_j^2]
=n\sigma_w^2.
$$

Setting this to 1 gives

$$
\sigma_w^2=\frac1n=\frac1{256},\qquad \sigma_w=\frac1{16}=0.0625.
$$

Now let $x_j=\mathrm{ReLU}(u_j)$. Although $x_j$ no longer has mean zero, $w_jx_j$ still does because the independent weight has mean zero. The variance calculation therefore needs the **second moment** of $x_j$, not $\mathrm{Var}(x_j)$. If the distribution of $u_j$ is symmetric about zero and $\mathbb E[u_j^2]=1$, then its positive and negative halves contribute equally to the second moment:

$$
\mathbb E[x_j^2]=\mathbb E[u_j^2\mathbf 1_{\{u_j>0\}}]=\tfrac12\mathbb E[u_j^2]=\tfrac12.
$$

Hence

$$
\mathrm{Var}(z)=\frac{n\sigma_w^2}{2}=1
\quad\Longrightarrow\quad
\sigma_w^2=\frac2n,\qquad
\sigma_w=\sqrt{\frac2{256}}=0.0884.
$$

This is the forward-pass reasoning behind [He et al. (2015)](/references/#he-2015) initialization. It is an initialization-scale calculation under independence and symmetry approximations—not a guarantee that every layer of a trained network will maintain unit variance. §3.8.1 derives the corresponding backward-pass argument, and Figure 3.3 shows what happens when the scale is badly chosen.
