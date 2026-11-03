---
title: "A · 线性代数"
description: 向量、内积与范数；线性映射、秩与投影；最小二乘法；特征值与谱定理；正定矩阵；奇异值分解；矩阵范数与条件数；矩阵的幂、谱半径与瞬态增长。
sidebar: { label: "A · 线性代数", order: 1 }
---

本书的每个模型都由矩阵乘积构成，而它的理论大多是谱理论：步长上限是特征值（单元 03），记忆的衰减是谱半径（单元 05），低秩结构是奇异值分解（单元 02 与 04）。本附录收录各单元所用的事实，证明简短且有启发性时附上证明。例题中的数字由 `labs/test_appendices.py` 检查。

| 小节 | 用于 |
|---|---|
| A.1 内积与范数 | 02（相似度、JL 引理）、03（最陡下降）、04（注意力分数） |
| A.2 秩与投影 | 02（瓶颈）、04（$QK^{\top}$ 的秩）、05（回忆的秩上限） |
| A.3 最小二乘法 | 01、02（线性读出）、06（拟合缩放定律） |
| A.4–A.5 特征值、正定性 | 03（海森矩阵、条件数）、05（稳定性）、10（协方差） |
| A.6 SVD | 02（Eckart–Young）、04（低秩注意力）、11（电路的秩） |
| A.7 范数与条件数 | 03（病态条件）、06（数值精度） |
| A.8 幂与谱半径 | 03（深层乘积）、05（递推、瞬态增长） |

## A.1 向量、内积与范数

向量是列向量，$x\in\mathbb{R}^n$。**内积（inner product）**为 $\langle x, y\rangle = x^{\top}y = \sum_ix_iy_i$，**欧几里得范数**为 $\lVert x\rVert = \sqrt{x^{\top}x}$。其他常见范数：$\lVert x\rVert_1 = \sum_i|x_i|$ 与 $\lVert x\rVert_\infty = \max_i|x_i|$。

**定理 A.1（柯西–施瓦茨，Cauchy–Schwarz）。** $|x^{\top}y|\le\lVert x\rVert\,\lVert y\rVert$，等号成立当且仅当 $x$ 与 $y$ 平行。

*证明。* 若 $y = 0$ 则无需证明。否则，对每个实数 $t$，$0\le\lVert x - ty\rVert^2 = \lVert x\rVert^2 - 2t\,x^{\top}y + t^2\lVert y\rVert^2$。一个永不为负的 $t$ 的二次式，判别式不为正：$4(x^{\top}y)^2 - 4\lVert x\rVert^2\lVert y\rVert^2\le0$。等号恰在某个 $t$ 使 $x - ty = 0$ 时成立。$\blacksquare$

书中用到的推断：**三角不等式** $\lVert x + y\rVert\le\lVert x\rVert + \lVert y\rVert$（两边平方，对交叉项用柯西–施瓦茨）；用来比较嵌入的**余弦相似度** $\cos\angle(x, y) = x^{\top}y/(\lVert x\rVert\lVert y\rVert)\in[-1, 1]$（单元 02）；以及在所有单位向量 $u$ 中，方向导数 $\nabla f^{\top}u$ 在 $u = \nabla f/\lVert\nabla f\rVert$ 时最大——梯度是最陡上升方向（单元 03，§3.2.5）。

**标准正交基。** 若 $q_i^{\top}q_j = \mathbb{1}[i = j]$，向量 $q_1,\dots,q_k$ 就是**标准正交的（orthonormal）**。把它们排成 $Q\in\mathbb{R}^{n\times k}$ 的各列，就是 $Q^{\top}Q = I_k$。当 $k = n$ 时，$Q$ 是**正交矩阵**：$Q^{-1} = Q^{\top}$，且对每个 $x$ 都有 $\lVert Qx\rVert = \lVert x\rVert$——正交映射是旋转与反射，保持长度与角度。

## A.2 线性映射、秩与投影

矩阵 $A\in\mathbb{R}^{m\times n}$ 是线性映射 $\mathbb{R}^n\to\mathbb{R}^m$。它的**列空间**（值域）$\mathcal{R}(A) = \{Ax\}$ 与**零空间** $\mathcal{N}(A) = \{x: Ax = 0\}$ 都是子空间，**秩（rank）** $r = \dim\mathcal R(A)$ 等于线性无关之列的数目，也等于线性无关之行的数目。

**定理 A.2（秩–零化度定理）。** $\dim\mathcal R(A) + \dim\mathcal N(A) = n$。

*证明。* 取 $\mathcal N(A)$ 的一组基 $v_1,\dots,v_k$，并扩充成 $\mathbb{R}^n$ 的一组基 $v_1,\dots,v_n$。向量 $Av_{k+1},\dots,Av_n$ 张成 $\mathcal R(A)$（因为 $i\le k$ 时 $Av_i = 0$），而且它们线性无关：若 $\sum_{i>k}c_iAv_i = 0$，则 $\sum_{i>k}c_iv_i\in\mathcal N(A)$，所以它是 $v_1,\dots,v_k$ 的线性组合，而整组基的独立性迫使每个 $c_i = 0$。因此 $\dim\mathcal R(A) = n - k$。$\blacksquare$

**乘积的秩。** $\mathrm{rank}(AB)\le\min(\mathrm{rank}\,A, \mathrm{rank}\,B)$：$AB$ 的各列落在 $\mathcal R(A)$ 中，且 $\mathcal N(B)\subseteq\mathcal N(AB)$。这就是为什么无论序列多长，注意力头的 $QK^{\top} = XW_QW_K^{\top}X^{\top}$ 的秩至多为 $d_k$（单元 04），也是为什么具有 $N$ 维状态的递推，从过去传到未来的独立数字至多 $N$ 个（单元 05，§5.5.4）。

**正交投影。** 令 $U\in\mathbb{R}^{n\times k}$ 的各行为子空间 $\mathcal S$ 的一组标准正交基。矩阵 $P = UU^{\top}$ 是到 $\mathcal S$ 的**正交投影**：$P^2 = P$、$P^{\top} = P$，而且对每个 $y$，残差 $y - Py$ 与 $\mathcal S$ 正交。

**定理 A.3（投影就是最近点）。** 对每个 $s\in\mathcal S$，$\lVert y - s\rVert^2 = \lVert y - Py\rVert^2 + \lVert Py - s\rVert^2$；因此 $Py$ 是 $\mathcal S$ 中离 $y$ 最近的唯一一点。

*证明。* $y - s = (y - Py) + (Py - s)$；第一项与 $\mathcal S$ 正交，第二项在 $\mathcal S$ 中，所以交叉项为零（勾股定理）。$\blacksquare$

**例题 A.1。** 把 $y = (1, 2, 2)$ 投影到 $a = (1, 1, 0)$ 张成的直线上。以 $u = a/\lVert a\rVert$，$Py = a\,(a^{\top}y)/(a^{\top}a) = (1, 1, 0)\cdot 3/2 = (1.5, 1.5, 0)$。残差 $(-0.5, 0.5, 2)$ 与 $a$ 正交，长度为 $\sqrt{4.5} = 2.121$：直线上没有任何一点更近。

## A.3 最小二乘法

给定列线性无关的 $X\in\mathbb{R}^{n\times p}$ 与 $y\in\mathbb{R}^n$，**最小二乘（least squares）**问题是 $\min_\beta\lVert y - X\beta\rVert^2$。由定理 A.3，最小化者使 $X\beta$ 成为 $y$ 在 $\mathcal R(X)$ 上的投影，所以残差与 $X$ 的每一列正交：

$$
X^{\top}(y - X\hat\beta) = 0 \quad\Longleftrightarrow\quad X^{\top}X\hat\beta = X^{\top}y \qquad\text{（正规方程）}
$$

当各列线性无关时，$X^{\top}X$ 可逆（A.5），$\hat\beta = (X^{\top}X)^{-1}X^{\top}y$。数值计算上，应通过 QR 分解或 SVD 求解，而不是组出 $X^{\top}X$——它的条件数是 $X$ 的平方（A.7）。

**例题 A.2。** 以直线 $y\approx\beta_0 + \beta_1x$ 拟合 $(0, 1), (1, 2), (2, 2)$。正规方程 $\begin{pmatrix}3 & 3\\3 & 5\end{pmatrix}\beta = \begin{pmatrix}5\\6\end{pmatrix}$ 给出 $\hat\beta = (7/6, 1/2) = (1.1667, 0.5)$。拟合值为 $(1.1667, 1.6667, 2.1667)$，残差 $(-0.1667, 0.3333, -0.1667)$ 总和为零——残差与常数列正交。

单元 06 的缩放定律拟合是非线性最小二乘；对纯幂律取对数 $\ln L = \ln A - \alpha\ln N$ 之后，就正好变成这个问题。

## A.4 特征值与谱定理

若非零的 $v$ 满足 $Av = \lambda v$，它就是**特征向量**，$\lambda$ 是**特征值**。特征值是特征多项式 $\det(\lambda I - A)$ 的根，所以 $n\times n$ 矩阵计重数有 $n$ 个特征值，可能是复数。两个恒等式反复用到：$\mathrm{tr}\,A = \sum_i\lambda_i$ 与 $\det A = \prod_i\lambda_i$。

若 $A$ 有 $n$ 个独立的特征向量，排成 $S$ 的各列，则 $A = S\Lambda S^{-1}$，其中 $\Lambda = \mathrm{diag}(\lambda_i)$，任何幂都很容易计算：$A^t = S\Lambda^tS^{-1}$。并非每个矩阵都可对角化（$\begin{pmatrix}0&1\\0&0\end{pmatrix}$ 就不行），而即使可对角化，$S$ 也可能条件很差——这是瞬态增长的来源（A.8）。

**定理 A.4（对称矩阵的谱定理）。** 若 $A = A^{\top}\in\mathbb{R}^{n\times n}$，则它的特征值全为实数，并存在正交矩阵 $Q$ 使 $A = Q\Lambda Q^{\top}$。

*证明概要。* 若 $Av = \lambda v$ 且 $v$ 为复向量，则 $\bar v^{\top}Av = \lambda\lVert v\rVert^2$；取共轭转置并利用对称性，同一个量等于 $\bar\lambda\lVert v\rVert^2$；所以 $\lambda$ 是实数。对应相异特征值的特征向量互相正交：$\lambda_1v_2^{\top}v_1 = v_2^{\top}Av_1 = (Av_2)^{\top}v_1 = \lambda_2v_2^{\top}v_1$。完整陈述：取一个单位特征向量 $q_1$；$A$ 把 $q_1$ 的正交补空间映到自身（若 $x\perp q_1$，则 $q_1^{\top}Ax = (Aq_1)^{\top}x = \lambda_1q_1^{\top}x = 0$），再对维度归纳即可。$\blacksquare$

**瑞利商（Rayleigh quotient）。** 对特征值为 $\lambda_{\min}\le\dots\le\lambda_{\max}$ 的对称矩阵 $A$，写 $x = Qc$，
$$
\frac{x^{\top}Ax}{x^{\top}x} = \frac{\sum_i\lambda_ic_i^2}{\sum_ic_i^2}\in[\lambda_{\min}, \lambda_{\max}],
$$
极值在对应的特征向量处达到。二次损失沿某个方向的曲率，就是其海森矩阵的瑞利商；最大曲率 $\lambda_{\max}$ 决定了梯度下降法的步长上限 $\eta < 2/\lambda_{\max}$（单元 03，§3.2.3 与 §3.5.4；附录 F）。

**例题 A.3。** $A = \begin{pmatrix}2 & 1\\1 & 2\end{pmatrix}$ 的特征值为 $1$ 与 $3$，特征向量为 $(1, -1)/\sqrt2$ 与 $(1, 1)/\sqrt2$；迹 $4 = 1 + 3$，行列式 $3 = 1\cdot3$。二次形式 $x^{\top}Ax$ 沿 $(1, 1)$ 的弯曲速度是沿 $(1, -1)$ 的三倍：条件数为 3。

## A.5 正定矩阵

若对所有 $x$ 都有 $x^{\top}Ax\ge0$，对称矩阵 $A$ 就是**半正定的**（$A\succeq0$）；若对 $x\ne0$ 严格成立，就是**正定的**（$A\succ0$）。由谱定理，这分别等于所有特征值 $\ge0$、$>0$。

它们出现在：
- **Gram 矩阵与协方差矩阵。** $X^{\top}X\succeq0$，因为 $x^{\top}X^{\top}Xx = \lVert Xx\rVert^2$；它恰在 $X$ 的各列线性无关时为正定。每个协方差矩阵 $\Sigma = \mathbb{E}[(z - \mu)(z - \mu)^{\top}]$ 都是半正定的，因为 $x^{\top}\Sigma x = \mathrm{Var}(x^{\top}z)\ge0$。
- **海森矩阵。** 二次可微函数为凸，当且仅当它的海森矩阵处处半正定（附录 F）；在临界点，正定的海森矩阵保证严格局部极小。
- **分解。** $A\succ0$ 有唯一的 **Cholesky 分解** $A = LL^{\top}$，$L$ 为下三角且对角线为正；也有唯一的正定平方根 $A^{1/2} = Q\Lambda^{1/2}Q^{\top}$。从 $N(\mu, \Sigma)$ 采样就是 $\mu + Lz$，其中 $z\sim N(0, I)$（附录 C）。

## A.6 奇异值分解

**定理 A.5（SVD）。** 每个 $A\in\mathbb{R}^{m\times n}$ 都可写成 $A = U\Sigma V^{\top} = \sum_{i=1}^{r}\sigma_iu_iv_i^{\top}$，其中 $U\in\mathbb{R}^{m\times m}$ 与 $V\in\mathbb{R}^{n\times n}$ 为正交矩阵，$\Sigma$ 为对角矩阵，$\sigma_1\ge\sigma_2\ge\dots\ge\sigma_r > 0$，且 $r = \mathrm{rank}\,A$。

*证明概要。* $A^{\top}A$ 是对称半正定的，所以 $A^{\top}A = V\,\mathrm{diag}(\sigma_i^2)\,V^{\top}$，$\sigma_i\ge0$。对 $\sigma_i > 0$，令 $u_i = Av_i/\sigma_i$。它们标准正交：$u_i^{\top}u_j = v_i^{\top}A^{\top}Av_j/(\sigma_i\sigma_j) = \mathbb{1}[i = j]$。把它们扩充成 $\mathbb{R}^m$ 的标准正交基。于是 $AV = U\Sigma$，即 $A = U\Sigma V^{\top}$。$\blacksquare$

从几何上读，每个线性映射都是一次旋转（$V^{\top}$）、一次沿座标轴的伸缩（$\Sigma$），再一次旋转（$U$）。奇异值就是伸缩倍数：$\sigma_1 = \max_{\lVert x\rVert = 1}\lVert Ax\rVert$。

**低秩近似。** 截取最大的 $k$ 项，$A_k = \sum_{i\le k}\sigma_iu_iv_i^{\top}$，在谱范数与 Frobenius 范数下都是最优的秩 $k$ 近似，误差分别为 $\sigma_{k+1}$ 与 $\sqrt{\sum_{i>k}\sigma_i^2}$（Eckart–Young 定理，证明於单元 02，§2.5.5）。

**伪逆。** $A^{+} = V\Sigma^{+}U^{\top}$，把非零奇异值取倒数。$A^{+}y$ 是 $Ax\approx y$ 的最小范数最小二乘解——在欠定线性问题上，从零出发的梯度下降法正是收敛到它，这是隐式偏差最简单的例子（单元 01，§1.5.4）。

**例题 A.4。** $A = \begin{pmatrix}3 & 0\\4 & 5\end{pmatrix}$。$A^{\top}A = \begin{pmatrix}25 & 20\\20 & 25\end{pmatrix}$ 的特征值为 45 与 5，所以 $\sigma_1 = \sqrt{45} = 6.708$、$\sigma_2 = \sqrt5 = 2.236$，右奇异向量为 $(1, 1)/\sqrt2$ 与 $(1, -1)/\sqrt2$。最优秩 1 近似在两种范数下的误差都是 $\sigma_2 = 2.236$；条件数为 $\sigma_1/\sigma_2 = 3$。注意 $A$ 本身的特征值是 3 与 5——对非对称矩阵，特征值与奇异值不同。

## A.7 矩阵范数与条件数

**谱范数** $\lVert A\rVert = \sigma_1$ 是 $A$ 伸长向量的最大倍数；**Frobenius 范数** $\lVert A\rVert_F = \sqrt{\sum_{ij}A_{ij}^2} = \sqrt{\sum_i\sigma_i^2}$ 是各元素的欧几里得范数。两者都满足次乘性：$\lVert AB\rVert\le\lVert A\rVert\,\lVert B\rVert$。这个不等式是所有关于深层乘积之论证的骨干——$L$ 个层雅可比矩阵乘积的范数可能随 $L$ 呈几何增长或缩小（单元 03，§3.7.4）。

可逆矩阵 $A$ 的**条件数（condition number）**为 $\kappa(A) = \sigma_1/\sigma_n$。它量测求解 $Ax = b$ 时相对误差被放大多少：若 $A(x + \delta x) = b + \delta b$，则
$$
\frac{\lVert\delta x\rVert}{\lVert x\rVert}\le\kappa(A)\,\frac{\lVert\delta b\rVert}{\lVert b\rVert},
$$
因为 $\lVert\delta x\rVert\le\lVert A^{-1}\rVert\lVert\delta b\rVert = \lVert\delta b\rVert/\sigma_n$ 且 $\lVert b\rVert\le\sigma_1\lVert x\rVert$。在单位舍入误差为 $u$ 的浮点运算中（32 比特约 $6\times10^{-8}$，bfloat16 约 $5\times10^{-4}$），预期会损失约 $\log_{10}\kappa$ 位有效数字（[Higham 2002](/zh-cn/references/#higham-2002)）。对优化而言，同一个数有第二个意义：对海森矩阵为 $H$ 的二次函数，$\kappa(H)$ 决定梯度下降法的收敛速率（附录 F）。

## A.8 矩阵的幂与谱半径

**谱半径**为 $\rho(A) = \max_i|\lambda_i|$。递推 $h_t = Ah_{t-1}$ 给出 $h_t = A^th_0$，所以它的长期行为由 $A^t$ 决定。

**定理 A.6。** $t\to\infty$ 时 $A^t\to0$，当且仅当 $\rho(A) < 1$。此外，对每个 $\epsilon > 0$ 都有常数 $c_\epsilon$ 使 $\lVert A^t\rVert\le c_\epsilon(\rho(A) + \epsilon)^t$，且 $\lim_t\lVert A^t\rVert^{1/t} = \rho(A)$（Gelfand 公式）。

*证明概要。* 若某个特征值 $|\lambda|\ge1$、特征向量为 $v$，则 $\lVert A^tv\rVert = |\lambda|^t\lVert v\rVert\not\to0$。反之，把 $A$ 写成 Jordan 形式 $A = SJS^{-1}$。特征值为 $\lambda$、大小为 $m$ 的 Jordan 块，其 $t$ 次幂的元素为 $\binom{t}{j}\lambda^{t-j}$，$j < m$，每个都被 $t$ 的多项式乘 $|\lambda|^t$ 所界定。多项式乘 $|\lambda|^t$ 至多为 $c_\epsilon(|\lambda| + \epsilon)^t$，在 $|\lambda| < 1$ 时趋于零。$\blacksquare$

常数 $c_\epsilon$ 在实践中很重要。当 $A$ 是**正规的（normal）**（$AA^{\top} = A^{\top}A$，例如对称矩阵）时，$\lVert A^t\rVert = \rho(A)^t$ 精确成立，衰减是单调的。若不是，$\lVert A^t\rVert$ 可能在最终衰减之前先增长许多步——**瞬态增长（transient growth）**（单元 05，§5.3.5；[Trefethen & Embree 2005](/zh-cn/references/#trefethen-2005)）。

**例题 A.5。** $A = \begin{pmatrix}0.5 & 2\\0 & 0.5\end{pmatrix}$ 的 $\rho(A) = 0.5$，然而
$$
\lVert A^t\rVert = 1,\ 2.118,\ 2.031,\ 1.510,\ 1.004,\ 0.627,\ 0.376,\ \dots\qquad(t = 0, 1, 2, \dots).
$$
前四步这个映射都在放大，直到 $t = 5$ 才开始收缩。原因是非对角的耦合：$A^t = \begin{pmatrix}0.5^t & 4t\,0.5^{t}\\0 & 0.5^t\end{pmatrix}$，其右上元素先上升，几何因子才胜出。因此，谱半径小于一的学习递推，仍可能在中等时程内放大扰动与梯度。

**线性系统的稳定性**——内部稳定与输入–输出稳定，以及为什么转移函数的极点不一定是全部的特征值——在附录 G 中展开。

## 习题

**A.1** 证明 $\mathrm{rank}(A^{\top}A) = \mathrm{rank}(A)$。*（提示：证明 $\mathcal N(A^{\top}A) = \mathcal N(A)$。）*

**A.2** 证明正交矩阵的特征值模为 1，且 $\lVert Q\rVert = 1$。

**A.3** 对 $A = \begin{pmatrix}4 & 0\\0 & 1\end{pmatrix}$，计算 $\kappa(A)$，以及 $b$ 的相对误差为 $10^{-3}$ 时 $x$ 的最大相对误差。

**A.4** 证明对秩为 $r$ 的矩阵，$\lVert A\rVert\le\lVert A\rVert_F\le\sqrt{r}\,\lVert A\rVert$。

**A.5** 对 $A = \begin{pmatrix}0.9 & 0\\0 & 0.5\end{pmatrix}$ 计算 $\lVert A^t\rVert$，并说明为什么没有瞬态增长。

**A.6** ★ 令 $A$ 为特征值 $\lambda_1\ge\dots\ge\lambda_n$ 的对称矩阵。证明 Courant–Fischer 刻画 $\lambda_k = \max_{\dim\mathcal S = k}\min_{x\in\mathcal S, x\ne0}x^{\top}Ax/x^{\top}x$。

**解答。** **A.3** $\kappa = 4$；相对误差至多 $4\times10^{-3}$。**A.5** $\lVert A^t\rVert = 0.9^t$：这个矩阵是对角的，因此是正规的。

## 延伸阅读

[Strang 2016](/zh-cn/references/#strang-2016) 提供几何直觉；[Axler 2024](/zh-cn/references/#axler-2024) 以证明为主；[Horn & Johnson 2012](/zh-cn/references/#horn-2012) 与 [Golub & Van Loan 2013](/zh-cn/references/#golub-2013) 供查阅；[Trefethen & Embree 2005](/zh-cn/references/#trefethen-2005) 讨论非正规矩阵；[Deisenroth et al. 2020](/zh-cn/references/#deisenroth-2020) 在机器学习的背景中讲述这些内容。
