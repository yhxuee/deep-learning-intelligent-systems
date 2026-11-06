---
title: "F · 优化"
description: 凸集与凸函数及其一阶与二阶刻画；光滑性、强凸性与 Polyak–Łojasiewicz 条件；梯度下降、动量与随机梯度的收敛速率，并证明凸情形的速率；牛顿法与预条件；拉格朗日乘子、KKT 条件与对偶；最大熵与吉布斯变分原理。
sidebar: { label: "F · 优化", order: 6 }
---

单元 03 从第一原理推导梯度下降法并证明它的主要保证：下降引理、不需凸性也能收敛到驻点、二次函数上的条件数速率、强凸性下的线性收敛、随机梯度的噪声下限，以及动量的速率（§3.5.2–§3.5.7）。本附录提供这些证明所假设的凸分析背景，把各种速率集中在一张表中，证明单元 03 没有证明的那一个速率，并补上受限优化——拉格朗日乘子、KKT 条件与对偶——本书用它们推导最大似然估计、softmax 输出，以及 KL 正则化后训练的最优解。例题中的数字由 `labs/test_appendices.py` 检查。

### 初学者图像：地形、规则、保证

一个优化问题有三项不能混为一谈的成分：

- **目标函数**定义地形，例如 $f(x)=x^2$；
- **算法**定义移动规则，例如 $x_{k+1}=x_k-\eta f'(x_k)$；
- **假设**决定能保证什么，例如凸性或光滑性。

同一个目标对某算法可能很容易，对另一个却很慢；同一个算法可能在某步长收敛、另一个步长发散。凸性表示局部极小也是全局极小；光滑性限制斜率能多突然地改变；强凸性排除平坦方向。这些词都不表示“深层网络是凸的”；它们提供精确的模型案例与理解困难地形的局部工具。

| 小节 | 用于 |
|---|---|
| F.1–F.2 凸性、光滑性 | 01（对数损失）、03（下降引理）、06（计算最优分配） |
| F.3 收敛速率 | 03、05（稳定递推）、06 |
| F.4 牛顿法、预条件 | 03（Adam）、06（二阶方法） |
| F.5 拉格朗日与 KKT | 01（MLE 二元语法）、06（Chinchilla 分配）、09（KL 最优解） |
| F.6 最大熵、吉布斯 | 01、04（softmax）、09（倾斜的参考分布） |

## F.1 凸集与凸函数

若集合 $C$ 包含任两点之间的线段，它就是**凸的**。凸集上的函数 $f$ 若满足
$$
f(\lambda x + (1 - \lambda)y)\le\lambda f(x) + (1 - \lambda)f(y)\qquad\text{对所有 }x, y\text{ 与 }\lambda\in[0, 1],
$$
它就是**凸函数**。

**命题 F.1（刻画）。** 对开凸集上的可微函数 $f$，下列等价：(i) $f$ 是凸的；(ii) 对所有 $x, y$，$f(y)\ge f(x) + \nabla f(x)^{\top}(y - x)$——函数字于它的切平面之上；(iii) $(\nabla f(x) - \nabla f(y))^{\top}(x - y)\ge0$——梯度是单调的。若 $f$ 二次可微，它们也等价于 (iv) 处处 $\nabla^2f(x)\succeq0$。

*(i)⇒(ii) 的证明。* 由凸性，$f(x + \lambda(y - x))\le f(x) + \lambda(f(y) - f(x))$。减去 $f(x)$、除以 $\lambda$ 并令 $\lambda\to0$：$\nabla f(x)^{\top}(y - x)\le f(y) - f(x)$。其他蕴含关系类似（[Boyd & Vandenberghe 2004](/zh-cn/references/#boyd-2004)，§3.1）。$\blacksquare$

(ii) 最有用的推断：**凸函数的每个驻点都是全局极小**，因为 $\nabla f(x^\star) = 0$ 给出对所有 $y$ 都有 $f(y)\ge f(x^\star)$。

**保持凸性的运算。** 非负和；与仿射映射的合成 $f(Ax + b)$；凸函数的逐点最大值；以及 $\mathrm{LSE}(z) = \ln\sum_je^{z_j}$（附录 B，命题 B.3）。所以线性 softmax 模型的交叉熵在其权重上是凸的，而二元语法的最大似然问题（单元 01，§1.3.3）是有唯一解的凸问题。深层网络在其参数上不是凸的：凸损失与非线性网络的合成破坏了凸性，这就是为什么单元 03 在没有凸性时的保证比较弱。

## F.2 光滑性、强凸性与 PL 条件

- 若 $\lVert\nabla f(x) - \nabla f(y)\rVert\le L\lVert x - y\rVert$，$f$ 就是 **$L$-光滑的**。对二次可微的 $f$，这等于 $-LI\preceq\nabla^2f\preceq LI$，并蕴含二次上界 $f(y)\le f(x) + \nabla f(x)^{\top}(y - x) + \frac L2\lVert y - x\rVert^2$（附录 B.3）。
- 若 $f(y)\ge f(x) + \nabla f(x)^{\top}(y - x) + \frac\mu2\lVert y - x\rVert^2$，$f$ 就是 **$\mu$-强凸的**；对二次可微的 $f$，即 $\nabla^2f\succeq\mu I$。强凸函数有唯一的极小点，并满足 $f(x) - f^\star\le\frac1{2\mu}\lVert\nabla f(x)\rVert^2$。
- 最后这个不等式，**Polyak–Łojasiewicz（PL）条件**，可以在*没有*凸性的情况下成立，而它正是梯度下降法线性收敛证明所需要的全部。在插值解的流形附近，过参数化模型常常满足局部的 PL 条件，这是它们尽管不凸、训练却收敛得很快的一种解释。

比值 $\kappa = L/\mu$ 是问题的**条件数**；对二次函数 $\frac12x^{\top}Hx$，它就是 $H$ 的条件数（附录 A.7）。

## F.3 收敛速率

对梯度下降法 $x_{k+1} = x_k - \eta\nabla f(x_k)$，$f^\star = \inf f$：

| 假设 | 步长 | $k$ 步之后的保证 | 证明 |
|---|---|---|---|
| $L$-光滑 | $1/L$ | $\min_{j<k}\lVert\nabla f(x_j)\rVert^2\le2L(f(x_0) - f^\star)/k$ | 单元 03，§3.5.3 |
| $L$-光滑、凸 | $1/L$ | $f(x_k) - f^\star\le L\lVert x_0 - x^\star\rVert^2/(2k)$ | 定理 F.2 |
| $L$-光滑、$\mu$-强凸（或 PL） | $1/L$ | $f(x_k) - f^\star\le(1 - 1/\kappa)^k(f(x_0) - f^\star)$ | 单元 03，§3.5.5 |
| 二次函数，特征值在 $[\mu, L]$ | $2/(L + \mu)$ | $\lVert x_k - x^\star\rVert\le\big(\frac{\kappa - 1}{\kappa + 1}\big)^k\lVert x_0 - x^\star\rVert$ | 单元 03，§3.5.4 |
| 二次函数，调好的重球动量 | — | 每步速率 $\frac{\sqrt\kappa - 1}{\sqrt\kappa + 1}$（渐近） | 单元 03，§3.5.7 |
| 随机梯度，方差 $\sigma^2$，强凸 | $\eta_k\propto1/k$ | $\mathbb{E}f(x_k) - f^\star = O(1/k)$ | [Bottou et al. 2018](/zh-cn/references/#bottou-2018) |

**定理 F.2（凸情形的速率）。** 若 $f$ 是凸的且 $L$-光滑，极小点为 $x^\star$，则步长 $\eta = 1/L$ 的梯度下降法满足 $f(x_k) - f^\star\le\frac{L\lVert x_0 - x^\star\rVert^2}{2k}$。

*证明。* 记 $g_j = \nabla f(x_j)$。下降引理给出 $f(x_{j+1})\le f(x_j) - \frac1{2L}\lVert g_j\rVert^2$，凸性给出 $f(x_j)\le f^\star + g_j^{\top}(x_j - x^\star)$。合起来，
$$
f(x_{j+1}) - f^\star\le g_j^{\top}(x_j - x^\star) - \tfrac1{2L}\lVert g_j\rVert^2 = \tfrac L2\big(\lVert x_j - x^\star\rVert^2 - \lVert x_{j+1} - x^\star\rVert^2\big),
$$
其中等式来自展开 $\lVert x_{j+1} - x^\star\rVert^2 = \lVert x_j - x^\star - g_j/L\rVert^2$。对 $j = 0,\dots,k-1$ 求和，右边伸缩相消，至多为 $\frac L2\lVert x_0 - x^\star\rVert^2$；由于 $f(x_j)$ 不增，左边至少为 $k(f(x_k) - f^\star)$。$\blacksquare$

**例题 F.1。** 对 $f(x, y) = \frac12(x^2 + 10y^2)$，$\mu = 1$、$L = 10$、$\kappa = 10$。每步误差的收缩倍数：$\eta = 1/L$ 时为 $1 - 1/\kappa = 0.9$；最优固定步长 $2/11$ 时为 $(\kappa - 1)/(\kappa + 1) = 0.818$；调好的重球动量为 $(\sqrt{10} - 1)/(\sqrt{10} + 1) = 0.519$。要把误差缩小 $10^6$ 倍，分别约需 131、69 与 21 步。在 $\kappa = 10^4$ 时，前两者约需 140 000 与 69 000 步，动量约需 690 步——这就是动量与预条件对病态条件问题之所以重要的原因（单元 03，§3.7.3）。

**随机梯度。** 以方差为 $\sigma^2$ 的无偏梯度估计与固定步长 $\eta$，梯度下降法收敛到极小值的一个邻域，其大小随 $\eta\sigma^2$ 增长，而不是收敛到极小值本身（单元 03，§3.5.6）。要精确收敛，需要满足 **Robbins–Monro 条件** $\sum_k\eta_k = \infty$、$\sum_k\eta_k^2 < \infty$ 的递减步长（[Robbins & Monro 1951](/zh-cn/references/#robbins-1951)）——前者让迭代能走任意远的距离，后者让累积的噪声有有限的方差。实践中的学习率调度（暖身后再以余弦或线性衰减；单元 03，§3.8.3）是在这些要求与有限预算之间的工程折衷。

## F.4 牛顿法与预条件

**牛顿法**精确地最小化局部二次模型：$x_{k+1} = x_k - \nabla^2f(x_k)^{-1}\nabla f(x_k)$。在二次函数上，无论条件如何，它一步就收敛。在海森矩阵正定且 Lipschitz 的极小点附近，它**二次收敛**：误差大致每步平方一次。

**例题 F.2。** 最小化 $f(x) = x - \ln x$（极小点在 $x = 1$）。牛顿更新为 $x_{k+1} = 2x_k - x_k^2$，从 $x_0 = 0.5$ 出发，误差 $1 - x_k$ 依序为 $0.25,\ 0.0625,\ 0.0039,\ 1.5\times10^{-5},\ 2.3\times10^{-10}$——每一个都是前一个的平方。

对有数百万参数的网络，牛顿法太昂贵（$O(N^2)$ 内存、$O(N^3)$ 求解），而且会被吸向海森矩阵不定的鞍点。**预条件梯度下降** $x_{k+1} = x_k - \eta P^{-1}\nabla f(x_k)$ 以较低成本保留这个想法：若 $P$ 近似海森矩阵，有效条件数就变成 $P^{-1/2}\nabla^2fP^{-1/2}$ 的条件数。Adam 除以每个梯度坐标的移动均方根，就是一个**对角**预条件器（单元 03，§3.3.8 与 §3.5.8）；Shampoo 之类的结构化预条件器则逐块近似海森矩阵（单元 03，§3.8.6）。

## F.5 受限优化：拉格朗日乘子、KKT 与对偶

**等式限制。** 要在 $g_i(x) = 0$ 的限制下最小化 $f(x)$，构造**拉格朗日函数** $\mathcal L(x, \lambda) = f(x) + \sum_i\lambda_ig_i(x)$。在限制梯度线性无关的局部极小点，存在乘子使
$$
\nabla f(x^\star) + \sum_i\lambda_i\nabla g_i(x^\star) = 0 .
$$
*理由。* 任何在一阶上保持限制成立的方向 $d$，对所有 $i$ 都有 $\nabla g_i^{\top}d = 0$；在极小点，对所有这样的 $d$ 都有 $\nabla f^{\top}d = 0$；所以 $\nabla f$ 落在 $\nabla g_i$ 张成的空间中。$\blacksquare$

*例子：最大似然类别分布。* 在 $\sum_kp_k = 1$ 的限制下最大化 $\sum_kc_k\ln p_k$。驻点条件给出 $c_k/p_k = \lambda$，所以 $p_k\propto c_k$，而限制确定 $p_k = c_k/n$——即单元 01 命题 1.1。

**不等式限制（KKT）。** 要在 $g_i(x)\le0$ 与 $h_j(x) = 0$ 的限制下最小化 $f(x)$，**Karush–Kuhn–Tucker 条件**为：驻点条件 $\nabla f + \sum_i\lambda_i\nabla g_i + \sum_j\nu_j\nabla h_j = 0$；原始可行性；对偶可行性 $\lambda_i\ge0$；以及**互补松弛** $\lambda_ig_i(x) = 0$——不紧的限制其乘子为零。对满足某个限制条件资格（例如存在严格可行点——Slater 条件）的凸问题，KKT 条件是全局极小的必要且充分条件。

**对偶。** **对偶函数** $q(\lambda, \nu) = \inf_x\mathcal L(x, \lambda, \nu)$ 是凹的，而且对每个 $\lambda\ge0$，它都是最优值的下界（**弱对偶**）：对可行的 $x$，$\mathcal L(x, \lambda, \nu)\le f(x)$。在 Slater 条件下，最优下界是紧的（**强对偶**）。对偶把单元 06 的计算最优分配——在 $6ND = C$ 的限制下最小化损失 $L(N, D)$——变成一行计算：在最优点，参数与数据每 FLOP 的边际损失下降相等（单元 06，命题 6.2）。

## F.6 最大熵与吉布斯变分原理

**最大熵。** 在有限集合上、期望值被规定为 $\mathbb{E}_p[f_j(x)] = m_j$ 的分布中，熵最大者的形式为
$$
p(x) = \frac{\exp\big(\sum_j\lambda_jf_j(x)\big)}{Z(\lambda)},
$$
乘子 $\lambda_j$ 被选来满足限制。*推导。* $-\sum p\ln p$ 加上这些限制与 $\sum p = 1$ 的拉格朗日函数，在 $-\ln p(x) - 1 + \sum_j\lambda_jf_j(x) + \nu = 0$ 时为驻点。$\blacksquare$ 没有限制时这是均匀分布；对方差固定的实数变量加上 $\mathbb{E}[x]$ 的限制，就是高斯分布（附录 D.4）。

**例题 F.3。** 已知一颗六面骰的平均是 4.5 而不是 3.5。最大熵分布为 $p_k\propto e^{\lambda k}$，$\lambda = 0.371$：$(0.054, 0.079, 0.114, 0.165, 0.240, 0.348)$，熵为 1.614 nats——与限制兼容、承诺最少的分布，略低于均匀骰子的 $\ln6 = 1.792$。

**吉布斯变分原理。** 对任何分数 $r(x)$ 与温度 $\tau > 0$，
$$
\max_p\ \Big(\mathbb{E}_p[r(x)] + \tau H(p)\Big) = \tau\ln\sum_xe^{r(x)/\tau},\qquad\text{在 }p^\star(x) = \frac{e^{r(x)/\tau}}{\sum_ye^{r(y)/\tau}}\text{ 处达到。}
$$
*证明。* $\mathbb{E}_p[r] + \tau H(p) = \tau\ln Z - \tau\,\mathrm{KL}(p\,\Vert\,p^\star)$，而 KL 在 $p = p^\star$ 时最小为零。$\blacksquare$

本书三个核心公式都是这个原理的化身。**Softmax** 是 logits 的熵正则化 argmax，它的对数归一化常数就是 LSE（附录 B.5）；温度就是 $\tau$。**注意力**权重是查询–键分数上的 softmax（单元 04）。而把熵换成对参考分布的 KL 散度，$\mathbb{E}_p[r] - \beta\,\mathrm{KL}(p\,\Vert\,\pi_{\text{ref}})$，就得到 KL 正则化后训练的倾斜最优解 $\pi^\star\propto\pi_{\text{ref}}e^{r/\beta}$（单元 09，命题 9.1）——证明相同，只是以 $\pi_{\text{ref}}$ 取代均匀分布。

## 习题

**F.1** 证明 $f(x) = \lVert Ax - b\rVert^2$ 是凸的，并以 $A$ 的奇异值表示 $L$ 与 $\mu$。

**F.2** 证明凸且 $L$-光滑的函数满足 $f(x) - f^\star\ge\frac1{2L}\lVert\nabla f(x)\rVert^2$。

**F.3** 对例题 F.1 以 $\eta = 1/L$ 从 $(1, 1)$ 开始做梯度下降。写出 $x_k$ 与 $y_k$ 的闭式。哪一个坐标限制了收敛？

**F.4** 以拉格朗日乘子求在 $a^{\top}x = 1$ 限制下 $\frac12\lVert x\rVert^2$ 的最小值，并把它诠释为投影。

**F.5** 证明牛顿法在可逆线性变量替换 $x = Ty$ 下不变，而梯度下降法不是。

**F.6** ★ 以拉格朗日乘子推导 $L(N, D) = E + AN^{-\alpha} + BD^{-\beta}$ 在 $6ND = C$ 限制下的计算最优分配，并与单元 06 命题 6.2 核对。

**解答。** **F.1** $L = 2\sigma_{\max}^2$、$\mu = 2\sigma_{\min}^2$（当 $A$ 列满秩时）。**F.3** $x_k = 0.9^k$，一步之后 $y_k = 0$：平坦方向 $x$ 限制了收敛。**F.4** $x^\star = a/\lVert a\rVert^2$，超平面上离原点最近的点。

## 延伸阅读

[Boyd & Vandenberghe 2004](/zh-cn/references/#boyd-2004) 讲凸性与对偶；[Nesterov 2018](/zh-cn/references/#nesterov-2018) 讲收敛速率与下界；[Bottou et al. 2018](/zh-cn/references/#bottou-2018) 讲机器学习中的随机优化；[Deisenroth et al. 2020](/zh-cn/references/#deisenroth-2020) 第 7 章为入门。
