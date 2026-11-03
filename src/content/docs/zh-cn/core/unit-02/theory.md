---
title: "2.5 一般理论"
description: 普遍逼近定理及其限制、Cover 函数计数定理、随机投影与 Johnson–Lindenstrauss 引理、以群平均得到不变性与等变线性映射，以及 Eckart–Young 定理的证明。
sidebar: { label: "2.5 一般理论", order: 5 }
---

## 2.5.1 普遍逼近——以及它没有保证的事

**定理 2.2（普遍逼近；[Cybenko 1989](/zh-cn/references/#cybenko-1989)、[Hornik et al. 1989](/zh-cn/references/#hornik-1989)）。** 令 $\sigma$ 为连续的 S 形函数。对紧致集 $K\subset\mathbb{R}^p$ 上的每个连续函数 $f$ 与每个 $\varepsilon>0$，存在 $H$ 与参数，使单隐藏层网络 $g(x) = \sum_{j=1}^{H}a_j\sigma(w_j^{\top}x + b_j)$ 满足 $\sup_{x\in K}|f(x) - g(x)|<\varepsilon$。

这个定理说：够宽的隐藏层基本上能表示任何函数。它没有说**需要多少**隐藏单元（对某些函数，需要的数量随 $p$ 指数成长）、需要多少**数据**才能找到正确的参数，也没有说**梯度下降法**能否找到它们。表示是学习的必要条件；定理只保证它在原则上不是瓶颈。本单元的每个实践问题——共享、样本效率、不变性——都存在于这个定理留下的空隙中。以 XOR 来说：隐藏层*能*计算 $x_1x_2$；问题在于数据与优化器是否会引导它去计算。

## 2.5.2 Cover 计数定理：为什么维度有帮助

**定理 2.3（[Cover 1965](/zh-cn/references/#cover-1965)）。** 令 $\mathbb{R}^p$ 中的 $N$ 个点处于一般位置（其中任意至多 $p$ 个点都线性无关）。能被通过原点的超平面分开的标注数为
$$
C(N, p) = 2\sum_{k=0}^{p-1}\binom{N-1}{k}.
$$
对仿射分类器 $w^{\top}x + b$，附加一个常数坐标，并使用 $C(N, p+1)$。

*证明概要。* 加入第 $N{+}1$ 个点。前 $N$ 个点的一种可分标注，若没有分隔超平面通过新点，就只能以一种方式延伸到新点；若有，就能以两种方式延伸；后者对应于这 $N$ 个点投影到 $(p-1)$ 维空间后的可分标注。由此得到递推 $C(N+1, p) = C(N, p) + C(N, p-1)$，且 $C(1, p) = 2$，其解就是上式（它是巴斯卡法则的累加）。$\blacksquare$

当 $N\le p$ 时，$C(N, p) = 2^N$：**每一种**标注都可分。当 $N = 2p$ 时，恰好一半的标注可分。所以线性读出的容量大约是每个维度两个点。以固定的非线性特征——随机特征（[Rahimi & Recht 2007](/zh-cn/references/#rahimi-2007)）、核函数或宽的随机隐藏层——把输入映射到更高维空间，几乎任何标注都能线性达到，这既是它有效的原因，也是它会记住噪声的原因。学习的表示追求更好的东西：让*真正的*标注变得简单的少数维度。

## 2.5.3 随机投影保留几何

一个嵌入需要多少维度，才能让 $n$ 个不同项目彼此保持距离？远少于 $n$。

**定理 2.4（Johnson–Lindenstrauss；[Johnson & Lindenstrauss 1984](/zh-cn/references/#johnson-1984)）。** 对任何 $0<\varepsilon<1$ 与任何 $n$ 个点 $x_1,\dots,x_n\in\mathbb{R}^D$，若 $d \ge 8\ln n/\varepsilon^2$，则元素独立服从 $N(0, 1/d)$ 的随机线性映射 $R:\mathbb{R}^D\to\mathbb{R}^d$ 以正的概率满足
$$
(1-\varepsilon)\lVert x_i - x_j\rVert^2 \le \lVert Rx_i - Rx_j\rVert^2 \le (1+\varepsilon)\lVert x_i - x_j\rVert^2 \quad\text{对所有 } i, j .
$$

*证明想法（[Dasgupta & Gupta 2003](/zh-cn/references/#dasgupta-2003)）。* 对固定的单位向量 $u$，$d\lVert Ru\rVert^2$ 是 $d$ 个独立标准正态平方的和，即 $\chi^2_d$。Chernoff 界给出 $\Pr(|\lVert Ru\rVert^2 - 1| > \varepsilon) \le 2e^{-d(\varepsilon^2/2 - \varepsilon^3/3)/2}$。把它应用到 $\binom n2$ 个差向量并取联合界；给定的 $d$ 让总失败概率小于 1。$\blacksquare$

维度只依赖 $\ln n$，不依赖 $n$ 或 $D$。一个推论对单元 11 很重要：在 $d$ 维中，可以放入指数多个*近似*正交（两两余弦至多 $\varepsilon$）的向量。所以只要同时只有少数特征活跃，网络能表示的特征就可以远多于它的维度——这就是**叠加（superposition）**假说（[Elhage et al. 2022](/zh-cn/references/#elhage-2022)）。

## 2.5.4 以平均得到不变性；等变映射

令有限群 $\mathcal{G}$ 作用在输入上（例如 $T$ 个位置的 $T!$ 种置换，或长度 $T$ 信号的 $T$ 种循环位移）。

**命题 2.2（群平均）。** 对任何函数 $\phi$，平均 $\bar\phi(x) = \frac{1}{|\mathcal G|}\sum_{g\in\mathcal G}\phi(g\cdot x)$ 是不变的：对每个 $h\in\mathcal G$，$\bar\phi(h\cdot x) = \bar\phi(x)$。

*证明。* $\bar\phi(h\cdot x) = \frac{1}{|\mathcal G|}\sum_g\phi(gh\cdot x)$，而当 $g$ 跑遍 $\mathcal G$ 时，$gh$ 也跑遍 $\mathcal G$。$\blacksquare$

把词嵌入加总成“词袋（bag of words）”就是置换群的平均构造：结果对词序不变。这对主题分类是好的表示，对“狗咬人”与“人咬狗”则是灾难（§2.7.1）。

若对所有 $g$ 都有 $F(g\cdot x) = g\cdot F(x)$，线性映射 $F$ 就是**等变的**。等变线性映射构成的空间比所有线性映射小得多，这就是它们样本效率的来源。书中两种情况一再出现。对长度 $T$ 信号的循环位移，等变线性映射恰好是循环卷积——$T$ 个参数而不是 $T^2$ 个（单元 04，定理 4.2）。对单一通道（$X\in\mathbb{R}^{T}$）时 $T$ 个位置的置换，它们恰好是 $X\mapsto \lambda X + \gamma\mathbf 1\mathbf 1^{\top}X$——两个参数（单元 04，定理 4.3）。有 $d_{\text{in}}$ 个输入通道与 $d_{\text{out}}$ 个输出通道（$X\in\mathbb{R}^{T\times d_{\text{in}}}$）时，它们恰好是 $X\mapsto XA + \mathbf 1\mathbf 1^{\top}XB$，其中 $A, B\in\mathbb{R}^{d_{\text{in}}\times d_{\text{out}}}$（等价地可用 $\frac1T\mathbf 1\mathbf 1^{\top}$，即平均而非总和）——共 $2d_{\text{in}}d_{\text{out}}$ 个参数，可以任意混合通道（[Zaheer et al. 2017](/zh-cn/references/#zaheer-2017)）。仍然贫乏的是*位置*结构：每个位置只看到它自己与所有位置的汇总，通过两种汇总方式。注意力就是摆脱这种贫乏的非线性出路。由对称性构建网络的一般理论是几何深度学习的主题（[Bronstein et al. 2021](/zh-cn/references/#bronstein-2021)）。

## 2.5.5 ★ 进阶：Eckart–Young 定理的证明

令 $A = \sum_i\sigma_iu_iv_i^{\top}$，$B$ 的秩至多为 $d$。$B$ 的零空间维度至少为 $n - d$，所以它与 $v_1,\dots,v_{d+1}$ 张成的 $(d+1)$ 维空间交于一个非零向量 $z$，将其归一化。于是 $Bz = 0$，且
$$
\lVert A - B\rVert_2^2 \ge \lVert(A - B)z\rVert^2 = \lVert Az\rVert^2 = \sum_{i\le d+1}\sigma_i^2(v_i^{\top}z)^2 \ge \sigma_{d+1}^2 .
$$
所以在谱范数下，没有秩 $d$ 的矩阵能胜过 $\sigma_{d+1}$，而 $A_d$ 恰好达到。对 Frobenius 范数，利用 Weyl 不等式 $\sigma_{i+j-1}(X+Y)\le\sigma_i(X)+\sigma_j(Y)$，取 $X = A - B$、$Y = B$、$j = d+1$：$\sigma_i(A - B)\ge\sigma_{i+d}(A)$，因此 $\lVert A - B\rVert_F^2 = \sum_i\sigma_i(A-B)^2\ge\sum_{i>d}\sigma_i(A)^2$。$\blacksquare$

## 2.5.6 摘要

| 结果 | 说了什么 | 对表示的意义 |
|---|---|---|
| 普遍逼近（定理 2.2） | 单隐藏层能表示任何连续函数 | 原则上表达能力不是瓶颈；样本与优化效率才是 |
| Cover（定理 2.3） | $p$ 维中约可打散 $2p$ 个点 | 更高维的特征让更多标注变成线性——包括噪声 |
| Johnson–Lindenstrauss（定理 2.4） | $O(\ln n/\varepsilon^2)$ 维即可保留 $n$ 个点的距离 | 嵌入可以很小；叠加是可能的 |
| 群平均（命题 2.2） | 平均产生不变性 | 嵌入袋对顺序不变——依任务而定，可能有用也可能致命 |
| Eckart–Young（定理 2.1） | 截断 SVD 是最优低秩近似 | 把嵌入看成共现统计的分解 |
