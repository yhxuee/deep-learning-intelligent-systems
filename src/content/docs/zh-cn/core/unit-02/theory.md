---
title: "2.5 一般理论"
description: 普遍逼近定理及其限制、Cover 函数计数定理、随机投影与 Johnson–Lindenstrauss 引理、以群平均得到不变性与等变线性映射，以及 Eckart–Young 定理的证明。
sidebar: { label: "2.5 一般理论", order: 5 }
---

本页各项结果回答的是不同问题。普遍逼近问网络*能表示什么*；Cover 定理计算线性读出*能分开哪些标注*；Johnson–Lindenstrauss 问投影*能保留多少距离*；群平均构建指定的对称性；Eckart–Young 则精确解出一项矩阵近似问题。没有任何一项结果能单独保证：梯度下降法会从有限数据学到我们想要的表示。

## 2.5.1 普遍逼近——以及它没有保证的事

**定理 2.2（普遍逼近；[Cybenko 1989](/zh-cn/references/#cybenko-1989)、[Hornik et al. 1989](/zh-cn/references/#hornik-1989)）。** 令 $\sigma$ 为连续的 S 形函数。对紧致集 $K\subset\mathbb{R}^p$ 上的每个连续函数 $f$ 与每个 $\varepsilon>0$，存在 $H$ 与参数，使单隐藏层网络 $g(x) = \sum_{j=1}^{H}a_j\sigma(w_j^{\top}x + b_j)$ 满足 $\sup_{x\in K}|f(x) - g(x)|<\varepsilon$。

这个定理谈的是紧致定义域上的连续函数，以及任意指定的近似精度。它没有说**需要多少**隐藏单元（对指定精度，有些目标函数族所需的单元数会随 $p$ 指数增加）、需要多少**数据**，也没有说**梯度下降法**能否找到参数。它只排除一种可能障碍——在无限宽极限下表达能力不足——却不提供实际学习保证。共享、样本效率与不变性等问题，全都存在于这项空白中。对 XOR 而言，隐藏层*能*建立分隔特征；数据与优化器是否会引导它找到，则是另一回事。

## 2.5.2 Cover 计数定理：为什么维度有帮助

**定理 2.3（[Cover 1965](/zh-cn/references/#cover-1965)）。** 令 $\mathbb{R}^p$ 中的 $N$ 个点处于一般位置（其中任意至多 $p$ 个点都线性无关）。能被通过原点的超平面分开的标注数为
$$
C(N, p) = 2\sum_{k=0}^{p-1}\binom{N-1}{k}.
$$
对仿射分类器 $w^{\top}x + b$，附加一个常数坐标，并使用 $C(N, p+1)$。

*证明概要。* 加入第 $N{+}1$ 个点。前 $N$ 个点的一种可分标注，若没有分隔超平面通过新点，就只能以一种方式延伸到新点；若有，就能以两种方式延伸；后者对应于这 $N$ 个点投影到 $(p-1)$ 维空间后的可分标注。由此得到递推 $C(N+1, p) = C(N, p) + C(N, p-1)$，且 $C(1, p) = 2$，其解就是上式（它是巴斯卡法则的累加）。$\blacksquare$

当 $N\le p$ 时，$C(N,p)=2^N$：一般位置上的每一种标注，都可由通过原点的超平面分开。当 $N=2p$ 时，全部标注中恰好有一半可分。这**不代表** $2p$ 个点可被击碎。击碎要求*每一种*标注都能实现；通过原点的线性分隔器之 VC 维度是 $p$，仿射分隔器则是 $p+1$。$2p$ 这个数字代表的是 Cover 可分标注比例的中点。

通过固定的非线性特征——随机特征（[Rahimi & Recht 2007](/zh-cn/references/#rahimi-2007)）、核函数或宽的随机隐藏层——把输入映射到更高维空间，可以让更多标注变得线性可达。这会增加表达容量，也包括拟合噪声的容量；它本身不能解释泛化。学习表示追求的是更具选择性的结果：让任务相关的标注变简单，而不是让每项差异都同样容易表达。

## 2.5.3 随机投影保留几何

一个嵌入需要多少维度，才能让 $n$ 个不同项目彼此保持距离？远少于 $n$。

**定理 2.4（Johnson–Lindenstrauss；[Johnson & Lindenstrauss 1984](/zh-cn/references/#johnson-1984)）。** 对任何 $0<\varepsilon<1$ 与任何 $n$ 个点 $x_1,\dots,x_n\in\mathbb R^D$，令随机线性映射 $R:\mathbb R^D\to\mathbb R^d$ 的元素独立服从 $N(0,1/d)$。以下不等式以正概率同时成立的一项充分条件是

$$
d>\frac{4\ln n}{\varepsilon^2/2-\varepsilon^3/3}=O(\varepsilon^{-2}\ln n).
$$

$$
(1-\varepsilon)\lVert x_i - x_j\rVert^2 \le \lVert Rx_i - Rx_j\rVert^2 \le (1+\varepsilon)\lVert x_i - x_j\rVert^2 \quad\text{对所有 } i, j .
$$

*证明想法（[Dasgupta & Gupta 2003](/zh-cn/references/#dasgupta-2003)）。* 对固定的单位向量 $u$，$d\lVert Ru\rVert^2$ 服从 $\chi_d^2$。Chernoff 界给出

$$
\Pr\!\left(\left|\lVert Ru\rVert^2-1\right|>\varepsilon\right)
\le2\exp\!\left[-\frac d2\left(\frac{\varepsilon^2}{2}-\frac{\varepsilon^3}{3}\right)\right].
$$

把它应用到少于 $n^2/2$ 个非零的两两差向量。根据上述 $d$ 条件，联合界小于一，所以至少存在一个投影能保留所有配对。零差向量本来就满足结论。$\blacksquare$

所需维度只随 $n$ 的对数增加，而且不依赖原始维度 $D$。把这个想法用在许多彼此正交的基底向量上，可以看出：$d$ 维空间能容纳指数多个归一化向量，而它们两两内积的绝对值都很小。这项几何事实启发了**叠加（superposition）**假说：若同时活跃的特征很少，网络或许能容忍干扰，编码比维度更多的稀疏特征（[Elhage et al. 2022](/zh-cn/references/#elhage-2022)）。Johnson–Lindenstrauss 证明的是几何上的可能性，不是训练后的网络一定会找到这种编码，也不保证它能无误使用。

## 2.5.4 以平均得到不变性；等变映射

令有限群 $\mathcal{G}$ 作用在输入上（例如 $T$ 个位置的 $T!$ 种置换，或长度 $T$ 信号的 $T$ 种循环位移）。

**命题 2.2（群平均）。** 对任何函数 $\phi$，平均 $\bar\phi(x) = \frac{1}{|\mathcal G|}\sum_{g\in\mathcal G}\phi(g\cdot x)$ 是不变的：对每个 $h\in\mathcal G$，$\bar\phi(h\cdot x) = \bar\phi(x)$。

*证明。* $\bar\phi(h\cdot x) = \frac{1}{|\mathcal G|}\sum_g\phi(gh\cdot x)$，而当 $g$ 跑遍 $\mathcal G$ 时，$gh$ 也跑遍 $\mathcal G$。$\blacksquare$

把词嵌入加总成“词袋（bag of words）”是一种简单的置换不变聚合，与平均构造密切相关：重新排列输入不会改变总和。它可能适合主题分类；若词序决定答案，例如“狗咬人”与“人咬狗”，就会造成灾难（§2.7.1）。

若对所有 $g$ 都有 $F(g\cdot x)=g\cdot F(x)$，线性映射 $F$ 就是**等变的**。把模型限制在等变映射中，可以大幅减少参数量与假设类别；当对称性假设正确时，这可能改善样本效率，但改善不会自动发生。书中有两个反复出现的例子。对长度 $T$ 信号的循环位移，等变线性映射恰好是循环卷积——$T$ 个参数，而非 $T^2$ 个（单元 04，定理 4.2）。对 $T\ge2$ 个位置、单一通道的置换（$X\in\mathbb R^T$），它们恰好是 $X\mapsto\lambda X+\gamma\mathbf1\mathbf1^{\top}X$——两个参数（单元 04，定理 4.3）。若有 $d_{\text{in}}$ 个输入通道与 $d_{\text{out}}$ 个输出通道，则形式为 $X\mapsto XA+\mathbf1\mathbf1^{\top}XB$，其中 $A,B\in\mathbb R^{d_{\text{in}}\times d_{\text{out}}}$，共 $2d_{\text{in}}d_{\text{out}}$ 个参数（[Zaheer et al. 2017](/zh-cn/references/#zaheer-2017)）。把总和改成平均，只会重新缩放 $B$。

这个置换等变线性函数族的位置结构仍很贫乏：每个位置只看见自己，以及所有位置的一项汇总。注意力会提供非线性、依内容而变的聚合方式。从对称性构建网络的更广泛理论，就是几何深度学习（[Bronstein et al. 2021](/zh-cn/references/#bronstein-2021)）。

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
| Cover（定理 2.3） | $N=2p$ 时，一般位置标注中有一半可分；VC 维度仍是 $p$ | 更高维的特征让更多标注变成线性——包括噪声——但不能击碎 $2p$ 个点 |
| Johnson–Lindenstrauss（定理 2.4） | $O(\ln n/\varepsilon^2)$ 维足以保留 $n$ 个点的两两距离 | 小型嵌入可以保留有限几何；这会启发叠加，但不能证明学习出的叠加 |
| 群平均（命题 2.2） | 平均产生不变性 | 嵌入袋对顺序不变——依任务而定，可能有用也可能致命 |
| Eckart–Young（定理 2.1） | 截断 SVD 是 Frobenius 范数下的最优低秩近似 | 精确理解低秩表格近似的视角，不代表每种嵌入目标 |
