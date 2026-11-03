---
title: "2.3 数学构建"
description: 线性可分性、把嵌入查找看成矩阵乘积及其散射加总梯度、秩 d 分解式二元语法及其梯度、神经 n 元语法、最优低秩近似，以及不变性与等变性的精确定义。
sidebar: { label: "2.3 数学构建", order: 3 }
---

## 2.3.1 线性读出与可分性

特征 $\phi(x)\in\mathbb{R}^d$ 上的二元线性分类器预测 $\mathrm{sign}(w^{\top}\phi(x) + b)$。若某个 $(w, b)$ 能完美分类一组有标签的数据，该数据在 $\phi$ 中就是**线性可分（linearly separable）**的。对概率输出，对应的是 **logistic**（两类）或 **softmax**（多类）读出：

$$
q(y\mid x) = \frac{\exp(w_y^{\top}\phi(x) + b_y)}{\sum_{y'}\exp(w_{y'}^{\top}\phi(x) + b_{y'})}.
$$

任两类之间的对数胜算 $\ln q(y\mid x) - \ln q(y'\mid x) = (w_y - w_{y'})^{\top}\phi(x) + (b_y - b_{y'})$ **对特征是线性的**。所以“目标在这个表示中很简单”对本书每个模型都有精确的意义：任务所需的对数胜算（接近）是抵达读出层之表示的线性函数。改变 $\phi$，就改变了哪些对数胜算函数可以达到。

## 2.3.2 嵌入查找是一个矩阵乘积

令 $E\in\mathbb{R}^{V\times d}$，$e_a\in\mathbb{R}^{V}$ 为符号 $a$ 的独热向量（可能与嵌入 $e_a$ 混淆时，记为 $\mathbf{u}_a$）。则

$$
E^{\top}\mathbf{u}_a = E[a] \qquad(E \text{ 的第 } a \text{ 行}).
$$

对一批索引 $x_1,\dots,x_N$，把独热向量叠成 $U\in\{0,1\}^{N\times V}$ 的各行；查出的嵌入就是 $H = UE\in\mathbb{R}^{N\times d}$。实现中从不真的形成 $U$——索引便宜得多——但这个恒等式告诉我们梯度是什么。

**命题 2.1（查找的梯度）。** 若标量损失 $L$ 只通过 $H = UE$ 依赖 $E$，且 $G = \partial L/\partial H$，则
$$
\frac{\partial L}{\partial E} = U^{\top}G,\qquad\text{即}\qquad \frac{\partial L}{\partial E[a]} = \sum_{i:\,x_i = a} G[i].
$$

*证明。* $H_{ik} = \sum_a U_{ia}E_{ak}$，所以 $\partial H_{ik}/\partial E_{ak} = U_{ia}$，由链式法则 $\partial L/\partial E_{ak} = \sum_i G_{ik}U_{ia} = (U^{\top}G)_{ak}$。$U^{\top}G$ 的第 $a$ 行，就是把符号 $a$ 出现位置上的 $G$ 各行加总。$\blacksquare$

在程序中这是一个**散射加总（scatter-add）**：`np.add.at(dE, x, G)`。改用单纯指定（`dE[x] = G`）是经典错误：当一个符号在同一批量中出现两次，第二个梯度会覆盖第一个，而不是加上去。主干项目阶段 02 的测试把散射加总与明确的 $U^{\top}G$ 相比较（例题 2.7）。

## 2.3.3 分解式二元语法及其梯度

以下式为配对“$a$ 接着 $b$”打分：

$$
z_{ab} = e_a^{\top}Me_b + c_b,\qquad q(b\mid a) = \mathrm{softmax}_b(z_{a\cdot}),
$$

两侧使用**同一张**表 $E$（绑定），加上一个 $d\times d$ 矩阵 $M$ 与输出偏置 $c$。以矩阵形式写，对查出列为 $H = E[x]$（$N\times d$）的一批数据，分数为 $Z = HME^{\top} + \mathbf 1c^{\top}$（$N\times V$）。因为 $HME^{\top}$ 的秩至多为 $d$，这是**二元语法对数概率表的秩 $d$ 模型**（差一个逐行归一化）。

令 $\Delta = \partial L/\partial Z = (Q - Y)/N$，其中 $Q$ 为预测概率、$Y$ 为独热目标（softmax–交叉熵的梯度，於单元 03 §3.3.5 推导）。对 $Z = HME^{\top} + \mathbf 1c^{\top}$ 微分：

$$
\frac{\partial L}{\partial c} = \Delta^{\top}\mathbf 1,\qquad
\frac{\partial L}{\partial M} = H^{\top}\Delta E,\qquad
\frac{\partial L}{\partial E}\Big|_{\text{输出侧}} = \Delta^{\top}HM,\qquad
\frac{\partial L}{\partial H} = \Delta EM^{\top}.
$$

最后一项通过查找抵达 $E$（命题 2.1），所以绑定表的总梯度为

$$
\frac{\partial L}{\partial E} = \Delta^{\top}HM + U^{\top}(\Delta EM^{\top}).
$$

每一项都来自规则“对 $Y = AXB$，$\partial L/\partial X = A^{\top}(\partial L/\partial Y)B^{\top}$”，也就是对每个元素两次应用单变量链式法则（习题 2.8）。`spine/stage02_embeddings.py` 正是实现这四行，并以有限差分检查。

**初始化与基线。** 当 $M = 0$、$c = 0$ 时，每个分数都是零、每个预测都是均匀的，所以初始损失恰好是 $\ln V$——单元 01 的基线，现在用来检查代码。

**最优值受计数所限。** 每个分解式二元语法都是一个二元语法，所以由命题 1.1，它的训练损失永远不可能低于计数式最大似然表。当 $d$ 增加到 $V$ 时，分解可以表示任何表，这个界就能达到。

## 2.3.4 神经 n 元语法

对 $k$ 个符号的上下文，把它们的嵌入串接成 $u = [e_{x_{t-k}};\dots;e_{x_{t-1}}]\in\mathbb{R}^{kd}$，并计算

$$
a = W_1^{\top}u + b_1,\quad h = \tanh(a),\quad z = W_2^{\top}h + b_2,\quad q = \mathrm{softmax}(z),
$$

其中 $W_1\in\mathbb{R}^{kd\times H}$、$W_2\in\mathbb{R}^{H\times V}$。其参数量为 $Vd + kdH + H + HV + V$：当 $V = 50$、$d = 16$、$k = 4$、$H = 128$ 时，为 $800 + 8{,}192 + 128 + 6{,}400 + 50 = 15{,}570$。相同上下文的计数表最多需要 $V^k = 6{,}250{,}000$ 行、每行 $V$ 个数。

单一样本的反向传递，记 $\delta_z = q - \mathbf{u}_y$：

$$
\frac{\partial L}{\partial W_2} = h\,\delta_z^{\top},\quad \frac{\partial L}{\partial b_2} = \delta_z,\quad
\delta_a = (W_2\delta_z)\odot(1 - h\odot h),\quad
\frac{\partial L}{\partial W_1} = u\,\delta_a^{\top},\quad
\frac{\partial L}{\partial u} = W_1\delta_a .
$$

梯度 $\partial L/\partial u$ 切成 $k$ 段、每段长 $d$，第 $j$ 段散射加总到上下文第 $j$ 个位置符号所对应的行。在同一个上下文中出现两次的符号（`aa`）会收到两段。因子 $1 - h\odot h$ 是 $\tanh$ 的导数。

[Bengio et al. (2003)](/zh-cn/references/#bengio-2003) 在词层级语料上使用这个架构（另加从 $u$ 到 $z$ 的直接线性连接），报告的测试困惑度大幅低于当时最好的平滑 n 元语法模型。我们的字符层级复现（例题 2.6）显示，在小数据量时有同样的效果，而在大数据量时效果消失——原论文不需要展示这一点，但它阐明了嵌入*为什么*有帮助。

## 2.3.5 最优低秩近似

为什么 $V\times V$ 表的秩 $d$ 模型会好？奇异值分解给出答案。任何实矩阵 $A\in\mathbb{R}^{m\times n}$ 都可写成 $A = \sum_{i=1}^{r}\sigma_iu_iv_i^{\top}$，其中 $u_i$ 两两标准正交、$v_i$ 两两标准正交，且 $\sigma_1\ge\sigma_2\ge\dots\ge\sigma_r>0$。

**定理 2.1（Eckart–Young，[Eckart & Young 1936](/zh-cn/references/#eckart-1936)）。** 对每个 $d < r$，截断 $A_d = \sum_{i=1}^{d}\sigma_iu_iv_i^{\top}$ 在所有秩至多为 $d$ 的矩阵 $B$ 中使 $\lVert A - B\rVert_F$ 最小，最小值为 $\sqrt{\sum_{i>d}\sigma_i^2}$。

（证明见 §2.5.5 ★。）如果共现统计矩阵的奇异值迅速衰减——少数强模式、许多弱模式——低秩表示就能捕捉大部分，而被舍弃的方向大多是有限计数造成的噪声。古典的*潜在语意分析（latent semantic analysis）*正是对词–文档矩阵计算这种截断（[Deerwester et al. 1990](/zh-cn/references/#deerwester-1990)）。学习的嵌入并不计算 SVD，但 [Levy & Goldberg (2014)](/zh-cn/references/#levy-2014) 证明，流行的 skip-gram 嵌入目标在最优点上，是对位移后的逐点互信息矩阵的隐式分解。嵌入学习与低秩分解，是从两个侧面看到的同一个想法。

## 2.3.6 不变性与等变性

表示让学习变容易的第二种方式，是**舍弃任务不依赖的东西**。令 $g$ 为输入的变换——平移一张图像、置换一个集合、替故事中的朋友换名字。

- 若 $\phi(g\cdot x) = \phi(x)$，表示 $\phi$ 对 $g$ 是**不变的（invariant）**：输入被变换时，表示不改变。
- 若 $f(g\cdot x) = g\cdot f(x)$，映射 $f$ 对 $g$ 是**等变的（equivariant）**：变换输入，会以对应的方式变换输出。

若目标是不变的（图像平移时类别不变），不变的表示就移除了学习者原本必须从数据中学到的干扰——每张训练图像的每个平移版本。若目标是等变的（每个像素的标签随图像平移），等变的层就能保留结构，同时在各位置之间共享参数。卷积是对平移等变的线性映射；单元 04 证明了这一点（定理 4.2），并证明没有位置信息的注意力对位置置换是等变的（命题 4.2）。§2.5.4 给出一般的构造：对群取平均就产生不变性。

硬币的另一面是 §2.7.1：**舍弃了任务所需信息的表示，下游任何学习者都无法修补**。不变性是一场赌注，赌的是被舍弃的信息无关紧要。
