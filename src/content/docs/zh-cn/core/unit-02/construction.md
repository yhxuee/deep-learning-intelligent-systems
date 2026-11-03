---
title: "2.3 数学构建"
description: 线性可分性、把嵌入查找看成矩阵乘积及其散射加总梯度、秩 d 分解式二元语法及其梯度、神经 n 元语法、最优低秩近似，以及不变性与等变性的精确定义。
sidebar: { label: "2.3 数学构建", order: 3 }
---

本节会把 §2.2 的图像转成矩阵方程。每个式子都请按照同一顺序阅读：

1. 说出每个符号代表的对象；
2. 检查每个因子的形状；
3. 找出哪些参数被重复使用；
4. 最后才沿着导数往回走。

反复出现的维度包括：$V$ 个词汇符号、嵌入宽度 $d$、批量大小 $N$、上下文长度 $k$，以及隐藏宽度 $H$。形状不符不只是记号上的小问题；它通常会显示出错误的转置，或送到错误位置的梯度。

## 2.3.1 线性读出与可分性

特征 $\phi(x)\in\mathbb R^d$ 上的二元线性分类器，依 $w^{\top}\phi(x)+b$ 的正负做预测。若至少有一组 $(w,b)$ 能把每个样本都分对，这组有标签数据就是**在 $\phi$ 中线性可分（linearly separable）**的。请注意，可分性属于“某个特定表示中的数据”；同一批有标签对象，在一种 $\phi$ 下可能可分，换一种则不可分。

若需要概率输出，两类使用 **logistic** 读出，多类则使用 **softmax** 读出：

$$
q(y\mid x) = \frac{\exp(w_y^{\top}\phi(x) + b_y)}{\sum_{y'}\exp(w_{y'}^{\top}\phi(x) + b_{y'})}.
$$

Softmax 明明含有指数函数，为什么仍说它是线性读出？比较两个类别时，共同分母会消掉：

$$
\ln\frac{q(y\mid x)}{q(y'\mid x)}
=(w_y-w_{y'})^{\top}\phi(x)+(b_y-b_{y'}).
$$

因此，每一对类别的**对数胜算**都对特征呈线性。“目标在这个表示中很简单”现在有了精确版本：任务需要的对数胜算，应该接近抵达读出层之表示的线性函数。改变 $\phi$ 会改变哪些对数胜算函数容易表达，但不会自行保证训练一定找得到它们。

## 2.3.2 嵌入查找是一个矩阵乘积

令 $E\in\mathbb R^{V\times d}$ 为嵌入表。为了避免用 $e_a$ 表示两种不同对象，我们以 $\mathbf u_a\in\mathbb R^V$ 表示符号 $a$ 的独热向量，并以 $e_a=E[a]\in\mathbb R^d$ 表示它学习出的嵌入。则

$$
E^{\top}\mathbf u_a=e_a \qquad(E\text{ 的第 }a\text{ 行，以列向量写出}).
$$

对批量中的索引 $x_1,\dots,x_N$，把它们的独热向量叠成 $U\in\{0,1\}^{N\times V}$ 的各行。所有查出的嵌入就是

$$
H=UE\in\mathbb R^{N\times d}.
$$

形状检查为 $(N\times V)(V\times d)=N\times d$。真正的实现会使用 `E[x]`，不会建立几乎全为零的矩阵 $U$；但把等价的矩阵乘积写出来，能让反向传递变得可见。

**命题 2.1（查找的梯度）。** 若标量损失 $L$ 只通过 $H=UE$ 依赖 $E$，且 $G=\partial L/\partial H\in\mathbb R^{N\times d}$，则
$$
\frac{\partial L}{\partial E} = U^{\top}G,\qquad\text{即}\qquad \frac{\partial L}{\partial E[a]} = \sum_{i:\,x_i = a} G[i].
$$

证明之前，先用文本读结果：符号 $a$ 每次出现时，都查找了同一行参数 $E[a]$，所以这些位置的梯度贡献都必须回到该行，并且加总起来。

*证明。* 因为 $H_{ik}=\sum_aU_{ia}E_{ak}$，所以 $\partial H_{ik}/\partial E_{ak}=U_{ia}$。链式法则给出 $\partial L/\partial E_{ak}=\sum_iG_{ik}U_{ia}=(U^{\top}G)_{ak}$。因此，$U^{\top}G$ 的第 $a$ 行，恰好加总了符号 $a$ 出现位置上的 $G$ 各行。$\blacksquare$

在程序中这是一个**散射加总（scatter-add）**：`np.add.at(dE, x, G)`。改用单纯指定（`dE[x] = G`）是经典错误：当一个符号在同一批量中出现两次，第二个梯度会覆盖第一个，而不是加上去。主干项目阶段 02 的测试把散射加总与明确的 $U^{\top}G$ 相比较（例题 2.7）。

## 2.3.3 分解式二元语法及其梯度

以下式为配对“$a$ 接着 $b$”打分：

$$
z_{ab} = e_a^{\top}Me_b + c_b,\qquad q(b\mid a) = \mathrm{softmax}_b(z_{a\cdot}),
$$

同一张表 $E$ 会出现两次：$e_a$ 表示观察到的输入，$e_b$ 表示一个候选输出。这就是权重绑定。矩阵 $M\in\mathbb R^{d\times d}$ 让输入与输出坐标交互，$c_b$ 则是所有输入共用的“下一个符号”偏置。

对一个批量，$H=E[x]\in\mathbb R^{N\times d}$，且

$$
Z=HME^{\top}+\mathbf1c^{\top}\in\mathbb R^{N\times V}.
$$

检查主要路径：$(N\times d)(d\times d)(d\times V)=N\times V$。交互项 $HME^{\top}$ 的秩至多为 $d$。更精确地说，这是**输入–输出交互 logit 矩阵**的低秩模型；加入的偏置 $\mathbf1c^{\top}$，以及 log-softmax 产生的逐行常数，都是额外的秩一项。直接把整张对数概率表称为“秩 $d$”，会隐藏这些限定。

令 $\Delta=\partial L/\partial Z=(Q-Y)/N$，其中 $Q$ 是预测概率，$Y$ 是独热目标。这项 softmax–交叉熵结果会在单元 03 §3.3.5 推导；此处先把 $\Delta$ 每一行读成“预测概率减去目标概率”。对 $Z=HME^{\top}+\mathbf1c^{\top}$ 微分，得到

$$
\frac{\partial L}{\partial c} = \Delta^{\top}\mathbf 1,\qquad
\frac{\partial L}{\partial M} = H^{\top}\Delta E,\qquad
\frac{\partial L}{\partial E}\Big|_{\text{输出侧}} = \Delta^{\top}HM,\qquad
\frac{\partial L}{\partial H} = \Delta EM^{\top}.
$$

绑定表 $E$ 会从两条路径收到梯度：

- **输出路径：** 每个候选向量都出现在 $E^{\top}$ 中；
- **输入路径：** 观察到的符号先经查找形成 $H=UE$。

上一个式子中的最后一项，是抵达 $H$ 的梯度。命题 2.1 会把它从输入路径散射回去，因此两条路径必须相加：

$$
\frac{\partial L}{\partial E} = \Delta^{\top}HM + U^{\top}(\Delta EM^{\top}).
$$

每个式子都来自规则“对 $Y=AXB$，$\partial L/\partial X=A^{\top}(\partial L/\partial Y)B^{\top}$”；它本身只是把标量链式法则逐元素应用（习题 2.8）。也可以用形状逐项稽核：例如 $H^{\top}\Delta E$ 的形状是 $(d\times N)(N\times V)(V\times d)=d\times d$，正好与 $M$ 相同。`spine/stage02_embeddings.py` 实现了这些式子，并以有限差分检查。

**初始化与基线。** 当 $M = 0$、$c = 0$ 时，每个分数都是零、每个预测都是均匀的，所以初始损失恰好是 $\ln V$——单元 01 的基线，现在用来检查代码。

**最优值受计数所限。** 每个分解式二元语法都是一个二元语法，所以由命题 1.1，它的训练损失永远不可能低于计数式最大似然表。当 $d$ 增加到 $V$ 时，分解可以表示任何表，这个界就能达到。

## 2.3.4 神经 n 元语法

对 $k$ 个符号的上下文，把它们的嵌入串接成 $u = [e_{x_{t-k}};\dots;e_{x_{t-1}}]\in\mathbb{R}^{kd}$，并计算

$$
a = W_1^{\top}u + b_1,\quad h = \tanh(a),\quad z = W_2^{\top}h + b_2,\quad q = \mathrm{softmax}(z),
$$

其中 $W_1\in\mathbb{R}^{kd\times H}$、$W_2\in\mathbb{R}^{H\times V}$。其参数量为 $Vd + kdH + H + HV + V$：当 $V = 50$、$d = 16$、$k = 4$、$H = 128$ 时，为 $800 + 8{,}192 + 128 + 6{,}400 + 50 = 15{,}570$。相同上下文的计数表最多需要 $V^k = 6{,}250{,}000$ 行、每行 $V$ 个数。

这五项依序计算嵌入表、第一层权重、第一层偏置、输出权重与输出偏置。因此，计数表最坏情况的存储量是 $V^{k+1}$ 个数值；不过，稀疏实现只会存储观察到的上下文。神经模型之所以较小，是因为许多上下文都必须通过同一组 $W_1$ 与 $W_2$；这项共享既是它的优点，也是限制。

对单一样本，从 $\delta_z=q-\mathbf u_y$ 开始反向传递，再以与前向方程相反的顺序往回走：

$$
\frac{\partial L}{\partial W_2} = h\,\delta_z^{\top},\quad \frac{\partial L}{\partial b_2} = \delta_z,\quad
\delta_a = (W_2\delta_z)\odot(1 - h\odot h),\quad
\frac{\partial L}{\partial W_1} = u\,\delta_a^{\top},\quad
\frac{\partial L}{\partial u} = W_1\delta_a .
$$

因子 $1-h\odot h$ 是 $\tanh$ 的逐元素导数。最后，把 $\partial L/\partial u$ 切成 $k$ 段，每段长 $d$。第 $j$ 段属于上下文位置 $j$ 所查找的嵌入，因此要散射加总到该符号的行。如果同一符号在一个上下文中出现两次，例如 `aa`，两段都必须加上；这与命题 2.1 的重复索引问题完全相同。

[Bengio et al. (2003)](/zh-cn/references/#bengio-2003) 在词级语料上使用这个架构（另加从 $u$ 到 $z$ 的直接线性连接），报告的测试困惑度大幅低于当时最好的平滑 n 元语法模型。我们的字符级复现（例题 2.6）显示，在小数据量时有同样的效果，而在大数据量时效果消失——原论文不需要展示这一点，但它阐明了嵌入*为什么*有帮助。

## 2.3.5 最优低秩近似

低秩模型在什么情况下能妥善近似一张大表？奇异值分解给出精确的诊断方式。任何秩为 $r$ 的实矩阵 $A\in\mathbb R^{m\times n}$ 都可写成

$$
A=\sum_{i=1}^{r}\sigma_i u_i v_i^{\top},
$$

其中 $u_i$ 彼此标准正交，$v_i$ 也彼此标准正交，而且 $\sigma_1\ge\sigma_2\ge\dots\ge\sigma_r>0$。每一项都是一个秩一模式，$\sigma_i$ 则衡量其强度。

**定理 2.1（Eckart–Young，[Eckart & Young 1936](/zh-cn/references/#eckart-1936)）。** 对每个 $d < r$，截断 $A_d = \sum_{i=1}^{d}\sigma_iu_iv_i^{\top}$ 在所有秩至多为 $d$ 的矩阵 $B$ 中使 $\lVert A - B\rVert_F$ 最小，最小值为 $\sqrt{\sum_{i>d}\sigma_i^2}$。

（证明见 §2.5.5 ★。）若共现矩阵的奇异值迅速衰减，定理便告诉我们：低秩近似的 Frobenius 误差很小。但它**没有**说明尾部为什么小，也没有证明被舍弃的方向就是噪声；这种解读还需要采样模型或留出数据证据。古典的*潜在语义分析（latent semantic analysis）*会对词–文档矩阵计算这项截断（[Deerwester et al. 1990](/zh-cn/references/#deerwester-1990)）。学习出的嵌入一般不会直接计算 SVD。对一个重要特例，[Levy & Goldberg (2014)](/zh-cn/references/#levy-2014) 证明，skip-gram 目标在最优点会隐式分解一个位移后的逐点互信息矩阵。因此，低秩分解是理解某些嵌入目标的有用视角，不是涵盖所有学习表示的恒等关系。

## 2.3.6 不变性与等变性

表示可能让学习变容易的第二种方式，是**舍弃任务不依赖的东西**。令 $g$ 为输入的变换——平移一张图像、置换一个集合，或一致地替故事中的朋友换名字。

- 若 $\phi(g\cdot x) = \phi(x)$，表示 $\phi$ 对 $g$ 是**不变的（invariant）**：输入被变换时，表示不改变。
- 若 $f(g\cdot x) = g\cdot f(x)$，映射 $f$ 对 $g$ 是**等变的（equivariant）**：变换输入，会以对应的方式变换输出。

两者的差异在于理想输出应该如何变化。图像分类时，平移输入不应改变类别，因此需要不变性。逐像素分割时，输出遮罩应该跟着图像一起平移，因此需要等变性。在单元 04 明确采用的边界惯例下，卷积是对平移等变的线性映射（定理 4.2）；该单元也证明，没有位置信息的注意力对位置置换是等变的（命题 4.2）。§2.5.4 则通过对有限群取平均，给出一般的不变构造。

硬币的另一面是 §2.7.1：**舍弃了任务所需信息的表示，下游任何学习者都无法修补**。因此，不变性是一项与任务相关的赌注。让数字识别器对小幅平移不变可能有帮助；若让它对所有旋转都不变，却可能把 `6` 与 `9` 视为相同，直接破坏答案。
