---
title: "2.3 数学构建"
description: 线性可分性、把嵌入查找看成矩阵乘积及其散射加法梯度、秩 d 分解式二元语法及其梯度、神经 n 元语法、最优低秩近似，以及不变性与等变性的精确定义。
sidebar: { label: "2.3 数学构建", order: 3 }
---

本节将 §2.2 的几何直觉精确转化为现代深度学习的张量代数系统。在阅读每一个矩阵方程时，请始终遵循一套四步思维协议：

1. **命名本体：** 说出每一个符号所代表的具体数学或物理对象；
2. **形状校验（Shape Check）：** 严格检查每个因子的张量维度（Dimensions）；
3. **参数复用识别：** 找出哪些底层权重在正向或反向传播的多个路径中被同时复用；
4. **沿着计算图回溯：** 搞清楚梯度反向流动的每一条分支与汇聚点。

全书反复出现的关键维度符号约定如下：词表大小 $V$、嵌入向量维度 $d$、批量大小 $N$、上下文长度 $k$，以及神经网络隐藏层宽度 $H$。维度检查是神经网络实现中最有效的调试工具，能及早发现张量转置错误或计算图连接错误。

## 2.3.1 线性读出与可分性

令输入对象 $x$ 映射至高维特征空间中的向量 $\phi(x) \in \mathbb{R}^d$。一个定义在该表示上的二元线性分类器，其决策边界由超平面参数 $(w, b)$ 决定，根据判别式 $w^{\top}\phi(x) + b$ 的代数符号进行正负类别预测。

若存在至少一组参数 $(w, b) \in \mathbb{R}^d \times \mathbb{R}$，能够使给定数据集中的每一个样本均被正确划分，则称该标注数据集**在表示 $\phi$ 下是线性可分的（Linearly Separable）**。
请注意：**可分性并不是原始数据对象自身的固有属性，而是数据在特定几何表示 $\phi$ 下的相互关系。** 同一组数据（例如 XOR 异或逻辑），在原始坐标下不可分，但在经过非线性变换 $\phi$ 投影到新空间后，就可以变得线性可分。

若任务需要输出校准的类别概率分布，二元分类使用 **Logistic（Sigmoid）** 读出层；多类别分类（如预测词表中 $V$ 个候选词）则使用 **Softmax** 读出层：

$$
q(y\mid x) = \frac{\exp(w_y^{\top}\phi(x) + b_y)}{\sum_{y'}\exp(w_{y'}^{\top}\phi(x) + b_{y'})}.
$$

### 为什么含有指数函数的 Softmax 依然被称为“线性读出”？
Softmax 运算中显然包含非线性的指数运算 $\exp(\cdot)$ 与归一化分母，为何它被归类为线性读出？
考察任意两个候选类别 $y$ 与 $y'$ 的概率比值，它们共同的归一化分母在相除时对消：

$$
\ln\frac{q(y\mid x)}{q(y'\mid x)}
=(w_y-w_{y'})^{\top}\phi(x)+(b_y-b_{y'}).
$$

取自然对数后，**任意一对类别之间的对数胜算（Log-Odds / Logit Difference），严格等于特征向量 $\phi(x)$ 的线性仿射函数。**
这给出了“任务在该表示下线性可解”的具体含义：一个优秀的表征 $\phi(x)$，使得不同类别之间的对数胜算曲面，正好与表示空间中的超平面平行。

## 2.3.2 嵌入查找是一个矩阵乘积

令 $E \in \mathbb{R}^{V\times d}$ 为模型的词嵌入矩阵（Embedding Table），其第 $a$ 行存储了词汇符号 $a$ 所对应的 $d$ 维连续几何向量。
为了避免符号混淆，我们用标准基向量 $\mathbf{u}_a \in \{0, 1\}^V$ 表示符号 $a$ 的 One-Hot（独热）编码，并用 $e_a = E[a] \in \mathbb{R}^d$ 表示其稠密嵌入向量。则符号查找可以严格写作转置矩阵与独热向量的矩阵乘积：

$$
E^{\top}\mathbf u_a=e_a \qquad(E\text{ 的第 }a\text{ 行，以列向量写出}).
$$

在一个包含 $N$ 个样本的批量中，将各样本输入符号 $x_1, \dots, x_N$ 的独热列向量堆叠为大矩阵 $U \in \{0, 1\}^{N\times V}$。则整批样本的查表过程在代数上等价于矩阵乘法：

$$
H=UE\in\mathbb R^{N\times d}.
$$

维度校验：$(N\times V) \times (V\times d) = N\times d$。在实际工程实现中，底层框架直接通过内存索引寻址（即 `E[x]`），不会在内存中真正分配几乎全为 0 的稀疏矩阵 $U$。但将其写作等价的矩阵乘法，能使反向传播的链式法则推导一目了然。

**命题 2.1（查找操作的梯度传播）。** 若标量损失 $L$ 仅通过批量表示 $H=UE$ 依赖于嵌入矩阵 $E$，且抵达 $H$ 的上游梯度记为 $G = \partial L/\partial H \in \mathbb{R}^{N\times d}$，则损失对底层嵌入矩阵 $E$ 的梯度严格为：

$$
\frac{\partial L}{\partial E} = U^{\top}G,\qquad\text{即}\qquad \frac{\partial L}{\partial E[a]} = \sum_{i:\,x_i = a} G[i].
$$

理解其物理意义：在同一个批量中，若符号 $a$ 在多个不同位置（如第 2 个与第 5 个样本）被重复查阅，这些位置获取的都是同一套参数 $E[a]$。根据多元微积分的链式法则，**所有引用了该参数的下游节点所传回的梯度分量，必须全部累加到该参数行之上。**

*证明。* 由前向定义，$H_{ik} = \sum_{v=1}^V U_{iv} E_{vk}$。对特定的参数元素求偏导：$\partial H_{ik} / \partial E_{vk} = U_{iv}$。
由链式法则，损失对 $E_{vk}$ 的总偏导为所有依赖路径之和：
$\frac{\partial L}{\partial E_{vk}} = \sum_{i=1}^N \frac{\partial L}{\partial H_{ik}} \frac{\partial H_{ik}}{\partial E_{vk}} = \sum_{i=1}^N G_{ik} U_{iv} = \sum_{i=1}^N (U^{\top})_{vi} G_{ik} = (U^{\top}G)_{vk}$。
将此矩阵形式展开到第 $a$ 行：由于 $U_{ia} = 1$ 当且仅当第 $i$ 个样本的输入符号为 $a$（其余为 0），因此矩阵乘积 $U^{\top}G$ 的第 $a$ 行正好等于所有输入等于 $a$ 的样本索引 $i$ 对应的梯度列 $G[i]$ 之和。$\blacksquare$

### 实现细节：散射加法（Scatter-Add）
在深度学习框架底层（NumPy / PyTorch），这项累加运算称为 **Scatter-Add（散射加法）**：`np.add.at(dE, x, G)` 或 `dE.index_add_(0, x, G)`。
在实现中需要特别注意：如果一个批量中多次出现同一个符号，使用普通索引赋值 `dE[x] = G` 会导致后面的梯度覆盖前面的梯度。正确的做法是使用散射累加，将所有出现位置的梯度加总起来。

## 2.3.3 分解式二元语法及其梯度

现在，我们用连续嵌入向量重构单元 01 的二元语法模型。我们为任意“前驱字符 $a$ 接续后继字符 $b$”的转移打分，分数记为未归一化的 logit $z_{ab}$：

$$
z_{ab} = e_a^{\top}Me_b + c_b,\qquad q(b\mid a) = \mathrm{softmax}_b(z_{a\cdot}),
$$

这项双线性打分公式包含清晰的几何含义：
1. **权重绑定（Weight Tying）：** 同一张嵌入矩阵 $E \in \mathbb{R}^{V\times d}$ 在输入端与输出端被同时复用——$e_a = E[a]$ 代表观察到的输入实体，$e_b = E[b]$ 代表候选预测对象；
2. **转移矩阵 $M \in \mathbb{R}^{d\times d}$：** 如果没有矩阵 $M$（即 $M = I$），则打分退化为简单内积 $z_{ab} = e_a^\top e_b + c_b$。为了让转移概率最大化，模型会被迫将频繁相邻的词（如 `q` 与 `u`）的嵌入向量拉向相同方向，但相邻的词在语义上往往截然不同。**矩阵 $M$ 充当了语义转移算子**：它将前驱向量 $e_a$ 投影为期望特征向量 $M^\top e_a$，再与后继候选 $e_b$ 进行内积比对；
3. **后继偏置 $c_b$：** 捕捉符号 $b$ 的边际独立出现频率。

在批量矩阵计算中，令输入索引查出的特征矩阵为 $H = E[x] \in \mathbb{R}^{N\times d}$，则整批样本在全部 $V$ 个候选词上的 logit 矩阵为：

$$
Z=HME^{\top}+\mathbf1c^{\top}\in\mathbb R^{N\times V}.
$$

张量维度核验：$(N\times d) \times (d\times d) \times (d\times V) + (N\times 1)(1\times V) = N\times V$。
核心交互项 $HME^{\top}$ 的代数秩最高不超过 $d$。我们用一个仅包含 $Vd + d^2 + V$ 个参数的低秩几何系统，逼近了原本包含 $V^2$ 个参数的完整离散表格。

令交叉熵损失传回的 logit 误差矩阵为 $\Delta = \partial L/\partial Z = (Q - Y)/N \in \mathbb{R}^{N\times V}$（此处 $Q$ 为 Softmax 预测概率矩阵，$Y$ 为标签的 One-Hot 矩阵，完整证明见单元 03 §3.3.5）。对矩阵方程 $Z = HME^{\top} + \mathbf{1}c^{\top}$ 求各参数的偏导数：

$$
\frac{\partial L}{\partial c} = \Delta^{\top}\mathbf 1,\qquad
\frac{\partial L}{\partial M} = H^{\top}\Delta E,\qquad
\frac{\partial L}{\partial E}\Big|_{\text{输出侧}} = \Delta^{\top}HM,\qquad
\frac{\partial L}{\partial H} = \Delta EM^{\top}.
$$

### 双路径梯度汇聚
由于我们对输入与输出采用了权重绑定，嵌入表 $E$ 同时承担了两个角色，因而会沿着计算图的两条不同分支同时收到反向梯度：
- **输出路径：** 每个候选向量作为矩阵 $E^\top$ 的一行参与了分类比对，直接贡献梯度 $\Delta^\top H M$；
- **输入路径：** 观测到的前驱符号查表形成了 $H$，抵达 $H$ 的梯度为 $\Delta E M^\top$，根据命题 2.1，该梯度必须通过 $U^\top$ 散射加法回输入列。

两条路径在代数上直接相加：

$$
\frac{\partial L}{\partial E} = \Delta^{\top}HM + U^{\top}(\Delta EM^{\top}).
$$

每个推导均遵循矩阵微分的通用法则：“若 $Y = AXB$，则 $\partial L/\partial X = A^\top (\partial L/\partial Y) B^\top$”。读者可用维度进行自检：例如 $H^\top \Delta E$ 的维度为 $(d\times N) \times (N\times V) \times (V\times d) = d\times d$，与矩阵 $M$ 的尺寸吻合。

## 2.3.4 神经 n 元语法

如果上下文包含 $k$ 个连续历史符号 $(x_{t-k}, \dots, x_{t-1})$，神经模型如何避免离散表格的组合爆炸？
经典的 Bengio 神经语言模型（[Bengio et al. 2003](/zh-cn/references/#bengio-2003)）将这 $k$ 个符号的嵌入向量拼接为长向量 $u = [e_{x_{t-k}}; \dots; e_{x_{t-1}}] \in \mathbb{R}^{kd}$，并通过单隐藏层多层感知机（MLP）进行非线性特征合成：

$$
a = W_1^{\top}u + b_1,\quad h = \tanh(a),\quad z = W_2^{\top}h + b_2,\quad q = \mathrm{softmax}(z),
$$

其中第一层投影权重为 $W_1 \in \mathbb{R}^{kd\times H}$，输出层权重为 $W_2 \in \mathbb{R}^{H\times V}$。
模型的总参数量为五项之和：$Vd + kdH + H + HV + V$。
以典型设定 $V = 50, d = 16, k = 4, H = 128$ 为例：
总参数量为 $800 + 8{,}192 + 128 + 6{,}400 + 50 = \mathbf{15{,}570}$ 个参数。
对比之下，同等上下文长度的传统离散计数表，其最坏情况需要 $V^k = 50^4 = \mathbf{6{,}250{,}000}$ 行、共计 3.125 亿个单元格。神经网络大幅压缩了参数规模，使长上下文建模具备了可操作性。

对单一样本，令输出误差向量为 $\delta_z = q - \mathbf{u}_y \in \mathbb{R}^V$。沿着前向计算图依序逆向回溯：

$$
\frac{\partial L}{\partial W_2} = h\,\delta_z^{\top},\quad \frac{\partial L}{\partial b_2} = \delta_z,\quad
\delta_a = (W_2\delta_z)\odot(1 - h\odot h),\quad
\frac{\partial L}{\partial W_1} = u\,\delta_a^{\top},\quad
\frac{\partial L}{\partial u} = W_1\delta_a .
$$

其中 $\odot$ 代表 Hadamard 逐元素相乘，项 $1 - h\odot h$ 是双曲正切激活函数 $\tanh'(a) = 1 - \tanh^2(a)$ 的导数。
最后，将梯度向量 $\partial L/\partial u \in \mathbb{R}^{kd}$ 均匀切分为 $k$ 个长度为 $d$ 的子切片。第 $j$ 个切片对应上下文历史中第 $j$ 个位置所查阅的嵌入列，再次通过 Scatter-Add 散射累加回该符号的嵌入参数中。

## 2.3.5 最优低秩近似与奇异值分解

低秩矩阵分解在什么情况下能够高保真地逼近一个庞大的离散表格？线性代数中的奇异值分解（Singular Value Decomposition, SVD）为我们提供了最严格的分析诊断工具。

任何秩为 $r$ 的实矩阵 $A \in \mathbb{R}^{m\times n}$ 均可精确分解为一组正交秩一外积矩阵的加权和：

$$
A=\sum_{i=1}^{r}\sigma_i u_i v_i^{\top},
$$

其中左奇异向量 $\{u_i\}$ 与右奇异向量 $\{v_i\}$ 分别构成标准正交基底，且奇异值按降序排列：$\sigma_1 \ge \sigma_2 \ge \dots \ge \sigma_r > 0$。

**定理 2.1（Eckart–Young 最优低秩逼近定理，[Eckart & Young 1936](/zh-cn/references/#eckart-1936)）。** 对于任意指定的正整数 $d < r$，在所有秩不超过 $d$ 的实矩阵集合 $\mathcal{M}_d = \{B \in \mathbb{R}^{m\times n} : \mathrm{rank}(B) \le d\}$ 中，截断奇异值矩阵 $A_d = \sum_{i=1}^{d}\sigma_iu_iv_i^{\top}$ 是在 Frobenius 范数意义下距离 $A$ 最近的唯一最优逼近，其最小逼近误差精确等于被舍弃的奇异值平方和的开方：

$$\min_{\mathrm{rank}(B)\le d} \lVert A - B\rVert_F = \lVert A - A_d \rVert_F = \sqrt{\sum_{i=d+1}^{r}\sigma_i^2}.$$

（完整证明见 §2.5.5 ★。）
若一个语言共现矩阵的奇异值光谱呈现出急剧的指数级衰减（即前 $d$ 个奇异值主导了整体能量，而后续奇异值迅速趋于零），Eckart-Young 定理保证了将维度压缩至 $d$ 维只会损失微不足道的信号。现代嵌入算法（如 Word2Vec 与 GloVe）在数学极限下本质上是在隐式分解某种经过重整化的点互信息（PMI）矩阵（[Levy & Goldberg 2014](/zh-cn/references/#levy-2014)）。

## 2.3.6 不变性与等变性：几何先验的设计

表征学习提高效率的另一条途径，是**通过架构设计滤除任务无关的干扰信息**。令 $g$ 代表输入空间上的某种对称变换操作（例如图像的平移、旋转、符号序列的置换）：

- **不变性（Invariance）：** 若对任意群操作 $g$，特征映射均满足 $\phi(g\cdot x) = \phi(x)$，则称表征 $\phi$ 对变换 $g$ 具有不变性。变换输入，特征输出保持不变；
- **等变性（Equivariance）：** 若映射函数满足 $f(g\cdot x) = g\cdot f(x)$，则称 $f$ 对群操作 $g$ 具有等变性。变换输入，输出以相应的方式同步变换。

两者的选择取决于下游任务的需求：
- 在图像分类任务中，物体出现在画面的左侧还是右侧不改变类别标签，因此分类器通常需要**平移不变性**；
- 在像素级图像分割或物体检测中，物体在画面中平移，其分割掩码必须同步平移，因此网络需要**平移等变性**（如卷积神经网络的平移等变性定理 4.2）；
- 在单元 04 中我们将看到：未注入位置编码的自注意力机制，对输入序列的顺序置换具有严格的**置换等变性（Permutation Equivariance）**（命题 4.2）。

需要注意不变性的权衡（§2.7.1）：**一旦表征在前期滤除了某项特征信息，下游网络就无法再恢复它。** 引入适当的归纳偏差是有益的：为手写数字识别模型引入适度的平移不变性是有益的；但若引入 180 度旋转不变性，模型就无法区分数字 `6` 与 `9`。
