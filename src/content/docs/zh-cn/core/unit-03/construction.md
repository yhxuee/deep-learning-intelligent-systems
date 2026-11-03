---
title: "3.3 数学构建：从链式法则到反向传播与现代优化器"
description: 计算图、前向与反向模式微分及其成本分析、矩阵运算与 softmax 的向量–雅可比乘积、MLP 的反向传播、随机梯度、动量、Adam 与 AdamW，以及自动微分引擎的设计。
sidebar: { label: "3.3 数学构建", order: 3 }
---

## 3.3.1 把程序看成计算图

任何从参数算出损失的程序，都会执行一连串**基本运算**——加法、乘法、$\tanh$、$\exp$、矩阵乘法——每一个都从先前的值产生中间值。记录实际执行的运算与其数值，就得到一条**求值轨迹（evaluation trace）**。循环会展开成重复运算；分支则记录实际走过的路径。分支边界是否可微，是另一个问题（§3.7.1）。

同一条轨迹会有两种读法。**前向传递**问：“这个节点算出什么值？”**反向传递**问：“若这个节点改变一点，最终损失会怎么改变？”分开这两个问题，会让以下符号容易理解许多。

以单一神经元搭配平方误差损失为例，它是本节的贯穿范例：

$$
z = w_1x_1 + w_2x_2 + b,\qquad a = \tanh z,\qquad L = \tfrac12(a - t)^2 .
$$

它的轨迹，每个中间值都有名字：

| 步骤 | 运算 | 在 $x = (1,2)$、$w = (0.5,-0.25)$、$b = 0.1$、$t = 0$ 时的值 |
|---|---|---|
| $v_1$ | $w_1\cdot x_1$ | 0.5 |
| $v_2$ | $w_2\cdot x_2$ | −0.5 |
| $v_3$ | $v_1 + v_2$ | 0.0 |
| $z$ | $v_3 + b$ | 0.1 |
| $a$ | $\tanh z$ | 0.099668 |
| $d$ | $a - t$ | 0.099668 |
| $L$ | $\tfrac12 d^2$ | 0.0049669 |

从每个值画一个箭头，指向每个直接使用它的后续值，就得到这次执行的**有向无环图**（directed acyclic graph, DAG），也就是**计算图（computation graph）**。输入与参数是起点；标量损失是终点。每条边 $p\to c$ 都带有一个由单一基本运算决定的**局部导数** $\partial c/\partial p$：若 $c=pq$，固定 $q$ 后就是 $q$；若 $c=\tanh p$，就是 $1-\tanh^2p$；若 $c=\tfrac12p^2$，就是 $p$。

由 §3.2.6 的链式法则，$\partial L/\partial w_1$ 是从 $w_1$ 到 $L$ 的**所有路径**上**局部导数乘积的总和**。这里只有一条路径 $w_1\to v_1\to v_3\to z\to a\to d\to L$，乘积为 $x_1\cdot1\cdot1\cdot(1-a^2)\cdot1\cdot d$。真实网络的路径多到天文数字。以下两种算法组织这个总和，使它永远不必被逐一枚举。

## 3.3.2 前向模式：让导数跟着值往前推

先固定起点空间中的一个方向——第一个例子只让 $w_1$ 增加一单位，其他起点不变。对每个中间值 $v$，定义其**切向量（tangent）** $\dot v=\partial v/\partial w_1$。设 $\dot w_1=1$，其他起点 $\dot u=0$。链式法则会在计算数值的同时，从父节点的切向量算出每个新切向量：

$$
\dot v_c = \sum_{p\,\in\,\text{parents}(c)} \frac{\partial v_c}{\partial v_p}\;\dot v_p .
$$

在轨迹上：$\dot v_1 = x_1\dot w_1 = 1$、$\dot v_2 = 0$、$\dot v_3 = 1$、$\dot z = 1$、$\dot a = (1-a^2)\dot z = 0.990066$、$\dot d = 0.990066$、$\dot L = d\,\dot d = 0.098678$。这就是 $\partial L/\partial w_1$。

更一般地说，任意起点切向量会指定方向 $r$，一次传递计算雅可比–向量乘积 $Jr$。前向模式可以流式数值与切向量，不需存储完整的反向轨迹，但仍要携带切向信息。一次标量方向传递只得到一个方向导数。若要分别取得 $\partial L/\partial w_2$ 与 $\partial L/\partial b$，就得换坐标起始值重跑：全部 $n$ 个参数偏导数需要 **$n$ 次传递**。这是依链式法则得到的精确微分，但当标量损失依赖许多参数时，传递次数仍不理想。

## 3.3.3 反向模式：从损失把导数往回拉

反向模式把问题倒过来。它不问一个起点方向如何改变所有后续值，而是问最终标量如何随每个中间值改变。定义**伴随值（adjoint）**

$$
\bar v = \frac{\partial L}{\partial v},
$$

即最终损失对该中间值的敏感度。上方的横线不是平均符号；它标记反向模式的导数。

**递推式的推导。** 从 $v_p$ 到 $L$ 的每一条路径，都以一条从 $v_p$ 到它某个子节点 $v_c$ 的边开始。依照这第一条边把路径分组。以 $p\to c$ 开头的路径，贡献为 $\frac{\partial v_c}{\partial v_p}$ 乘上从 $v_c$ 到 $L$ 所有路径的总和，而那正是 $\bar v_c$。对所有子节点加总：

$$
\boxed{\;\bar v_p = \sum_{c\,\in\,\text{children}(p)} \bar v_c\;\frac{\partial v_c}{\partial v_p}\;}
\qquad\text{且}\qquad \bar L = 1 .
$$

要计算这个递推，某节点的所有子节点都必须先送回贡献，该节点才能继续往前传递伴随值。因此要依**反向拓扑排序（topological order）**走访——从损失往参数走。抵达起点时，这次执行所使用的每个参数都有 $\bar\theta_j=\partial L/\partial\theta_j$。“一次得到”是指一次反向走访，不是单一常数时间的机器指令。

<div class="dlis-widget not-content" data-widget="backprop-neuron">交互图需要 JavaScript；下方的表格包含相同的信息。</div>

**在轨迹上**（局部导数使用存储的前向值）：

| 伴随值 | 计算方式 | 值 |
|---|---|---|
| $\bar L$ | 起始值 | 1 |
| $\bar d$ | $\bar L\cdot d$ | 0.099668 |
| $\bar a$ | $\bar d\cdot1$ | 0.099668 |
| $\bar z$ | $\bar a\,(1-a^2)$ | 0.098678 |
| $\bar v_3,\ \bar b$ | $\bar z\cdot 1$ | 0.098678 |
| $\bar v_1,\ \bar v_2$ | $\bar v_3\cdot1$ | 0.098678 |
| $\bar w_1$ | $\bar v_1\,x_1$ | 0.098678 |
| $\bar w_2$ | $\bar v_2\,x_2$ | 0.197356 |

一次反向扫描就得到全部三个参数的梯度。注意每个实现都必须遵守的两个特点：

- **累加。** 如果一个值有多个子节点，它的伴随值是*总和*，每个子节点贡献一份。所以实现会把伴随值初始化为零，并把每份贡献**加上去**（$\mathrel{+}=$）。若写成赋值（$=$），就只留下最后一个子节点的贡献——这是最常见的自动微分错误。
- **存储中间值。** 局部导数（$d$、$1-a^2$、$x_1$、$x_2$）要用到前向传递的值，这些值必须保留到反向传递用到它们为止。

**算法（反向模式微分）。**

1. *前向：* 执行程序，记录每个运算、它的输入与输出。
2. 把所有伴随值设为 0；设 $\bar L = 1$。
3. *反向：* 对每个记录的运算 $c = \varphi(p_1,\dots,p_r)$，依相反顺序：对每个输入，$\bar p_i \mathrel{+}= \bar c\cdot\partial\varphi/\partial p_i$。
4. 读出参数的 $\bar\theta$。

:::note[历史]
反向模式微分最早在数值分析中发展——见于 Linnainmaa 1970 年的硕士论文与他 1976 年关于舍入误差传播的论文（[Linnainmaa 1976](/zh-cn/references/#linnainmaa-1976)）——后来由 [Rumelhart、Hinton & Williams 1986](/zh-cn/references/#rumelhart-1986) 以“反向传播（back-propagation）”之名推广到多层神经网络的训练。一般理论见 [Griewank & Walther 2008](/zh-cn/references/#griewank-2008)。
:::

## 3.3.4 反向模式的成本

**时间。** 反向传递会走访每个记录的运算一次，并对每条输入边应用局部 VJP 规则、累加一项贡献。在标准成本模型中，若每项基本运算的导数规则成本至多是原运算的固定倍数（通常更少：$\tanh'$ 会重用存储的输出），对整条轨迹加总可得

$$
\text{cost}(\nabla L) \;\le\; c\cdot\text{cost}(L),
$$

其中 $\operatorname{cost}(\nabla L)$ 包含前向与反向工作，而 $c$ 不会额外多出参数数量这个因子。这就是**廉价梯度原理（cheap gradient principle）**；[Griewank & Walther（2008）](/zh-cn/references/#griewank-2008)对指定的基本运算集合与成本模型给出明确上界。这项陈述谈的是比例：较大模型的前向传递较昂贵，梯度当然也较昂贵；反向模式省掉的是“为每个参数各跑一次前向计算”。

**内存。** 代价是存储：反向传递需要的前向值。对神经网络而言，主要是批量中每个样本在每一层的活化值——往往是训练中最大的内存开销。检查点重算（§3.7.6）以重新计算换取内存。

**一般情况下的前向 vs. 反向。** 对 $f:\mathbb{R}^n\to\mathbb{R}^m$，以坐标为起始方向的前向模式，每次会给出雅可比矩阵的一列（column）；以坐标为起始伴随值的反向模式，每次给出一行（row）。所以建立*完整*雅可比矩阵，需要 $n$ 次前向扫描或 $m$ 次反向扫描。训练损失有 $m=1$，反向模式一次就能取得完整梯度；若函数输入很少、输出很多，前向模式可能更合适。通常我们只需要乘积 $Jr$ 或 $J^\top v$，不会真的建立完整雅可比矩阵。例题 3.8 会计算扫描次数。

## 3.3.5 向量与矩阵：雅可比矩阵与向量–雅可比乘积

真实网络处理的是向量与矩阵，实现也在这个层次上进行，而不是标量。对 $y = f(x)$，$x\in\mathbb{R}^n$、$y\in\mathbb{R}^m$，**雅可比矩阵（Jacobian）** $J\in\mathbb{R}^{m\times n}$ 的元素为 $J_{ij} = \partial y_i/\partial x_j$，合成函数的链式法则则变成矩阵乘法 $J_{f\circ g} = J_fJ_g$。

对一个向量值运算，伴随值递推逐元素写出为 $\bar x_j = \sum_i\bar y_i\,\partial y_i/\partial x_j$，即

$$
\bar x = J^{\top}\bar y .
$$

这是**向量–雅可比乘积（vector–Jacobian product, VJP）**的列向量写法：较传统的行记号写成 $\bar y^\top J$，其转置就是 $J^\top\bar y$。反向模式不必建立可能极大的 $J$；每个运算只需提供一条把输入伴随值 $\bar y$ 映成 $J^\top\bar y$ 的规则。以下四条规则涵盖本书验证的大部分情况。

**(a) 线性映射。** $y = Wx$，$W\in\mathbb{R}^{m\times n}$，即 $y_i = \sum_jW_{ij}x_j$。则 $\partial y_i/\partial x_j = W_{ij}$ 且 $\partial y_i/\partial W_{ij} = x_j$（$W_{ij}$ 不影响其他输出）。因此

$$
\bar x_j = \sum_i\bar y_iW_{ij}\;\Longrightarrow\;\bar x = W^{\top}\bar y,
\qquad
\bar W_{ij} = \bar y_i\,x_j\;\Longrightarrow\;\bar W = \bar y\,x^{\top},
\qquad
\bar b = \bar y \text{（对 } y = Wx + b\text{）}.
$$

对以行存储的批量，$Y = XW$，$X\in\mathbb{R}^{B\times n}$、$W\in\mathbb{R}^{n\times m}$，同样的推理得到

$$
\bar W = X^{\top}\bar Y,\qquad \bar X = \bar Y\,W^{\top}.
$$

**形状规则：** 一个张量的伴随值与该张量形状相同。$\bar W$ 必须是 $n\times m$；$X^\top$（$n\times B$）与 $\bar Y$（$B\times m$）唯一能得到这个形状的乘积就是 $X^\top\bar Y$。在算任何数值之前先检查形状，就能抓到大部分的转置错误。

**(b) 逐元素非线性。** $y_i = \varphi(x_i)$ 的雅可比矩阵是对角的，所以 $\bar x = \bar y\odot\varphi'(x)$。常用导数：$\tanh' = 1-\tanh^2$；logistic 函数 $\sigma' = \sigma(1-\sigma)$；$\mathrm{ReLU}'(x) = \mathbb 1[x>0]$（在 $0$ 处依惯例取值；§3.7.1）。

**(c) softmax 加交叉熵。** 对分数 $s\in\mathbb{R}^V$ 与目标类别 $y$，$p = \mathrm{softmax}(s)$、$L = -\log p_y$。把 $L$ 写成容易微分的形式：

$$
L = -s_y + \log\sum_{j}e^{s_j}.
$$

则 $\partial L/\partial s_k = -\mathbb 1[k = y] + e^{s_k}/\sum_je^{s_j} = p_k - \mathbb 1[k=y]$，即

$$
\boxed{\;\bar s = p - e_y\;}
$$

其中 $e_y$ 是目标的 one-hot 向量。每个分数梯度都在 $[-1,1]$ 内，而且各分量总和为零，因为对所有分数加上同一常数不会改变 softmax。这条简洁规则计算便宜，但前向损失仍必须使用稳定的 log-sum-exp；直接计算 $e^{s_j}$ 可能溢位。单独的 softmax 雅可比矩阵为 $\mathrm{diag}(p)-pp^\top$；单元 04 会在注意力中使用其 VJP。

**(d) 复制、广播与加总。** 如果一个值被复制到多个用途（扇出），它的伴随值就是各用途贡献的总和。若 $y=\sum_i x_i$，则每个 $\bar x_i=\bar y$。这些运算彼此转置：前向的广播或重复使用，会在反向变成加总；前向的归约，则在反向变成广播。这就是数组形式的累加规则。

## 3.3.6 多层网络中的反向传播

一个有 $L$ 个隐藏层的多层感知器（multilayer perceptron, MLP）计算

$$
h_0 = x,\qquad z_\ell = W_\ell\,h_{\ell-1} + b_\ell,\quad h_\ell = \varphi(z_\ell)\ \ (\ell = 1,\dots,L),\qquad \hat y = W_{L+1}h_L + b_{L+1},
$$

接着计算损失 $\ell(\hat y, y)$。由上往下应用规则 (a)、(b)、(c)：

**算法（MLP 的反向传播，单一样本）。**

1. 前向传递；存储每个 $z_\ell$ 与 $h_\ell$。
2. $\bar{\hat y} = \partial\ell/\partial\hat y$（例如平方误差 $\tfrac12\|\hat y - y\|^2$ 时为 $\hat y - y$；softmax 交叉熵时为 $p - e_y$）。
3. 输出层：$\bar W_{L+1} = \bar{\hat y}\,h_L^{\top}$、$\bar b_{L+1} = \bar{\hat y}$、$\bar h_L = W_{L+1}^{\top}\bar{\hat y}$。
4. 对 $\ell = L, L-1, \dots, 1$：
   $\bar z_\ell = \bar h_\ell\odot\varphi'(z_\ell)$；$\ \bar W_\ell = \bar z_\ell\,h_{\ell-1}^{\top}$；$\ \bar b_\ell = \bar z_\ell$；$\ \bar h_{\ell-1} = W_\ell^{\top}\bar z_\ell$。

对一个批量，参数在样本间共用，所以它们的梯度是各样本梯度的**总和**（或平均）——由批量规则 $\bar W = X^{\top}\bar Y$ 一次算出。例题 3.3 对一个 2–2–1 网络手算这个算法，并以有限差分检查其中一个元素。

看第 4 步：反向信号 $\bar h_{\ell-1} = W_\ell^{\top}\bigl(\bar h_\ell\odot\varphi'(z_\ell)\bigr)$ 每经过一层，就乘上一个转置的权重矩阵与一个导数的对角矩阵。经过许多层，这个矩阵乘积可能以等比方式缩小或放大。这个观察推动了 §3.7.4 与 §3.8，并在单元 05 以“随*时间*传递的梯度”再次出现。

## 3.3.7 随机梯度

训练损失是对全部 $N$ 个样本的平均，所以它的梯度要扫过整个数据集。取而代之，从中均匀随机地（放回地）抽取 $B$ 个索引 $i_1,\dots,i_B$ 组成一个**小批量（minibatch）**，并使用

$$
g_B = \frac1B\sum_{b=1}^{B}\nabla\ell_{i_b}(\theta).
$$

**在上述采样方式下，它是无偏的（unbiased）。** 固定 $\theta$ 时，每个放回抽样的索引都在 $\{1,\dots,N\}$ 上均匀，所以 $\mathbb E[\nabla\ell_{i_b}]=N^{-1}\sum_i\nabla\ell_i=\nabla L$，由线性性得 $\mathbb E[g_B]=\nabla L$。“固定 $\theta$”很重要：更新后，参数就改变了。

**它的噪声以 $1/\sqrt B$ 缩小。** 令 $\Sigma = \mathrm{Cov}(\nabla\ell_i)$ 为单一采样梯度的协方差。$B$ 个样本彼此独立，而 $B$ 个独立复本之平均的协方差为 $\Sigma/B$：

$$
\mathrm{Cov}(g_B) = \frac{\Sigma}{B},\qquad \mathbb E\|g_B - \nabla L\|^2 = \frac{\mathrm{tr}\,\Sigma}{B}.
$$

把 $B$ 增为四倍，这项均方根抽样误差减半，算术工作则增为四倍；并行硬件可能改变实际耗时的比例。**随机梯度下降法（stochastic gradient descent, SGD）**在更新中以 $g_B$ 取代 $\nabla L$。每次更新都比全数据梯度便宜，却带有噪声。在 §3.5.6 明确列出的假设下，这种噪声会为固定步长造成误差底限，而递减步长可以移除它（[Robbins & Monro 1951](/zh-cn/references/#robbins-1951)；[Bottou et al. 2018](/zh-cn/references/#bottou-2018)）。

实际的数据加载器常在每个**训练轮次（epoch）**打乱一次，再以不放回方式取用样本。若固定 $\theta$，完整一轮的平均就是全数据梯度；但训练时，相邻小批量彼此依赖，而且参数不断改变。因此，上面的 $\Sigma/B$ 证明精确适用于彼此独立的放回抽样，不能原封不动套到每一种数据加载实现。

## 3.3.8 动量与自适应步长

§3.2.3 说明安全步长受*最大*曲率限制。如果不同方向的曲率差异很大，对陡峭方向安全的步长，对平坦方向就太小，于是梯度下降法会在陡峭方向上来回锯齿，在平坦方向上缓慢爬行（图 3.2，以及 §3.5.4 的精确分析）。由此产生两类补救方法。

**动量（momentum；重球法）。** 维持一个累积过去梯度的“速度”（[Polyak 1964](/zh-cn/references/#polyak-1964)）：

$$
v_{k+1} = \beta v_k - \eta\nabla L(\theta_k),\qquad \theta_{k+1} = \theta_k + v_{k+1},\qquad 0\le\beta<1 .
$$

在 $v_0=0$ 时展开速度，可得 $v_{k+1}=-\eta\sum_{j=0}^{k}\beta^{k-j}\nabla L(\theta_j)$，也就是过去梯度的指数加权和，有效窗口约为 $1/(1-\beta)$ 步（与单元 05 的移动平均是同一个数学对象）。符号持续一致的梯度分量会累积；交替的分量则可能抵销。“重球”名称来自惯性与摩擦的物理类比，但真正的定义仍是上方的离散更新。对特征值落在已知区间内的正定二次函数，调好的动量会把最坏情况收缩因子从 $(\kappa-1)/(\kappa+1)$ 改善到 $(\sqrt\kappa-1)/(\sqrt\kappa+1)$（§3.5.7）——当 $\kappa=10$，从 0.818 降到 0.519。

![在曲率分别为 1 与 10 的二次函数上，从 (1, 1) 出发的梯度下降法与重球动量路径：梯度下降法在陡峭方向上锯齿前进；动量用较少步数到达最小值。](/figures/u03-quadratic-paths.svg)

*图 3.2——病态条件（$\kappa = 10$）：使用最优固定步长的梯度下降法 vs. 调好参数的动量。例题 3.5 给出数字。*

<div class="dlis-widget not-content" data-widget="gd-2d">交互图需要 JavaScript；上方的静态图与表格包含相同的信息。</div>

**逐坐标的自适应步长：Adam。** 第二个想法是给每个坐标各自的步长，依它的梯度通常有多大来缩放。Adam（[Kingma & Ba 2015](/zh-cn/references/#kingma-2015)）维护梯度及其逐元素平方的指数移动平均：

$$
\begin{aligned}
m_k &= \beta_1m_{k-1} + (1-\beta_1)g_k, & v_k &= \beta_2v_{k-1} + (1-\beta_2)g_k^{2},\\
\hat m_k &= \frac{m_k}{1-\beta_1^{k}}, & \hat v_k &= \frac{v_k}{1-\beta_2^{k}}, \qquad
\theta_{k} = \theta_{k-1} - \eta\,\frac{\hat m_k}{\sqrt{\hat v_k}+\epsilon}.
\end{aligned}
$$

*为什么要偏差校正（bias correction）？* 由 $m_0 = 0$，$m_k = (1-\beta_1)\sum_{j=1}^{k}\beta_1^{k-j}g_j$。若梯度为常数 $g_j = g$，这个和等于 $(1-\beta_1^{k})g$——早期太小，因为平均是从零开始的。除以 $1-\beta_1^k$ 正好消除这个偏差；$v_k$ 亦同。

*这个更新在做什么。* 忽略 $\epsilon$ 时，比值 $\hat m/\sqrt{\hat v}$ 没有单位：把所有梯度乘上常数 $c>0$，比值不变。第一步有 $\hat m_1=g_1$、$\hat v_1=g_1^2$，所以非零坐标会移动

$$
-\eta\frac{g_1}{|g_1|+\epsilon},
$$

只有在 $|g_1|\gg\epsilon$ 时，才近似 $-\eta\operatorname{sign}(g_1)$（例题 3.7）。之后若梯度方向持续一致，$|\hat m|$ 可以接近 $\sqrt{\hat v}$；符号改变或变动很大时，这个比值会缩小。这是一种实用的逐坐标归一化启发法，不是反曲率的精确估计，也不是普遍的收敛保证。

**AdamW：解耦的权重衰减。** 在损失加入 $\tfrac\lambda2\|\theta\|^2$，会在梯度加入 $\lambda\theta$。Adam 接着把这一项混入两个动差估计，并对它做自适应缩放，所以所得更新不等同于普通 SGD 权重衰减所产生的简单乘法收缩。AdamW 改为在自适应梯度变换之外施加收缩（[Loshchilov & Hutter 2019](/zh-cn/references/#loshchilov-2019)）：

$$
\theta_k = \theta_{k-1} - \eta\Bigl(\frac{\hat m_k}{\sqrt{\hat v_k}+\epsilon} + \lambda\,\theta_{k-1}\Bigr).
$$

等价地说，显示的公式会先把旧参数乘上 $1-\eta\lambda$，再应用自适应更新。AdamW 是 Transformer 常用的优化器，也是主干项目相关阶段采用的选择；流行的工程惯例不等于它对每种模型都最优。[AdamW Atlas 节点](/zh-cn/atlas/nodes/opt-adamw/)列出了它的不变量与常见实现错误。

## 3.3.9 构建引擎

实验室要求你为标量实现反向模式。设计直接由 §3.3.3 而来：

- **一种节点类型**（`Value`）存储它的数值（`data`）、它的伴随值（`grad`，初始为 0）、对父节点的参照，以及一个小函数：给定自己的伴随值，依规则 $\bar p \mathrel{+}= \bar c\,\partial c/\partial p$ 把贡献加到父节点的伴随值上。
- **每个运算**（`+`、`*`、`tanh`……）建立一个新节点，并附上正确的局部导数函数，同时捕捉它需要的值（例如 $\tanh$ 捕捉输出 $t$，使用 $1-t^2$）。
- **`backward()`** 对输出可达的计算图建立拓扑排序，把输出的伴随值设为 1，再以相反顺序调用每个节点的局部函数。实验室要求使用迭代走访，因为递推深度优先搜索可能在深达 10,000 个节点的图上使语言调用堆叠溢位。
- **在这个引擎中，归零是调用者的责任。** 伴随值会刻意累加，不论是在同一张扇出图内，还是在共用叶节点的多次反向调用之间。因此，除非本来就想跨批量累积梯度，优化循环必须在计算下一次更新前重设参数伴随值。

[实验室](https://github.com/yhxuee/deep-learning-intelligent-systems/tree/main/labs/unit-03-autodiff)正是测试这些性质：例题的数字、扇出下的累加（`a*a`、`a+a`）、在随机表达式图上与中心差分一致、线性性、深层图，以及跨图的累加。通过测试，验证的是这个有明确范围的标量实现。PyTorch 与 JAX 等张量框架采用相同的反向模式想法，每个基本运算有一条 VJP 规则，此外还要处理广播、设备、突变规则、控制流程，以及大量性能优化。从单元 04 起，主干项目就使用这些框架。
