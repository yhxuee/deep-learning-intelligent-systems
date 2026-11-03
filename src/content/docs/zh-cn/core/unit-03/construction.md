---
title: "3.3 数学构建：从链式法则到反向传播与现代优化器"
description: 计算图、前向与反向模式微分及其成本分析、矩阵运算与 softmax 的向量–雅可比乘积、MLP 的反向传播、随机梯度、动量、Adam 与 AdamW，以及自动微分引擎的设计。
sidebar: { label: "3.3 数学构建", order: 3 }
---

## 3.3.1 把程序看成计算图

任何从参数算出损失的程序，都执行一连串**基本运算**——加法、乘法、$\tanh$、$\exp$、矩阵乘法——每一个都从先前的值产生一个新的中间值。按顺序写下这些值，就得到一条**求值轨迹（evaluation trace）**。

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

从每个值画一个箭头，指向每个由它算出来的值，就得到一个**有向无环图**（directed acyclic graph, DAG）：**计算图（computation graph）**。输入与参数是它的起点；损失是终点。每个箭头 $p\to c$ 都带着一个**局部导数** $\partial c/\partial p$，只依赖该基本运算与它的输入：对 $c = p\cdot q$ 是 $q$；对 $c = \tanh p$ 是 $1-\tanh^2 p$；对 $c = \tfrac12p^2$ 是 $p$。

由 §3.2.6 的链式法则，$\partial L/\partial w_1$ 是从 $w_1$ 到 $L$ 的**所有路径**上**局部导数乘积的总和**。这里只有一条路径 $w_1\to v_1\to v_3\to z\to a\to d\to L$，乘积为 $x_1\cdot1\cdot1\cdot(1-a^2)\cdot1\cdot d$。真实网络的路径多到天文数字。以下两种算法组织这个总和，使它永远不必被逐一枚举。

## 3.3.2 前向模式：让导数跟着值往前推

固定一个输入方向——例如我们要对 $w_1$ 的导数。对每个中间值 $v$，定义它的**切向量（tangent）** $\dot v = \partial v/\partial w_1$。设 $\dot w_1 = 1$，其他起点 $\dot u = 0$。由链式法则，每个值的切向量在计算该值的同时，由其父节点的切向量算出：

$$
\dot v_c = \sum_{p\,\in\,\text{parents}(c)} \frac{\partial v_c}{\partial v_p}\;\dot v_p .
$$

在轨迹上：$\dot v_1 = x_1\dot w_1 = 1$、$\dot v_2 = 0$、$\dot v_3 = 1$、$\dot z = 1$、$\dot a = (1-a^2)\dot z = 0.990066$、$\dot d = 0.990066$、$\dot L = d\,\dot d = 0.098678$。这就是 $\partial L/\partial w_1$。

前向模式（forward mode）简单、不需要额外内存，但一次传递只得到对**一个**输入方向的导数。要得到 $\partial L/\partial w_2$ 与 $\partial L/\partial b$，就得换起始值再跑一次：**$n$ 个参数要 $n$ 次传递**，次数上并不比有限差分好（虽然是精确的）。

## 3.3.3 反向模式：从损失把导数往回拉

把问题倒过来。对每个中间值 $v$，定义它的**伴随值（adjoint）**

$$
\bar v = \frac{\partial L}{\partial v},
$$

即最终损失对该中间值的敏感度。

**递推式的推导。** 从 $v_p$ 到 $L$ 的每一条路径，都以一条从 $v_p$ 到它某个子节点 $v_c$ 的边开始。依照这第一条边把路径分组。以 $p\to c$ 开头的路径，贡献为 $\frac{\partial v_c}{\partial v_p}$ 乘上从 $v_c$ 到 $L$ 所有路径的总和，而那正是 $\bar v_c$。对所有子节点加总：

$$
\boxed{\;\bar v_p = \sum_{c\,\in\,\text{children}(p)} \bar v_c\;\frac{\partial v_c}{\partial v_p}\;}
\qquad\text{且}\qquad \bar L = 1 .
$$

要计算它，在处理父节点之前必须先有所有子节点的伴随值：以**反向拓扑排序（topological order）** 走访节点——从损失往参数走。走到起点时，我们就一次得到了**每个**参数的 $\bar\theta_j = \partial L/\partial\theta_j$。

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

**时间。** 反向传递中，每个记录的运算只走访一次，每条输入边计算一个局部导数、做一次乘加。基本运算的局部导数成本至多是该运算本身的小常数倍（往往更少：$\tanh'$ 直接重用存储的 $\tanh$）。对所有运算加总：

$$
\text{cost}(\nabla L) \;\le\; c\cdot\text{cost}(L),
$$

其中小常数 $c$ **与参数数量 $n$ 无关**。这就是*廉价梯度原理（cheap gradient principle）*；[Griewank & Walther（2008）](/zh-cn/references/#griewank-2008)在标准的运算计数模型下给出并证明了明确的上界。这回答了 §3.1 的预测：标量损失的精确梯度只要几次前向传递的成本，不论模型有十个参数还是一千亿个。

**内存。** 代价是存储：反向传递需要的前向值。对神经网络而言，主要是批量中每个样本在每一层的活化值——往往是训练中最大的内存开销。检查点重算（§3.7.6）以重新计算换取内存。

**一般情况下的前向 vs. 反向。** 对 $f:\mathbb{R}^n\to\mathbb{R}^m$，前向模式每次传递给出雅可比矩阵的一列（column；全部要 $n$ 次）；反向模式每次给出一行（row；$m$ 次）。训练损失的 $m = 1$，所以反向模式快了 $n$ 倍。对一个输入、多个输出的函数，则是前向模式胜出。例题 3.8 计算两者的次数。

## 3.3.5 向量与矩阵：雅可比矩阵与向量–雅可比乘积

真实网络处理的是向量与矩阵，实现也在这个层次上进行，而不是标量。对 $y = f(x)$，$x\in\mathbb{R}^n$、$y\in\mathbb{R}^m$，**雅可比矩阵（Jacobian）** $J\in\mathbb{R}^{m\times n}$ 的元素为 $J_{ij} = \partial y_i/\partial x_j$，合成函数的链式法则则变成矩阵乘法 $J_{f\circ g} = J_fJ_g$。

对一个向量值运算，伴随值递推逐元素写出为 $\bar x_j = \sum_i\bar y_i\,\partial y_i/\partial x_j$，即

$$
\bar x = J^{\top}\bar y .
$$

这是一个**向量–雅可比乘积（vector–Jacobian product, VJP）**。反向模式从不建立 $J$（它可能极大）；每个运算只需要一条有效计算 $J^{\top}\bar y$ 的规则。以下四条规则涵盖了本书中你要验证的大部分情况。

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

其中 $e_y$ 是目标的 one-hot 向量。梯度就是“预测概率减去真相”——有界、便宜、数值上温和。（单独的 softmax 雅可比矩阵为 $\mathrm{diag}(p) - pp^{\top}$；单元 04 的注意力机制会用到它。）

**(d) 复制与加总。** 如果一个向量被用在多处（扇出），它的伴随值是各处伴随值的总和——又是累加规则。反过来，对 $y = \sum_ix_i$，每个 $\bar x_i = \bar y$。两者互为转置：前向传递中的广播，在反向传递中变成加总。

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

**它是无偏的（unbiased）。** 每个抽到的索引在 $\{1,\dots,N\}$ 上均匀，所以 $\mathbb E[\nabla\ell_{i_b}] = \frac1N\sum_i\nabla\ell_i = \nabla L$，由线性性得 $\mathbb E[g_B] = \nabla L$。

**它的噪声以 $1/\sqrt B$ 缩小。** 令 $\Sigma = \mathrm{Cov}(\nabla\ell_i)$ 为单一采样梯度的协方差。$B$ 个样本彼此独立，而 $B$ 个独立复本之平均的协方差为 $\Sigma/B$：

$$
\mathrm{Cov}(g_B) = \frac{\Sigma}{B},\qquad \mathbb E\|g_B - \nabla L\|^2 = \frac{\mathrm{tr}\,\Sigma}{B}.
$$

批量变为四倍，典型误差减半，成本则变为四倍。**随机梯度下降法（stochastic gradient descent, SGD）** 在更新中以 $g_B$ 取代 $\nabla L$。每一步便宜而有噪声；§3.5.6 会说明噪声为常数步长的 SGD 能多接近最小值设下了底限，以及为什么递减的步长能消除它（[Robbins & Monro 1951](/zh-cn/references/#robbins-1951)；[Bottou et al. 2018](/zh-cn/references/#bottou-2018)）。实践中，数据在每个**训练轮次（epoch）** 打乱一次后依序取用（不放回抽样），行为类似。

## 3.3.8 动量与自适应步长

§3.2.3 说明安全步长受*最大*曲率限制。如果不同方向的曲率差异很大，对陡峭方向安全的步长，对平坦方向就太小，于是梯度下降法会在陡峭方向上来回锯齿，在平坦方向上缓慢爬行（图 3.2，以及 §3.5.4 的精确分析）。由此产生两类补救方法。

**动量（momentum；重球法）。** 维持一个累积过去梯度的“速度”（[Polyak 1964](/zh-cn/references/#polyak-1964)）：

$$
v_{k+1} = \beta v_k - \eta\nabla L(\theta_k),\qquad \theta_{k+1} = \theta_k + v_{k+1},\qquad 0\le\beta<1 .
$$

展开速度得到 $v_{k+1} = -\eta\sum_{j\le k}\beta^{k-j}\nabla L(\theta_j)$——过去梯度的指数加权和，记忆约 $1/(1-\beta)$ 步（与单元 05 的移动平均是同一个对象）。符号保持一致的梯度分量会累积；符号交替的分量（如锯齿）会互相抵消。物理上，它是一颗在损失地形中受摩擦力滚动的有质量的球。在条件数为 $\kappa$ 的二次函数上，调好的动量把收敛率从 $(\kappa-1)/(\kappa+1)$ 改善到 $(\sqrt\kappa - 1)/(\sqrt\kappa+1)$（§3.5.7）——当 $\kappa = 10$，每步从 0.818 降到 0.519。

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

*这个更新在做什么。* 比值 $\hat m/\sqrt{\hat v}$ 没有单位：把每个梯度乘上常数 $c>0$，比值不变（除了 $\epsilon$）。第一步时 $\hat m_1 = g_1$、$\hat v_1 = g_1^2$，所以每个坐标都恰好移动 $\eta\,\mathrm{sign}(g_1)$，不论梯度大小（例题 3.7）。因此 Adam 的行为像逐坐标、类似符号的方法，在梯度噪声大的地方（$|\hat m|\ll\sqrt{\hat v}$）步长缩小。

**AdamW：解耦的权重衰减。** 常见的正则化（regularization）是在损失加上 $\tfrac\lambda2\|\theta\|^2$，也就是在梯度加上 $\lambda\theta$。在 Adam 内部，这一项会被 $\sqrt{\hat v}$ 除，所以梯度大的参数反而被正则化得*较少*——这不是惩罚项的本意。AdamW 把权重衰减（weight decay）直接作用在参数上，放在自适应缩放之外（[Loshchilov & Hutter 2019](/zh-cn/references/#loshchilov-2019)）：

$$
\theta_k = \theta_{k-1} - \eta\Bigl(\frac{\hat m_k}{\sqrt{\hat v_k}+\epsilon} + \lambda\,\theta_{k-1}\Bigr).
$$

它是 Transformer 与主干项目的默认优化器。[AdamW Atlas 节点](/zh-cn/atlas/nodes/opt-adamw/)列出了它的不变量与常见实现错误。

## 3.3.9 构建引擎

实验室要求你为标量实现反向模式。设计直接由 §3.3.3 而来：

- **一种节点类型**（`Value`）存储它的数值（`data`）、它的伴随值（`grad`，初始为 0）、对父节点的参照，以及一个小函数：给定自己的伴随值，依规则 $\bar p \mathrel{+}= \bar c\,\partial c/\partial p$ 把贡献加到父节点的伴随值上。
- **每个运算**（`+`、`*`、`tanh`……）建立一个新节点，并附上正确的局部导数函数，同时捕捉它需要的值（例如 $\tanh$ 捕捉输出 $t$，使用 $1-t^2$）。
- **`backward()`** 建立图的拓扑排序（以迭代方式——递推的深度优先搜索在 10,000 个节点深的图上会使调用堆叠溢位），把输出的伴随值设为 1，再以相反顺序调用每个节点的局部函数。
- **归零是调用者的责任。** 共用叶节点的不同图之间，伴随值会在多次反向传递中累加；优化循环在每一步之前把它们设为零。

[实验室](https://github.com/yhxuee/deep-learning-intelligent-systems/tree/main/labs/unit-03-autodiff)正是测试这些性质：例题的数字、扇出下的累加（`a*a`、`a+a`）、在随机表达式图上与中心差分一致、线性性、深层图，以及跨图的累加。一旦你的引擎通过这些测试，你就**赢得了使用抽象的资格**：PyTorch 与 JAX 等框架在张量上实现同一个算法，每个运算一条 VJP 规则；从单元 04 起，主干项目就改用它们。
