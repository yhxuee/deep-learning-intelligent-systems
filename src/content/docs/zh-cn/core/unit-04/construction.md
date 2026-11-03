---
title: "4.3 数学构建：从注意力到 Transformer 区块"
description: 矩阵形式的缩放点积注意力；因果遮罩与置换等变性及其证明；位置编码（含 RoPE）；多头注意力；反向传递；MLP、残差流与层归一化；仅解码器语言模型；以 KV 缓存生成。
sidebar: { label: "4.3 数学构建", order: 3 }
---

## 4.3.1 矩阵形式的注意力

把所有位置的查询、键与值堆成行。在矩阵形式中，把投影矩阵存成 §4.2.7 的转置（代码也这样做）会比较方便：

$$
Q = XW_Q\in\mathbb{R}^{T\times d_k},\qquad K = XW_K\in\mathbb{R}^{T\times d_k},\qquad V = XW_V\in\mathbb{R}^{T\times d_v},
$$

其中 $W_Q, W_K\in\mathbb{R}^{d\times d_k}$、$W_V\in\mathbb{R}^{d\times d_v}$。全部 $T^2$ 个分数是一次矩阵乘法，整层就是

$$
\boxed{\;\mathrm{Attn}(Q,K,V) = \mathrm{softmax}\!\Bigl(\frac{QK^{\top}}{\sqrt{d_k}}\Bigr)V\;}
$$

其中 softmax 对**每一行**分别作用。分步骤来看：

1. **分数** $S = QK^{\top}/\sqrt{d_k}\in\mathbb{R}^{T\times T}$：$S_{ij}$ 是查询 $i$ 与键 $j$ 的匹配程度。
2. **权重** $A = \mathrm{softmax}_{\text{行}}(S)$：$A$ 的每一行都是位置上的概率分布。
3. **输出** $Y = AV$：$Y$ 的第 $i$ 行是值向量以 $A_{i\cdot}$ 加权的平均。

每一步都值得检查形状：$T\times d_k$ 乘 $d_k\times T$ 得 $T\times T$；$T\times T$ 乘 $T\times d_v$ 得 $T\times d_v$。softmax 必须对**键**的索引 $j$（沿每一行）作用：对查询索引归一化，是 [Atlas](/zh-cn/atlas/nodes/mech-attention/) 中五个经典注意力错误的第一个。

当 $Q$、$K$、$V$ 都来自同一个序列时，这一层是**自注意力（self-attention）**。当查询来自一个序列、键与值来自另一个序列时（翻译解码器读取编码后的原文，或文本模型读取图像特征），则是**交叉注意力（cross-attention）**。数学完全相同。

## 4.3.2 因果遮罩

语言模型被训练来从词元 $1,\dots,t$ 预测词元 $t+1$。如果位置 $t$ 能注意位置 $t+1$，模型就能直接复制答案，训练信号也就毫无意义。修正方法是**因果遮罩（causal mask）**：在 softmax 之前，把矩阵 $M$ 加到分数上，其中 $j\le i$ 时 $M_{ij} = 0$、$j>i$ 时 $M_{ij} = -\infty$：

$$
A = \mathrm{softmax}_{\text{行}}(S + M).
$$

因为 $e^{-\infty} = 0$，被遮罩的元素权重恰好为 0，每一列在剩下的位置上仍然总和为 1。

**命题 4.1（因果性）。** 有因果遮罩时，输出 $y_i$ 只依赖 $x_1,\dots,x_i$。

*证明。* $q_i$ 只由 $x_i$ 算出，$k_j, v_j$ 只由 $x_j$ 算出。$A$ 的第 $i$ 行只在 $j\le i$ 处非零，每一个都是 $\exp(q_i^{\top}k_j/\sqrt{d_k})$ 除以对 $j'\le i$ 的总和——所以只涉及 $x_1,\dots,x_i$。于是 $y_i = \sum_{j\le i}A_{ij}v_j$。$\blacksquare$

因果性让仅解码器模型能**在一次前向传递中训练序列的所有位置**：对每个 $t$，位置 $t$ 同时预测词元 $t+1$，而没有任何一个能看到自己的答案。它也是一个可以精确测试的不变量（实验室会扰动未来的输入，检查过去的输出是否不变）。

两个陷阱：在 softmax **之后**才应用遮罩（把权重设为零却不重新归一化），会同时破坏归一化*与*因果性，因为归一化的总和已经包含了未来的分数；而差一位的遮罩（$j < i$ 而非 $j\le i$）不是遮住当前词元，就是往另一边偏移、泄漏下一个词元。

:::note[主张检查]
- **主张。** 加上因果遮罩后，输出 $y_i$ 只依赖 $x_1,\dots,x_i$（命题 4.1）。
- **假设。** 遮罩是在 softmax 之前加到分数上的；查询、键与值都是逐位置计算的；模型中没有其他运算会混合不同位置。
- **检验。** 扰动一个未来的输入，检查每一个较早的输出在浮点精度内都不变——这正是实验室所测试的不变量（实验 4.4）。
- **限制。** 证明涵盖的是如上所写的注意力。任何其他混合位置的运算——沿序列做的归一化、错位的缓存——都可能在能执行、也能训练的代码中破坏因果性，而阅读代码未必看得出来（§4.7.7）。
:::


## 4.3.3 注意力把输入当成集合

**置换矩阵（permutation matrix）** $P\in\{0,1\}^{T\times T}$ 的每一行与每一列恰好有一个 1；$PX$ 重新排列 $X$ 的各行。置换矩阵是正交的：$P^{\top}P = I$。

**命题 4.2（置换等变性）。** 没有遮罩、也没有位置信息时，自注意力对每个置换 $P$ 满足 $\mathrm{Attn}(PX) = P\,\mathrm{Attn}(X)$。

*证明。* 重新排列 $X$ 的行，就重新排列了 $Q, K, V$ 的行：$Q' = PQ$、$K' = PK$、$V' = PV$。分数变成 $S' = PQK^{\top}P^{\top}/\sqrt{d_k} = PSP^{\top}$：$S$ 的行**与**列以同一个置换重新排列。逐行的 softmax 与此可交换——重排列会重排输出的行，而在一行之内重排元素，该行的 softmax 也以相同方式重排——所以 $A' = PAP^{\top}$。最后 $Y' = A'V' = PAP^{\top}PV = PAV = PY$。$\blacksquare$

所以，没有额外信息时，自注意力对“狗咬人”与“人咬狗”算出*相同*的输出，只是列出的顺序不同：它看到的是词元的**集合**，而不是序列。对集合（点云中的点、分子中的原子、场景中的物体）而言，这正是正确的对称性（§4.5.3）。对语言而言，则必须加入顺序。

## 4.3.4 加入位置

**绝对位置嵌入。** 为每个位置 $t$（直到最大长度）学习一个向量 $p_t\in\mathbb{R}^d$，加到词元嵌入上：$x_t\leftarrow x_t + p_t$（如 GPT-2，[Radford et al. 2019](/zh-cn/references/#radford-2019)）。简单有效；超过最大长度的位置则完全没有嵌入。

**正弦位置编码（positional encoding）。** 使用固定向量，其坐标位于等比间隔的频率 $\omega_i = 10000^{-2i/d}$（[Vaswani et al. 2017](/zh-cn/references/#vaswani-2017)）：

$$
p_{t,2i} = \sin(\omega_it),\qquad p_{t,2i+1} = \cos(\omega_it).
$$

每一对 $(\sin\omega_it,\cos\omega_it)$ 都是单位圆上的一点，从 $t$ 移到 $t+k$ 就把它**旋转**角度 $\omega_ik$：由和角公式，

$$
\begin{pmatrix}\sin\omega_i(t+k)\\\cos\omega_i(t+k)\end{pmatrix} = \begin{pmatrix}\cos\omega_ik & \sin\omega_ik\\-\sin\omega_ik & \cos\omega_ik\end{pmatrix}\begin{pmatrix}\sin\omega_it\\\cos\omega_it\end{pmatrix}.
$$

一个固定的线性映射就能把 $t$ 的编码变成 $t+k$ 的编码，让线性层容易读出相对位移。

**旋转位置嵌入（RoPE）。** 不把位置加到输入上，而是旋转查询与键本身（[Su et al. 2024](/zh-cn/references/#su-2024)）。把 $q$ 与 $k$ 分成二维的一对对；在位置 $m$，把查询的第 $i$ 对以旋转 $R(m\theta_i)$ 转 $m\theta_i$ 角；键在其位置 $n$ 也同样处理。对其中一对，

$$
\bigl(R(m\theta)q\bigr)^{\top}\bigl(R(n\theta)k\bigr) = q^{\top}R(m\theta)^{\top}R(n\theta)\,k = q^{\top}R\bigl((n-m)\theta\bigr)\,k ,
$$

这里用到 $R(a)^{\top} = R(-a)$ 与 $R(-a)R(b) = R(b-a)$。**分数只通过两个位置的差 $n - m$ 依赖位置。** 例题 4.5 以数值检查这点。RoPE 作用在查询与键上，不作用在值上；旋转保持范数，所以它改变的是*哪些*键匹配，而不是分数的尺度。

## 4.3.5 多个注意力头

一层注意力对每个查询只计算**一个**位置上的概率分布。一个位置往往同时需要好几种信息：前一个词元、句子的主词、对应的左括号。单一分布可以把权重分给多个位置，但那样得到的是它们值的混合，而且单一的双线性分数 $q^{\top}k$ 必须同时编码所有关系。

**多头注意力（multi-head attention）** 并行执行 $H$ 个注意力运算，每个都有自己投影到维度 $d_h = d/H$ 之较小空间的投影，再把输出组合起来：

$$
\mathrm{head}_h = \mathrm{Attn}\bigl(XW_Q^{(h)}, XW_K^{(h)}, XW_V^{(h)}\bigr),\qquad
\mathrm{MHA}(X) = \bigl[\mathrm{head}_1,\dots,\mathrm{head}_H\bigr]\,W_O ,
$$

其中 $W_Q^{(h)}, W_K^{(h)}, W_V^{(h)}\in\mathbb{R}^{d\times d_h}$、$W_O\in\mathbb{R}^{d\times d}$。每个注意力头可以学一种自己的关系，例如“注意前一个词元”或“注意与我相符的词元”——单元 11 会在主干模型中找出这样的头。

**成本。** 实现上，各头的投影就是三个 $d\times d$ 矩阵的列区块，所以参数量为 $4d^2$（加上偏置），**与 $H$ 无关**；算术量也和一个全宽的头差不多。改变的是*表达能力*：每个查询有 $H$ 个独立的分布，而不是一个。

## 4.3.6 穿过注意力的反向传递

训练需要损失对 $Q$、$K$、$V$ 的梯度（再由单元 03 §3.3.5 的线性映射规则，得到对 $W_Q, W_K, W_V$ 的梯度）。给定 $\bar Y = \partial L/\partial Y$，沿着 §4.3.1 的三个步骤往回走。

**输出，$Y = AV$。** 线性映射规则给出

$$
\bar V = A^{\top}\bar Y,\qquad \bar A = \bar YV^{\top}.
$$

**逐行 softmax，$A_{i\cdot} = \mathrm{softmax}(S_{i\cdot})$。** 每一行都是独立的 softmax，雅可比矩阵为 $\mathrm{diag}(a) - aa^{\top}$（单元 03，§3.3.5）。第 $i$ 行的 VJP 为

$$
\bar S_{i\cdot} = \bigl(\mathrm{diag}(a_i) - a_ia_i^{\top}\bigr)\bar A_{i\cdot} = a_i\odot\bigl(\bar A_{i\cdot} - (a_i^{\top}\bar A_{i\cdot})\,\mathbf 1\bigr).
$$

写成矩阵形式，$\odot$ 为逐元素相乘，行总和在每一行上广播：

$$
\bar S = A\odot\Bigl(\bar A - \mathrm{rowsum}\bigl(A\odot\bar A\bigr)\Bigr).
$$

被遮罩的元素 $A_{ij} = 0$，所以自动有 $\bar S_{ij} = 0$：没有梯度流向未来。

**分数，$S = QK^{\top}/\sqrt{d_k}$。** 再对两个因子应用线性映射规则：

$$
\bar Q = \frac{\bar SK}{\sqrt{d_k}},\qquad \bar K = \frac{\bar S^{\top}Q}{\sqrt{d_k}} .
$$

例题 4.4 在一个三个词元的例子上完成每一步，并以有限差分检查其中一个元素。有两个观察之后会用到。反向传递需要 $A$（或能重算它），一个 $T\times T$ 矩阵——这正是考量 IO 的核心要解决的内存问题（§4.8.1）。而当 $A$ 的某一行几乎是 one-hot 时，该行的 $\bar S$ 几乎为零——这就是促成 $\sqrt{d_k}$ 缩放的梯度消失。

## 4.3.7 为什么需要逐位置的 MLP

给定权重后，注意力的输出 $y_i = \sum_jA_{ij}v_j$ 是值向量的**线性**组合；它在位置之间*搬运*信息，对信息本身却几乎不做计算。所以每个 Transformer 区块在注意力之后，都接一个**逐位置前馈网络（position-wise feed-forward network）**，以共用的权重分别作用在每个位置上：

$$
\mathrm{MLP}(u) = W_2\,\varphi(W_1u + b_1) + b_2,\qquad W_1\in\mathbb{R}^{4d\times d},\ W_2\in\mathbb{R}^{d\times4d},
$$

其中 $\varphi$ 是 GELU 之类的非线性函数。隐藏宽度 $4d$ 是原始 Transformer 与 GPT-2 的惯例。这种分工是区块的核心设计概念：**注意力在位置之间传递信息；MLP 在一个位置之内转换信息。**

MLP 不是可有可无的。对只由注意力组成、没有 MLP 与跳接的网络，[Dong et al.（2021）](/zh-cn/references/#dong-2021)证明其输出随深度以*双重指数*速度收敛到秩一矩阵——每个位置最后都是同一个向量——并显示跳接与 MLP 能抵抗这种崩塌。

## 4.3.8 残差流与层归一化

各子层都包在**残差连接**中（单元 03，§3.8.2；[He et al. 2016](/zh-cn/references/#he-2016)）：每个子层计算一个*加到*其输入上的更新。因此每个位置都带着一个向量——**残差流（residual stream）**——每个注意力层与 MLP 都以累加方式从中读取、向其写入。这个观点由 [Elhage et al.（2021）](/zh-cn/references/#elhage-2021) 为了可解释性研究而发展，让每个子层的贡献能被分开研究（单元 11）。

**层归一化（layer normalization；[Ba et al. 2016](/zh-cn/references/#ba-2016)）** 把每个词元的向量在其 $d$ 个特征上重新缩放为平均 0、方差 1，再应用可学习的增益 $\gamma$ 与偏置 $\beta$：

$$
\mathrm{LN}(u) = \gamma\odot\frac{u - \mu(u)}{\sqrt{\sigma^2(u)+\epsilon}} + \beta .
$$

放在哪里是真正的设计选择。原始 Transformer 在加上残差*之后*才归一化（**后置归一化，post-LN**：$x\leftarrow\mathrm{LN}(x + F(x))$）。GPT-2 与大多数现代模型归一化每个子层的*输入*（**前置归一化，pre-LN**：$x\leftarrow x + F(\mathrm{LN}(x))$），让残差路径保持为未归一化的恒等映射。[Xiong et al.（2020）](/zh-cn/references/#xiong-2020)显示后置归一化在初始化时会在输出附近产生很大的梯度，因此需要学习率暖身，而前置归一化不需要暖身也表现良好。

## 4.3.9 Transformer 区块与仅解码器语言模型

**区块**（前置归一化、仅解码器，如 GPT-2）：

$$
\begin{aligned}
X &\leftarrow X + \mathrm{MHA}_{\text{causal}}\bigl(\mathrm{LN}_1(X)\bigr),\\
X &\leftarrow X + \mathrm{MLP}\bigl(\mathrm{LN}_2(X)\bigr).
\end{aligned}
$$

**语言模型。** 对词元序列 $t_1,\dots,t_T$：

1. **嵌入：** $X^{(0)}_i = E[t_i] + P[i]$，其中词元嵌入表 $E\in\mathbb{R}^{V\times d}$，位置表 $P$（或在注意力内改用 RoPE 而不用 $P$）。
2. **处理：** 对 $\ell = 1,\dots,L$，$X^{(\ell)} = \mathrm{Block}_\ell\bigl(X^{(\ell-1)}\bigr)$。
3. **反嵌入：** logits $Z = \mathrm{LN}_f\bigl(X^{(L)}\bigr)E^{\top}\in\mathbb{R}^{T\times V}$，重用（“绑定”）嵌入表（[Press & Wolf 2017](/zh-cn/references/#press-2017)）。
4. **损失：** 由 $Z$ 的第 $i$ 行预测 $t_{i+1}$ 的平均交叉熵，对 $i = 1,\dots,T-1$——在每个位置应用单元 03 的 softmax 交叉熵。

多亏因果遮罩（命题 4.1），一次前向传递就算出全部 $T-1$ 个预测，每一个都只用到自己的前缀。**参数量：** 每个区块 $12d^2 + 13d$（含偏置与层归一化），加上绑定嵌入的 $Vd$、学习的位置的 $T_{\max}d$，以及最后层归一化的 $2d$。例题 4.7 推导这些项并复现 GPT-2 small 的 124,439,808 个参数。

## 4.3.10 生成与 KV 缓存

**生成**时，模型预测下一个词元的分布，采样（或选出）一个，接到序列后面，再重复。每一步都对越来越长的前缀重算整个前向传递，在长度 $t$ 时每个新词元要花 $O(t^2d)$ 的注意力计算。

因果性让其中大部分工作是多余的。词元 $1,\dots,t$ 的键与值在词元 $t+1$ 到来时不会改变（它们只依赖自己的前缀）。所以每一层都维护一个存放所有过去键与值的 **KV 缓存（KV cache）**。每一步只为新词元：计算 $q, k, v$；把 $k, v$ 接到缓存后面；以 $q$ 对整个缓存做注意力；继续到下一层。这样每一步每层只需 $O(t\,d)$ 的注意力计算，而不是 $O(t^2d)$。

正确实现由两个不变量定义：**缓存解码产生的 logits 与对同一前缀做完整前向传递的结果完全相同**（在浮点舍入内），以及**新词元的位置索引是 $t+1$**，不是 1——忘记递增它是经典的 KV 缓存错误（[KV 缓存](/zh-cn/atlas/nodes/mech-kv-cache/)）。缓存的内存——每条序列 $2\times L\times t\times d$ 个数——随上下文增长；单元 05 把它当作未压缩的记忆来研究，单元 06 则当作推理成本来研究。
