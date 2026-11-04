---
title: "8.3 数学构建"
description: 从“这个程序究竟生成哪个分布”出发，推导祖先采样、变量替换、ELBO、去噪声分数、流匹配、精确高斯速度场与数值采样。
sidebar: { label: "8.3 数学构建", order: 3 }
---

上一节把三件事分开：学到的分布、从中采样的规则，以及最后实际出现的样本。现在我们把这个区分写成数学。每一步都反复问同一个问题：

> 假如这个程序能精确执行，它究竟会生成哪个分布？

这个问题能避免一种常见混淆：模型定义的分布可能没有问题，但近似采样器会扭曲它；反过来，采样器也可能很精确，只是它抽的是经过刻意修改的分布，而不是训练时的原分布。

## 8.3.1 祖先采样及其修改

假设自回归模型在每个位置 $t$ 都给出归一化的条件分布 $q_t(\,\cdot\mid x_{<t})$。**祖先采样（ancestral sampling）**从空前缀开始，先抽 $X_1\sim q_1$，把结果接到前缀，再抽 $X_2\sim q_2(\cdot\mid X_1)$，如此继续。

**命题 8.1。** 对固定长度 $T$，祖先采样产生特定序列 $x_{1:T}$ 的概率为

$$
\Pr(X_{1:T}=x_{1:T})
=\prod_{t=1}^{T}q_t(x_t\mid x_{<t})
=q(x_{1:T}).
$$

*证明。* “算法返回 $x_{1:T}$”表示第 1 步抽到 $x_1$，第 2 步在看过 $x_1$ 后抽到 $x_2$，依此类推。条件概率的乘法法则给出

$$
\Pr(X_{1:T}=x_{1:T})
=\Pr(X_1=x_1)\prod_{t=2}^{T}
\Pr(X_t=x_t\mid X_{<t}=x_{<t}).
$$

这些因子正是模型的条件分布；它们的乘积就是 $q(x_{1:T})$ 的链式法则分解。$\blacksquare$

若序列长度可变，必须加入序列结束符号 EOS。此时一个完整结果是 $(x_1,\ldots,x_L,\mathrm{EOS})$，其概率也包含最后“停止”的概率。若略去 EOS，样本空间就被悄悄改变了：程序虽然指定了符号，却没有指定序列何时结束。

**修改解码规则时，什么改变了？** 令某个前缀下的 logits 为 $a_k$。温度 $\tau>0$ 产生新的局部条件分布

$$
\widetilde q_\tau(k\mid x_{<t})
=\frac{\exp(a_k/\tau)}
{\sum_j\exp(a_j/\tau)}.
$$

当 $\tau\to0^+$，概率集中到最大 logit；若最大值不只一个，极限概率会由并列最大者共享，除非另外指定破除平手的规则。当 $\tau\to\infty$，有限词表会趋近均匀分布。$\tau=1$ 则还原原本的条件分布。

Top-$k$ 保留概率最高的 $k$ 个符号。**核采样（nucleus sampling）**或 top-$p$ 则先排列符号，使 $q(k_1)\ge q(k_2)\ge\cdots$，再选择满足下式的最小 $m$：

$$
\sum_{i=1}^{m}q(k_i)\ge p.
$$

令 $S_p=\{k_1,\ldots,k_m\}$，抽样分布为

$$
\widetilde q_p(k)
=
\begin{cases}
\displaystyle\frac{q(k)}{\sum_{j\in S_p}q(j)}, & k\in S_p,\\[6pt]
0, & k\notin S_p.
\end{cases}
$$

分母不可省略：截断删掉了一些结果，重新归一化才会让剩余数值再次成为概率。

在每个前缀应用上述任一规则，仍然是在做精确的祖先采样——但抽的是*新*序列分布

$$
\widetilde q(x_{1:T})
=\prod_{t=1}^{T}\widetilde q_t(x_t\mid x_{<t}),
$$

一般不再是原本的 $q$。局部温度也不同于把完整序列概率整体取次方：

$$
\prod_t
\frac{q(x_t\mid x_{<t})^{1/\tau}}
{\sum_k q(k\mid x_{<t})^{1/\tau}}
\ne
\frac{q(x_{1:T})^{1/\tau}}
{\sum_{y_{1:T}}q(y_{1:T})^{1/\tau}}.
$$

左式每一步都有依前缀而变的归一化常数；右式只有一个对所有完整序列计算的归一化常数。贪心解码走得更远：它把每个条件分布换成集中在某个最大值上的点质量。指定平手规则后，它是确定性的，而且不保证找到全局概率最高的完整序列（§8.2）。

## 8.3.2 变量替换与标准化流

自回归模型一次构建一个条件分布。**标准化流（normalizing flow）**采取另一条路：先抽一个简单随机向量 $Z$，再以可逆映射一次变换整个向量。

令 $\mathbb R^n$ 中的 $Z\sim p_Z$、$X=g(Z)$。假设 $g$ 是可微双射，逆映射 $f=g^{-1}$ 也可微，而且雅可比行列式几乎处处非零。概率必须守恒：$x$ 周围的一小块区域，与其在 $z=f(x)$ 周围的原像，承载相同概率：

$$
p_X(x)\,d x\approx p_Z(f(x))\,d z.
$$

在局部，逆映射把体积改变为 $d z=\left|\det J_f(x)\right|d x$。代入便得到**变量替换公式**

$$
\boxed{
p_X(x)=p_Z(f(x))\left|\det J_f(x)\right|
}.
$$

行列式不是装饰性的修正项。密度是“每单位体积的概率”；如果映射把一小块区域拉伸成两倍体积，同一份概率就会被摊薄成一半。

**一维检查。** 令 $Z\sim N(0,1)$，且 $X=\mu+\sigma Z$、$\sigma>0$。此时 $f(x)=(x-\mu)/\sigma$，且 $|f'(x)|=1/\sigma$，所以

$$
p_X(x)=\frac{1}{\sigma}
\phi\left(\frac{x-\mu}{\sigma}\right).
$$

熟悉的高斯缩放因子，正是一个雅可比。

实用的流会组合多个映射，$g=g_L\circ\cdots\circ g_1$。行列式在组合下相乘，因此取对数后相加。若把逆映射逐层执行得到 $z=f(x)$，则

$$
\ln p_X(x)
=\ln p_Z(z)
+\sum_{\ell=1}^{L}
\ln\left|\det J_{f_\ell}\right|.
$$

这同时提供精确密度计算与直接的最大似然训练。真正困难的是架构：任意神经网络通常既不可逆，行列式也不便宜。耦合层的解法是让一组坐标保持不变，再用第一组坐标算出的参数去变换另一组。它的雅可比是三角矩阵，行列式只需把对角元素相乘（[Dinh et al. 2017](/zh-cn/references/#dinh-2017)；[Rezende & Mohamed 2015](/zh-cn/references/#rezende-2015)）。

这里的“精确似然”以数学模型与数值计算都正确为前提：映射必须真的可逆、行列式必须正确求值，而且基底密度必须已知。这些限制换来可处理性，也约束了架构。同维度的平滑双射还会保留较多拓扑性质；后面的章节会说明这为何重要。

## 8.3.3 潜变量与证据下界

**潜变量模型**先抽取未观察到的编码 $Z\sim p(z)$，再由解码器抽取数据 $X\sim p_\theta(x\mid Z)$。观察数据的密度要把所有可能解释平均起来：

$$
p_\theta(x)=\int p_\theta(x\mid z)p(z)\,d z.
$$

这个积分通常难以处理——不是依定义必然无法计算，而是对神经解码器而言，往往成本太高或没有闭式。后验

$$
p_\theta(z\mid x)
=\frac{p_\theta(x\mid z)p(z)}{p_\theta(x)}
$$

含有同一个麻烦的归一化常数。变分自编码器引入可处理的编码器 $q_\phi(z\mid x)$ 来近似它（[Kingma & Welling 2014](/zh-cn/references/#kingma-2014)）。

为了推导训练目标，在联合密度中乘上再除以编码器：

$$
\begin{aligned}
\ln p_\theta(x)
&=\ln\mathbb E_{q_\phi(z\mid x)}
\left[
\frac{p_\theta(x,z)}
{q_\phi(z\mid x)}
\right]\\
&\ge
\mathbb E_{q_\phi(z\mid x)}
\left[
\ln p_\theta(x,z)-\ln q_\phi(z\mid x)
\right],
\end{aligned}
$$

第二行使用 Jensen 不等式。再展开 $p_\theta(x,z)=p_\theta(x\mid z)p(z)$，便得到**证据下界（evidence lower bound, ELBO）**

$$
\boxed{
\mathcal L_{\mathrm{ELBO}}(x)
=
\mathbb E_{q_\phi(z\mid x)}
\left[\ln p_\theta(x\mid z)\right]
-
\mathrm{KL}
\left(q_\phi(z\mid x)\,\Vert\,p(z)\right)
}.
$$

名称是字面意义：$\mathcal L_{\mathrm{ELBO}}(x)\le\ln p_\theta(x)$。下列精确恒等式揭示差距有多大：

$$
\ln p_\theta(x)
=\mathcal L_{\mathrm{ELBO}}(x)
+\mathrm{KL}
\left(q_\phi(z\mid x)\,\Vert\,p_\theta(z\mid x)\right).
$$

因此，当且仅当编码器几乎处处等于模型的真正后验时，这个界才取等号。

ELBO 的两项各有精确含义。

- 期望对数似然要求解码器在拿到由数据推断的编码后，给该笔观察高概率。只有在观察模型是固定方差的高斯分布时，它才会化成平方误差重建项。Bernoulli、类别、离散化 logistic 或可学习方差的解码器，会产生不同损失。
- KL 项抑制每笔观察的编码器分布偏离所选先验太远。这有助于让先验样本可供使用，却不能单独保证生成结果逼真：解码器、变分分布族、优化与泛化仍然都会影响结果。

如何对随机采样微分？对角高斯编码器可写成

$$
q_\phi(z\mid x)
=N\!\left(
\mu_\phi(x),
\operatorname{diag}(\sigma_\phi^2(x))
\right).
$$

再令

$$
\varepsilon\sim N(0,I),
\qquad
Z=\mu_\phi(x)+\sigma_\phi(x)\odot\varepsilon.
$$

随机性现在由不含参数的 $\varepsilon$ 注入，而 $Z$ 是 $\phi$ 的可微函数。这个**重新参数化技巧（reparameterization trick）**让我们能以一般反向传播，对期望值的 Monte Carlo 估计求梯度。它改变的是梯度估计方式，不是模型分布。

## 8.3.4 扩散：学会消除噪声

扩散从一个容易执行、却难以逆转的操作开始：不断把噪声加入数据，直到几乎没有结构。为避免符号冲突，本节把干净样本写成 $X_{\mathrm{data}}$；到了 §8.3.5，$X_0$ 会改指噪声，$X_1$ 才指数据。

在噪声等级 $t$，考虑高斯破坏过程

$$
X_t=\alpha_t X_{\mathrm{data}}+\sigma_t\varepsilon,
\qquad
\varepsilon\sim N(0,I).
$$

调度从接近 $(\alpha_0,\sigma_0)=(1,0)$ 开始，终点的信号则足够小，使 $X_T$ 接近一个已知噪声分布。离散扩散模型以许多 Markov 转移实现这件事；上式总结了选定时间的边际破坏分布（[Sohl-Dickstein et al. 2015](/zh-cn/references/#sohl-dickstein-2015)；[Ho et al. 2020](/zh-cn/references/#ho-2020)）。

训练网络预测当次实际加入的噪声：

$$
\mathcal L(\theta)
=\mathbb E_{t,X_{\mathrm{data}},\varepsilon}
\left[
\left\|
\varepsilon_\theta(X_t,t)-\varepsilon
\right\|^2
\right].
$$

平方误差回归告诉我们，总体最优解是 $\varepsilon_\theta^\star(x,t)=\mathbb E[\varepsilon\mid X_t=x]$。但生成时不会有人告诉模型真正的 $\varepsilon$，这个量为何有用？固定干净数据时，破坏分布是高斯分布，所以

$$
\nabla_x\ln p_t(x\mid X_{\mathrm{data}})
=-\frac{x-\alpha_tX_{\mathrm{data}}}{\sigma_t^2}
=-\frac{\varepsilon}{\sigma_t}.
$$

在积分号下对边缘分布 $p_t(x)=\int p_t(x\mid x_{\mathrm{data}})p_{\mathrm{data}}(x_{\mathrm{data}})\,d x_{\mathrm{data}}$ 微分，再除以 $p_t(x)$。在通常的正则条件下，

$$
\begin{aligned}
\nabla_x\ln p_t(x)
&=
\mathbb E\left[
\nabla_x\ln p_t(x\mid X_{\mathrm{data}})
\mid X_t=x
\right]\\
&=-\frac{1}{\sigma_t}
\mathbb E[\varepsilon\mid X_t=x].
\end{aligned}
$$

因此

$$
\boxed{
\mathbb E[\varepsilon\mid X_t=x]
=-\sigma_t\nabla_x\ln p_t(x)
}.
$$

向量 $\nabla_x\ln p_t(x)$ 称为**分数（score）**：它指向带噪声数据的对数密度增加最快的方向。因此，对这个高斯破坏过程与噪声预测参数化而言，最优去噪声器与分数含有相同信息（[Vincent 2011](/zh-cn/references/#vincent-2011)）。这是一个有条件的恒等式，并不是宣称所有噪声模型下的每个去噪声器，字面上都是分数网络。

在所有相关噪声等级上都准确的分数，配合已指定的调度，会定义反向时间随机微分方程。相关的 probability-flow ODE 在理想连续时间模型中具有相同的单一时间边缘分布（[Song et al. 2021](/zh-cn/references/#song-2021)）。实践中，网络误差、端点近似、时间离散化与数值求解器，都会使生成样本偏离这个理想。

## 8.3.5 流匹配

扩散通过学习分数来掌握密度如何变化。**流匹配（flow matching）**则直接学习速度场。

独立抽取噪声与数据：

$$
X_0\sim N(0,I),
\qquad
X_1\sim p_{\mathrm{data}},
$$

再以直线条件路径连接每一组配对：

$$
X_t=(1-t)X_0+tX_1,
\qquad
\dot X_t=X_1-X_0.
$$

同一个位置 $X_t=x$ 可能落在许多采样直线上，每一条线的速度不同。确定性的 ODE 在 $(x,t)$ 只能有一个速度，因此定义**边缘速度场**

$$
v^\star(x,t)=\mathbb E[X_1-X_0\mid X_t=x].
$$

**命题 8.2（流匹配；[Lipman et al. 2023](/zh-cn/references/#lipman-2023)、[Liu et al. 2023](/zh-cn/references/#liu-2023)）。** 假设相关密度与速度场足够正则，使条件期望、连续方程与 ODE 流都存在，而且密度演化具有唯一解。

1. 总体损失

   $$
   \mathcal L(\theta)
   =
   \mathbb E
   \left[
   \left\|
   v_\theta(X_t,t)-(X_1-X_0)
   \right\|^2
   \right]
   $$

   在 $v_\theta(x,t)=v^\star(x,t)$ 几乎处处成立时最小。
2. 若 $Y(0)\sim N(0,I)$ 且 $\dot Y(t)=v^\star(Y(t),t)$，则 $Y(t)$ 与 $X_t$ 的边缘分布相同。特别是 $Y(1)\sim p_{\mathrm{data}}$。

*第 1 部分的证明。* 令 $V=X_1-X_0$，且 $m(X_t,t)=\mathbb E[V\mid X_t,t]$。加上再减去 $m$：

$$
\begin{aligned}
\mathbb E\|V-v_\theta\|^2
&=\mathbb E\|V-m\|^2
+\mathbb E\|m-v_\theta\|^2\\
&\quad+
2\mathbb E[(V-m)^\top(m-v_\theta)].
\end{aligned}
$$

因为 $\mathbb E[V-m\mid X_t,t]=0$，交叉项为零。第一项不依赖 $\theta$；第二项非负，而且只在 $v_\theta=m$ 几乎处处成立时为零。$\blacksquare$

这个分解也说明为什么最优可达的回归损失不必等于零：

$$
\mathcal L_{\min}
=\mathbb E
\left[
\|V-\mathbb E[V\mid X_t,t]\|^2
\right].
$$

独立配对会产生彼此交叉的条件路径。即使已知 $(X_t,t)$，交叉处的目标速度仍有不确定性；网络学到的是它们的条件平均。

*第 2 部分的证明概要。* 令 $p_t$ 为 $X_t$ 的密度。条件路径所承载的概率遵守守恒律；在每个位置平均其速度可得

$$
\partial_t p_t(x)
+\nabla\cdot
\left(p_t(x)v^\star(x,t)\right)
=0,
$$

这就是**连续方程（continuity equation）**。由 ODE $\dot Y=v^\star(Y,t)$ 输运的密度满足同一方程，而且从同一个 $p_0$ 出发。在上述唯一性条件下，两条密度路径因此都是 $p_t$。$\blacksquare$

这个结果有一个细微后果：训练时使用从采样配对直接取得的简单目标 $X_1-X_0$，但生成时走的是平均后的边缘场 $v^\star$。各个生成 ODE 轨迹通常**不会**重走某一组训练配对的直线。

## 8.3.6 高斯数据的精确速度场

高斯目标让我们能逐项检查命题 8.2。令

$$
X_0\sim N(0,I),
\qquad
X_1\sim N(\mu,\sigma^2I),
$$

两者独立，且 $\sigma>0$。沿着直线插值，$X_t=(1-t)X_0+tX_1$。平均会线性相加，所以 $\mathbb E[X_t]=t\mu$。由于独立，协方差项也直接相加：

$$
\operatorname{Var}(X_t)
=\left((1-t)^2+t^2\sigma^2\right)I
=s_t^2I,
\qquad
s_t=\sqrt{(1-t)^2+t^2\sigma^2}.
$$

现在令 $V=X_1-X_0$。其平均为 $\mu$，而

$$
\begin{aligned}
\operatorname{Cov}(V,X_t)
&=
\operatorname{Cov}
\left(
X_1-X_0,\,
(1-t)X_0+tX_1
\right)\\
&=\left(t\sigma^2-(1-t)\right)I.
\end{aligned}
$$

由于 $(V,X_t)$ 联合高斯，其条件平均是仿射函数：

$$
\boxed{
v^\star(x,t)
=\mu+
\frac{t\sigma^2-(1-t)}{s_t^2}
(x-t\mu)
}.
$$

这个 ODE 不需数值方法就能求解。给定初始值 $Y(0)=y_0$，猜测 $Y(t)=t\mu+s_ty_0$。由于

$$
\frac{\dot s_t}{s_t}
=\frac{t\sigma^2-(1-t)}{s_t^2},
$$

对猜测路径微分可得

$$
\dot Y(t)
=\mu+
\frac{t\sigma^2-(1-t)}{s_t^2}
\left(Y(t)-t\mu\right)
=v^\star(Y(t),t).
$$

因为 $s_0=1$，它具有正确初始值。终点则有 $s_1=\sigma$，所以 $Y(1)=\mu+\sigma y_0$。若 $y_0\sim N(0,I)$，便精确得到 $Y(1)\sim N(\mu,\sigma^2I)$。注意这个解保留了什么：每个初始标准化坐标 $y_0$ 都还在，只是随 $s_t$ 连续缩放。这个流改变点云的平均与分散程度，并不会把每个样本都送到平均。

\`stage08_sampling.py\` 中的 \`gaussian_velocity\` 实现这个精确场。相应测试会积分许多初始噪声样本，再检查终点平均与标准差。通过测试表示“速度场加上求解器”在选定容差下通过检查；它不是对每一种求解器设定的证明。

## 8.3.7 采样就是求解常微分方程

速度网络训练完成后，生成需要求解

$$
\dot x(t)=v_\theta(x(t),t),
\qquad
x(0)\sim p_0.
$$

Euler 方法把 $[0,1]$ 分成 $n$ 步，步长 $h=1/n$：

$$
x_{j+1}=x_j+h\,v_\theta(x_j,t_j).
$$

在标准的平滑与 Lipschitz 条件下，Euler 每一步的局部截断误差为 $O(h^2)$，在固定时间区间累积后的**全局误差为 $O(h)$**。“一阶”描述的是步长缩小时最终误差如何下降，不是每单位时间随机加入多少误差。

精确高斯场揭示一个剧烈的一步失败。在 $t=0$，$v^\star(x,0)=\mu-x$。因此，用步长 $h=1$ 做一次 Euler 更新会得到

$$
x_{\mathrm{Euler}}(1)=x(0)+\mu-x(0)=\mu.
$$

每个初始样本都塌缩到平均，尽管精确 ODE 解的终点是 $\mu+\sigma x(0)$。速度场是正确的；错的是数值近似。面对其他场时，大步长也可能过冲、移动平均、制造瑕疵或遗漏模态。因此，步数较少并不普遍只意味着“更模糊”或只意味着“多样性更低”。

最好把两种误差分开：

$$
\text{生成误差}
\quad\longleftarrow\quad
\underbrace{v_\theta-v^\star}_{\text{学到的速度场误差}}
\quad+\quad
\underbrace{\text{数值积分误差}}_{\text{求解器与步长调度}}.
$$

增加求解器步数只会降低第二项，而且这还假设方法稳定且收敛；它无法修复有偏的学习速度场。高阶方法在每一步多次评估速度场，以消去低阶误差；自适应方法则把评估花在轨迹较难的区域。真正相关的成本通常是神经网络函数评估次数，而不只是时间区间被切成几段。更直的路径（§8.8.1）、更好的求解器，以及蒸馏成少步生成器（§8.8.3），处理的是这个精度与计算量问题的不同部分。
