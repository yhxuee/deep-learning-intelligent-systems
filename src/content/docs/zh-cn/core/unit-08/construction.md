---
title: "8.3 数学构建"
description: 祖先采样的精确性、核采样、变量替换公式与标准化流、潜变量模型与证据下界、把扩散看成去噪声，以及流匹配与连续方程和精确的高斯速度场。
sidebar: { label: "8.3 数学构建", order: 3 }
---

## 8.3.1 祖先采样及其修改

**命题 8.1。** 以条件分布 $q(x_t\mid x_{<t})$ 做祖先采样，返回序列 $x_{1:T}$ 的概率为 $q(x_{1:T}) = \prod_t q(x_t\mid x_{<t})$。

*证明。* 各步是只依赖目前为止序列的采样；某个特定序列的概率是各步概率的乘积，由链式法则它等于 $q(x_{1:T})$。$\blacksquare$

温度、top-$k$ 与核采样把每个条件分布换成修改过的 $\tilde q(x_t\mid x_{<t})$，因此恰好从*另一个*序列分布 $\prod_t\tilde q(x_t\mid x_{<t})$ 采样。阈值 $p$ 的核心是 $S_p = \{k_1,\dots,k_m\}$，即依概率递减排列的最可能符号，$m$ 是使 $\sum_{i\le m}q(k_i)\ge p$ 的最小索引；截断后的分布在 $S_p$ 上为 $q(k)/\sum_{j\in S_p}q(j)$，其余为零。注意像 $\tau < 1$ 这样的逐步修改，*不等于*从重新归一化的 $q(x_{1:T})^{1/\tau}$ 采样序列；前者是局部的，后者是全域的。

## 8.3.2 变量替换与标准化流

令 $\mathbb{R}^n$ 中的 $z\sim p_Z$，且 $x = g(z)$，$g$ 可逆且可微，$f = g^{-1}$。则

$$
p_X(x) = p_Z(f(x))\,\big|\det J_f(x)\big| .
$$

一维中：若 $x = \mu + \sigma z$、$z\sim N(0,1)$，则 $p_X(x) = \frac{1}{\sigma}\phi\big(\frac{x-\mu}{\sigma}\big)$——$1/\sigma$ 就是雅可比，拉伸空间会让密度变稀。**标准化流**组合许多雅可比行列式很便宜的简单可逆映射——例如耦合层：以一个依另一半座标决定参数的仿射映射转换一半座标，使雅可比为三角矩阵（[Dinh et al. 2017](/zh-cn/references/#dinh-2017)；[Rezende & Mohamed 2015](/zh-cn/references/#rezende-2015)）。对数似然 $\ln p_X(x) = \ln p_Z(f(x)) + \sum_\ell\ln|\det J_{f_\ell}|$ 于是是精确的，可以直接最大化——单元 01 的原则用在连续密度上。代价在架构上：每一层都必须可逆，且行列式容易计算。

## 8.3.3 潜变量与证据下界

**潜变量模型**先抽 $z\sim p(z)$ 再抽 $x\sim p_\theta(x\mid z)$，所以 $p_\theta(x) = \int p_\theta(x\mid z)p(z)\,dz$——一个我们算不出来的积分。对任何分布 $q_\phi(z\mid x)$，

$$
\ln p_\theta(x) = \underbrace{\mathbb{E}_{q_\phi(z\mid x)}\big[\ln p_\theta(x\mid z)\big] - \mathrm{KL}\big(q_\phi(z\mid x)\,\Vert\,p(z)\big)}_{\text{ELBO}(x)} + \mathrm{KL}\big(q_\phi(z\mid x)\,\Vert\,p_\theta(z\mid x)\big).
$$

*推导。* 写成 $\ln p_\theta(x) = \mathbb{E}_{q}[\ln p_\theta(x)] = \mathbb{E}_q\big[\ln\frac{p_\theta(x,z)}{p_\theta(z\mid x)}\big] = \mathbb{E}_q\big[\ln\frac{p_\theta(x,z)}{q(z\mid x)}\big] + \mathbb{E}_q\big[\ln\frac{q(z\mid x)}{p_\theta(z\mid x)}\big]$，再展开 $p_\theta(x,z) = p_\theta(x\mid z)p(z)$。$\blacksquare$

最后一项是 KL 散度，所以非负：ELBO 是对数似然的下界，当 $q_\phi$ 等于真正的后验时取等号。**变分自编码器**以最大化 ELBO 同时训练编码器 $q_\phi(z\mid x)$ 与解码器 $p_\theta(x\mid z)$，并使用*重新参数化* $z = \mu_\phi(x) + \sigma_\phi(x)\odot\varepsilon$、$\varepsilon\sim N(0, I)$，让梯度能穿过采样步骤（[Kingma & Welling 2014](/zh-cn/references/#kingma-2014)）。第一项奖励重建，第二项让编码靠近先验，使得抽 $z\sim p(z)$ 再解码能得到合理的数据。

## 8.3.4 扩散：学会消除噪声

固定一个逐步破坏数据的**前向过程**：$x_t = \alpha_tx_0 + \sigma_t\varepsilon$、$\varepsilon\sim N(0, I)$，$\alpha_t$ 从 1 降到接近 0、$\sigma_t$ 上升，使 $x_T$ 基本上是纯噪声（[Sohl-Dickstein et al. 2015](/zh-cn/references/#sohl-dickstein-2015)；[Ho et al. 2020](/zh-cn/references/#ho-2020)）。训练网络从带噪声的样本预测噪声：

$$
\mathcal L(\theta) = \mathbb{E}_{t, x_0, \varepsilon}\big\lVert\varepsilon_\theta(x_t, t) - \varepsilon\big\rVert^2 .
$$

这又是回归。它的最优解是 $\mathbb{E}[\varepsilon\mid x_t]$，而由一个古典恒等式（Tweedie 公式），这个条件平均与带噪声数据分布的**分数（score）** $\nabla_x\ln p_t(x)$ 成正比：$\mathbb{E}[\varepsilon\mid x_t] = -\sigma_t\nabla\ln p_t(x_t)$。所以去噪声器*就是*分数估计器（[Vincent 2011](/zh-cn/references/#vincent-2011)）。生成时反转这个过程，可以是随机的（反向时间 SDE），也可以是确定的（边际相同的 ODE），两者都由学到的分数驱动（[Song et al. 2021](/zh-cn/references/#song-2021)）。

## 8.3.5 流匹配

选择噪声与数据的一种配对——这里是独立的：$x_0\sim N(0, I)$、$x_1\sim p_{\text{data}}$——以及直线路径 $x_t = (1-t)x_0 + tx_1$。它的**条件速度**是 $\dot x_t = x_1 - x_0$。定义**边缘速度场**

$$
v^\star(x, t) = \mathbb{E}\big[x_1 - x_0\ \big|\ x_t = x\big].
$$

**命题 8.2（流匹配；[Lipman et al. 2023](/zh-cn/references/#lipman-2023)、[Liu et al. 2023](/zh-cn/references/#liu-2023)）。**
(a) 回归损失 $\mathcal L(\theta) = \mathbb{E}_{t, x_0, x_1}\lVert v_\theta(x_t, t) - (x_1 - x_0)\rVert^2$ 在 $v_\theta = v^\star$ 时最小。
(b) 若 $x(0)\sim N(0, I)$ 且 $\dot x = v^\star(x, t)$，则对每个 $t$，$x(t)$ 与 $x_t$ 同分布；特别是 $x(1)\sim p_{\text{data}}$。

*(a) 的证明。* 对固定的 $(x, t)$，损失是在给定 $x_t = x$ 下预测随机向量 $x_1 - x_0$ 的期望平方误差；期望平方误差的最小解就是条件平均。$\blacksquare$

*(b) 的证明概要。* 令 $p_t$ 为 $x_t$ 的密度。概率沿着路径守恒，所以 $p_t$ 满足**连续方程（continuity equation）** $\partial_tp_t + \nabla\cdot(p_tv^\star) = 0$，其中正是平均速度 $v^\star$（对配对取平均条件连续方程）。常微分方程 $\dot x = v^\star(x, t)$ 依同一方程输运密度；初始密度相同，由解的唯一性得到相同的 $p_t$。$\blacksquare$

(a) 部分说明了训练为什么不需要知道 $v^\star$：以*条件*目标 $x_1 - x_0$ 为回归目标（每个抽到的配对都有），在期望上就得到*边缘*场。最小损失不是零：它等于给定 $x_t$ 时 $x_1 - x_0$ 的平均条件方差——路径会交叉，在交叉处网络必须取平均。主干项目的二维模型把损失从 5.27 降到 3.81，而不是 0（例题 8.8）。

## 8.3.6 高斯数据的精确速度场

令 $x_1\sim N(\mu, \sigma^2I)$。则 $x_t\sim N(t\mu, s_t^2I)$，$s_t^2 = (1-t)^2 + t^2\sigma^2$，且 $x_1 - x_0$ 与 $x_t$ 联合高斯，$\mathrm{Cov}(x_1 - x_0, x_t) = (t\sigma^2 - (1-t))I$。高斯的条件平均是线性的：

$$
v^\star(x, t) = \mu + \frac{t\sigma^2 - (1-t)}{s_t^2}\,(x - t\mu).
$$

这个常微分方程有明确解 $x(t) = t\mu + s_tx_0$：微分得 $\dot x = \mu + \dot s_t x_0 = \mu + \frac{\dot s_t}{s_t}(x - t\mu)$，而 $\dot s_t/s_t = (t\sigma^2 - (1-t))/s_t^2$。在 $t = 1$ 时，$x(1) = \mu + \sigma x_0\sim N(\mu, \sigma^2I)$ 恰好成立。`stage08_sampling.py` 实现了这个场（`gaussian_velocity`），主干项目的测试从 20,000 个噪声样本积分它，并检查结果的平均与标准差。

## 8.3.7 采样就是求解常微分方程

有了学到的 $v_\theta$，生成就是数值积分。$n$ 步、步长 $h = 1/n$ 的 Euler 方法更新 $x\leftarrow x + h\,v_\theta(x, t)$。它每单位时间的误差是 $O(h)$（单元 05，§5.3.9），而且误差是系统性的，不是随机的：对高斯场，一步 Euler 就把*每个*噪声样本送到 $\mu$——样本完全没有分散——因为 $v^\star(x, 0) = \mu - x$ 直指平均（例题 8.7）。步数少而大，采样就快，样本就模糊、多样性较低。高阶求解器、更直的学习路径（§8.8.1）以及蒸馏成少步数采样器（§8.8.3），都在处理这个取舍。
