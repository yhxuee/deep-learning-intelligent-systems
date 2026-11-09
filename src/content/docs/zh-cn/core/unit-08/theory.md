---
title: "8.5 一般理论"
description: 分布之间的最优传输与直线路径为何重要、分数匹配及其与去噪声的等价、采样中的离散化误差、生成模型总览表，以及似然度量与没度量的东西。
sidebar: { label: "8.5 一般理论", order: 5 }
---

## 8.5.1 分布之间的几何输运：最优传输与直线流

§8.3.2–§8.3.6 的连续生成方法，本质上都是在构建一个时变速度场向量 $v(x, t)$，将简单的先验高斯噪声分布 $p_0$ 沿着轨迹平滑推动至复杂的真实数据分布 $p_1$。在所有可能将 $p_0$ 变形为 $p_1$ 的连续速度场中，**最优传输理论（Optimal Transport, OT）**挑选出了能量消耗最少的那一个：在 $p_0$ 与 $p_1$ 的所有联合分布 $\pi(x_0, x_1)$ 中，最小化二阶 Wasserstein 距离 $\mathbb{E}_{(x_0, x_1)\sim\pi}\lVert x_1 - x_0\rVert^2$（[Villani 2009](/zh-cn/references/#villani-2009)）。

**最优传输的几何性质：**
OT 的理论最优解（位移插值，Displacement Interpolation）保证每个粒子以恒定速度沿**严格不相交的直线路径**匀速行进：$\dot{x}_t = x_1 - x_0$。

**独立流匹配的弯曲困境与整流（Rectification）：**
然而，基础的流匹配方法（§8.3.5）采用了独立耦合先验 $q(x_0, x_1) = p_0(x_0) p_1(x_1)$。虽然对任一单独配对而言条件路径 $x_t = (1-t)x_0 + t x_1$ 是一条直线，但**来自不同噪声与数据对的直线轨迹在高维空间中频繁交叉**！
- 当两条轨迹在 $(x, t)$ 处相交时，边缘速度场 $v^\star(x, t) = \mathbb{E}[x_1 - x_0 \mid x_t = x]$ 被迫对截然不同的速度向量进行平均；
- 这种平均效应使得宏观速度场发生剧烈旋转与弯曲，导致数值常微分方程（ODE）求解器在采样时必须采用数十步甚至上百步微小步长才能避免轨道偏离。

理论给出两种优雅的解法：
1. **OT 迷你批量配对：** 在训练批量内通过求解小型线性指派问题（Hungarian 算法或 Sinkhorn 迭代），将几何距离最近的噪声与数据点进行配对；
2. **整流算法（Reflow / Rectified Flow，[Liu et al. 2023](/zh-cn/references/#liu-2023)）：** 利用已训练的第一代流模型从 $x_0$ 数值模拟生成端点 $x(1)$，构建全新的确定性配对 $(x_0, x(1))$。理论证明，以这种自生成配对重新训练第二代流，可以在严格不增加输运成本的前提下**彻底解开交叉路径，将轨迹强力拉直（Straightening the Flow）** **[D]**。直化的速度场使得 Euler 单步或二步采样即可达到顶级生成质量（§8.5.3）。

## 8.5.2 分数匹配、能量爬坡与去噪声等价性

连续分布的**分数函数（Score Function）**定义为对数概率密度的梯度向量：
$s(x) = \nabla_x\ln p(x) = \frac{\nabla_x p(x)}{p(x)}$。
分数向量直观上就是**指向概率密度更高区域的局部爬坡方向**。

**分数的核心数学优势：**
若密度由能量模型定义 $p(x) = \frac{1}{Z} e^{-E(x)}$，对其求空间梯度：
$\nabla_x \ln p(x) = \nabla_x \big(-E(x) - \ln Z\big) = -\nabla_x E(x)$。
极其昂贵且通常在数学上无法解析积分的配分函数常数 $Z = \int e^{-E(x)}dx$ 在梯度运算下完全消失！这使得我们能够为任意未归一化的复杂能量模型学习梯度场（[Hyvärinen 2005](/zh-cn/references/#hyvarinen-2005)）。

但实践困境在于：我们只有离散的数据样本，不知道真实连续分布的 $\nabla\ln p(x)$。**去噪声分数匹配（Denoising Score Matching）**通过人工加噪巧妙解决了这一僵局。

**命题 8.3（去噪声分数匹配；[Vincent 2011](/zh-cn/references/#vincent-2011)）。** 令 $\tilde x = x + \sigma\varepsilon$、$\varepsilon\sim N(0, I)$，$p_\sigma$ 为 $\tilde x$ 的密度。则
$$
\mathbb{E}\Big\lVert s_\theta(\tilde x) + \frac{\varepsilon}{\sigma}\Big\rVert^2 = \mathbb{E}\big\lVert s_\theta(\tilde x) - \nabla\ln p_\sigma(\tilde x)\big\rVert^2 + \text{常数}.
$$

*证明概要。* 展开两个平方；它们只在交叉项上不同。利用 $\nabla\ln p_\sigma(\tilde x) = \mathbb{E}[-\varepsilon/\sigma\mid\tilde x]$（在积分号下对 $p_\sigma(\tilde x) = \int p(x)N(\tilde x; x, \sigma^2I)\,dx$ 微分），得 $\mathbb{E}[s_\theta(\tilde x)^{\top}\varepsilon/\sigma] = -\mathbb{E}[s_\theta(\tilde x)^{\top}\nabla\ln p_\sigma(\tilde x)]$，所以交叉项相同。$\blacksquare$

所以预测加入的噪声，在差一个常数的意义下，就等于估计带噪声数据的分数——这是 §8.3.4 背后的恒等式。在许多噪声等级上的分数模型，结合 Langevin 动力学或反向时间 SDE，就是扩散的分数观点（[Song & Ermon 2019](/zh-cn/references/#song-2019)；[Song et al. 2021](/zh-cn/references/#song-2021)）；扩散、分数式与流匹配模型，是同一族输运的不同参数化（[Albergo & Vanden-Eijnden 2023](/zh-cn/references/#albergo-2023)）**[D]**。

## 8.5.3 抽样误差如何依赖步长

以步长 $h$ 的 Euler 步积分 $\dot x = v(x, t)$，每步的局部误差量级为 $h^2\lVert\partial_tv + (\nabla v)v\rVert$，终点的整体误差量级为 $h$，其常数与*路径的曲率*成正比 **[D]**。若场产生等速的直线路径，Euler 一步就精确。这就是为什么“变直”（§8.5.1）与少步数采样是同一个研究问题。高阶求解器降低误差的幂次；单元 05 的稳定性条件（§5.3.4）也适用，因为在某些噪声调度下，常微分方程在 $t = 0$ 或 $t = 1$ 附近可能是刚性的（[Karras et al. 2022](/zh-cn/references/#karras-2022)）**[E]**。

## 8.5.4 生成模型总览

| 家族 | 采样 | 似然 | 训练信号 | 主要成本 |
|---|---|---|---|---|
| 自回归 | 精确，每词元一步 | 精确 | 交叉熵 | 循序生成 |
| 标准化流 | 一次传递 | 精确 | 最大似然 | 可逆架构 |
| VAE | 一次传递 | 下界（ELBO） | 重建 + KL | 模糊的样本、后验崩塌 |
| GAN（[Goodfellow et al. 2014](/zh-cn/references/#goodfellow-2014)） | 一次传递 | 无 | 对抗赛局 | 训练不稳、遗漏众数 |
| 扩散／分数 | 许多步（SDE 或 ODE） | 通过 ODE（昂贵） | 去噪声回归 | 许多次网络评估 |
| 流匹配 | 许多步（ODE） | 通过 ODE（昂贵） | 速度回归 | 许多次网络评估 |

没有任何家族在每个面向都胜出。文本使用自回归，因为精确且便宜的每词元似然很重要，而且词元是离散的；图像使用扩散与流匹配，因为回归训练稳定、样本质量高，而许多步的成本则由 §8.8 的方法处理。

## 8.5.5 似然度量了什么

留出数据的对数似然（单元 01）仍是密度模型有原则的评分。对生成器而言，它在两个方向上都不完整（[Theis et al. 2016](/zh-cn/references/#theis-2016)）**[D]/[E]**：模型可以有极佳的似然却有差的样本（它把一点质量分散到垃圾上，同时很好地涵盖了所有数据），也可以有极佳的样本却有差或无定义的似然（它完美涵盖少数众数，却漏掉其余）。以样本为基础的指标，例如真实与生成图像特征统计之间的 Fréchet 距离（[Heusel et al. 2017](/zh-cn/references/#heusel-2017)），度量质量与涵盖度，代价是依赖一个特征提取器。单元 10 把指标的选择当成评估设计的决定来处理。
