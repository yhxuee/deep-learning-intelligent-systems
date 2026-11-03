---
title: "8.5 一般理论"
description: 分布之间的最优传输与直线路径为何重要、分数匹配及其与去噪声的等价、采样中的离散化误差、生成模型总览表，以及似然度量与没度量的东西。
sidebar: { label: "8.5 一般理论", order: 5 }
---

## 8.5.1 分布之间的输运

§8.3.2–§8.3.6 的每个方法，都构建一个映射——或一个流——把简单的分布推到数据分布上。这样的映射有很多。**最优传输（optimal transport）**挑出移动质量最少的那一个：在 $p_0$ 与 $p_1$ 的所有配对中，最小化 $\mathbb{E}\lVert x_1 - x_0\rVert^2$（[Villani 2009](/zh-cn/references/#villani-2009)）。它的解让每个粒子以等速沿直线移动，而且粒子的路径不会交叉。

以*独立*配对做的流匹配（§8.3.5）使用直线的条件路径，但它们会交叉，所以边缘场 $v^\star$ 是弯曲的，常微分方程需要许多步。理论给出两种补救：选择更接近最优传输的配对（例如在小批量内解一个小型指派问题来配对噪声与数据），或**整流（rectify）**——以学到的流产生配对 $(x_0, x(1))$，再以它们训练新的流，这可证明不会增加输运成本，并让路径变直（[Liu et al. 2023](/zh-cn/references/#liu-2023)）**[D]**。更直的路径允许更少的积分步数（§8.5.3）。

## 8.5.2 分数匹配与去噪声

密度的**分数**是 $s(x) = \nabla_x\ln p(x)$。它不依赖归一化常数，所以能为未归一化的模型学习（[Hyvärinen 2005](/zh-cn/references/#hyvarinen-2005)）。从样本学习它似乎需要 $\nabla\ln p$，而我们没有；技巧在于加噪声。

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
