---
title: "9.3 数学构建"
description: 马尔可夫决策过程、带基线与优势的策略梯度定理、价值函数与贝尔曼方程、KL 正则化奖励最大化的闭式最优解、由偏好得到的奖励模型、PPO 的截断目标函数、DPO 的推导，以及可验证奖励与群组基线。
sidebar: { label: "9.3 数学构建", order: 3 }
---

## 9.3.1 马尔可夫决策过程

一个**马尔可夫决策过程（Markov decision process，MDP）**包含状态 $s$、动作 $a$、转移分布 $P(s'\mid s, a)$、奖励 $r(s, a)$ 与折扣 $\gamma\in[0, 1]$（[Sutton & Barto 2018](/zh-cn/references/#sutton-2018)）。策略 $\pi_\theta(a\mid s)$ 产生轨迹；从时间 $t$ 起的**回报（return）**是 $G_t = \sum_{k\ge0}\gamma^kr_{t+k}$，目标函数为 $J(\theta) = \mathbb{E}_{\tau\sim\pi_\theta}[G_0]$。

文本生成是一个结构特别简单的 MDP：状态是提示加上目前已生成的词元，动作是下一个词元，转移以确定的方式把它接上去，而奖励通常只在最后出现一次。主干项目的算术任务是一步的 MDP——一个*情境式多臂老虎机（contextual bandit）*。

## 9.3.2 策略梯度定理

**定理 9.1（策略梯度，回合式折扣目标）。** 令 $J(\theta) = \mathbb{E}_{\tau\sim\pi_\theta}[G_0]$，其中对有限长度的回合 $G_t = \sum_{k\ge0}\gamma^kr_{t+k}$。对任何不依赖 $a_t$ 的基线 $b(s_t)$，
$$
\nabla_\theta J(\theta) = \mathbb{E}_{\tau\sim\pi_\theta}\Big[\sum_{t}\gamma^t\,\nabla_\theta\ln\pi_\theta(a_t\mid s_t)\,\big(G_t - b(s_t)\big)\Big].
$$

*证明。* 轨迹概率为 $p_\theta(\tau) = p(s_0)\prod_t\pi_\theta(a_t\mid s_t)P(s_{t+1}\mid s_t, a_t)$；只有策略因子依赖 $\theta$，所以 $\nabla\ln p_\theta(\tau) = \sum_t\nabla\ln\pi_\theta(a_t\mid s_t)$。于是 $\nabla J = \nabla\int p_\theta(\tau)G_0(\tau) = \mathbb{E}[G_0\sum_t\nabla\ln\pi_\theta(a_t\mid s_t)]$。两个改进。第一，时间 $t$ 之前得到的奖励在抽出 $a_t$ 之前就已确定，因此它与 $\nabla\ln\pi(a_t\mid s_t)$ 的乘积期望为零；$G_0 = \sum_k\gamma^kr_k$ 在第 $t$ 步那一项中剩下的是 $\sum_{k\ge t}\gamma^kr_k = \gamma^tG_t$（因果性）。第二， $\mathbb{E}_{a_t}[b(s_t)\nabla\ln\pi_\theta(a_t\mid s_t)\mid s_t] = b(s_t)\nabla\sum_{a}\pi_\theta(a\mid s_t) = 0$，因此任何与动作无关的基线都可以减去（其权重 $\gamma^t$ 是常数）。$\blacksquare$

**因子 $\gamma^t$。** 它很容易被漏掉，而许多实现是刻意省略它：$\sum_t\nabla\ln\pi(a_t\mid s_t)(G_t - b)$ 是*另一个*目标函数的梯度——它把每个造访到的状态都当成回合的起点来加权。$\gamma = 1$ 时两者一致。*例子：*一个两步的回合，$t = 0$ 只有一个动作；$t = 1$ 时是 logit 为 $\theta = 0$ 的伯努利动作，若为 1 则得到奖励 1。以 $\gamma = 0.5$，$J = 0.5\,\sigma(\theta)$，$\partial J/\partial\theta = 0.125$。定理给出 0.125；省略 $\gamma^t$ 的式子给出 0.25。主干项目的单步任务只有 $t = 0$，此时 $\gamma^0 = 1$，因此无法揭露这个差异；`test_stage09_discounted_policy_gradient_carries_gamma_t` 检查了两步的情况。

对有限动作集上 logits 为 $z$ 的 softmax 策略，$\nabla_z\ln\pi(a) = e_a - \pi$（单元 03，§3.3.5）。因此其梯度即为 REINFORCE 估计的单样本损失是 $\ell = -(G - b)\ln\pi(a)$，且 $\partial\ell/\partial z = (G - b)(\pi - e_a)$——朝抽到之动作的交叉熵，以其优势缩放。`spine/stage09_post_training.py` 正好实现了这一行，并以有限差分穿过整个 Transformer 检查它。

## 9.3.3 价值函数与贝尔曼方程

在 $\pi$ 下一个状态的**价值（value）**是 $V^\pi(s) = \mathbb{E}_\pi[G_t\mid s_t = s]$，**动作价值（action value）**是 $Q^\pi(s, a) = \mathbb{E}_\pi[G_t\mid s_t = s, a_t = a]$。它们满足**贝尔曼方程（Bellman equation）**（[Bellman 1957](/zh-cn/references/#bellman-1957)）：

$$
V^\pi(s) = \sum_a\pi(a\mid s)\Big[r(s, a) + \gamma\sum_{s'}P(s'\mid s, a)V^\pi(s')\Big],
$$

对固定策略是一个线性方程组；把平均换成 $\max_a$，就是以动态规划求解的最优性方程。$V^\pi(s_t)$ 是最常用的状态依赖基线，但一般而言*不是*使方差最小的那一个：§9.5.1 推导出最优解，是以分数平方加权的动作价值平均，可能与 $V^\pi$ 相去甚远——对 $p = 0.8$、动作 1 得到奖励 1 的伯努利策略，$V = 0.8$，最优基线却是 0.2，并使估计的方差为零（[Greensmith et al. 2004](/zh-cn/references/#greensmith-2004)）。**优势** $A^\pi(s, a) = Q^\pi(s, a) - V^\pi(s)$ 衡量一个动作比策略的平均好多少。演员–评论家（actor–critic）方法以第二个网络学习 $V$，并以 $r_t + \gamma V(s_{t+1}) - V(s_t)$ 或其平滑版本（[Schulman et al. 2016](/zh-cn/references/#schulman-2016)）作为优势——以一点偏差换取比原始回报低得多的方差。

## 9.3.4 KL 正则化目标函数及其最优解

固定一个状态，把 $\pi(\cdot\mid s)$ 写成 $\pi$。考虑

$$
J_\beta(\pi) = \sum_a\pi(a)r(a) - \beta\sum_a\pi(a)\ln\frac{\pi(a)}{\pi_{\text{ref}}(a)} .
$$

**命题 9.1。** 在所有分布中，$J_\beta$ 的最大化者为 $\pi^\star(a) = \pi_{\text{ref}}(a)\,e^{r(a)/\beta}/Z$，其中 $Z = \sum_a\pi_{\text{ref}}(a)e^{r(a)/\beta}$，最大值为 $\beta\ln Z$。

*证明。* 改写为 $J_\beta(\pi) = -\beta\sum_a\pi(a)\ln\frac{\pi(a)}{\pi_{\text{ref}}(a)e^{r(a)/\beta}} = -\beta\,\mathrm{KL}(\pi\,\Vert\,\pi^\star) + \beta\ln Z$。由吉布斯不等式（单元 01，定理 1.1），KL 项非负，且恰在 $\pi = \pi^\star$ 时为零。$\blacksquare$

倾斜后的分布保留参考分布在奖励相同的动作之间的相对偏好，而每多 $\Delta r$ 单位的奖励，就把该动作的胜算乘上 $e^{\Delta r/\beta}$。主干项目的测试检查对 $J_\beta$ 做梯度上升会收敛到 $\pi^\star$，精确到小数点后六位。把这个关系反过来，就得到 DPO 需要的那一步：

$$
r(a) = \beta\ln\frac{\pi^\star(a)}{\pi_{\text{ref}}(a)} + \beta\ln Z .
$$

## 9.3.5 奖励模型与 PPO

**奖励模型。** 收集附带成对回应的提示，以及人类（或 AI）对哪一个较好的标注。以最大化 Bradley–Terry 对数似然 $\sum\ln\sigma\big(r_\phi(y_w) - r_\phi(y_l)\big)$ 拟合 $r_\phi$——对奖励差做逻辑回归（[Christiano et al. 2017](/zh-cn/references/#christiano-2017)；[Stiennon et al. 2020](/zh-cn/references/#stiennon-2020)）。只有差值可被辨识：每个奖励都加上一个常数不会改变任何事。

**PPO。** 接着以 KL 惩罚针对 $r_\phi$ 优化策略。因为每批样本会被重复用于好几个梯度步，正在更新的策略与产生样本的策略不同；近端策略优化（Proximal Policy Optimization，PPO）以重要性比值 $\rho = \pi_\theta(a\mid s)/\pi_{\text{old}}(a\mid s)$ 修正，并在目标函数 $\mathbb{E}[\min(\rho A, \mathrm{clip}(\rho, 1-\epsilon, 1+\epsilon)A)]$ 中把它**截断（clip）**在 $[1-\epsilon, 1+\epsilon]$（[Schulman et al. 2017](/zh-cn/references/#schulman-2017)）。截断移除的是把被抽到之动作的比值推到区间之外的*诱因*；它不是硬性限制。未被抽到之动作的比值，以及相邻策略之间的 KL 散度，都不受它约束，大幅更新仍可能发生——这就是为什么实现会监控 KL 的估计值，并在它变大时提早结束该轮。这条三阶段流程——监督微调、奖励模型、带 KL 的 PPO——产生了第一批被广泛使用、能遵循指令的助理（[Ouyang et al. 2022](/zh-cn/references/#ouyang-2022)）。

## 9.3.6 直接偏好优化

把 §9.3.4 反解出的最优解代入 Bradley–Terry 模型。对同一个提示的两个回应而言，$\beta\ln Z$ 项相等而相消：

$$
P(y_w\succ y_l) = \sigma\Big(\beta\ln\frac{\pi(y_w)}{\pi_{\text{ref}}(y_w)} - \beta\ln\frac{\pi(y_l)}{\pi_{\text{ref}}(y_l)}\Big).
$$

在此模型下最大化观测到之偏好的似然，就得到 **DPO 损失**（[Rafailov et al. 2023](/zh-cn/references/#rafailov-2023)）：

$$
\mathcal L_{\text{DPO}} = -\mathbb{E}\Big[\ln\sigma\big(u\big)\Big],\qquad u = \beta\Big[\big(\ln\pi(y_w) - \ln\pi_{\text{ref}}(y_w)\big) - \big(\ln\pi(y_l) - \ln\pi_{\text{ref}}(y_l)\big)\Big].
$$

没有奖励模型、训练时不需采样、没有 PPO：它是成对数据上的分类损失。它对对数概率的梯度是 $-\beta\sigma(-u)\big(\nabla\ln\pi(y_w) - \nabla\ln\pi(y_l)\big)$——提高偏好的回应、降低无偏好的回应，权重是目前模型把两者排错的程度。对单一词元的答案，两个对数概率梯度中的 softmax 项相消，在 logits 上只剩 $-\beta\sigma(-u)(e_{y_w} - e_{y_l})$；主干项目以有限差分检查它。

## 9.3.7 可验证奖励与群组基线

当正确性能由程序检查——算术、单元测试、证明检查器——奖励就不需要人，也不需要学习出来的模型。对同一个提示采样 $G$ 个答案，以它们的平均奖励作为基线，得到优势 $A_i = r_i - \bar r$（可再除以该群组的标准差），不需要价值网络（[Shao et al. 2024](/zh-cn/references/#shao-2024)）。因为 $\bar r$ 包含 $r_i$ 本身，这个基线并不独立于 $a_i$：期望上它把梯度缩放了 $(G-1)/G$。**留一法（leave-one-out）**基线 $b_i = \frac{1}{G-1}\sum_{j\ne i}r_j$ 独立于 $a_i$，使估计量保持无偏；它与群组平均只差优势上的一个因子 $G/(G-1)$。除以群组标准差则引入另一个依数据而定的缩放。以这种方式在数学与程序上训练大型模型，并让它在作答前写出长串推理，带来了强大的推理表现（[DeepSeek-AI 2025](/zh-cn/references/#deepseek-2025)）**[E]**。主干项目的阶段 09 是这个配方的单词元版本，以批量上的留一法作为基线；它用有错检查器的那次执行，则是警示版本（§9.7.1）。因为一个批量中的 64 个提示各不相同，这个基线是常数，而不是状态的函数——无偏，但降低方差的效果不如逐提示的基线。（v2.1 之前的版本使用单纯的批量平均，使奖励梯度向下偏差 1/64；`test_stage09_batch_mean_baseline_is_biased_and_leave_one_out_is_not` 检查两者。）

:::note[主张检查]
- **主张。** 留一法基线让策略梯度估计量保持无偏。
- **假设。** $G$ 个样本是独立抽出的，因此 $b_i$ 不依赖 $a_i$。
- **检验。** 在一个小问题上精确计算期望梯度，再与各种基线下估计量的期望值比较（`test_stage09_batch_mean_baseline_is_biased_and_leave_one_out_is_not`）。
- **限制。** 无偏不等于方差最小：使方差最小的基线是以分数平方加权的平均奖励（命题 9.2），一般而言它既不同于群组平均，也不同于 $V^\pi$。
:::

