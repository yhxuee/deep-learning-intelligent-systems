---
title: "9.3 数学构建"
description: 马尔可夫决策过程、带基线与优势的策略梯度定理、价值函数与贝尔曼方程、KL 正则化奖励最大化的闭式最优解、由偏好得到的奖励模型、PPO 的截断目标函数、DPO 的推导，以及可验证奖励与群组基线。
sidebar: { label: "9.3 数学构建", order: 3 }
---

## 9.3.1 马尔可夫决策过程

**马尔可夫决策过程（Markov decision process，MDP）**形式化“动作会影响下一步”的决策问题（[Sutton & Barto 2018](/zh-cn/references/#sutton-2018)）：

$$
(\mathcal S,\mathcal A,P,r,\gamma).
$$

| 组件 | 意义 | 文本生成实例 |
|---|---|---|
| 状态 $s_t$ | 行动前可用的信息 | 提示加上已生成前缀 |
| 动作 $a_t$ | 一次选择 | 下一个词元 |
| 转移 $P(s_{t+1}\mid s_t,a_t)$ | 世界如何改变 | 接上词元，或执行工具 |
| 奖励 $r_t$ | 测得的结果 | 验证器分数、偏好模型分数、任务结果 |
| 折扣 $\gamma$ | 后期奖励的相对权重 | 短有限答案可用 $1$；持续任务常小于 $1$ |

**马尔可夫性质**表示：状态含有足够信息，使下一状态的分布在给定 $(s_t,a_t)$ 后，不必再查更早历史。它不是说世界没有记忆。若历史很重要，有效状态就必须摘要它；语言模型前缀直接包含先前所有词元，因此做到这点。

策略 $\pi_\theta(a\mid s)$ 与转移规则共同产生轨迹。从时间 $t$ 起的回报为

$$
G_t
=r_t+\gamma r_{t+1}+\gamma^2r_{t+2}+\cdots
=\sum_{k=0}^{T-t-1}\gamma^k r_{t+k},
$$

回合式目标函数则是

$$
J(\theta)
=\mathbb E_{\tau\sim p_\theta(\tau)}[G_0].
$$

这个期望同时涵盖策略与环境的随机性。对有限回合，数学上可以取 $\gamma=1$；对无限视野，$\gamma<1$ 能让有界奖励的总和收敛，也表达时间偏好。

文本生成的词元转移是确定的：接上所选词元，就固定了下一个前缀。工具使用较不简单，因为同一查询可能返回随机或随时间改变的观察。主干算术任务在一个词元后结束，没有动作会影响后续决策。因此它是 **情境式多臂老虎机（contextual bandit）**，也就是 MDP 的单步边界情况。

:::note[多臂老虎机，还是 MDP？]
若每个提示都独立抽出，一个答案只得到一个分数，使用多臂老虎机模型即可。若动作会改变下一个观察或之后可选的选项，连续 MDP 结构才不可省略。把所有后训练问题都称为“强化学习”，不会让这个差别消失。
:::

## 9.3.2 策略梯度定理

微分之前，先辨认哪些对象依赖 $\theta$：

- 初始状态分布固定；
- 环境转移与奖励不直接依赖 $\theta$；
- 策略可微，而且在使用对数概率之处分配正概率；
- 微分可与有限轨迹总和或积分交换。

这些假设在最后的估计量中看不见，却决定了它的意义。若奖励模型一起更新、环境会随策略改变，或数据来自另一个策略，就可能需要额外项或校正。

**定理 9.1（策略梯度，回合式折扣目标）。** 令 $J(\theta) = \mathbb{E}_{\tau\sim\pi_\theta}[G_0]$，其中对有限长度的回合 $G_t = \sum_{k\ge0}\gamma^kr_{t+k}$。对任何不依赖 $a_t$ 的基线 $b(s_t)$，
$$
\nabla_\theta J(\theta) = \mathbb{E}_{\tau\sim\pi_\theta}\Big[\sum_{t}\gamma^t\,\nabla_\theta\ln\pi_\theta(a_t\mid s_t)\,\big(G_t - b(s_t)\big)\Big].
$$

*证明，第一步：对轨迹微分。* 轨迹概率是

$$
p_\theta(\tau)
=p(s_0)\prod_t
\pi_\theta(a_t\mid s_t)
P(s_{t+1}\mid s_t,a_t).
$$

只有策略因子依赖 $\theta$，所以

$$
\nabla_\theta\log p_\theta(\tau)
=\sum_t\nabla_\theta
\log\pi_\theta(a_t\mid s_t).
$$

分数函数恒等式于是给出

$$
\nabla_\theta J
=\mathbb E_\tau\left[
G_0\sum_t\nabla_\theta
\log\pi_\theta(a_t\mid s_t)
\right].
$$

*第二步：移除过去的奖励。* 时间 $t$ 之前收到的奖励，在抽出 $a_t$ 前就已确定。固定 $s_t$ 后，它与动作分数之乘积的期望为零。$G_0=\sum_k\gamma^k r_k$ 在第 $t$ 步留下的是

$$
\sum_{k\ge t}\gamma^k r_k
=\gamma^tG_t.
$$

这就是因果性或从当下开始的回报步骤。

*第三步：减去基线。* 固定 $s_t$ 后，

$$
\mathbb E_{a_t}\left[
b(s_t)\nabla_\theta\log\pi_\theta(a_t\mid s_t)
\right]
=b(s_t)\nabla_\theta\sum_a\pi_\theta(a\mid s_t)
=0.
$$

乘上 $\gamma^t$ 仍然是零，所以可以减去与动作无关的基线。$\blacksquare$

这条定理是精确的期望恒等式。REINFORCE 以采样轨迹代替期望。因此，定理在其假设下建立无偏性，却不保证低方差、安全探索或正确奖励。

**因子 $\gamma^t$。** 它很容易被漏掉，而许多实现是刻意省略它：$\sum_t\nabla\ln\pi(a_t\mid s_t)(G_t - b)$ 是*另一个*目标函数的梯度——它把每个造访到的状态都当成回合的起点来加权。$\gamma = 1$ 时两者一致。*例子：*一个两步的回合，$t = 0$ 只有一个动作；$t = 1$ 时是 logit 为 $\theta = 0$ 的伯努利动作，若为 1 则得到奖励 1。以 $\gamma = 0.5$，$J = 0.5\,\sigma(\theta)$，$\partial J/\partial\theta = 0.125$。定理给出 0.125；省略 $\gamma^t$ 的式子给出 0.25。主干项目的单步任务只有 $t = 0$，此时 $\gamma^0 = 1$，因此无法揭露这个差异；`test_stage09_discounted_policy_gradient_carries_gamma_t` 检查了两步的情况。

对有限动作集上 logits 为 $z$ 的 softmax 策略，$\nabla_z\ln\pi(a) = e_a - \pi$（单元 03，§3.3.5）。因此其梯度即为 REINFORCE 估计的单样本损失是 $\ell = -(G - b)\ln\pi(a)$，且 $\partial\ell/\partial z = (G - b)(\pi - e_a)$——朝抽到之动作的交叉熵，以其优势缩放。`spine/stage09_post_training.py` 正好实现了这一行，并以有限差分穿过整个 Transformer 检查它。

以梯度**下降**实现时，损失带有上面的负号；描述对期望回报做梯度**上升**时，估计量则是正的。混用这两种正负号惯例，常会写出稳定学到相反策略的程序。

## 9.3.3 价值函数与贝尔曼方程

在 $\pi$ 下一个状态的**价值（value）**是 $V^\pi(s) = \mathbb{E}_\pi[G_t\mid s_t = s]$，**动作价值（action value）**是 $Q^\pi(s, a) = \mathbb{E}_\pi[G_t\mid s_t = s, a_t = a]$。它们满足**贝尔曼方程（Bellman equation）**（[Bellman 1957](/zh-cn/references/#bellman-1957)）：

$$
V^\pi(s) = \sum_a\pi(a\mid s)\Big[r(s, a) + \gamma\sum_{s'}P(s'\mid s, a)V^\pi(s')\Big],
$$

对固定策略，这是一个线性方程组。拆开一次递推后，意思会更清楚：

$$
\text{目前价值}
=\mathbb E[
\text{立即奖励}
+\gamma\,\text{下一状态价值}
].
$$

例如，状态 $s$ 只有一个动作，没有立即奖励，接着确定地移到价值为 1 的终止前状态 $u$。若 $\gamma=0.9$，

$$
V(s)=0+0.9V(u)=0.9.
$$

反复代入就会把递推展开回期望折扣奖励总和。对固定 $\pi$，这是**策略评估（policy evaluation）**。把策略加权平均换成 $\max_a$，是在问最优动作，并得到贝尔曼最优性方程；那是**控制（control）**。悄悄混用两条方程，就改变了问题。

$V^\pi(s_t)$ 是最常用的状态依赖基线，但一般而言*不是*使方差最小的那一个。§9.5.1 推导出的最优解，是以分数平方加权的动作价值平均，可能与 $V^\pi$ 相去甚远。对 $p=0.8$、动作 1 得到奖励 1 的伯努利策略，$V=0.8$，最优常数基线却是 0.2，并使这个一维估计量的方差为零（[Greensmith et al. 2004](/zh-cn/references/#greensmith-2004)）。

**优势**

$$
A^\pi(s,a)=Q^\pi(s,a)-V^\pi(s)
$$

衡量一个动作比策略在该状态下的平均好多少。演员–评论家（actor–critic）方法以**评论者**学习价值近似，再更新策略**行动者**。时间差分残差

$$
\delta_t
=r_t+\gamma V_\phi(s_{t+1})-V_\phi(s_t)
$$

在 $V_\phi=V^\pi$ 时，条件平均等于正确优势；使用近似评论者时则可能有偏。广义优势估计以参数 $\lambda$ 组合多种时间差分视野，在方差与偏差间取舍（[Schulman et al. 2016](/zh-cn/references/#schulman-2016)）；它不会普遍地只“加入一点”固定大小的偏差。

## 9.3.4 KL 正则化目标函数及其最优解

固定一个状态，把 $\pi(\cdot\mid s)$ 写成 $\pi$。考虑

$$
J_\beta(\pi) = \sum_a\pi(a)r(a) - \beta\sum_a\pi(a)\ln\frac{\pi(a)}{\pi_{\text{ref}}(a)} .
$$

假设 $\beta>0$、奖励有限，而且优化只在 $\pi_{\text{ref}}$ 的支撑集上进行。否则，只要 $\pi$ 在参考概率为零之处分配正质量，前向 KL 就是无限大。

**命题 9.1。** 在有支撑的分布中，$J_\beta$ 的最大化者为

$$
\pi^\star(a)
=\frac{\pi_{\text{ref}}(a)e^{r(a)/\beta}}{Z},
\qquad
Z=\sum_a\pi_{\text{ref}}(a)e^{r(a)/\beta},
$$

最大值为 $\beta\ln Z$。

*证明。* 因为 $\pi_{\text{ref}}(a)e^{r(a)/\beta}=Z\pi^\star(a)$，

$$
\begin{aligned}
J_\beta(\pi)
&=-\beta\sum_a\pi(a)
\log\frac{\pi(a)}
{\pi_{\text{ref}}(a)e^{r(a)/\beta}}\\
&=-\beta\sum_a\pi(a)
\log\frac{\pi(a)}
{Z\pi^\star(a)}\\
&=-\beta\,\mathrm{KL}(\pi\Vert\pi^\star)
+\beta\log Z.
\end{aligned}
$$

由吉布斯不等式（单元 01，定理 1.1），KL 非负，且恰在 $\pi=\pi^\star$ 时为零。$\blacksquare$

归一化常数不只是簿记。它让概率总和为一，也给出最优目标函数值。胜算形式通常最好解读：

$$
\frac{\pi^\star(a)}{\pi^\star(b)}
=\frac{\pi_{\text{ref}}(a)}
       {\pi_{\text{ref}}(b)}
\exp\!\left(
\frac{r(a)-r(b)}{\beta}
\right).
$$

奖励相等的动作保留参考胜算；奖励优势 $\Delta r$ 则把胜算乘上 $e^{\Delta r/\beta}$。若 $\Delta r=1$ 且 $\beta=0.5$，乘子是 $e^2\approx7.39$。这不表示较好动作的概率变成 1，因为参考胜算与所有其他动作仍然存在。

主干项目的测试检查对 $J_\beta$ 做梯度上升会收敛到 $\pi^\star$，精确到小数点后六位。把这个关系反过来，就得到 DPO 需要的那一步：

$$
r(a) = \beta\ln\frac{\pi^\star(a)}{\pi_{\text{ref}}(a)} + \beta\ln Z .
$$

加法项 $\beta\log Z$ 依赖状态，却不依赖动作。比较同一提示的两个回应时，它会相消。

## 9.3.5 奖励模型与 PPO

经典语言模型 RLHF 流水线分开三种数据与目标函数：

| 阶段 | 数据 | 目标函数 |
|---|---|---|
| 监督微调 | 提示—回应示范 | 模仿有用的回应分布 |
| 奖励建模 | 提示加上偏好与非偏好回应 | 预测成对偏好 |
| 策略优化 | 不断改变的策略所产生的样本 | 提高学到的奖励，同时控制漂移 |

分开各阶段，才能看见各自的失败模式。奖励模型在验证集表现好，不代表优化它之后，在分布外仍能保留人类偏好。

**奖励模型。** 收集提示 $x$、回应 $(y_w,y_l)$，以及人类或 AI 对 $y_w$ 较佳的标注。以最小化下式拟合 $r_\phi$：

$$
\mathcal L_{\text{RM}}
=-\mathbb E\left[
\log\sigma\!\left(
r_\phi(x,y_w)-r_\phi(x,y_l)
\right)
\right].
$$

这就是对奖励差做逻辑回归（[Christiano et al. 2017](/zh-cn/references/#christiano-2017)；[Stiennon et al. 2020](/zh-cn/references/#stiennon-2020)）。只有差值能被识别：对所有回应奖励加入同一个依状态常数，不会改变比较。奖励尺度也会影响策略优化，因为它改变了奖励与 KL 之间的取舍。

**PPO。** 假设一批样本由旧策略 $\pi_{\text{old}}$ 产生，但优化已把目前策略移到 $\pi_\theta$。每个动作的重要性比值

$$
\rho_t(\theta)
=\frac{\pi_\theta(a_t\mid s_t)}
       {\pi_{\text{old}}(a_t\mid s_t)}
$$

校正这项分布改变的一阶代理目标。近端策略优化（Proximal Policy Optimization，PPO）使用

$$
L_{\text{clip}}(\theta)
=\mathbb E\left[
\min\!\left(
\rho_t A_t,\;
\operatorname{clip}(\rho_t,1-\epsilon,1+\epsilon)A_t
\right)
\right]
$$

（[Schulman et al. 2017](/zh-cn/references/#schulman-2017)）。

解读最小值之前，要先看 $A_t$ 的正负：

- 若 $A_t>0$，把抽到动作的比值提高到 $1+\epsilon$ 以上，不再增加代理收益；
- 若 $A_t<0$，把比值降到 $1-\epsilon$ 以下，也不再增加代理收益。

截断移除的是采样代理中的一项*诱因*，不是对参数、未抽到动作或总 KL 的硬限制。共享神经网络的一次更新可以移动许多其他概率。因此，实现通常还会监控 KL 估计，在它过大时提早结束该轮，并另加参考策略 KL 惩罚。

监督微调、奖励模型与带 KL 的 PPO 流水线曾用于具影响力的指令遵循系统（[Ouyang et al. 2022](/zh-cn/references/#ouyang-2022)）**[E]**。这项历史结果支持流水线可行，不表示每个偏好学习问题都必须使用 PPO。

## 9.3.6 直接偏好优化

对回应序列 $y=(y_1,\ldots,y_T)$，

$$
\log\pi_\theta(y\mid x)
=\sum_{t=1}^{T}
\log\pi_\theta(y_t\mid x,y_{<t}).
$$

DPO 使用这个完整序列对数概率，通常会用注意力遮罩排除提示与填补词元。它不是只比较最后一个词元。

现在把 §9.3.4 反解出的最优解代入 Bradley–Terry 模型。对同一个提示的两个回应而言，$\beta\log Z(x)$ 项相等而相消：

$$
P(y_w\succ y_l) = \sigma\Big(\beta\ln\frac{\pi(y_w)}{\pi_{\text{ref}}(y_w)} - \beta\ln\frac{\pi(y_l)}{\pi_{\text{ref}}(y_l)}\Big).
$$

在此模型下最大化观测到之偏好的似然，就得到 **DPO 损失**（[Rafailov et al. 2023](/zh-cn/references/#rafailov-2023)）：

$$
\mathcal L_{\text{DPO}} = -\mathbb{E}\Big[\ln\sigma\big(u\big)\Big],\qquad u = \beta\Big[\big(\ln\pi(y_w) - \ln\pi_{\text{ref}}(y_w)\big) - \big(\ln\pi(y_l) - \ln\pi_{\text{ref}}(y_l)\big)\Big].
$$

这里没有另行训练的奖励模型、每个优化步骤内不需要 on-policy 采样，也没有 PPO：更新是在已收集成对数据上的分类损失。它对对数概率的梯度是 $-\beta\sigma(-u)\big(\nabla\ln\pi(y_w) - \nabla\ln\pi(y_l)\big)$——提高偏好的回应、降低无偏好的回应，权重是目前模型把两者排错的程度。对单一词元答案，两个对数概率梯度中的 softmax 项相消，在 logits 上只剩 $-\beta\sigma(-u)(e_{y_w} - e_{y_l})$；主干项目以有限差分检查它。

初始化时 $\pi=\pi_{\text{ref}}$，所以 $u=0$、损失为 $\log2\approx0.6931$、logistic权重为 $\sigma(0)=1/2$。若目前策略已经提高赢家的相对对数胜算，$u>0$，该配对权重就变小；若排序方向错误，$u<0$，权重便增大。

这里的“没有奖励模型”是指策略优化期间，不必执行一个另行训练的标量网络。推导仍假设：

1. 对某个潜在奖励而言，偏好符合 Bradley–Terry 形式；
2. 目标策略与固定参考之间满足 KL 正则化最优解的关系；
3. 赢家与输家使用相同提示，因此配分函数项能相消；
4. 离线偏好配对涵盖想改善的行为。

违反这些假设，不一定会让数值优化失败；它会改变我们能把损失解读成正在估计什么。DPO 对固定数据集而言也是离线的，因此不会自动为更新后策略产生的新行为收集反馈。

## 9.3.7 可验证奖励与群组基线

当正确性能由程序检查——算术、单元测试、证明检查器——奖励就不需要人，也不需要学习出来的模型。对同一个提示采样 $G$ 个答案，以它们的平均奖励作为基线，得到优势 $A_i = r_i - \bar r$（可再除以该群组的标准差），不需要价值网络（[Shao et al. 2024](/zh-cn/references/#shao-2024)）。因为 $\bar r$ 包含 $r_i$ 本身，这个基线并不独立于 $a_i$：期望上它把梯度缩放了 $(G-1)/G$。**留一法（leave-one-out）**基线 $b_i = \frac{1}{G-1}\sum_{j\ne i}r_j$ 独立于 $a_i$，使估计量保持无偏；它与群组平均只差优势上的一个因子 $G/(G-1)$。除以群组标准差则引入另一个依数据而定的缩放。以这种方式在数学与程序上训练大型模型，并让它在作答前写出长串推理，带来了强大的推理表现（[DeepSeek-AI 2025](/zh-cn/references/#deepseek-2025)）**[E]**。主干项目的阶段 09 是这个配方的单词元版本，以批量上的留一法作为基线；它用有错检查器的那次执行，则是警示版本（§9.7.1）。因为一个批量中的 64 个提示各不相同，这个基线是常数，而不是状态的函数——无偏，但降低方差的效果不如逐提示的基线。（v2.1 之前的版本使用单纯的批量平均，使奖励梯度向下偏差 1/64；`test_stage09_batch_mean_baseline_is_biased_and_leave_one_out_is_not` 检查两者。）

不用概率论也能直接看出缩放。因为

$$
\bar r=\frac1G\left(r_i+\sum_{j\ne i}r_j\right),
$$

群组平均优势为

$$
r_i-\bar r
=\frac{G-1}{G}
\left(
r_i-\frac1{G-1}\sum_{j\ne i}r_j
\right)
=\frac{G-1}{G}(r_i-b_i).
$$

因此，对固定群组而言，群组平均与留一法优势指向同一方向，却相差已知因子。在优化器中，尺度会与学习率、截断及归一化交互；“方向相同”不会让每次训练都完全一样。

群组也必须含有信息：

- 若所有采样回应都失败，二元奖励全是 0，置中优势也全是 0；
- 若所有回应都通过，奖励全是 1，置中优势仍全是 0；
- 除非加入另一个信号，只有群组内奖励有差异时才会学习。

若还要把优势除以群组标准差，全相等群组的分母也是零；实现必须明订惯例，例如跳过归一化或加入 $\varepsilon$ 下限。这造成探索需求：只有策略抽到好与坏两种结果后，验证器才能区分它们。若正确解的概率几乎为零，相对基线无法凭空发明解答。

:::note[主张检查]
- **主张。** 留一法基线让策略梯度估计量保持无偏。
- **假设。** $G$ 个样本是独立抽出的，因此 $b_i$ 不依赖 $a_i$。
- **检验。** 在一个小问题上精确计算期望梯度，再与各种基线下估计量的期望值比较（`test_stage09_batch_mean_baseline_is_biased_and_leave_one_out_is_not`）。
- **限制。** 无偏不等于方差最小：使方差最小的基线是以分数平方加权的平均奖励（命题 9.2），一般而言它既不同于群组平均，也不同于 $V^\pi$。
:::

