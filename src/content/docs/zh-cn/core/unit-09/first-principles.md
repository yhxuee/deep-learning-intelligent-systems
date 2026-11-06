---
title: "9.2 第一原理"
description: 动作、奖励与策略；为什么采样挡住了反向传播，以及分数函数技巧如何绕过它；基线；连续决策；对预训练模型的 KL 锚定；以及以偏好取代奖励来学习。
sidebar: { label: "9.2 第一原理", order: 2 }
---

## 9.2.1 从目标值到行动后果

监督学习会在预测之前提供理想输出 $y$。若模型给它概率 $p_\theta(y\mid x)$，每个样本的交叉熵损失就是 $-\log p_\theta(y\mid x)$。目标值直接告诉学习器该提高哪个输出。

奖励反馈的接口不同：学习器先选择，再观察这个选择值多少。

| 符号 | 名称 | 主干算术范例 |
|---|---|---|
| $s$ | 状态或背景 | 以“Now Ben has ”结尾的提示 |
| $a$ | 动作 | 抽出的一个下一字符 |
| $\pi_\theta(a\mid s)$ | 策略 | 模型在各字符上的分布 |
| $r(s,a)$ | 奖励 | 字符是 $x+y$ 的数字时为 $1$，否则为 $0$ |

先考虑最简单情况：提示来自固定分布 $d(s)$，一个动作就结束一回合。目标函数为

$$
J(\theta)
=\mathbb E_{s\sim d}
  \mathbb E_{a\sim\pi_\theta(\cdot\mid s)}
  [r(s,a)].
$$

两层期望都很重要。外层决定哪些提示算数；内层则平均目前策略可能做出的行动。策略在某次幸运采样拿到奖励 1，不代表它已经达成 $J=1$。

奖励不必可微。它可以包含相等测试、执行编译器、玩一局游戏，或请人比较两个答案。困难在于：$\theta$ 改变的是*选到某动作的概率*，但实际观察到的奖励附着在已抽出的离散动作上。

:::tip[不要硬造出不存在的差别]
若一支程序能在训练前便宜地产生唯一正确答案，它也能提供监督目标值。这个玩具例子仍采用奖励学习，是为了让估计量可稽核。更强的动机来自结果延迟、许多动作序列都能成功，或反馈只提供排序的情况。
:::

## 9.2.2 为什么反向传播停在样本上

要以梯度上升改善 $J$，我们需要 $\nabla_\theta J$。一个看似自然的计算图是

$$
\theta\longrightarrow \pi_\theta
\longrightarrow \text{抽取 }a
\longrightarrow r(s,a).
$$

一般反向传播无法穿过离散采样决策。参数微小改变时，抽到的符号通常完全不变，偶尔又会突然换成另一个。奖励程序本身也可能不可微。把抽到的词元 ID 当成平滑数值，会凭空捏造问题原本没有的导数。

出路是对*每个可能动作的概率*微分。先固定一个状态，并假设有限动作集合与奖励函数本身不依赖 $\theta$：

$$
\begin{aligned}
\nabla_\theta J_s
&=\nabla_\theta\sum_a\pi_\theta(a\mid s)r(s,a)\\
&=\sum_a r(s,a)\nabla_\theta\pi_\theta(a\mid s)\\
&=\sum_a\pi_\theta(a\mid s)r(s,a)
   \nabla_\theta\log\pi_\theta(a\mid s)\\
&=\mathbb E_{a\sim\pi_\theta}
\left[r(s,a)\nabla_\theta\log\pi_\theta(a\mid s)\right].
\end{aligned}
$$

第三行使用

$$
\nabla_\theta \pi_\theta
=\pi_\theta\nabla_\theta\log\pi_\theta,
$$

它直接来自 $p>0$ 时的 $\nabla\log p=(\nabla p)/p$。

这就是**分数函数（score-function）**或 **REINFORCE** 恒等式（[Williams 1992](/zh-cn/references/#williams-1992)）。这里的“score”是概率分数 $\nabla_\theta\log\pi_\theta(a\mid s)$，**不是**奖励分数。单一样本梯度

$$
\widehat g
=r(s,a)\nabla_\theta\log\pi_\theta(a\mid s)
$$

虽然嘈杂，期望值却等于精确梯度。平均多个独立样本可以降低蒙特卡洛噪声。

### 三动作的完整计算

令 $\pi=(1/3,1/3,1/3)$，奖励为 $r=(1,0,0)$，则 $J=1/3$。若策略是 logit $z$ 上的 softmax，单元 03 给出

$$
\frac{\partial J}{\partial z_i}
=\pi_i(r_i-J),
$$

因此

$$
\nabla_z J
=\left(\frac{2}{9},-\frac{1}{9},-\frac{1}{9}\right).
$$

提高受奖动作的 logit 时，其他动作必须下降，因为概率总和固定为一。REINFORCE 不需要枚举每个动作，也能估计这个向量。面对巨大词表或长轨迹时，避免完整枚举正是关键。

把 REINFORCE 读成**对策略自身样本的奖励加权模仿**很有用，但不完整：负优势会明确降低抽到动作的概率；连续决策中的数据会依赖策略；off-policy 样本还需要校正。

## 9.2.3 基线：和预期相比

估计量可以无偏，却仍然嘈杂到无法有效学习。假设每个奖励都接近 100，而动作之间真正有意义的差别只有 0.1。把分数向量乘上大约 100，会加入巨大波动，却没有多少信息能指出哪个动作较好。

减去一个可以依赖状态、但不能依赖所抽动作的**基线（baseline）** $b(s)$：

$$
\nabla_\theta J_s
=\mathbb E_{a\sim\pi_\theta}
\left[(r(s,a)-b(s))
\nabla_\theta\log\pi_\theta(a\mid s)\right].
$$

为什么期望值不变？固定 $s$ 后，

$$
\begin{aligned}
\mathbb E_a\left[
b(s)\nabla_\theta\log\pi_\theta(a\mid s)
\right]
&=b(s)\sum_a\pi_\theta(a\mid s)
  \nabla_\theta\log\pi_\theta(a\mid s)\\
&=b(s)\nabla_\theta\sum_a\pi_\theta(a\mid s)\\
&=b(s)\nabla_\theta 1=0.
\end{aligned}
$$

因此，基线改变的是估计量方差，不是理想期望值。差值 $r-b$ 是一个**优势估计（advantage estimate）**：正值表示比比较点好，负值表示较差。

回到三动作例子。当 $b=0$ 时，第一个 logit 的单样本梯度方差约为 $0.0988$；改成 $b=1/3$ 后，平均仍为 $2/9$，方差却降到 $0.0247$——只剩四分之一。例题 9.2 会列出每一种结果。

以下三项提醒可以避免常见错误：

1. **动作独立性是条件式的。**学到的价值 $V(s)$ 可以使用，因为它只看状态；偷看已抽动作的基线却可能改变平均。
2. **价值很方便，却不一定让方差最小。**最优常数基线还取决于分数的平方范数；§9.5 会推导。
3. **批量平均可能包含样本本身。**批量大小为 $B$ 时，减去普通奖励批量平均，会把期望梯度缩成 $(B-1)/B$。主干项目改为让每个样本使用其余 $B-1$ 个样本的平均——也就是留一法基线。群组相对方法也可以比较同一提示的多个答案（[Shao et al. 2024](/zh-cn/references/#shao-2024)）。

:::note[合理性检查]
减去基线不是“把坏奖励改成负奖励”，而是把梯度估计量置中。底层目标函数 $J$ 仍然使用原本的奖励。
:::

## 9.2.4 连续决策

语言模型的答案通常包含许多词元。会使用工具的智能体也会改变下一个状态：搜索查询决定观察到哪个网页，而这项观察又影响下一个动作。一条**轨迹（trajectory）**

$$
\tau=(s_0,a_0,r_0,s_1,a_1,r_1,\ldots)
$$

的概率为

$$
p_\theta(\tau)
=p(s_0)\prod_t
\pi_\theta(a_t\mid s_t)
P(s_{t+1}\mid s_t,a_t).
$$

假设环境转移 $P$ 不依赖 $\theta$。对数导数中只有策略因子会留下来：

$$
\nabla_\theta\log p_\theta(\tau)
=\sum_t\nabla_\theta\log\pi_\theta(a_t\mid s_t).
$$

对折扣回报

$$
R(\tau)=\sum_{t=0}^{T-1}\gamma^t r_t,
\qquad 0\le\gamma\le1,
$$

分数函数恒等式先得到 $\mathbb E[R\sum_t\nabla\log\pi_t]$。发生在动作 $a_t$ 之前的奖励不可能由该动作造成，其期望分数贡献为零。移除这些项后，得到从当下开始的回报形式：

$$
\nabla_\theta J
=\mathbb E_\tau\left[
\sum_{t=0}^{T-1}
\gamma^t
\nabla_\theta\log\pi_\theta(a_t\mid s_t)
\bigl(G_t-b(s_t)\bigr)
\right],
$$

其中

$$
G_t=\sum_{k=t}^{T-1}\gamma^{k-t}r_k.
$$

对这里写出的目标函数，外面的 $\gamma^t$ 不能漏掉；漏掉就会改变后期决策的权重。有些教材把它吸收到折扣状态造访分布中；两种惯例都可以，但必须前后一致。

若一个 200 词元的答案只在结尾收到一个奖励，同一个标量结果起初会乘上许多词元的分数向量。正确答案该归功于哪个词元？失败又该怪哪个？这就是**功劳分配问题（credit-assignment problem）**。价值与优势利用未来回报的预测，让比较更局部（§9.3.3），却不会让因果歧义凭空消失。

主干项目的答案只有一个动作，因此刻意拿掉轨迹中的功劳分配问题。这项简化让我们先隔离估计量与奖励设计错误，再把它们和长期视野组合。

## 9.2.5 锚定在模型已知的东西上

只为度量奖励优化的策略可能远离有用的预训练行为、塌缩到狭窄的输出集合，或利用评估器怪癖。一种常见目标函数会惩罚策略偏离固定的**参考策略** $\pi_{\text{ref}}$：

$$
J_\beta(\theta)
=\mathbb E_{s\sim d,\;a\sim\pi_\theta}[r(s,a)]
-\beta\,
\mathbb E_{s\sim d}
\left[
\mathrm{KL}\!\left(
\pi_\theta(\cdot\mid s)
\Vert
\pi_{\mathrm{ref}}(\cdot\mid s)
\right)
\right].
$$

固定一个状态，并允许任意策略时，§9.3.4 会证明最优解是

$$
\pi^\star(a\mid s)
=\frac{
\pi_{\mathrm{ref}}(a\mid s)
\exp(r(s,a)/\beta)}
{\sum_{a'}
\pi_{\mathrm{ref}}(a'\mid s)
\exp(r(s,a')/\beta)}.
$$

奖励以指数方式让参考分布**倾斜（tilt）**：

- 小 $\beta$ 让奖励差异主导；
- 大 $\beta$ 让策略更靠近参考；
- 所有奖励相等时，复现参考分布；
- 若参考策略给某动作的概率恰为零，这个前向 KL 解也不能给它正概率。

这是一个取舍参数，不是普遍安全保证。小 KL 仍可能掩盖罕见但重要提示上的严重改变；大 KL 也可能阻止必要修正。方向同样重要：$\mathrm{KL}(\pi_\theta\Vert\pi_{\mathrm{ref}})$ 不能随意换成反方向。

主干项目在完整条件分布上计算 KL，不只看抽到的单一动作。因此，对这个玩具词表，KL 梯度是精确的。大型语言模型系统常以样本或词元对数比估计序列层级 KL，因而加入额外方差与实现选择。

## 9.2.6 没有奖励时：偏好

有些任务有明确验证器：单元测试通过、定理检查器接受，或算术答案相符。然而，实用、文体与无害等性质通常没有一支完整的程序测试。一种常见接口改用比较：

> 对提示 $x$，回应 $y_w$ 优于回应 $y_l$。

比较不会自动可靠。标注者可能意见不同、误解提示、采用不同价值，或受长度与呈现方式影响。偏好数据是一种需要规程与留出稽核的度量，不是直接读取人类意图。

第一条路以 **Bradley–Terry** 似然拟合标量奖励模型：

$$
P(y_w\succ y_l\mid x)
=\sigma\!\left(
r_\phi(x,y_w)-r_\phi(x,y_l)
\right),
$$

其中 $\sigma(u)=1/(1+e^{-u})$（[Bradley & Terry 1952](/zh-cn/references/#bradley-1952)）。接着针对这个学到的代理奖励优化策略，形成通常称为 RLHF 的“奖励建模加强化学习”流水线（[Christiano et al. 2017](/zh-cn/references/#christiano-2017)；[Ouyang et al. 2022](/zh-cn/references/#ouyang-2022)）。

第二条路利用 KL 正则化最优解，把显式奖励模型从策略训练目标中消去。**直接偏好优化（direct preference optimization，DPO）**比较策略相对于参考模型，将赢家提高得比输家多多少（[Rafailov et al. 2023](/zh-cn/references/#rafailov-2023)）。它没有让奖励假设消失：推导内仍含 Bradley–Terry 偏好模型与 KL 正则化策略关系。§9.3.6 会写出每一步代数。

这带出一个重要区分：

| 反馈对象 | 观察到什么？ | 推断什么？ |
|---|---|---|
| 标量奖励 | 一次采样结果的一个数字 | 哪些动作该变得更可能 |
| 成对偏好 | 赢家与输家 | 潜在效用差或直接策略比值 |
| 监督目标 | 理想输出本身 | 不需要比较模型 |

## 9.2.7 目前为止

| 信号 | 更新时问的问题 | 主干观察 |
|---|---|---|
| 目标字符 | “提高这个指定字符” | 预训练：贪心准确率 $0.3175$ |
| 精确程序化奖励 | “提高抽到而且通过的动作” | REINFORCE：留出贪心准确率 $0.82$ |
| 正确 $\succ$ 错误配对 | “相对于输家与参考，提高赢家” | DPO：留出贪心准确率 $0.78$ |
| 有错的奖励 $\mathbf1\{\text{答案}\ge x+y\}$ | “提高任何通过错误规则的采样数字” | 训练奖励 $0.99$，真实准确率 $0.09$ |

算法没有自己发现理想目标，而是转换供给它的反馈。因此，一份后训练报告至少要分开四条曲线：优化损失、度量训练奖励、相对于参考的漂移，以及不重用训练奖励实现的留出任务指标。

**下一节：**[§9.3 数学构造](/zh-cn/core/unit-09/construction/)把单一动作恒等式推广到 MDP，推导价值与贝尔曼方程，再在不跳过假设的情况下构建 PPO 与 DPO。
