---
title: "H · 马尔可夫链与决策过程"
description: 马尔可夫链、平稳分布与混合；马尔可夫决策过程、价值函数与作为线性方程组的贝尔曼方程；压缩映射与价值迭代；折扣占用分布与策略梯度定理的回合形式与占用形式；基线及其偏差与方差；熵与 KL 正则化的决策问题。
sidebar: { label: "H · 马尔可夫决策过程", order: 8 }
---

单元 09 把后训练当作强化学习来处理，而且主要处理单步问题，也就是答案只是一个词元。本附录以一般形式给出多步理论：马尔可夫结构、贝尔曼方程、价值迭代为什么收敛、折扣因子放在正确位置的策略梯度定理，以及基线在什么条件下不会让梯度产生偏差。例题中的数字由 `labs/test_appendices.py` 检查，策略梯度的部分则由 `spine/test_spine.py` 检查。

| 小节 | 用于 |
|---|---|
| H.1 马尔可夫链 | 01（n-gram 模型）、07（序列模型）、08（祖先采样） |
| H.2–H.3 MDP、贝尔曼方程、价值迭代 | 09（§9.3.1、§9.3.3） |
| H.4 策略梯度定理 | 09（§9.2.2、§9.3.2） |
| H.5 基线 | 09（§9.2.3、§9.3.7、§9.5.1） |
| H.6 正则化决策问题 | 09（§9.3.4、§9.5.2） |

## H.1 马尔可夫链

若有限集合中的状态序列 $s_0, s_1, \dots$ 满足 $\Pr(s_{t+1}\mid s_0,\dots,s_t) = \Pr(s_{t+1}\mid s_t)$，它就是**马尔可夫链**。时间齐次的链由行随机的**转移矩阵** $P_{ij} = \Pr(s_{t+1} = j\mid s_t = i)$ 描述。状态上的分布写成行向量 $\mu_t$，依 $\mu_{t+1} = \mu_tP$ 演化。

二元语法模型是字符上的马尔可夫链；$n$-gram 模型是 $(n-1)$ 元组上的马尔可夫链；而自回归 Transformer 是以整个前缀为状态的马尔可夫链。祖先采样（单元 08，§8.3.1）就是执行这条链。

**平稳分布**满足 $\pi = \pi P$。若链是**不可约的**（每个状态都能到达其他每个状态）且**非周期的**（回到某状态的时间不限于某个 $d > 1$ 的倍数），则 $\pi$ 唯一，且从任何初始分布出发都有 $\mu_t\to\pi$（[Levin & Peres 2017](/zh-cn/references/#levin-2017)）。速度由 $P$ 第二大的特征值模决定：到 $\pi$ 的距离大致像 $|\lambda_2|^t$ 那样缩小。

**例题 H.1。** 两状态链 $P = \begin{pmatrix}0.9 & 0.1\\0.3 & 0.7\end{pmatrix}$ 的平稳分布为 $(0.75, 0.25)$——一般而言，若离开状态 1 的转换概率为 $a$、离开状态 2 的为 $b$，平稳分布为 $(b, a)/(a + b)$——第二特征值为 $1 - a - b = 0.6$。从状态 1 出发，位于状态 1 的概率为 $0.75 + 0.25\times0.6^t$：$1, 0.9, 0.84, 0.804, \dots$

若 $\pi_iP_{ij} = \pi_jP_{ji}$，链就相对于 $\pi$ 满足**细致平衡**；对 $i$ 求和即可证明此时 $\pi$ 是平稳的。马尔可夫链蒙地卡罗方法就是为所要的 $\pi$ 构造具有这个性质的链。

## H.2 马尔可夫决策过程与贝尔曼方程

**马尔可夫决策过程**加上了动作与奖励：在状态 $s$ 时代理采取动作 $a$，得到奖励 $r(s, a)$，并移到 $s'\sim P(\cdot\mid s, a)$。**策略** $\pi(a\mid s)$ 选择动作。以折扣 $\gamma\in[0, 1)$，从时间 $t$ 起的**回报**为 $G_t = \sum_{k\ge0}\gamma^kr_{t+k}$，状态的**价值**为 $V^\pi(s) = \mathbb{E}_\pi[G_t\mid s_t = s]$（[Sutton & Barto 2018](/zh-cn/references/#sutton-2018)；[Puterman 1994](/zh-cn/references/#puterman-1994)）。

在固定策略下，状态构成一条马尔可夫链，其转移矩阵为 $P_\pi(s, s') = \sum_a\pi(a\mid s)P(s'\mid s, a)$，期望奖励为 $r_\pi(s) = \sum_a\pi(a\mid s)r(s, a)$。把回报拆成第一个奖励与其余部分，就得到**贝尔曼期望方程**
$$
V^\pi = r_\pi + \gamma P_\pi V^\pi\quad\Longleftrightarrow\quad V^\pi = (I - \gamma P_\pi)^{-1}r_\pi .
$$
逆矩阵存在，因为 $\rho(\gamma P_\pi)\le\gamma < 1$（随机矩阵的谱半径为 1），其 Neumann 级数 $\sum_t\gamma^tP_\pi^t$ 就是期望的折扣造访次数。

**例题 H.2。** 两个状态，$P_\pi = \begin{pmatrix}0.9 & 0.1\\0.5 & 0.5\end{pmatrix}$，状态 1 的奖励为 0、状态 2 为 1，$\gamma = 0.9$。解 $\begin{pmatrix}0.19 & -0.09\\-0.45 & 0.55\end{pmatrix}V = \begin{pmatrix}0\\1\end{pmatrix}$ 得 $V = (45/32, 95/32) = (1.406, 2.969)$。状态 2 的价值是状态 1 的两倍以上，虽然每次造访只付 1，因为从状态 1 很少到达它。

**动作价值** $Q^\pi(s, a) = r(s, a) + \gamma\sum_{s'}P(s'\mid s, a)V^\pi(s')$ 满足 $V^\pi(s) = \sum_a\pi(a\mid s)Q^\pi(s, a)$，而**优势**为 $A^\pi = Q^\pi - V^\pi$。**贝尔曼最优性方程**
$$
V^\star(s) = \max_a\Big[r(s, a) + \gamma\sum_{s'}P(s'\mid s, a)V^\star(s')\Big]
$$
刻画最优价值；任何相对于 $V^\star$ 为贪心的策略都是最优的。

## H.3 压缩映射与价值迭代

把最优性方程的右边写成作用在价值函数上的算子 $\mathcal T$。

**定理 H.1。** $\mathcal T$ 在上确界范数下是 $\gamma$-压缩：$\lVert\mathcal TU - \mathcal TV\rVert_\infty\le\gamma\lVert U - V\rVert_\infty$。

*证明。* 对每个 $s$，$|\max_af_a - \max_ag_a|\le\max_a|f_a - g_a|$。应用到 $f_a = r + \gamma\sum_{s'}PU$ 与 $g_a = r + \gamma\sum_{s'}PV$，差至多为 $\gamma\max_a\sum_{s'}P(s'\mid s, a)|U(s') - V(s')|\le\gamma\lVert U - V\rVert_\infty$。$\blacksquare$

由 Banach 不动点定理，$\mathcal T$ 有唯一的不动点 $V^\star$，而**价值迭代** $V_{k+1} = \mathcal TV_k$ 以 $\lVert V_k - V^\star\rVert_\infty\le\gamma^k\lVert V_0 - V^\star\rVert_\infty$ 收敛到它。同样的论证，把 $\max_a$ 换成 $\pi$ 下的平均，就证明了迭代期望方程会收敛到 $V^\pi$。

**例题 H.3。** 对例题 H.2，从 $V_0 = 0$ 出发，界 $0.9^k\times2.969 < 10^{-3}$ 需要 $k\ge76$；实际误差在 $k = 71$ 时第一次降到 $10^{-3}$ 以下。折扣是使问题良定的原因：当 $\gamma\to1$，压缩变弱，迭代次数像 $1/(1 - \gamma)$ 那样增长。

**策略迭代**交替进行 $V^\pi$ 的精确评估（一次线性求解）与贪心改进；**策略改进定理**保证每个新策略在每个状态上都至少一样好，而对有限 MDP，这个过程在有限步内终止于最优策略。

## H.4 策略梯度定理

令 $\pi_\theta$ 对 $\theta$ 可微，$J(\theta) = \mathbb{E}_{\pi_\theta}[G_0]$，回合从固定的初始分布开始。

**定理 H.2（回合形式）。** 对有限长度的回合与任何不依赖 $a_t$ 的基线 $b(s_t)$，
$$
\nabla_\theta J = \mathbb{E}\Big[\sum_t\gamma^t\,\nabla_\theta\ln\pi_\theta(a_t\mid s_t)\,\big(G_t - b(s_t)\big)\Big].
$$
证明即单元 09 定理 9.1：对轨迹分布应用分数函数恒等式，再移除早于 $t$ 的奖励（它们不依赖 $a_t$），第 $t$ 步那一项剩下 $\sum_{k\ge t}\gamma^kr_k = \gamma^tG_t$。

**折扣占用分布。** 定义 $d^\pi_\gamma(s) = (1 - \gamma)\sum_t\gamma^t\Pr(s_t = s)$，一个状态上的概率分布，较早的造访权重较大。

**定理 H.3（占用形式）。** 对 $\gamma < 1$，
$$
\nabla_\theta J = \frac{1}{1 - \gamma}\,\mathbb{E}_{s\sim d^\pi_\gamma,\ a\sim\pi_\theta(\cdot\mid s)}\big[Q^\pi(s, a)\,\nabla_\theta\ln\pi_\theta(a\mid s)\big].
$$
*证明概要。* 对 $V^\pi(s) = \sum_a\pi(a\mid s)Q^\pi(s, a)$ 微分并用 $Q^\pi = r + \gamma PV^\pi$：
$\nabla V^\pi(s) = \sum_a\nabla\pi(a\mid s)Q^\pi(s, a) + \gamma\sum_a\pi(a\mid s)\sum_{s'}P(s'\mid s, a)\nabla V^\pi(s')$。展开递推，第一项在所有未来状态上累积，每个以 $\gamma^t\Pr(s_t = s)$ 加权；把权重归一化就得到 $d^\pi_\gamma$。$\blacksquare$

两种形式一致：定理 H.2 中的 $\gamma^t$ 正是定理 H.3 的占用加权，而 $\mathbb{E}[\gamma^tG_t\mid s_t, a_t] = \gamma^tQ^\pi(s_t, a_t)$。许多实现**省略 $\gamma^t$**，并从未折扣的造访分布采样状态；结果不是 $J$ 的梯度，而是另一个加权方式不同之目标的梯度（$\gamma = 1$ 时两者一致）。**例题 H.4**（单元 09，§9.3.2）：$t = 0$ 只有一个动作；$t = 1$ 是 logit 为 $\theta = 0$ 的伯努利动作，只有为 1 时才得到奖励；$\gamma = 0.5$。真实导数为 $0.5\,\sigma'(0) = 0.125$；不含 $\gamma^t$ 的式子给出 $0.25$。

## H.5 基线：偏差与方差

**无偏性。** 只要 $\mathbb{E}[b\,\nabla\ln\pi_\theta(a_t\mid s_t)] = 0$，减去 $b$ 就不改变梯度。若 $b$ 是 $s_t$ 的函数（以及任何在抽出 $a_t$ 之前就已确定、或与它独立抽出的东西），这就成立，因为 $\sum_a\pi(a\mid s)\nabla\ln\pi(a\mid s) = \nabla\sum_a\pi(a\mid s) = 0$。若 $b$ 依赖 $a_t$——包括通过样本自己的奖励——它就不成立。

**批量与群组平均。** 以 $B$ 个独立样本，并对每个样本都用基线 $\bar r = \frac1B\sum_jr_j$，$\bar r$ 中样本自身奖励的那一份与 $a_i$ 相关，而对独立同分布样本，
$$
\mathbb{E}\Big[\frac1B\sum_i(r_i - \bar r)\nabla\ln\pi(a_i)\Big] = \Big(1 - \frac1B\Big)\nabla J .
$$
**留一法**基线 $b_i = \frac1{B-1}\sum_{j\ne i}r_j$ 独立于 $a_i$ 且无偏，它与批量平均只差优势上的一个因子 $B/(B - 1)$。除以群组标准差的群组相对方法，会引入另一个依数据而定的缩放。主干项目使用留一法基线（[勘误](/zh-cn/start/errata/)，K03）。

**方差。** 在单一座标的常数基线中，使方差最小的选择是以分数加权的平均 $b^\star = \mathbb{E}[r\,s^2]/\mathbb{E}[s^2]$，$s = \partial\ln\pi$；对状态依赖基线与整个梯度，$b^\star(s) = \mathbb{E}_a[Q^\pi(s, a)\lVert\nabla\ln\pi(a\mid s)\rVert^2]/\mathbb{E}_a[\lVert\nabla\ln\pi(a\mid s)\rVert^2]$（[Greensmith et al. 2004](/zh-cn/references/#greensmith-2004)）。价值 $V^\pi(s)$ 是常见的实用选择，而不是最优选择。**例题 H.5：**对 logit 参数化、$p = 0.8$、动作 1 奖励为 1 的伯努利策略，$V = 0.8$ 而 $b^\star = 0.2$，后者使单样本估计对任一动作都恰好是 $0.16$（单元 09，§9.5.1）。

## H.6 熵与 KL 正则化的决策问题

在每一步的奖励上加上 $\tau H(\pi(\cdot\mid s))$——或减去 $\beta\,\mathrm{KL}(\pi(\cdot\mid s)\,\Vert\,\pi_{\text{ref}}(\cdot\mid s))$——就得到正则化的 MDP。单步时，吉布斯变分原理（附录 F.6）解出它：$\pi^\star\propto\pi_{\text{ref}}\,e^{r/\beta}$，值为 $\beta\ln Z$（单元 09，命题 9.1）。多步时，同样的原理在每个状态上适用，只是把 $r$ 换成动作价值，得到**软贝尔曼方程**
$$
V^\star(s) = \beta\ln\sum_a\pi_{\text{ref}}(a\mid s)\exp\Big(\frac{r(s, a) + \gamma\,\mathbb{E}_{s'}V^\star(s')}{\beta}\Big),
$$
其算子同样是 $\gamma$-压缩，而最优策略为 $\pi^\star(a\mid s)\propto\pi_{\text{ref}}(a\mid s)\exp(Q^\star(s, a)/\beta)$。精确解保有 $\pi_{\text{ref}}$ 的全支撑；实践中多样性的丧失来自小的 $\beta$、受限的策略类别与不完整的优化（单元 09，§9.5.2；附录 D.7）。

## 习题

**H.1** 求 $P = \begin{pmatrix}0.5 & 0.5 & 0\\0.25 & 0.5 & 0.25\\0 & 0.5 & 0.5\end{pmatrix}$ 的平稳分布，并检查细致平衡。

**H.2** 证明 $(I - \gamma P_\pi)^{-1}$ 的元素非负，且每一行的总和为 $1/(1 - \gamma)$。

**H.3** 对例题 H.2，分别以 $\gamma = 0.5$ 与 $\gamma = 0.99$ 计算 $V^\pi$。哪个状态的价值改变较多？为什么？

**H.4** 证明期望算子 $V\mapsto r_\pi + \gamma P_\pi V$ 在上确界范数下是 $\gamma$-压缩。

**H.5** 明确计算 $d^\pi_\gamma$ 与 $Q^\pi$，在例题 H.4 上验证定理 H.3。

**H.6** ★ 证明 H.6 的软贝尔曼算子是 $\gamma$-压缩。*（提示：log-sum-exp 在上确界范数下是 1-Lipschitz 的。）*

**解答。** **H.1** $\pi = (0.25, 0.5, 0.25)$；细致平衡成立（每对相邻状态都有 $0.25\times0.5 = 0.5\times0.25$）。**H.3** $\gamma = 0.5$：$V = (0.125, 1.375)$；$\gamma = 0.99$：$V = (16.39, 18.05)$——两者都随有效时程 $1/(1-\gamma)$ 增长，而状态 1 的价值从状态 2 的 9% 升到 91%，因为它最终造访状态 2 的价值，如今几乎与立即造访一样多。

## 延伸阅读

[Sutton & Barto 2018](/zh-cn/references/#sutton-2018) 第 3–4 章与第 13 章；[Puterman 1994](/zh-cn/references/#puterman-1994) 为 MDP 的完整理论；[Levin & Peres 2017](/zh-cn/references/#levin-2017) 讲马尔可夫链与混合；[Greensmith et al. 2004](/zh-cn/references/#greensmith-2004) 讲策略梯度的方差降低。
