---
title: "1.3 数学构建"
description: 总体风险与经验风险、交叉熵 = 熵 + KL 与 Gibbs 不等式、以拉格朗日乘子推导最大似然二元语法、把加 α 平滑看成 Dirichlet 后验均值、Hoeffding 不等式、挑选的代价，以及留出数据损失的标准误。
sidebar: { label: "1.3 数学构建", order: 3 }
---

本节把 §1.2 转化为数学。每个小节陈述一个结果、完整证明，并说明它对二元语法的意义。

## 1.3.1 设定：分布、预测器与风险

数据是从未知的**数据分布** $p_{\text{data}}$ 抽出的配对 $(x, y)$：对语言模型而言，$x$ 是上下文（对二元语法来说就是前一个字符），$y$ 是下一个字符。**预测器** $q$ 把每个 $x$ 对应到 $y$ 的 $V$ 种可能值上的一个分布 $q(\cdot\mid x)$。$q$ 在单一样本上的**损失**是 $\ell(q; x, y) = -\ln q(y\mid x)$。

我们在意的是在*新*样本上的期望损失，即**总体风险（population risk）**：

$$
L(q) = \mathbb{E}_{(x,y)\sim p_{\text{data}}}\big[-\ln q(y\mid x)\big].
$$

我们无法计算它，因为 $p_{\text{data}}$ 未知。我们能计算的是样本 $\mathcal{D} = \{(x_i, y_i)\}_{i=1}^n$ 上的**经验风险（empirical risk）**：

$$
\hat L(q) = \frac{1}{n}\sum_{i=1}^n -\ln q(y_i\mid x_i).
$$

若样本是从 $p_{\text{data}}$ 抽出的 i.i.d. 样本，而且 $q$ 是在*没看过* $\mathcal{D}$ 的情况下选定的，则 $\hat L(q)$ 是 $L(q)$ 的无偏估计：由期望值的线性，$\mathbb{E}[\hat L(q)] = L(q)$。这一行就是留出数据评估之所以有效的原因，也是训练损失无效的原因：当 $q$ 是*利用* $\mathcal{D}$ 选出时，这个期望值论证就不再适用（§1.3.6）。

**经验风险最小化（empirical risk minimization, ERM）**从一个模型族 $\mathcal{Q}$ 中选出让训练集上 $\hat L$ 最小的 $q$。学习理论问的是：ERM 选出的 $q$ 什么时候也有小的 $L$？

## 1.3.2 交叉熵、熵与 KL 散度

固定一个上下文 $x$ 并在记号中省略它。令 $p$ 为 $y$ 的真实分布，$q$ 为模型的分布。$q$ 在 $p$ 下的期望对数损失是**交叉熵（cross-entropy）**

$$
H(p, q) = -\sum_{y} p(y)\ln q(y).
$$

加上再减去 $\sum_y p(y)\ln p(y)$，得到

$$
H(p, q) = \underbrace{-\sum_y p(y)\ln p(y)}_{H(p)\ \text{（熵）}} \;+\; \underbrace{\sum_y p(y)\ln\frac{p(y)}{q(y)}}_{\mathrm{KL}(p\,\Vert\, q)\ \text{（KL 散度）}} .
$$

**定理 1.1（Gibbs 不等式）。** $\mathrm{KL}(p\,\Vert\,q)\ge 0$，等号成立当且仅当 $q = p$。

*证明。* $\ln$ 是凹函数，所以由 Jensen 不等式，对任何正随机变量 $Z$ 有 $\mathbb{E}_p[\ln Z] \le \ln \mathbb{E}_p[Z]$。取 $Z = q(y)/p(y)$，范围限于 $p(y) > 0$ 的结果：

$$
-\mathrm{KL}(p\,\Vert\,q) = \sum_{y:\,p(y)>0} p(y)\ln\frac{q(y)}{p(y)} \le \ln\sum_{y:\,p(y)>0} p(y)\frac{q(y)}{p(y)} = \ln\sum_{y:\,p(y)>0} q(y) \le \ln 1 = 0 .
$$

Jensen 不等式取等号需要 $q(y)/p(y)$ 在 $p>0$ 处为常数，最后一步取等号则需要 $q$ 在 $p = 0$ 处没有概率质量；两者合起来就是 $q = p$。$\blacksquare$

**推论 1.1。** 期望对数损失 $H(p,q)$ 恰在 $q = p$ 时最小，最小值为熵 $H(p)$。每个模型的总体风险都可分解为

$$
L(q) = \underbrace{\mathbb{E}_x\big[H(p(\cdot\mid x))\big]}_{\text{不可约减}} + \underbrace{\mathbb{E}_x\big[\mathrm{KL}\big(p(\cdot\mid x)\,\Vert\, q(\cdot\mid x)\big)\big]}_{\text{可约减}} .
$$

这有两个贯穿全书的后果。第一，没有任何模型的留出数据损失能低于数据的条件熵：存在一个下限，而且通常不是零（单元 06 把它称为缩放定律中的*不可约减损失* $E$）。第二，超出下限的**额外**损失是一个 KL 散度——度量模型的信念离真相多远。对数损失之所以是 §1.2.3 所说的恰当评分，正是因为推论 1.1。

## 1.3.3 最大似然二元语法

二元语法模型族是所有满足 $q(b\mid a)\ge0$、且对每个 $a$ 都有 $\sum_b q(b\mid a) = 1$ 的表 $q(b\mid a)$。序列 $x_1,\dots,x_n$ 上的训练对数损失为

$$
(n-1)\hat L(q) = -\sum_{t=2}^{n}\ln q(x_t\mid x_{t-1}) = -\sum_{a}\sum_{b} c(a\to b)\ln q(b\mid a),
$$

这里把相同的配对归并在一起。总和拆成彼此独立的问题，每一行 $a$ 一个。（$n$ 个字符的序列有 $n - 1$ 个转移，因此以 $n - 1$ 归一化得到每个转移的训练损失 $\hat L$；若在前面加上序列开始符号，就有 $n$ 个转移，并以 $n$ 归一化。）

**命题 1.1。** 对每个 $c(a)>0$ 的 $a$，在 $\sum_b q(b\mid a) = 1$ 的限制下，$-\sum_b c(a\to b)\ln q(b\mid a)$ 的最小值发生在 $q(b\mid a) = c(a\to b)/c(a)$。

*以拉格朗日乘子证明。* 记 $c_b = c(a\to b)$、$q_b = q(b\mid a)$，最小化 $F = -\sum_b c_b \ln q_b + \lambda(\sum_b q_b - 1)$。令 $\partial F/\partial q_b = -c_b/q_b + \lambda = 0$ 得 $q_b = c_b/\lambda$；由限制条件得 $\lambda = \sum_b c_b = c(a)$。目标函数对 $q$ 是凸的（$-\ln$ 的总和），所以这个驻点就是全局最小值。$\blacksquare$

*不用微积分的第二种证明。* 令 $\hat p_b = c_b/c(a)$。则 $-\sum_b c_b\ln q_b = c(a)\,H(\hat p, q) = c(a)\,[H(\hat p) + \mathrm{KL}(\hat p\,\Vert\,q)]$，由 Gibbs 不等式，它在 $q = \hat p$ 时最小。**最大似然就是最小化从经验分布到模型的 KL 散度。** 这个观点可推广到本书的每个模型：以交叉熵训练神经网络，就是在 KL 意义下把 $q_\theta$ 拟合到经验分布。

由此得到 §1.2.8 与单元 03 主干项目测试所用的主张：**任何二元语法模型——计数的、平滑的或神经的——在一份训练文本上的训练损失，都不可能低于该文本的最大似然表。** 主干项目阶段 03 验证其以梯度训练的二元语法从上方收敛到这个值，而且从不低于它。

## 1.3.4 把平滑看成贝叶斯推断

为什么加 $\alpha$ 平滑不是随意发明的权宜之计，而是具有深厚数学根基的推断方法？答案来自贝叶斯统计：**平滑本质上是带有对称先验的贝叶斯预测分布。**

### 参数空间上的先验：Dirichlet 分布

在最大似然估计中，我们把条件概率向量视为某个未知的“固定真值”，仅凭样本频率去逼近它。而贝叶斯视角则把二元语法表中的每一行 $\theta = (\theta_1,\dots,\theta_V)$（其中 $\theta_b = q(b\mid a)$ 且 $\sum_b \theta_b = 1, \theta_b \ge 0$）视为一个定义在**概率单形（probability simplex）**上的未知随机向量。

在观测到具体字符计数之前，学习者需要表达对 $\theta$ 的初始信念——这就是**先验分布（prior distribution）**。对于多项式／类别分布，最自然的选择是参数全为 $\alpha$ 的**对称 Dirichlet 分布**：

$$
\pi(\theta) \propto \prod_b \theta_b^{\alpha-1}, \qquad \theta_b\ge 0,\ \textstyle\sum_b\theta_b = 1 .
$$

Dirichlet 分布是 Beta 分布在多维空间的推广。超参数 $\alpha > 0$ 扮演了“虚拟计数强度”的角色：
- 当 $\alpha = 1$ 时，概率密度 $\pi(\theta)$ 在整个单形上处处为常数，代表**均匀先验（uniform prior）**，即对所有可能的概率分布赋予完全相同的初始权重；
- 当 $\alpha > 1$ 时，密度集中在单形中心（$\theta_b \approx 1/V$），先验强烈偏好均匀平坦的分布；
- 当 $\alpha < 1$ 时，密度集中在单形的顶点与边界，先验相信多数类别的概率接近于零（稀疏性）。

### 共轭性与后验更新

当给定当前字符 $a$、后续字符 $b$ 出现的观测计数为 $c = (c_1, \dots, c_V)$ 时，这组计数的似然函数形式为多项式乘积 $\prod_b \theta_b^{c_b}$。

由贝叶斯定理，后验分布正比于先验与似然的乘积：
$\pi(\theta \mid c) \propto \pi(\theta) \cdot p(c \mid \theta) \propto \Big(\prod_b \theta_b^{\alpha-1}\Big) \cdot \Big(\prod_b \theta_b^{c_b}\Big)$。
将指数项合并后得到：

$$
\pi(\theta\mid c) \propto \prod_b \theta_b^{c_b + \alpha - 1}.
$$

注意到，后验分布与先验分布具有完全相同的函数形式，只是参数由原来的 $\alpha$ 变成了 $\beta_b = c_b + \alpha$。这种“先验与后验属于同一分布族”的优雅代数性质称为**共轭性（conjugacy）**，而 Dirichlet 分布正是多项分布的共轭先验。这意味着观测到数据后，更新信念的过程简化为纯粹的**计数相加**。

### 后验均值与贝叶斯最优预测

在对数损失下，我们应该如何利用后验分布 $\pi(\theta \mid c)$ 来对“下一个字符”做出预测？

我们有两种选择：
1. **最大后验估计（MAP）：** 寻找使后验概率密度最大的点 $\hat{\theta}_{\mathrm{MAP}} = \arg\max_\theta \pi(\theta \mid c)$。对于 Dirichlet 分布，其众数在 $\frac{c_b + \alpha - 1}{c(a) + \alpha V - V}$（当 $\alpha \ge 1$）。如果 $\alpha = 1$，MAP 恰好退化为未平滑的最大似然估计（MLE），依然无法解决零计数问题。
2. **全贝叶斯预测分布（Posterior Mean）：** 不把预测押在单一的点估计上，而是将所有可能的 $\theta$ 依据后验分布取加权平均。在对数损失下，数学上可以证明这种“平均掉所有参数不确定性”的策略正是期望风险最小的贝叶斯最优预测：
   $q(X_{n+1}=b \mid c) = \int \theta_b \, \pi(\theta \mid c) \, d\theta = \mathbb{E}[\theta_b \mid c]$。

根据 Dirichlet 分布的积分性质，参数为 $\beta_b$ 的随机变量的数学期望为 $\beta_b / \sum_{b'} \beta_{b'}$。代入 $\beta_b = c_b + \alpha$，即得**后验均值**：

$$
\mathbb{E}[\theta_b \mid c] = \frac{c_b + \alpha}{c(a) + \alpha V},
$$

这与加 $\alpha$ 平滑公式**完全恒等**！

### 重新审视拉普拉斯继起法则

现在我们可以清楚看清拉普拉斯继起法则的完整数学全貌：
在二元分类问题中（$V = 2$，记成功为 1，失败为 0），二维 Dirichlet 分布即为一维 Beta 分布。拉普拉斯在 1814 年假设未知成功概率 $p \sim \mathrm{Uniform}(0, 1) = \mathrm{Beta}(1, 1)$。在观测到 $n$ 次试验中包含 $s$ 次成功后，后验分布更新为 $\mathrm{Beta}(s + 1, n - s + 1)$。

对下一次试验成功的预测概率，正是该后验分布的期望值：$\mathbb{E}[p \mid s, n] = \frac{s + 1}{(s + 1) + (n - s + 1)} = \frac{s + 1}{n + 2}$。这正是拉普拉斯在两个世纪前写下的继起法则。加 $\alpha$ 平滑正是拉普拉斯继起法则在词表大小为 $V$、先验强度为 $\alpha$ 时的自然推广。

平滑绝非工程上修补零调试的随意补丁；它代表了一个在看到任何数据前、秉持对称 Dirichlet 先验的理性学习者所做出的严格贝叶斯推断。先验的选择，在机器学习中被称为**归纳偏差（inductive bias）**——即算法在看到数据之前，对解的世界所抱持的结构性假设。平滑引入了最基础的归纳偏差：世界并非非黑即白，未见不等于不可能。而在后续章节中，我们将看到更强大的归纳偏差（例如单元 02 的嵌入向量相似性、单元 04 的自注意力置换等变性），它们将彻底解放现代深度学习的泛化能力。

## 1.3.5 测得的损失可能离真值多远？

对一个*固定的*预测器 $q$，留出数据损失 $\hat L(q)$ 是 $n$ 个平均为 $L(q)$ 的 i.i.d. 项的平均。它可能有多接近真值？

**定理 1.2（Hoeffding 不等式，[Hoeffding 1963](/zh-cn/references/#hoeffding-1963)）。** 若 $Z_1,\dots,Z_n$ 彼此独立且 $Z_i\in[a,b]$，则对每个 $\varepsilon > 0$

$$
\Pr\Big(\Big|\tfrac{1}{n}\textstyle\sum_i Z_i - \mathbb{E}\big[\tfrac1n\sum_i Z_i\big]\Big| \ge \varepsilon\Big) \le 2\exp\!\Big(-\frac{2n\varepsilon^2}{(b-a)^2}\Big).
$$

（证明见 §1.5.1 ★。）令右侧等于 $\delta$ 并解出：以至少 $1-\delta$ 的概率，

$$
|\hat L(q) - L(q)| \le (b-a)\sqrt{\frac{\ln(2/\delta)}{2n}} .
$$

对数损失一般来说没有上界——概率接近零时损失极大——但平滑后的模型有：使用加 $\alpha$ 平滑时，每个概率至少是 $\alpha/(c(a) + \alpha V)$，所以损失至多是 $\ln((c(a)+\alpha V)/\alpha)$。更重要的是，这个界显示了**速率**：留出数据估计的误差以 $1/\sqrt n$ 的速度缩小。留出数据变成四倍，不确定性减半。例题 1.6 代入了数字。

## 1.3.6 在许多模型中挑选需要新的数据

Hoeffding 不等式适用于在看到数据*之前*就固定的预测器。假设改为在同一份数据上评估 $K$ 个预测器 $q_1,\dots,q_K$ 并保留最好的那个。每一个都有 $\delta/K$ 的小概率出现大偏差；由**联合界（union bound）**（$\Pr(\cup_k A_k)\le\sum_k\Pr(A_k)$），以至少 $1-\delta$ 的概率，*全部* $K$ 个估计都在其真值的

$$
\varepsilon_K = (b-a)\sqrt{\frac{\ln K + \ln(2/\delta)}{2n}}
$$

范围内。在 $K$ 个候选中挑选的代价只随 $\sqrt{\ln K}$ 成长——但它确实是代价。如果你尝试一千种设定并回报最好的验证分数，这个分数最多会乐观 $\varepsilon_K$，只有**只使用一次的测试集**能给出无偏的数字。把同样的论证应用在训练集上，“候选”就是所有可能的参数值，这正是 §1.5.1 均匀收敛理论的起点。

这个界也说明了为什么*适应性（adaptive）*重复使用比联合界所暗示的更糟：如果第 500 个候选的选择取决于前 499 个的分数，候选就不再是事先固定的，一个有心的分析者用远少于 $e^{n}$ 次的尝试就能过拟合验证集（[Dwork et al. 2015](/zh-cn/references/#dwork-2015)）。这就是为什么基准测试的分数会随多年的重复使用而上漂（§1.7.3；[Recht et al. 2019](/zh-cn/references/#recht-2019)）。

## 1.3.7 留出数据损失的标准误

Hoeffding 给的是最坏情况的界。实践中我们回报的是**标准误（standard error）**。若每个样本的损失 $Z_i$ 彼此独立、方差为 $\sigma^2$，则平均值的方差为 $\sigma^2/n$，所以

$$
\mathrm{se}(\hat L) = \frac{\sigma}{\sqrt n},
$$

以样本标准差代替 $\sigma$ 来估计。近似的 95% 区间为 $\hat L \pm 1.96\,\mathrm{se}$（由中央极限定理）。

“独立”这个词在这里很关键。一个故事中每个字符的损失并不独立：在主干语料上，二元语法下相邻字符损失的相关系数是 $-0.32$。诚实的采样单位是**故事**，因为被独立抽出的是故事，而不是字符。以每个故事的平均损失计算标准误（例题 1.7）是最简单的正确做法；单元 10 会发展自助法（bootstrap），它能处理任何统计量。

**要多少留出样本才能分辨两个模型？** 假设模型 B 平均每个样本比模型 A 好 $d$ nats，而两者每个样本损失*差*的标准差为 $\sigma_d$。显著性水平 5%、统计功效 80% 的双尾检验大约需要

$$
n \approx \Big(\frac{(1.96 + 0.84)\,\sigma_d}{d}\Big)^2
$$

个独立样本。注意是 $\sigma_d$，而不是任一模型损失的标准差：在*相同*样本上评估两个模型（配对）能抵销每个样本共有的难度，让 $\sigma_d$ 小得多（单元 10，§10.3）。

## 1.3.8 单位：nats、bits 与困惑度

文献中会看到同一个数字的三种等价报告方式：

| 量 | 公式 | 主干二元语法（留出数据） |
|---|---|---|
| 以 nats／字符计的损失 | $\hat L$ | 2.124 |
| bits／字符 | $\hat L/\ln 2$ | 3.065 |
| 困惑度（perplexity） | $e^{\hat L}$ | 8.37 |

困惑度有一个直观的读法：困惑度 8.37 的模型，平均而言就像在 8.37 个等可能的字符中均匀挑选一样不确定。均匀模型的困惑度恰好是 $V$。因为困惑度是指数，损失的小差异会变成看起来很大的困惑度差异；本书回报 nats。
