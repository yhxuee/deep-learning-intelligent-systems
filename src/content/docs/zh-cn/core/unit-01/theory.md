---
title: "1.5 一般理论"
description: 有限模型族的均匀收敛与 VC 维度、Hoeffding 不等式的证明、偏差–方差分解、没有免费午餐定理，以及古典容量论证为何无法解释现代网络。
sidebar: { label: "1.5 一般理论", order: 5 }
---

§1.3 一次只分析一个模型。学习理论问一个更强的问题：如果我们在整个模型族中选出训练损失最小的模型，它的留出数据损失什么时候会接近训练损失？本节给出古典的答案、证明其中关键的部分，然后说明它们在哪里不再能解释实践。

## 1.5.1 均匀收敛

令 $\mathcal{H}$ 为一个预测器族，$\ell\in[0,1]$ 为有界损失。ERM 返回 $\hat h = \arg\min_{h\in\mathcal H}\hat L(h)$。因为 $\hat h$ 取决于训练样本，Hoeffding 不等式不能直接应用在它身上。它可以应用在每一个*固定的* $h$ 上；如果这个界对所有 $h\in\mathcal H$ **同时**成立，那它对 $\hat h$ 也成立。

**定理 1.3（有限模型族）。** 若 $|\mathcal H| = K$，则对大小为 $n$ 的 i.i.d. 样本，以至少 $1-\delta$ 的概率，对值域在 $[0, 1]$ 的损失，每个 $h\in\mathcal H$ 都满足
$$
\big|L(h) - \hat L(h)\big| \le \varepsilon,\qquad \varepsilon = \sqrt{\frac{\ln K + \ln(2/\delta)}{2n}} = \sqrt{\frac{\ln(2K/\delta)}{2n}} .
$$

*证明。* 对每个固定的 $h$，双尾 Hoeffding 不等式（定理 1.2）给出 $\Pr(|L(h) - \hat L(h)|\ge\varepsilon)\le 2e^{-2n\varepsilon^2} = \delta/K$。对 $K$ 个成员取联合界（§1.3.6），使*任一*偏差达到 $\varepsilon$ 的概率至多为 $\delta$。$\blacksquare$

下面两个方向都需要：上偏差界定所选 $\hat h$ 的风险，下偏差界定比较对象 $h^\star$ 的经验风险。（单尾陈述只需 $\ln(K/\delta)$，但无法支持这个推论。）

**推论 1.2。** 令 $h^\star$ 为 $\mathcal H$ 中总体风险最小者。以至少 $1-\delta$ 的概率，$L(\hat h) \le L(h^\star) + 2\sqrt{(\ln K + \ln(2/\delta))/(2n)}$。

*证明。* $L(\hat h)\le \hat L(\hat h) + \varepsilon \le \hat L(h^\star) + \varepsilon \le L(h^\star) + 2\varepsilon$，在两个方向上各使用一次双尾均匀界，并使用 $\hat L(\hat h)\le\hat L(h^\star)$（ERM）。$\blacksquare$

有两个特点值得强调。额外风险只通过 $\ln K$——一种**容量（capacity）**量度——依赖模型族，并以 $1/\sqrt{n}$ 缩小。而且论证完全没有用到 ERM 选了*哪一个* $h$；它对族中最差的成员也成立。这是它的优点（假设很少），也是它的弱点，§1.5.4 会说明。

**无限模型族。** 二元语法表的参数是连续的，所以 $K$ 是无限大。古典的补救方法是以组合方式度量容量。对二元分类，$\mathcal H$ 的 **VC 维度** $d_{\mathrm{VC}}$ 是 $\mathcal H$ 能以全部 $2^m$ 种方式标注（“打散，shatter”）的最大点数。$\mathbb{R}^k$ 中的线性分类器有 $d_{\mathrm{VC}} = k+1$。统计学习的基本定理说：一个模型族可由 ERM 学习，当且仅当其 VC 维度有限，且额外风险的量级为 $\sqrt{(d_{\mathrm{VC}} + \ln(1/\delta))/n}$（[Vapnik 1998](/zh-cn/references/#vapnik-1998)；[Shalev-Shwartz & Ben-David 2014](/zh-cn/references/#shalev-shwartz-2014)，第 6 章）。上面有限族的证明就是模板：把 $\ln K$ 换成计算该族在 $n$ 个点上*实际不同*行为数量的量。

### ★ 进阶：Hoeffding 不等式的证明

令 $Z_i\in[a,b]$ 彼此独立，$\mu_i = \mathbb{E}Z_i$，$S = \sum_i(Z_i - \mu_i)$。对任何 $s>0$，把 Markov 不等式应用在 $e^{sS}$ 上，得到
$$
\Pr(S\ge n\varepsilon) \le e^{-sn\varepsilon}\,\mathbb{E}[e^{sS}] = e^{-sn\varepsilon}\prod_i\mathbb{E}[e^{s(Z_i-\mu_i)}].
$$
**Hoeffding 引理**为每个因子给出上界：平均为零的变量 $X\in[a',b']$ 满足 $\mathbb{E}[e^{sX}]\le e^{s^2(b'-a')^2/8}$。（证明概要：由 $e^{sx}$ 的凸性，$e^{sX}\le\frac{b'-X}{b'-a'}e^{sa'} + \frac{X-a'}{b'-a'}e^{sb'}$；取期望值，把结果写成 $e^{\varphi(u)}$，其中 $u = s(b'-a')$，并检查 $\varphi(0) = \varphi'(0) = 0$、$\varphi''\le 1/4$，由泰勒定理得 $\varphi(u)\le u^2/8$。）因此 $\Pr(S\ge n\varepsilon)\le\exp(-sn\varepsilon + ns^2(b-a)^2/8)$。在 $s = 4\varepsilon/(b-a)^2$ 处对 $s$ 取最小值，得到 $\exp(-2n\varepsilon^2/(b-a)^2)$。下尾完全相同；因子 2 来自两尾相加。$\blacksquare$

## 1.5.2 偏差–方差分解

对平方损失，一个学习*程序*的期望误差（对训练集取平均）有精确的分解。令 $y = f(x) + \varepsilon$，$\mathbb{E}\varepsilon = 0$、$\mathrm{Var}\,\varepsilon = \sigma^2$，并令 $\hat f_{\mathcal D}$ 为在随机训练集 $\mathcal{D}$ 上训练出的预测器。在固定的测试点 $x$，记 $\bar f(x) = \mathbb{E}_{\mathcal D}[\hat f_{\mathcal D}(x)]$。

**命题 1.2。**
$$
\mathbb{E}_{\mathcal D,\varepsilon}\big[(y - \hat f_{\mathcal D}(x))^2\big] = \underbrace{\sigma^2}_{\text{噪声}} + \underbrace{(f(x) - \bar f(x))^2}_{\text{偏差}^2} + \underbrace{\mathbb{E}_{\mathcal D}\big[(\hat f_{\mathcal D}(x) - \bar f(x))^2\big]}_{\text{方差}} .
$$

*证明。* 写成 $y - \hat f = \varepsilon + (f - \bar f) + (\bar f - \hat f)$，平方后取期望值。交叉项消失：$\varepsilon$ 与 $\mathcal D$ 独立且平均为零；$f - \bar f$ 是常数；而由 $\bar f$ 的定义，$\mathbb{E}_{\mathcal D}[\bar f - \hat f] = 0$。$\blacksquare$

**噪声**项是不可约减的下限（相当于推论 1.1 中的熵）。**偏差**度量*平均*训练出的模型离真相多远——模型族越灵活，偏差越小。**方差**度量训练出的模型随训练集不同而改变多少——它随灵活度增加而变大，随数据增加而变小。§1.2.7 的平滑就是一种偏差–方差取舍：α → ∞ 时方差为零（均匀模型忽略数据）而偏差很大；α → 0 时在大量数据的极限下偏差为零，但在小样本上方差很高。由这个图像得出的古典预测是：测试误差随容量增加呈 U 形（[Geman et al. 1992](/zh-cn/references/#geman-1992)；[Hastie et al. 2009](/zh-cn/references/#hastie-2009)，§7.3）。

## 1.5.3 没有免费午餐

有没有某种学习程序在每个问题上都是最好的？没有。

**定理 1.4（没有免费午餐，有限版本）。** 令 $\mathcal{X}$ 为有限集合，标签为二元，并固定任一学习者。对*全部* $2^{|\mathcal X|}$ 个目标函数 $f:\mathcal X\to\{0,1\}$ 取平均，训练集以 $f$ 标注。则在任何不在训练集中的点上，学习者的期望准确率恰好是 $1/2$。

*证明。* 固定训练输入集 $S$ 与测试点 $x\notin S$。学习者在 $x$ 的预测只取决于 $S$ 的标签。对 $S$ 的每一种标注，与之兼容的目标函数中，恰好一半有 $f(x) = 1$、一半有 $f(x) = 0$，因为 $x$ 上的值不受限制。所以无论学习者预测什么，它恰好对一半的目标函数是对的。$\blacksquare$

这个定理并不是说学习不可能；它说的是，**若不对哪些目标较可能做出假设**，学习就不可能（[Wolpert 1996](/zh-cn/references/#wolpert-1996)）。每个成功的学习者，都是在某些问题上获胜，代价是在它隐含地认为不太可能的其他问题上落败。光滑性、局部性、稀疏性、对称性，以及“相似的输入有相似的输出”，都是这样的假设——**归纳偏差**。单元 02 谈如何通过表示来选择它们；单元 04 由对称性论证推导出其中之一（注意力的置换等变性）。

## 1.5.4 古典理论停在哪里

把定理 1.3 的逻辑应用在现代网络上。一个有数百万参数的网络能完全拟合随机标签（§1.1），所以它在 $n = 50{,}000$ 个点上的有效容量，至少等于任意标签的信息量；任何“训练误差 + 容量项”形式的界于是都变得空洞——它超过 100% 的错误率。然而同一个网络以真实标签训练时泛化得很好。过去十年的三个观察让这个谜题更加尖锐：

- **记住随机标签**（[Zhang et al. 2017](/zh-cn/references/#zhang-2017)）：标准架构与优化器能拟合随机标签与随机像素；显式正则化对泛化既非必要也非充分。
- **双重下降（double descent）**（[Belkin et al. 2019](/zh-cn/references/#belkin-2019)；[Nakkiran et al. 2020](/zh-cn/references/#nakkiran-2020)）：当模型大小增加、越过恰好能内插训练数据的那一点时，测试误差先上升（古典的 U 形），然后*再次下降*，常常低于古典的最小值。同样的形状也出现在以训练时间或数据量为横轴的情况。
- **良性过拟合（benign overfitting）**（[Bartlett et al. 2020](/zh-cn/references/#bartlett-2020)）：在参数远多于样本的线性回归中，最小范数的内插解完全拟合了噪声，却仍几乎最优地预测——前提是数据的协方差有许多小方向，能无害地吸收噪声。

共通的线索是：最坏情况的容量是错误的量。重要的是训练程序找到的是**哪一个**内插解。从小初始值出发的梯度下降法，对简单的解有*隐式偏差（implicit bias）*（在线性模型中是最小范数；单元 03 §3.5 与单元 06 会回到这一点），而真实数据的结构让简单的解存在。泛化是数据、架构与优化器的共同性质——而不只是参数量的性质。精确的理论仍是开放的研究领域（§1.8）；*不*开放的，是本单元赖以建立的实践规则：**在来自同一来源、从未用于任何决策的留出数据上度量。**

## 1.5.5 理论摘要

| 结果 | 说了什么 | 假设 |
|---|---|---|
| 推论 1.1 | 对数损失 = 不可约减的熵 + KL 额外损失 | — |
| Hoeffding（定理 1.2） | 留出数据估计误差为 $O(1/\sqrt n)$ | i.i.d.、有界损失、模型事先固定 |
| 均匀收敛（定理 1.3） | ERM 的额外风险为 $O(\sqrt{\ln K / n})$ | i.i.d.、有界损失、有限或 VC 维度有限的模型族 |
| 偏差–方差（命题 1.2） | 误差 = 噪声 + 偏差² + 方差 | 平方损失 |
| 没有免费午餐（定理 1.4） | 没有学习者在所有目标上都最好 | 对目标均匀平均 |
| 现代证据（§1.5.4） | 容量界对大型网络是空洞的；优化器找到什么才重要 | 实验性，部分理论 |
