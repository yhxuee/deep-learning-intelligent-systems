---
title: "3.5 一般理论：梯度下降法为什么收敛、收敛多快"
description: 泰勒展开与 Hessian 矩阵；光滑性与下降引理；收敛到驻点；在二次函数上的精确分析与条件数；强凸性；随机梯度与噪声底限；动量与预条件化。
sidebar: { label: "3.5 一般理论", order: 5 }
---

§3.2 分析了一条抛物线。本节证明关于梯度下降法一般能说什么——同样重要的是，不能说什么。所有证明只用到泰勒定理、柯西–施瓦茨不等式与线性代数。这些内容的标准参考书为 [Nocedal & Wright 2006](/zh-cn/references/#nocedal-2006)、[Boyd & Vandenberghe 2004](/zh-cn/references/#boyd-2004)、[Nesterov 2018](/zh-cn/references/#nesterov-2018)，随机情况则见 [Bottou、Curtis & Nocedal 2018](/zh-cn/references/#bottou-2018)。

## 3.5.1 二阶的图像：Hessian 矩阵与局部曲率

在 §3.2 中，我们利用梯度构建了损失函数的一阶局部线性近似。然而线性模型完全无法感知“地势的弯曲”。为了理解优化算法的收敛极限，必须进入二阶泰勒展开的世界。

对二次连续可微的损失函数 $L(\theta)$，泰勒定理在局部将损失展开为：
$$
L(\theta + \delta) = L(\theta) + \nabla L(\theta)^{\top}\delta + \tfrac12\,\delta^{\top}H(\theta)\,\delta + o(\|\delta\|^2),
\qquad H_{ij} = \frac{\partial^2L}{\partial\theta_i\,\partial\theta_j}.
$$

**Hessian 矩阵（海森矩阵）的几何解构：**
$H$ 是一个 $n \times n$ 的实对称矩阵。根据线性代数的谱定理（Spectral Theorem），$H$ 必然存在一组由正交单位特征向量构成的基底 $q_1,\dots,q_n$，以及对应的实数特征值 $\lambda_1,\dots,\lambda_n$：
$H = Q\Lambda Q^{\top} = \sum_{i=1}^n \lambda_i q_i q_i^\top$。

若将位移向量投影到特征基底下 $\delta = \sum_i \alpha_i q_i$，二次项精确解耦为无交互的独立平方和：
$\frac{1}{2} \delta^\top H \delta = \frac{1}{2} \sum_{i=1}^n \lambda_i \alpha_i^2$。

这意味着：**在高维参数空间的任意局部，损失曲面沿着特征向量 $q_i$ 方向的行为，完全等价于一条曲率为 $\lambda_i$ 的一维抛物线！**
- **正特征值（$\lambda_i > 0$）：** 沿该方向曲面向上弯曲，如山谷碗底（凸曲率）；
- **负特征值（$\lambda_i < 0$）：** 沿该方向曲面向下弯曲，如山脊穹顶（凹曲率）；
- **零特征值（$\lambda_i = 0$）：** 平坦方向（退化山谷）。
- **鞍点（Saddle Point）：** 同时存在正特征值与负特征值。在几百万至数千亿维度的高维非凸深度学习损失曲面中，驻点（$\nabla L = 0$）几乎从来不是局部极小值，而是绝大多数特征方向均为负或近零的**高阶鞍点**（[Dauphin et al. 2014](/zh-cn/references/#dauphin-2014)）。

## 3.5.2 光滑性与下降引理

如果损失曲面的曲率无界剧烈震荡，任何固定步长的局部梯度更新都可能引发灾难性数值爆炸。为了给出优化的数学保证，我们需要限制曲率的上限——这就是**光滑性（Smoothness）**。

**定义。** 若 $L$ 的梯度是 $\beta$-Lipschitz 连续的，即对所有 $x, y$ 满足 $\|\nabla L(x) - \nabla L(y)\| \le \beta\|x - y\|$，则称 $L$ 为 **$\beta$-光滑（$\beta$-smooth）**。
在二次可微条件下，这等价于 Hessian 矩阵的最大特征值处处受限：$\lambda_{\max}(H(x)) \le \beta$。也就是说，曲面在任何方向上向上的弯曲程度都不会超过一条开口系数为 $\beta/2$ 的二次抛物线。

**引理 3.1（下降引理 / Descent Lemma）。** 若 $L$ 为 $\beta$-光滑，则对所有 $x, y$
$$
L(y) \le L(x) + \nabla L(x)^{\top}(y - x) + \frac\beta2\|y - x\|^2 .
$$

*证明。* 考虑连接 $x$ 与 $y$ 的线段路径 $\gamma(s) = x + s(y-x)$，其中 $s\in[0,1]$。由微积分基本定理：
$$
L(y) - L(x) - \nabla L(x)^{\top}(y-x) = \int_0^1\bigl(\nabla L(x + s(y-x)) - \nabla L(x)\bigr)^{\top}(y-x)\,ds .
$$
由柯西–施瓦茨不等式与梯度的 $\beta$-Lipschitz 光滑性，被积函数上界为：
$\|\nabla L(x + s(y-x)) - \nabla L(x)\| \cdot \|y-x\| \le \beta s \|y-x\|^2$。
将 $\beta s \|y-x\|^2$ 沿 $s \in [0, 1]$ 积分，得 $\frac{\beta}{2}\|y-x\|^2$。整理即得下降引理。$\blacksquare$

**推论（梯度下降单步的下降保证）。** 在下降引理中令下一步更新为 $y = x - \eta\nabla L(x)$：
$$
L\bigl(x - \eta\nabla L(x)\bigr) \le L(x) - \eta\Bigl(1 - \frac{\beta\eta}{2}\Bigr)\|\nabla L(x)\|^2 .
$$

这项基本不等式彻底确立了梯度下降的安全工作区间：
- 只要学习率满足 $0 < \eta < \frac{2}{\beta}$，系数 $\eta(1 - \frac{\beta\eta}{2}) > 0$，因此只要梯度非零，**损失保证在每一步严格单调下降**！
- 当 $\eta = \frac{1}{\beta}$ 时，系数取得最大值 $\frac{1}{2\beta}$，单步理论保证下降量最大：$L(x_{k+1}) \le L(x_k) - \frac{1}{2\beta}\|\nabla L(x_k)\|^2$。

:::note[主张检查]
- **主张。** 每一个 $0 < \eta < 2/\beta$ 的梯度下降步都会降低损失。
- **假设。** $L$ 在整条路径上都是 $\beta$-光滑的，梯度是精确的（全批量），而且 $\beta$ 已知。
- **检验。** 在最大特征值已知的二次函数上，分别以略低于与略高于 $2/\lambda_{\max}$ 的步长执行（实验 3.1）。
- **限制。** ReLU 网络并非处处平滑（§3.7.1），小批量梯度会带来噪声（§3.5.6），而曲率在训练中会改变；所以这个界限是选择学习率的指引，而不是直接决定学习率（§3.7.8）。
:::


## 3.5.3 没有凸性时能保证什么

神经网络的损失不是凸的。那还能证明什么？

**定理 3.1（收敛到驻点）。** 设 $L$ 为 $\beta$-光滑且下有界于 $L^*$。以 $\eta = 1/\beta$ 做梯度下降，满足

$$
\min_{0\le k<K}\|\nabla L(\theta_k)\|^2 \;\le\; \frac{2\beta\,\bigl(L(\theta_0) - L^*\bigr)}{K}.
$$

*证明。* 由推论，$\|\nabla L(\theta_k)\|^2 \le 2\beta\bigl(L(\theta_k) - L(\theta_{k+1})\bigr)$。对 $k = 0,\dots,K-1$ 加总，右边伸缩相消为 $2\beta\bigl(L(\theta_0) - L(\theta_K)\bigr) \le 2\beta\bigl(L(\theta_0) - L^*\bigr)$。$K$ 个数的最小值至多是它们的平均。$\blacksquare$

**它说了什么。** $K$ 步之后，某个迭代值的梯度范数至多为 $O(1/\sqrt K)$，常数与参数数量无关。**它没说什么。** 梯度小代表近似的*驻点（stationary point）*——可能是最小值、鞍点，也可能是平坦的高原。这里完全没说最小值是全局的，也没说它会泛化。对非凸损失，这些都是真正的开放问题，以实证与特殊情况的理论来研究（§3.7.2）。

## 3.5.4 精确分析二次函数：条件数与“狭长峡谷”困境

在 Hessian 正定的极小值 $\theta^*$ 附近，任何光滑损失函数都可以被二阶泰勒展开极佳地近似。因此，研究梯度下降法在纯二次模型
$$
L(\theta) = \tfrac12(\theta - \theta^*)^{\top}H(\theta - \theta^*),\qquad 0<\lambda_{\min}\le\lambda_i\le\lambda_{\max},
$$
上的动力学，就精确揭示了它在一般光滑非凸曲面上的局部收敛本质。

### 误差动力学的完全解耦
由于梯度为 $\nabla L(\theta) = H(\theta - \theta^*)$，定义第 $k$ 步的参数误差为 $e_k = \theta_k - \theta^*$。梯度下降迭代给出：
$e_{k+1} = e_k - \eta H e_k = (I - \eta H)e_k$。

将误差向量在 Hessian 的正交特征向量基底展开：$e_k = \sum_i c_{k,i}q_i$。由于 $H q_i = \lambda_i q_i$，算子作用在特征向量上化为标量相乘：
$$
c_{k+1,i} = (1-\eta\lambda_i)\,c_{k,i}\qquad\Longrightarrow\qquad c_{k,i} = (1-\eta\lambda_i)^k\,c_{0,i}.
$$

**惊人的正交解耦结论：** 高维二次函数上的多变量优化，被完美拆解成了 $n$ 个互不干扰、各自独立沿着特征轴演化的一维抛物线！

### 几何困境：病态条件与狭长峡谷
每个特征方向都要求 $|1 - \eta\lambda_i| < 1$ 才能保证收敛。由此导出优化的两大冲突法则：
1. **稳定性铁律：** 为了防止在最陡峭方向上爆炸，全局学习率必须受制于**最大特征值**：$\boxed{\eta < \frac{2}{\lambda_{\max}}}$。
2. **进展瓶颈：** 最平缓特征方向的收敛速度取决于 $(1 - \eta\lambda_{\min})$。当我们为了最陡方向的安全而被迫选取极小的 $\eta$ 时，平缓方向的误差衰减比率极其接近 1，几乎停滞不前！

定义**条件数（Condition Number）**为最大与最小曲率之比：
$$
\kappa = \frac{\lambda_{\max}}{\lambda_{\min}} \ge 1.
$$

**狭长峡谷（Narrow Ravine）的几何图景：**
当 $\kappa \gg 1$ 时（例如 $\kappa = 10^4$），等高线是极度拉长的狭长椭圆。
- 负梯度方向垂直于等高线，因此在峡谷的绝大部分区域，**负梯度几乎直指对面陡峭的崖壁，而非指向远处沿着峡谷底部的真实极小值点**！
- 传统梯度下降法被迫在两侧峭壁之间剧烈“锯齿状振荡”（Zig-zagging），同时以极其缓慢的速度沿着平坦的峡谷底部蠕动。

### 最优步长与收敛率
最差情况的收敛速率由最慢和最震荡的方向共同决定：$r(\eta) = \max\bigl(|1-\eta\lambda_{\min}|,\ |1-\eta\lambda_{\max}|\bigr)$。
当两端误差衰减幅度平衡时取得最优步长：$1 - \eta\lambda_{\min} = \eta\lambda_{\max} - 1$，由此解得：
$$
\eta^* = \frac{2}{\lambda_{\min}+\lambda_{\max}},\qquad r^* = \frac{\lambda_{\max}-\lambda_{\min}}{\lambda_{\max}+\lambda_{\min}} = \frac{\kappa-1}{\kappa+1},\qquad \kappa = \frac{\lambda_{\max}}{\lambda_{\min}} .
$$

若要将误差缩小至原始的 $\varepsilon$ 倍，所需迭代步数满足 $(r^*)^k \le \varepsilon$，即 $k = \ln(1/\varepsilon)/(-\ln r^*)$。当条件数 $\kappa$ 较大时，泰勒近似 $-\ln\frac{\kappa-1}{\kappa+1} \approx \frac{2}{\kappa}$，得到：
$$
k \approx \frac\kappa2\,\ln\frac1\varepsilon .
$$

**核心结论：梯度下降法的迭代步数随条件数 $\kappa$ 线性增长！**
若损失曲面中不同特征方向的曲率相差 $10^4$ 倍，无论工程师如何精细调节学习率，基础梯度下降法都需要成千上万倍的漫长步数。

这正是为什么后续的发明如此不可或缺：
- **动量法（Polyak Momentum，§3.5.7）：** 利用物理惯性抵消横跨崖壁的高频振荡，将迭代步数依赖从 $O(\kappa)$ 奇迹般地降低到 $O(\sqrt{\kappa})$；
- **自适应学习率（Adam/AdamW，§3.3.8）：** 针对不同坐标轴的历史梯度平方进行对角预条件化，使各轴步长动态适应局部曲率；
- **初始化与残差连接（§3.7）：** 从网络架构源头压制特征值发散，保持条件数 $\kappa \approx 1$。

## 3.5.5 强凸性与线性收敛

**定义。** 若对所有 $x, y$ 有 $L(y)\ge L(x) + \nabla L(x)^{\top}(y-x) + \frac\mu2\|y-x\|^2$，则称 $L$ 为 **$\mu$-强凸（strongly convex）**（$\mu > 0$）——一个二次下界，是引理 3.1 的镜像。对二次可微的 $L$，这表示 Hessian 的所有特征值至少为 $\mu$。

**Polyak–Łojasiewicz 不等式。** 把定义两边对 $y$ 取最小值。左边的最小值为 $L^*$；右边是 $y$ 的二次函数，在 $y = x - \nabla L(x)/\mu$ 取最小值 $L(x) - \frac{1}{2\mu}\|\nabla L(x)\|^2$。因此

$$
\tfrac12\|\nabla L(x)\|^2 \;\ge\; \mu\bigl(L(x) - L^*\bigr):
$$

*损失值离最优值越远，梯度就越大。*

**定理 3.2（线性收敛）。** 若 $L$ 为 $\beta$-光滑且 $\mu$-强凸，以 $\eta = 1/\beta$ 做梯度下降满足

$$
L(\theta_{k+1}) - L^* \;\le\; \Bigl(1 - \frac{\mu}{\beta}\Bigr)\bigl(L(\theta_k) - L^*\bigr).
$$

*证明。* 引理 3.1 的推论给出 $L(\theta_{k+1}) \le L(\theta_k) - \frac1{2\beta}\|\nabla L(\theta_k)\|^2$，PL 不等式则给出 $\|\nabla L\|^2 \ge 2\mu(L(\theta_k) - L^*)$。两边减去 $L^*$。$\blacksquare$

比值 $\beta/\mu$ 又是条件数；这个定理是 §3.5.4 在一般函数上的对应。证明只用到 PL 不等式，所以同样的线性速率对任何满足它的平滑函数都成立——包括某些非凸函数。

## 3.5.6 随机梯度：进展加上噪声

把 $\nabla L$ 换成无偏估计 $g$，满足 $\mathbb E[g] = \nabla L(\theta)$ 且 $\mathbb E\|g - \nabla L(\theta)\|^2 \le \sigma^2$（对小批量，$\sigma^2 = \mathrm{tr}\,\Sigma/B$，§3.3.7）。

**引理 3.2（一步的期望进展）。** 对 $\beta$-光滑的 $L$ 与一步 $\theta' = \theta - \eta g$，

$$
\mathbb E\bigl[L(\theta')\bigr] \;\le\; L(\theta) - \eta\Bigl(1 - \frac{\beta\eta}{2}\Bigr)\|\nabla L(\theta)\|^2 + \frac{\beta\eta^2}{2}\,\sigma^2 .
$$

*证明。* 以 $y - x = -\eta g$ 应用引理 3.1，得 $L(\theta') \le L(\theta) - \eta\nabla L^{\top}g + \frac{\beta\eta^2}{2}\|g\|^2$。取期望值：由无偏性 $\mathbb E[\nabla L^{\top}g] = \|\nabla L\|^2$，且 $\mathbb E\|g\|^2 = \|\nabla L\|^2 + \mathbb E\|g - \nabla L\|^2 \le \|\nabla L\|^2 + \sigma^2$（交叉项的平均为零）。整理各项即得。$\blacksquare$

第一项是 §3.5.2 的确定性进展；第二项是**噪声的代价**，与 $\eta^2\sigma^2$ 成正比。对 $K$ 步加总、取 $\eta\le1/\beta$（使 $1 - \beta\eta/2 \ge \tfrac12$），并像定理 3.1 一样伸缩相消：

$$
\frac1K\sum_{k<K}\mathbb E\|\nabla L(\theta_k)\|^2 \;\le\; \frac{2\bigl(L(\theta_0) - L^*\bigr)}{\eta K} + \beta\eta\,\sigma^2 .
$$

解读这两项：

- **步长固定时，SGD 有噪声底限（noise floor）** $\beta\eta\sigma^2$：它不会收敛到驻点，而是在一个区域内徘徊，区域大小随步长与梯度噪声增长。
- **缩小步长能降低底限，但会减慢进展。** 满足 $\eta_k\to0$、$\sum_k\eta_k = \infty$、$\sum_k\eta_k^2<\infty$ 的调度（例如 $\eta_k\propto1/k$）能让两项都趋于零——这是 [Robbins & Monro（1951）](/zh-cn/references/#robbins-1951) 的经典条件。
- **批量大小与步长互相取舍。** 因为 $\sigma^2\propto1/B$，底限与 $\eta/B$ 成正比。固定 $\eta/B$ 就固定了底限，这是“学习率随批量大小缩放”的理论动机（§3.8.4）。

## 3.5.7 ★ 二次函数上的动量

把 §3.3.8 的重球更新应用到 §3.5.4 的二次函数。在每个特征方向上，系数满足一个二项递推：

$$
c_{k+1} = (1 + \beta - \eta\lambda)\,c_k - \beta\,c_{k-1},
$$

其解是 $r^k$ 的组合，$r$ 为 $r^2 - (1+\beta-\eta\lambda)r + \beta = 0$ 的根。若两根为复数，它们的乘积为 $\beta$ 且模长相等，所以 $|r| = \sqrt\beta$——**与 $\lambda$ 无关**。两根为复数或重根的条件是 $(1+\beta-\eta\lambda)^2\le4\beta$，即

$$
(1-\sqrt\beta)^2 \;\le\; \eta\lambda \;\le\; (1+\sqrt\beta)^2 .
$$

选择 $\eta$ 与 $\beta$，让整个区间 $[\lambda_{\min}, \lambda_{\max}]$ 恰好落入其中：$\eta\lambda_{\min} = (1-\sqrt\beta)^2$、$\eta\lambda_{\max} = (1+\sqrt\beta)^2$。两式相除得 $\kappa = \bigl(\tfrac{1+\sqrt\beta}{1-\sqrt\beta}\bigr)^2$，所以

$$
\sqrt\beta = \frac{\sqrt\kappa-1}{\sqrt\kappa+1},\qquad \eta = \frac{4}{(\sqrt{\lambda_{\max}}+\sqrt{\lambda_{\min}})^2},\qquad \text{速率 } \frac{\sqrt\kappa-1}{\sqrt\kappa+1}.
$$

迭代次数现在随 $\sqrt\kappa$ 而非 $\kappa$ 增长——当 $\kappa = 10^4$，少了一百倍。例题 3.5 对 $\kappa = 10$ 检查了这些数字。Nesterov 加速梯度法对一般平滑强凸函数也达到同样的 $\sqrt\kappa$ 依赖，而对这类函数的一阶方法而言，这个依赖是最优的（[Nesterov 2018](/zh-cn/references/#nesterov-2018)）。

## 3.5.8 ★ 预条件化

**预条件化（preconditioning）** 的一步 $\theta \leftarrow \theta - \eta P^{-1}\nabla L$，就是在由 $P$ 定义的几何下做最陡下降（§3.2.5）。在二次函数上，误差迭代变成 $e_{k+1} = (I - \eta P^{-1}H)e_k$，所以重要的是 $P^{-1}H$ 的条件数：

- $P = H$（牛顿法）使 $P^{-1}H = I$，一步收敛——代价是建立并求逆一个 $n\times n$ 矩阵，$n$ 很大时不可能。
- 等于 $H$ 对角部分的对角 $P$，能去除与坐标轴对齐的病态条件，但无法去除旋转过的病态条件。

Adam 的分母 $\sqrt{\hat v}$ 是一个由梯度大小（而非曲率）建立的对角 $P$。两者在某些情况下相关，在其他情况下无关，所以“Adam 是对角二阶方法”只是一种启发式诠释，不是定理。以可负担的成本建立有结构、非对角预条件器的优化器，在 §3.8.6 讨论。

## 3.5.9 ★ 反向模式成本的正式叙述

设一个程序以一连串基本运算计算 $L$，每个运算的局部偏导数都能以至多该运算本身常数倍的成本计算。则梯度可以用至多“计算 $L$ 之成本的小常数倍”（与 $n$ 无关）的运算数算出，另加与存储的中间值数量成正比的内存。[Griewank & Walther（2008）](/zh-cn/references/#griewank-2008)在数种成本模型下给出精确叙述与常数。内存项可以用重新计算来换：只存储选定的检查点，在反向传递时重算各段——内存可降到运算数的对数级，计算量增加对数倍（他们的“revolve”调度）；对 $\ell$ 层网络，也可以约多一次前向传递换得 $O(\sqrt{\ell})$ 的内存（[Chen et al. 2016](/zh-cn/references/#chen-2016)）。
