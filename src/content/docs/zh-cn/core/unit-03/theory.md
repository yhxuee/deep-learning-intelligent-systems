---
title: "3.5 一般理论：梯度下降法为什么收敛、收敛多快"
description: 泰勒展开与 Hessian 矩阵；光滑性与下降引理；收敛到驻点；在二次函数上的精确分析与条件数；强凸性；随机梯度与噪声底限；动量与预条件化。
sidebar: { label: "3.5 一般理论", order: 5 }
---

§3.2 分析了一条抛物线。本节证明关于梯度下降法一般能说什么——同样重要的是，不能说什么。所有证明只用到泰勒定理、柯西–施瓦茨不等式与线性代数。这些内容的标准参考书为 [Nocedal & Wright 2006](/zh-cn/references/#nocedal-2006)、[Boyd & Vandenberghe 2004](/zh-cn/references/#boyd-2004)、[Nesterov 2018](/zh-cn/references/#nesterov-2018)，随机情况则见 [Bottou、Curtis & Nocedal 2018](/zh-cn/references/#bottou-2018)。

## 3.5.1 二阶的图像：Hessian 矩阵

对二次可微的 $L$，泰勒定理在 §3.2.4 的局部线性模型上加入二次项：

$$
L(\theta + \delta) = L(\theta) + \nabla L(\theta)^{\top}\delta + \tfrac12\,\delta^{\top}H(\theta)\,\delta + o(\|\delta\|^2),
\qquad H_{ij} = \frac{\partial^2L}{\partial\theta_i\,\partial\theta_j}.
$$

**Hessian 矩阵（海森矩阵）** $H$ 是对称的，由谱定理，它有实数特征值 $\lambda_1,\dots,\lambda_n$ 与一组正交单位特征向量基底 $q_1,\dots,q_n$：$H = Q\Lambda Q^{\top}$。沿方向 $q_i$，损失像抛物线 $\tfrac12\lambda_i t^2$ 一样弯曲。正特征值是碗状方向，负特征值是山丘状方向（两者同时出现就是鞍点），而特征值的*分散程度*——不只是它们的大小——将决定梯度下降法的速度。

## 3.5.2 光滑性与下降引理

**定义。** 若 $L$ 的梯度是 $\beta$-Lipschitz，即对所有 $x, y$ 有 $\|\nabla L(x) - \nabla L(y)\| \le \beta\|x - y\|$，则称 $L$ 为 **$\beta$-光滑（$\beta$-smooth）**。对二次可微的 $L$，这等价于 Hessian 的所有特征值处处落在 $[-\beta, \beta]$。光滑性限制了斜率能变化多快，也就限制了线性模型能错多少。

**引理 3.1（下降引理）。** 若 $L$ 为 $\beta$-光滑，则对所有 $x, y$

$$
L(y) \le L(x) + \nabla L(x)^{\top}(y - x) + \frac\beta2\|y - x\|^2 .
$$

*证明。* 沿线段 $x + s(y-x)$，$s\in[0,1]$，由微积分基本定理：

$$
L(y) - L(x) - \nabla L(x)^{\top}(y-x) = \int_0^1\bigl(\nabla L(x + s(y-x)) - \nabla L(x)\bigr)^{\top}(y-x)\,ds .
$$

由柯西–施瓦茨不等式与光滑性，被积函数至多为 $\beta s\|y-x\|\cdot\|y-x\|$。把 $\beta s$ 从 0 积到 1 得 $\beta/2$。$\blacksquare$

**推论（梯度下降法的一步）。** 令 $y = x - \eta\nabla L(x)$：

$$
L\bigl(x - \eta\nabla L(x)\bigr) \le L(x) - \eta\Bigl(1 - \frac{\beta\eta}{2}\Bigr)\|\nabla L(x)\|^2 .
$$

每一个 $0 < \eta < 2/\beta$ 的步都会降低损失（除非梯度为零）。保证的下降量在 $\eta = 1/\beta$ 时最大，等于 $\|\nabla L\|^2/(2\beta)$。这是抛物线结果 $\eta < 2/a$ 的多变量、最坏情况版本。

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

**它说了什么。** $K$ 步之后，某个迭代值的梯度范数至多为 $O(1/\sqrt K)$，常数与参数数量无关。**它没说什么。** 梯度小代表近似的*驻点（stationary point）*——可能是最小值、鞍点，也可能是平坦的高原。这里完全没说最小值是全域的，也没说它会泛化。对非凸损失，这些都是真正的开放问题，以实证与特殊情况的理论来研究（§3.7.2）。

## 3.5.4 精确分析二次函数：条件数

在 Hessian 正定的最小值 $\theta^*$ 附近，损失近似于二次函数。所以梯度下降法在

$$
L(\theta) = \tfrac12(\theta - \theta^*)^{\top}H(\theta - \theta^*),\qquad 0<\lambda_{\min}\le\lambda_i\le\lambda_{\max},
$$

上的行为，描述了它在任何平滑损失上的*局部*行为。在这里一切都可以精确计算。

梯度为 $H(\theta - \theta^*)$，所以误差 $e_k = \theta_k - \theta^*$ 满足 $e_{k+1} = (I - \eta H)e_k$。在特征基底下展开误差 $e_k = \sum_ic_{k,i}q_i$。因为 $Hq_i = \lambda_iq_i$，

$$
c_{k+1,i} = (1-\eta\lambda_i)\,c_{k,i}\qquad\Longrightarrow\qquad c_{k,i} = (1-\eta\lambda_i)^k\,c_{0,i}.
$$

每个特征方向都是 §3.2.3 抛物线的独立复本。因此：

1. **稳定性：** 从任何起点收敛的充要条件是对所有 $i$ 有 $|1 - \eta\lambda_i| < 1$，即 $\boxed{\eta < 2/\lambda_{\max}}$。
2. **速率：** 误差像 $r(\eta)^k$ 一样缩小，其中 $r(\eta) = \max\bigl(|1-\eta\lambda_{\min}|,\ |1-\eta\lambda_{\max}|\bigr)$。
3. **最优固定步长：** 当两项相等时 $r(\eta)$ 最小，即 $1 - \eta\lambda_{\min} = \eta\lambda_{\max} - 1$：

$$
\eta^* = \frac{2}{\lambda_{\min}+\lambda_{\max}},\qquad r^* = \frac{\lambda_{\max}-\lambda_{\min}}{\lambda_{\max}+\lambda_{\min}} = \frac{\kappa-1}{\kappa+1},\qquad \kappa = \frac{\lambda_{\max}}{\lambda_{\min}} .
$$

**条件数（condition number）** $\kappa$ 单独决定了可达到的最优速率。要把误差缩小 $\varepsilon$ 倍需要 $k = \ln(1/\varepsilon)/(-\ln r^*)$ 步；由于 $\kappa$ 很大时 $-\ln\frac{\kappa-1}{\kappa+1}\approx\frac2\kappa$，

$$
k \approx \frac\kappa2\,\ln\frac1\varepsilon .
$$

**迭代次数随条件数线性增长。** 曲率范围相差 $10^4$ 倍的损失，不论步长怎么选，都比条件良好的损失多需要数千倍的步数。因此，让 $\kappa$ 变小——通过初始化、归一化、预条件化或架构——是实践优化的主要目标之一（§3.8）。

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
- 等于 $H$ 对角部分的对角 $P$，能去除与座标轴对齐的病态条件，但无法去除旋转过的病态条件。

Adam 的分母 $\sqrt{\hat v}$ 是一个由梯度大小（而非曲率）建立的对角 $P$。两者在某些情况下相关，在其他情况下无关，所以“Adam 是对角二阶方法”只是一种启发式诠释，不是定理。以可负担的成本建立有结构、非对角预条件器的优化器，在 §3.8.6 讨论。

## 3.5.9 ★ 反向模式成本的正式叙述

设一个程序以一连串基本运算计算 $L$，每个运算的局部偏导数都能以至多该运算本身常数倍的成本计算。则梯度可以用至多“计算 $L$ 之成本的小常数倍”（与 $n$ 无关）的运算数算出，另加与存储的中间值数量成正比的内存。[Griewank & Walther（2008）](/zh-cn/references/#griewank-2008)在数种成本模型下给出精确叙述与常数。内存项可以用重新计算来换：只存储选定的检查点，在反向传递时重算各段——内存可降到运算数的对数级，计算量增加对数倍（他们的“revolve”调度）；对 $\ell$ 层网络，也可以约多一次前向传递换得 $O(\sqrt{\ell})$ 的内存（[Chen et al. 2016](/zh-cn/references/#chen-2016)）。
