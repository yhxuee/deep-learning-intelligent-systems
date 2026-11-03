---
title: "3.5 一般理论：梯度下降法为什么收敛、收敛多快"
description: 泰勒展开与 Hessian 矩阵；光滑性与下降引理；近似驻点；二次函数与条件数的精确分析；强凸性；随机梯度与噪声；动量、预条件化与反向模式成本。
sidebar: { label: "3.5 一般理论", order: 5 }
---

§3.2 追踪了梯度下降法在一条抛物线上的行为。现在问题变得更难：其中哪些现象能延伸到一般的多变量损失？

本节的论证分三层。第一，光滑性把一个局部梯度步变成不等式。第二，把这个不等式沿时间加总，得到收敛界限。第三，加入正定二次函数或强凸性等额外结构，把较弱的保证升级为几何速率。阅读时要留意量词：“前几步中最好的一个迭代点”、“对随机执行取平均”与“每个迭代点”是三种不同主张。

证明会用到泰勒定理、柯西–施瓦茨不等式、期望值与线性代数。标准参考书包括 [Nocedal & Wright（2006）](/zh-cn/references/#nocedal-2006)、[Boyd & Vandenberghe（2004）](/zh-cn/references/#boyd-2004)、[Nesterov（2018）](/zh-cn/references/#nesterov-2018)；随机优化则见 [Bottou、Curtis & Nocedal（2018）](/zh-cn/references/#bottou-2018)。

## 3.5.1 二阶图像：Hessian 矩阵

梯度记录局部斜率。再对梯度微分一次，得到 **Hessian 矩阵（海森矩阵）**：

$$
H_{ij}(\theta)=\frac{\partial^2L}{\partial\theta_i\,\partial\theta_j}.
$$

对二次可微的损失，泰勒定理把 §3.2.4 的局部线性模型延伸为

$$
L(\theta+\delta)
=L(\theta)+\nabla L(\theta)^\top\delta
+\tfrac12\delta^\top H(\theta)\delta
+o(\|\delta\|^2).
$$

依序读这四项：目前的函数值、一阶变化、二次曲率，以及当 $\delta\to0$ 时相对于 $\|\delta\|^2$ 可忽略的余项。这是**局部**描述；它不表示同一个 Hessian 能代表离 $\theta$ 很远的损失地形。

当混合偏导数连续时，$H$ 是对称矩阵。谱定理因此给出实数特征值 $\lambda_i$ 与正交单位特征基底 $q_i$：

$$
H=Q\Lambda Q^\top,\qquad
\delta=\sum_i a_iq_i,\qquad
\delta^\top H\delta=\sum_i\lambda_i a_i^2.
$$

最后一个等式最有用。在特征坐标中，二次项没有交叉项；沿 $q_i$ 方向的弯曲就像 $\tfrac12\lambda_i t^2$。正的 $\lambda_i$ 代表向上弯，负值代表向下弯。如果梯度也等于零，同时出现正、负特征值就能判定为鞍点；若梯度不为零，Hessian 不定只描述曲率，不能单独分类该点。

稍后会用到两种不同的频谱摘要。最大绝对曲率限制安全步长；对正定二次函数，最大与最小曲率的比值则控制调好步长后的最优最坏情况速率。

## 3.5.2 光滑性与下降引理

**定义。** 若在一个凸区域内，$L$ 的梯度是 $\beta$-Lipschitz：

$$
\|\nabla L(x)-\nabla L(y)\|\le\beta\|x-y\|,
$$

就称 $L$ 在该区域为 **$\beta$-光滑（$\beta$-smooth）**。平滑不代表凸；它只表示斜率不能任意快速改变。若 $L$ 二次连续可微，这个条件等价于整个区域内都有 $\|H(x)\|_2\le\beta$，也等价于 Hessian 的每个特征值都落在 $[-\beta,\beta]$。

**引理 3.1（下降引理）。** 若 $L$ 在包含 $x$ 到 $y$ 线段的凸区域内为 $\beta$-光滑，则

$$
L(y)\le L(x)+\nabla L(x)^\top(y-x)+\frac{\beta}{2}\|y-x\|^2.
$$

右边是线性预测，再加上一份二次余裕，容许梯度在移动途中发生变化。

*证明。* 令 $d=y-x$，沿直线 $\gamma(s)=x+sd$ 移动，其中 $0\le s\le1$。微积分基本定理给出

$$
L(y)-L(x)=\int_0^1\nabla L(x+sd)^\top d\,ds.
$$

减去固定的线性预测 $\nabla L(x)^\top d$：

$$
L(y)-L(x)-\nabla L(x)^\top d
=\int_0^1\bigl(\nabla L(x+sd)-\nabla L(x)\bigr)^\top d\,ds.
$$

柯西–施瓦茨不等式把被积函数限制在两个范数的乘积内；光滑性再把第一个范数限制为 $\beta s\|d\|$，所以

$$
\bigl(\nabla L(x+sd)-\nabla L(x)\bigr)^\top d
\le\beta s\|d\|^2.
$$

把 $s$ 从 0 积到 1，就得到因子 $1/2$。$\blacksquare$

**推论（一次精确梯度更新）。** 代入 $y=x-\eta\nabla L(x)$：

$$
L\bigl(x-\eta\nabla L(x)\bigr)
\le L(x)-\eta\left(1-\frac{\beta\eta}{2}\right)\|\nabla L(x)\|^2.
$$

因此，只要 $\nabla L(x)\ne0$，每个满足 $0<\eta<2/\beta$ 的步长都保证降低损失。这个界限所保证的下降量在 $\eta=1/\beta$ 时最大，为 $\|\nabla L(x)\|^2/(2\beta)$。这是抛物线条件 $\eta<2/a$ 的多变量、最坏情况版本。

:::note[主张检查]
- **主张。** 以 $0<\eta<2/\beta$ 做一次精确梯度更新会降低损失。
- **假设。** 更新经过的整条线段都位于 $L$ 为 $\beta$-光滑的区域；梯度精确；所用的 $\beta$ 确实是有效上界。
- **检验。** 对 $\lambda_{\max}$ 已知的二次函数，比较略低于与略高于 $2/\lambda_{\max}$ 的步长（实验 3.1）。
- **限制。** ReLU 网络并非处处可微（§3.7.1），小批量会把精确梯度换成随机估计（§3.5.6），曲率也会在训练中改变。这个界限是指引，不是自动产生的学习率设定（§3.7.8）。
:::

## 3.5.3 没有凸性时能保证什么

神经网络的损失通常是非凸的。单靠光滑性无法保证全局最小值，却能保证梯度下降法会找到至少一个梯度很小的迭代点。

**定理 3.1（近似驻点界限）。** 设 $L$ 为 $\beta$-光滑，且下有界于某个数 $L^*$。取 $\eta=1/\beta$，则

$$
\min_{0\le k<K}\|\nabla L(\theta_k)\|^2
\le\frac{2\beta\bigl(L(\theta_0)-L^*\bigr)}{K}.
$$

*证明。* 在推论中取 $\eta=1/\beta$，得到

$$
\|\nabla L(\theta_k)\|^2
\le2\beta\bigl(L(\theta_k)-L(\theta_{k+1})\bigr).
$$

把不等式从 $k=0$ 加到 $K-1$。每个中间损失都以正号出现一次、负号出现一次：

$$
\sum_{k=0}^{K-1}\|\nabla L(\theta_k)\|^2
\le2\beta\bigl(L(\theta_0)-L(\theta_K)\bigr)
\le2\beta\bigl(L(\theta_0)-L^*\bigr).
$$

两边除以 $K$。$K$ 个非负数中的最小值不可能大于它们的平均。$\blacksquare$

前 $K$ 个迭代点中**最好的一个**，其梯度平方为 $O(1/K)$，所以梯度范数为 $O(1/\sqrt K)$。公式没有明写参数数量，但常数 $\beta$ 与 $L(\theta_0)-L^*$ 本身仍可能随模型规模改变。

这个定理刻意比“迭代点会收敛”更弱。它只指出至少有一个近似驻点；没有说最后一个迭代点就是它、整个序列有极限，也没有说小梯度点必为最小值。驻点可能是局部最小值、鞍点或平坦高原，而上述任何叙述都不保证泛化。

## 3.5.4 精确分析二次函数：条件数

若要得到精确速率，可先专注于正定二次函数：

$$
L(\theta)
=\tfrac12(\theta-\theta^*)^\top H(\theta-\theta^*),
\qquad
0<\lambda_{\min}\le\lambda_i\le\lambda_{\max}.
$$

这个模型对二次函数完全精确；在 Hessian 正定的平滑最小值附近，它是第一个局部近似。如果 Hessian 沿路变化很大，就不能期待它描述一整段很长的轨迹。

令 $e_k=\theta_k-\theta^*$。由于 $\nabla L(\theta_k)=He_k$，

$$
e_{k+1}=(I-\eta H)e_k.
$$

在 Hessian 的特征基底中展开 $e_k=\sum_i c_{k,i}q_i$。每个方向都独立演化：

$$
c_{k+1,i}=(1-\eta\lambda_i)c_{k,i},
\qquad
c_{k,i}=(1-\eta\lambda_i)^kc_{0,i}.
$$

矩阵问题现在变成每个特征值各自对应一份一维抛物线。可直接得到三项结果。

1. **从每个起点都稳定。** 所有分量都收敛到零的充要条件为

   $$
   0<\eta<\frac{2}{\lambda_{\max}}.
   $$

2. **最坏情况的参数收缩。** 在欧几里得范数下，

   $$
   \|e_k\|\le r(\eta)^k\|e_0\|,
   \qquad
   r(\eta)=\max_i|1-\eta\lambda_i|.
   $$

   若只知道区间 $[\lambda_{\min},\lambda_{\max}]$，最坏情况会出现在端点。

3. **在这个极小极大意义下的最优固定步长。** 令两个端点的绝对值平衡：

   $$
   1-\eta\lambda_{\min}
   =\eta\lambda_{\max}-1.
   $$

   得到

   $$
   \eta^*=\frac{2}{\lambda_{\min}+\lambda_{\max}},
   \qquad
   r^*=\frac{\lambda_{\max}-\lambda_{\min}}
   {\lambda_{\max}+\lambda_{\min}}
   =\frac{\kappa-1}{\kappa+1},
   \qquad
   \kappa=\frac{\lambda_{\max}}{\lambda_{\min}}.
   $$

**条件数（condition number）** $\kappa$ 控制最优的最坏情况固定步长速率。把 $H$ 整体乘上一个常数，会改变合适的学习率，却不改变 $\kappa$ 或调好后的收缩因子。要把最坏情况参数误差缩小 $\varepsilon$ 倍，需要

$$
k=\frac{\ln(1/\varepsilon)}{-\ln r^*}.
$$

当 $\kappa$ 很大时，

$$
-\ln\frac{\kappa-1}{\kappa+1}\approx\frac{2}{\kappa},
\qquad
k\approx\frac{\kappa}{2}\ln\frac1\varepsilon.
$$

因此，最坏情况迭代次数随 $\kappa$ 线性增长。这**不**表示每个初始化都需要这么久；如果 $e_0$ 在缓慢特征方向上没有分量，实际速率可能更快。它真正表示的是：对跨越该频谱区间的所有起始误差，没有任何固定步长能改善这项保证。初始化、归一化、预条件化与架构都会以不同方式尝试改善有效条件数。

## 3.5.5 强凸性与线性收敛

二次函数很特殊，但它的几何速率可以延伸到更广的函数类别。

**定义。** 若 $\mu>0$，且定义域中的每个 $x,y$ 都满足

$$
L(y)\ge L(x)+\nabla L(x)^\top(y-x)+\frac{\mu}{2}\|y-x\|^2,
$$

就称 $L$ 为 **$\mu$-强凸（strongly convex）**。这是一个二次下界，是下降引理中二次上界的镜像。对二次可微的 $L$，它表示处处都有 $H(x)\succeq\mu I$。

假设最小值点存在，并把最小函数值写成 $L^*$。强凸性可推出 **Polyak–Łojasiewicz（PL）不等式**。把二次下界对 $y$ 最小化；它在 $y=x-\nabla L(x)/\mu$ 取得最小值 $L(x)-\|\nabla L(x)\|^2/(2\mu)$。真正的最小值不可能低于这个全局下界，因此

$$
\frac12\|\nabla L(x)\|^2
\ge\mu\bigl(L(x)-L^*\bigr).
$$

用白话说：有了这项结构，函数值远高于最优值就会迫使梯度很大。这正是前一个非凸定理缺少的联系。

**定理 3.2（函数值的线性收敛）。** 若 $L$ 为 $\beta$-光滑且 $\mu$-强凸，以 $\eta=1/\beta$ 做梯度下降，则

$$
L(\theta_{k+1})-L^*
\le\left(1-\frac{\mu}{\beta}\right)
\bigl(L(\theta_k)-L^*\bigr).
$$

*证明。* 下降引理给出

$$
L(\theta_{k+1})
\le L(\theta_k)-\frac{1}{2\beta}\|\nabla L(\theta_k)\|^2.
$$

以 PL 不等式把梯度平方替换成至少 $2\mu(L(\theta_k)-L^*)$，再从两边减去 $L^*$。$\blacksquare$

反复应用这个界限，得到

$$
L(\theta_k)-L^*
\le\left(1-\frac{\mu}{\beta}\right)^k
\bigl(L(\theta_0)-L^*\bigr).
$$

这里的“线性收敛”是指几何衰减，不是图上画出一条直线。比值 $\beta/\mu$ 又是条件数。注意，证明直接用到的是光滑性加 PL 不等式，而不是凸性；因此，某些满足 PL 的平滑非凸函数也有相同的函数值速率，强凸性只是取得 PL 的一个充分条件。

## 3.5.6 随机梯度：进展加上噪声

现在把精确梯度换成随机估计 $g_k$。在给定目前迭代点 $\theta_k$ 的条件下，假设

$$
\mathbb E_k[g_k]=\nabla L(\theta_k),
\qquad
\mathbb E_k\|g_k-\nabla L(\theta_k)\|^2\le\sigma^2.
$$

下标提醒我们：取期望时固定 $\theta_k$，只对新一次采样的随机性平均。若批量大小为 $B$、样本独立且放回采样，单笔样本梯度的协方差为 $\Sigma(\theta_k)$，则方差项为 $\operatorname{tr}\Sigma(\theta_k)/B$。不放回采样还有有限总体修正；样本彼此相关时，也未必遵守简单的 $1/B$ 定律。

**引理 3.2（一步的期望进展）。** 对 $\beta$-光滑损失与更新 $\theta_{k+1}=\theta_k-\eta g_k$，

$$
\mathbb E_k[L(\theta_{k+1})]
\le L(\theta_k)
-\eta\left(1-\frac{\beta\eta}{2}\right)
\|\nabla L(\theta_k)\|^2
+\frac{\beta\eta^2}{2}\sigma^2.
$$

*证明。* 在下降引理中代入 $y-x=-\eta g_k$：

$$
L(\theta_{k+1})
\le L(\theta_k)-\eta\nabla L(\theta_k)^\top g_k
+\frac{\beta\eta^2}{2}\|g_k\|^2.
$$

无偏性给出 $\mathbb E_k[\nabla L^\top g_k]=\|\nabla L\|^2$。展开 $g_k=\nabla L+(g_k-\nabla L)$，可得

$$
\mathbb E_k\|g_k\|^2
=\|\nabla L\|^2
+\mathbb E_k\|g_k-\nabla L\|^2
\le\|\nabla L\|^2+\sigma^2,
$$

因为交叉项的条件期望为零。代回并整理各项即得。$\blacksquare$

若使用固定的 $\eta\le1/\beta$，取完整期望、对 $K$ 步加总、让损失像望远镜一样逐项相消，再应用下界 $L^*$：

$$
\frac1K\sum_{k=0}^{K-1}
\mathbb E\|\nabla L(\theta_k)\|^2
\le
\frac{2\bigl(L(\theta_0)-L^*\bigr)}{\eta K}
+\beta\eta\sigma^2.
$$

第一项会随步数增加而缩小，第二项不会。因此，这个上界在平均梯度平方中留下 $O(\eta\sigma^2)$ 的渐近**噪声项**。当非零噪声持续存在时，固定步长 SGD 通常会在解附近波动，而非精确停在解上。

但不要把这句话反过来当成普遍不可能定理。不等式是上界，没有证明每次执行一定碰到这个底限；若梯度噪声在最优值附近消失，固定步长方法仍可能精确收敛。

对递减步长 $\eta_k\le1/\beta$，同一个证明给出加权界限

$$
\frac{\sum_{k<K}\eta_k\,
\mathbb E\|\nabla L(\theta_k)\|^2}
{\sum_{k<K}\eta_k}
\le
\frac{
2\bigl(L(\theta_0)-L^*\bigr)
+\beta\sigma^2\sum_{k<K}\eta_k^2}
{\sum_{k<K}\eta_k}.
$$

经典 Robbins–Monro 条件为

$$
\eta_k\to0,\qquad
\sum_k\eta_k=\infty,\qquad
\sum_k\eta_k^2<\infty.
$$

在上述一致方差界限下，这些条件会让右边趋近零；$\eta_k\propto1/k$ 是标准例子。若要对迭代点本身提出更强的收敛主张，还需要额外假设。

最后，若独立批量近似 $\sigma^2\propto1/B$ 成立，固定步长的噪声项就随 $\eta/B$ 缩放。这是让 $\eta$ 随 $B$ 增加的理论动机，却不保证线性缩放永远稳定：曲率仍会限制 $\eta$，而且批量够大后，方差模型不再是完整故事（§3.8.4）。

## 3.5.7 ★ 二次函数上的动量

把 §3.3.8 的重球更新应用到 §3.5.4 的正定二次函数。沿曲率为 $\lambda$ 的特征方向，误差系数满足

$$
c_{k+1}=(1+\beta-\eta\lambda)c_k-\beta c_{k-1}.
$$

尝试解 $c_k=r^k$，得到特征方程

$$
r^2-(1+\beta-\eta\lambda)r+\beta=0.
$$

当两根互为共轭复数时，其乘积为 $\beta$，所以每个根的模长都是 $\sqrt\beta$。在重根边界，频谱半径相同，但重根可能带来 $k r^k$ 之类的多项式前因子。两根为复数或重根的条件是

$$
(1+\beta-\eta\lambda)^2\le4\beta,
$$

等价于

$$
(1-\sqrt\beta)^2
\le\eta\lambda
\le(1+\sqrt\beta)^2.
$$

选择 $\eta$ 与 $\beta$，让整个曲率区间恰好塞进两个端点之间：

$$
\eta\lambda_{\min}=(1-\sqrt\beta)^2,
\qquad
\eta\lambda_{\max}=(1+\sqrt\beta)^2.
$$

两式相除并求解，得到

$$
\sqrt\beta
=\frac{\sqrt\kappa-1}{\sqrt\kappa+1},
\qquad
\eta
=\frac{4}
{(\sqrt{\lambda_{\max}}+\sqrt{\lambda_{\min}})^2}.
$$

因此，最坏情况的渐近根因子为

$$
\rho_{\mathrm{HB}}
=\frac{\sqrt\kappa-1}{\sqrt\kappa+1}.
$$

它的迭代数依赖为 $O(\sqrt\kappa\log(1/\varepsilon))$，优于梯度下降法的 $O(\kappa\log(1/\varepsilon))$。当 $\kappa=10^4$，这在增长阶上约改善一百倍；它并不承诺每个有限步执行都恰好早一百倍结束。初期瞬态、重根前因子与参数失调都会改变实测步数；例题 3.5 在 $\kappa=10$ 时展示了这一点。

这段重球推导只针对二次函数；把同一组固定参数搬到一般平滑强凸目标，不会自动继承全局加速保证。Nesterov 加速梯度法采用不同更新，能对这个更广的函数类别达到相同的 $\sqrt\kappa$ 依赖。在标准一阶预言机模型下，这个依赖对该函数类别是最优的（[Nesterov 2018](/zh-cn/references/#nesterov-2018)）。

## 3.5.8 ★ 预条件化

令 $P$ 为对称正定矩阵。**预条件化（preconditioning）** 更新为

$$
\theta\leftarrow\theta-\eta P^{-1}\nabla L(\theta).
$$

若以 $P$-范数 $\|d\|_P^2=d^\top Pd$ 衡量距离，这就是最陡下降（§3.2.5）。对二次函数，

$$
e_{k+1}=(I-\eta P^{-1}H)e_k.
$$

$P^{-1}H$ 外观看来不一定对称，但它与对称矩阵 $P^{-1/2}HP^{-1/2}$ 相似，两者有相同的正特征值。因此，真正重要的是这些**预条件化后**特征值的分散程度。

- 若 $P=H$ 且 $\eta=1$，则 $P^{-1}H=I$，精确二次函数一步就能解完。对大型模型而言，建立、存储并求解一个稠密的 $n\times n$ Hessian 通常不可行。
- 若 $H$ 为对角矩阵且 $P=\operatorname{diag}(H)$，预条件化后的 Hessian 恰好是 $I$。如果困难方向相对坐标轴旋转过，一个对角 $P$ 通常无法消除它们。

Adam 的分母 $\sqrt{\hat v}+\epsilon$，作用像由过去梯度平方建立、随时间变化的对角重新缩放，而不是 Hessian 本身。在某些问题中，梯度尺度与曲率相关；在另一些问题中则不相关。因此，“Adam 是对角二阶方法”只是启发式类比，不是定理。带结构的非对角预条件器会在 §3.8.6 讨论。

## 3.5.9 ★ 反向模式成本的正式叙述

“一次反向传递得到所有参数梯度”是在谈**成本比值**，不是说绝对成本与模型大小无关。

假设一个程序用一连串基本运算计算标量 $L$，而每个运算的局部偏导数，都能以相对于该运算至多常数倍的成本算出。反向模式自动微分便能用“前向计算成本的小常数倍”算出完整梯度。这个倍数不会只因输入坐标更多而增长；但模型变大时，前向程序本身当然也可能更昂贵。[Griewank & Walther（2008）](/zh-cn/references/#griewank-2008) 在多种成本模型下给出精确叙述。

基本反向传递会保留局部导数规则所需的前向中间值，所以内存可能随实际执行的计算长度增长。**检查点法（checkpointing）**以重新计算换取内存：只存选定状态，反向传递需要时重跑省略的前向区段，使用后再丢弃。不同调度位于不同的时间–空间取舍点。在相应模型下，*revolve* 之类的二项式调度能以对数级内存换取对数级重新计算因子；对 $\ell$ 层链，较简单的分段方式可用 $O(\sqrt\ell)$ 个存储活化值，代价约为多一次前向传递（[Chen et al. 2016](/zh-cn/references/#chen-2016)）。

所以，反向模式移除了逐坐标微分额外付出的“参数数量”因子；它没有让计算或内存变成免费。
