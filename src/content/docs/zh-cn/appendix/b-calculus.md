---
title: "B · 多变量与矩阵微积分"
description: 梯度、雅可比矩阵与多变量链式法则；泰勒定理与海森矩阵；矩阵微分与反向传播所用的恒等式；softmax、log-sum-exp 与交叉熵的导数；积分的变量替换；积分号下求导；有限差分及其误差。
sidebar: { label: "B · 微积分", order: 2 }
---

单元 03 从头构建反向模式微分；本附录把它所依赖的微积分集中陈述，并补上使“穿过线性代数的反向传播”变得机械化的矩阵恒等式。例题中的数字由 `labs/test_appendices.py` 检查。

### 初学者图像：导数是一项局部预测

导数首先不是搬动符号的规则，而是输入有小变化时，输出如何改变的最优线性预测。若 $f(3)=9$、$f'(3)=6$，则

$$
f(3+h)\approx 9+6h
$$

对多个输入，梯度预测标量变化 $\Delta f\approx\nabla f^{\top}\Delta x$；雅可比矩阵预测向量变化 $\Delta y\approx J\Delta x$。反向传播反复回答同一题：**已知输出端的敏感度，每个输入应分到多少敏感度？**

形状是第一个错误检测器。若 $x\in\mathbb R^n$、$y=f(x)\in\mathbb R^m$，则 $J_f$ 是 $m\times n$。在相乘雅可比矩阵以前，先把形状写在每个因子旁。这个习惯能在做任何代数以前抓到大量推导与程序错误。

| 小节 | 用于 |
|---|---|
| B.1–B.2 梯度、雅可比矩阵、链式法则 | 03（反向传播）、04、05（BPTT） |
| B.3 泰勒定理、海森矩阵 | 03（步长、下降引理）、06（精度） |
| B.4 矩阵微分 | 02（嵌入）、03、04（注意力的反向传播） |
| B.5 Softmax 与交叉熵 | 01、03、04、07、09 |
| B.6 积分与变量替换 | 08（流）、09（分数函数梯度） |
| B.7 有限差分 | 03（梯度检查）、每一个主干项目测试 |

## B.1 导数与梯度

对 $f:\mathbb{R}^n\to\mathbb{R}$，**梯度（gradient）** $\nabla f(x)\in\mathbb{R}^n$ 是偏导数构成的向量。当 $f$ 在 $x$ 可微时，它是使线性模型在一阶上精确的那个向量：
$$
f(x + h) = f(x) + \nabla f(x)^{\top}h + o(\lVert h\rVert).
$$
沿单位向量 $u$ 的**方向导数**是 $\nabla f(x)^{\top}u$；由柯西–施瓦茨（附录 A，定理 A.1），它在 $u = \nabla f/\lVert\nabla f\rVert$ 时最大，这就是 $-\nabla f$ 为最陡下降方向的原因。

**例题 B.1。** $f(x, y) = x^2y + \sin y$ 的 $\nabla f = (2xy,\ x^2 + \cos y)$。在 $(1, 0)$：$\nabla f = (0, 2)$，所以一阶近似下，$x$ 变动时 $f$ 不变，$y$ 变动时 $f$ 以速率 2 增加。

所有偏导数存在并不保证可微；偏导数连续才保证。本书用到的每个函数，除了孤立的折点——ReLU 在 0、最大池化中的最大值——之外都可微；在折点上，框架返回**次微分**中的一个元素，这个选择几乎处处无害（单元 03，§3.7.1）。

## B.2 雅可比矩阵与链式法则

对 $f:\mathbb{R}^n\to\mathbb{R}^m$，**雅可比矩阵（Jacobian）** $J_f(x)\in\mathbb{R}^{m\times n}$ 的元素为 $\partial f_i/\partial x_j$，且 $f(x + h) = f(x) + J_f(x)h + o(\lVert h\rVert)$。

**定理 B.1（链式法则）。** 若 $g$ 在 $x$ 可微、$f$ 在 $g(x)$ 可微，则 $J_{f\circ g}(x) = J_f(g(x))\,J_g(x)$。

*证明概要。* $f(g(x + h)) = f(g(x) + J_gh + o(h)) = f(g(x)) + J_f(J_gh + o(h)) + o(J_gh + o(h)) = f(g(x)) + J_fJ_gh + o(\lVert h\rVert)$。$\blacksquare$

对以标量损失结尾的合成 $L = f_k\circ\dots\circ f_1$，梯度是乘积 $\nabla L^{\top} = J_{f_k}J_{f_{k-1}}\cdots J_{f_1}$。矩阵乘法满足结合律，所以这个乘积可以用任何顺序计算，而顺序决定成本：

- **前向模式**从右边开始乘，先算 $J_{f_1}$：每一趟计算一个雅可比–向量乘积 $Jv$，所以对 $n$ 个输入的完整梯度需要 $n$ 趟。
- **反向模式**从左边、从标量损失开始乘：每一趟计算一个向量–雅可比乘积 $u^{\top}J$，所以完整梯度只需一次反向传递，成本是前向传递的小常数倍（单元 03，§3.3.4 与 §3.5.9）。

中间量 $v$ 的**伴随值（adjoint）**为 $\bar v = \partial L/\partial v$。反向模式就是规则 $\bar v_p = \sum_{c\,\in\,\text{children}(p)}\bar v_c\,\partial v_c/\partial v_p$：伴随值在从 $p$ 到损失的每一条路径上*累加*（单元 03，§3.3.3）。

## B.3 泰勒定理与海森矩阵

对二次连续可微的 $f:\mathbb{R}^n\to\mathbb{R}$，**海森矩阵（Hessian）** $\nabla^2 f(x)$ 是二阶偏导数构成的对称矩阵。

**定理 B.2（带余项的泰勒定理）。** 对每个 $x, h$，存在 $x$ 到 $x + h$ 线段上的 $\xi$，使
$$
f(x + h) = f(x) + \nabla f(x)^{\top}h + \tfrac12h^{\top}\nabla^2 f(\xi)\,h .
$$

*证明。* 对 $\phi(s) = f(x + sh)$ 应用一维带拉格朗日余项的泰勒定理，其导数为 $\phi'(s) = \nabla f(x + sh)^{\top}h$ 与 $\phi''(s) = h^{\top}\nabla^2f(x + sh)h$。$\blacksquare$

**推断：二次上界。** 若海森矩阵的特征值从不超过 $L$——等价地，梯度是 $L$-Lipschitz 的——则
$$
f(y)\le f(x) + \nabla f(x)^{\top}(y - x) + \tfrac L2\lVert y - x\rVert^2 .
$$
对 $y$ 最小化右式，得到步骤 $y = x - \nabla f(x)/L$ 与保证的下降量 $\lVert\nabla f(x)\rVert^2/(2L)$：这就是单元 03（§3.5.2）与附录 F 中每个收敛证明背后的**下降引理**。

**二阶条件。** 在临界点（$\nabla f = 0$），正定的海森矩阵表示严格局部极小，负定表示局部极大，不定则表示**鞍点**。在高维中，随机函数的临界点里鞍点远多于极小点，而带噪声的梯度方法会逃离它们（单元 03，§3.7.2）。

## B.4 矩阵微分

对矩阵求梯度，最可靠的方式是用**微分**：把 $df$ 写成 $dX$ 的线性函数，再从 $df = \mathrm{tr}(G^{\top}dX) = \sum_{ij}G_{ij}\,dX_{ij}$ 读出梯度 $G$。规则与标量微积分相同，只是要保持不可交换因子的顺序（[Magnus & Neudecker 2019](/zh-cn/references/#magnus-2019)）：
$$
d(XY) = (dX)Y + X(dY),\qquad d(X^{-1}) = -X^{-1}(dX)X^{-1},\qquad d\ln\det X = \mathrm{tr}(X^{-1}dX).
$$

| 函数 | 微分 | 梯度 |
|---|---|---|
| $a^{\top}x$ | $a^{\top}dx$ | $a$ |
| $x^{\top}Ax$ | $x^{\top}(A + A^{\top})dx$ | $(A + A^{\top})x$ |
| $\lVert Ax - b\rVert^2$ | $2(Ax - b)^{\top}A\,dx$ | $2A^{\top}(Ax - b)$ |
| $\mathrm{tr}(AX)$ | $\mathrm{tr}(A\,dX)$ | $A^{\top}$ |
| $\ln\det X$（$X\succ0$） | $\mathrm{tr}(X^{-1}dX)$ | $X^{-\top}$ |

**穿过矩阵乘积的反向传播。** 若 $Y = XW$，上游梯度为 $\bar Y = \partial L/\partial Y$，则 $dL = \mathrm{tr}(\bar Y^{\top}dY) = \mathrm{tr}(\bar Y^{\top}(dX)W) + \mathrm{tr}(\bar Y^{\top}X\,dW)$，循环置换迹即得
$$
\bar X = \bar YW^{\top},\qquad \bar W = X^{\top}\bar Y .
$$
主干项目手写反向传播中的每一个线性层，就是这两行。**形状检查**能抓到大部分错误：$\bar X$ 必须与 $X$ 同形状，而用手边的矩阵凑出这个形状，通常只有一种乘法方式。

**例题 B.2（查找的梯度）。** 嵌入查找是 $H = UE$，$U$ 的每一行是独热向量，所以 $\bar E = U^{\top}\bar H$：$\bar E$ 的每一行，是查找到它的那些上游行的*总和*（单元 02，命题 2.1）。

## B.5 Softmax、log-sum-exp 与交叉熵

$z\in\mathbb{R}^K$ 的 **softmax** 是 $p_k = e^{z_k}/\sum_je^{z_j}$；**log-sum-exp** 是 $\mathrm{LSE}(z) = \ln\sum_je^{z_j}$，所以 $\ln p_k = z_k - \mathrm{LSE}(z)$。

**命题 B.3。** (i) $\nabla\mathrm{LSE}(z) = p$。(ii) softmax 的雅可比矩阵为 $J = \mathrm{diag}(p) - pp^{\top}$。(iii) 对交叉熵损失 $\ell = -\ln p_y$，$\nabla_z\ell = p - e_y$。(iv) $\nabla^2\mathrm{LSE} = \mathrm{diag}(p) - pp^{\top}\succeq0$，所以 LSE 是凸的。

*证明。* (i) $\partial\mathrm{LSE}/\partial z_k = e^{z_k}/\sum_je^{z_j} = p_k$。(iii) $\ell = -z_y + \mathrm{LSE}(z)$，所以 $\nabla\ell = -e_y + p$。(ii) 与 (iv) 把 (i) 再微分一次：$\partial p_k/\partial z_j = p_k(\mathbb{1}[j = k] - p_j)$。这个矩阵是一个协方差——$Y\sim p$ 时独热向量 $e_Y$ 的协方差——因此是半正定的。$\blacksquare$

**例题 B.3。** $z = (1, 0, 0)$ 给出 $p = (0.5761, 0.2119, 0.2119)$，且
$$
J = \begin{pmatrix}0.2442 & -0.1221 & -0.1221\\-0.1221 & 0.1670 & -0.0449\\-0.1221 & -0.0449 & 0.1670\end{pmatrix}.
$$
每一行总和为零，因为对所有 logit 加上同一个常数不会改变任何事。目标为 $y = 2$ 时，交叉熵梯度为 $p - e_2 = (0.5761, -0.7881, 0.2119)$：提高目标的 logit，并依概率比例降低其他 logit。

**数值稳定性。** 直接计算 $e^{z_k}$，在 32 比特运算中 $z_k\gtrsim 89$ 就会溢位。改用 $\mathrm{LSE}(z) = m + \ln\sum_je^{z_j - m}$，其中 $m = \max_jz_j$；平移后的指数至多为 1。对 $z = (1000, 999)$，这给出 $1000 + \ln(1 + e^{-1}) = 1000.3133$，而天真的公式会返回无穷大。交叉熵应以 $\mathrm{LSE}(z) - z_y$ 计算，而不是对一个算出来的概率取 $-\ln$——那个概率可能下溢成 0。

**书中用到的其他导数。** logistic 函数 $\sigma'(u) = \sigma(u)(1 - \sigma(u))$；$\tanh'(u) = 1 - \tanh^2u$；层归一化与 GELU 请见主干项目的 `model.py`，其导数以有限差分检查。

## B.6 积分、变量替换与积分号下求导

**变量替换。** 若 $g:\mathbb{R}^n\to\mathbb{R}^n$ 是可微的双射且其反函数可微，则
$$
\int_{g(\Omega)}f(y)\,dy = \int_\Omega f(g(x))\,\big|\det J_g(x)\big|\,dx .
$$
应用到概率密度，就是标准化流背后的公式 $p_Y(y) = p_X(g^{-1}(y))\,|\det J_{g^{-1}}(y)|$（单元 08，§8.3.2；附录 C）。

**高斯积分。** $\int_{-\infty}^{\infty}e^{-x^2/2}dx = \sqrt{2\pi}$。*证明：*把它平方，得 $\iint e^{-(x^2 + y^2)/2}dx\,dy$，换成极坐标，其雅可比行列式为 $r$：$\int_0^{2\pi}\int_0^\infty e^{-r^2/2}r\,dr\,d\theta = 2\pi$。$\blacksquare$

**积分号下求导。** 若 $p_\theta(x)$ 及其对 $\theta$ 的导数连续，且被一个可积函数所控制，则 $\nabla_\theta\int f(x)p_\theta(x)\,dx = \int f(x)\nabla_\theta p_\theta(x)\,dx$。两个推断支撑了单元 08 与 09：

- **分数函数恒等式。** $\nabla_\theta\mathbb{E}_{p_\theta}[f] = \mathbb{E}_{p_\theta}[f\,\nabla_\theta\ln p_\theta]$，用到 $\nabla p = p\nabla\ln p$。它不需要 $f$ 的导数——这是 REINFORCE 的基础（单元 09）。
- **重参数化（reparameterization）。** 若 $x = g_\theta(\varepsilon)$，$\varepsilon$ 来自固定分布，则 $\nabla_\theta\mathbb{E}[f(x)] = \mathbb{E}_\varepsilon[\nabla_\theta f(g_\theta(\varepsilon))]$。它需要 $f$ 可微，但通常方差低得多——这是变分自编码器所用的梯度（单元 08，§8.3.3）。

## B.7 有限差分

要检查导数，就把它与差商比较。**前向差分** $(f(x + h) - f(x))/h$ 的截断误差为 $\tfrac h2f''(x) + O(h^2)$；**中心差分** $(f(x + h) - f(x - h))/(2h)$ 的误差为 $\tfrac{h^2}{6}f'''(x) + O(h^4)$，因为偶数阶项相消。

浮点舍入会加入约 $\epsilon_{\text{mach}}|f|/h$ 的误差，随 $h$ 变小而增大。平衡两者，中心差分的最优 $h\approx\epsilon_{\text{mach}}^{1/3}$——64 比特运算中约为 $6\times10^{-6}$。

**例题 B.4。** 对 $f = \exp$ 在 $0$（真实导数为 1），64 比特运算中中心差分的误差为：

| $h$ | $10^{-1}$ | $10^{-2}$ | $10^{-3}$ | $10^{-5}$ | $10^{-8}$ | $10^{-11}$ |
|---|---|---|---|---|---|---|
| 误差 | $1.67\times10^{-3}$ | $1.67\times10^{-5}$ | $1.67\times10^{-7}$ | $1.2\times10^{-11}$ | $6\times10^{-9}$ | $8\times10^{-8}$ |

前三个精确地遵循 $h^2/6$；在 $h\approx10^{-5}$ 以下，舍入误差接手，误差又增大。这就是为什么主干项目的梯度检查使用 64 比特运算与 $h = 10^{-6}$，并比较*相对*误差（单元 03，§3.7.5）。

## 习题

**B.1** 计算 $f(x) = x/\lVert x\rVert$ 的雅可比矩阵，并证明 $J_fx = 0$。加以诠释。

**B.2** 对 $Y = XW + \mathbf 1b^{\top}$ 推导 $\partial L/\partial X$ 与 $\partial L/\partial W$，并求 $\partial L/\partial b$。

**B.3** 证明对任何概率向量 $q$，$\nabla_z\big(-\sum_kq_k\ln p_k\big) = p - q$（软目标）。

**B.4** 以微分证明 $\nabla_X\,\mathrm{tr}(X^{\top}AX) = (A + A^{\top})X$。

**B.5** 对 $f(x) = x^3$ 在 $x = 1$、$h = 0.1$，计算前向与中心差分的误差，并与预测的 $\tfrac h2f''$ 与 $\tfrac{h^2}6f'''$ 比较。

**B.6** ★ 证明线性 softmax 模型 $z = Wx$ 的交叉熵损失对 $\mathrm{vec}(W)$ 的海森矩阵为 $(\mathrm{diag}(p) - pp^{\top})\otimes xx^{\top}$，并推断损失在 $W$ 上是凸的。

**解答。** **B.1** $J = (I - uu^{\top})/\lVert x\rVert$，其中 $u = x/\lVert x\rVert$：归一化忽略沿 $x$ 方向的改变。**B.2** $\bar X = \bar YW^{\top}$、$\bar W = X^{\top}\bar Y$、$\bar b = \bar Y^{\top}\mathbf 1$。**B.5** 前向：$(1.331 - 1)/0.1 - 3 = 0.31$（预测 $0.3$）；中央：$(1.331 - 0.729)/0.2 - 3 = 0.01$（预测 $0.01$，对三次函数精确）。

## 延伸阅读

[Magnus & Neudecker 2019](/zh-cn/references/#magnus-2019) 讲矩阵微分；[Griewank & Walther 2008](/zh-cn/references/#griewank-2008) 讲自动微分；[Deisenroth et al. 2020](/zh-cn/references/#deisenroth-2020) 第 5 章讲机器学习中的向量微积分；[Higham 2002](/zh-cn/references/#higham-2002) 讲浮点误差。
