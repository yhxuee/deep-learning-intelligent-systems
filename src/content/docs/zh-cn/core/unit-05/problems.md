---
title: "5.10 习题与延伸阅读"
description: 二十道分级习题——计算、推导、证明、实现与设计问题——附提示、部分解答与附注释的阅读清单。
sidebar: { label: "5.10 习题", order: 10 }
---

难度：★ 常规应用 · ★★ 需要结合多个概念 · ★★★ 证明或开放式设计。标记 **[不使用 AI]** 的习题用于独立练习与评估；其余题目可以在自己先尝试后，再搭配 [AI 导师](/zh-cn/mentor/)探索。部分题目的简答列在清单之后。

## 计算

**5.1 ★ [不使用 AI]** 以 $\alpha = 0.7$、$h_0 = 10$ 对输入 $(10, 12, 10, 30, 10)$ 做 EMA。写出 $h_1,\dots,h_5$，以及尖峰在 $h_5$ 中残留的比例。

**5.2 ★** 对 $\alpha = 0.95$ 计算精确半衰期与近似值 $\ln2/(1-\alpha)$。证明当 $\alpha\in[0.5,1)$ 时，近似的相对误差小于 $1-\alpha$。*提示：利用 $-\ln(1-\varepsilon) = \varepsilon + \varepsilon^2/2 + \dots$ 比较 $-\ln\alpha$ 与 $1-\alpha$。*

**5.3 ★** 证明 EMA 中信息的平均年龄 $\sum_{j\ge0}j(1-\alpha)\alpha^j$ 等于 $\alpha/(1-\alpha)$。*提示：对 $\sum_j\alpha^j = 1/(1-\alpha)$ 关于 $\alpha$ 微分。*

**5.4 ★ [不使用 AI]** 传感器以 10 Hz 采样，记忆的时间常数应为 2 秒。写出精确的 $\alpha$、前向欧拉的 $\alpha$ 与后向欧拉的 $\alpha$（§5.6.3）。哪个最接近精确值？

**5.5 ★** 求 $A = \begin{pmatrix}0.8&-0.6\\0.6&0.8\end{pmatrix}$ 的特征值、谱半径与振荡周期。这个递推稳定吗？脉冲响应会如何？

**5.6 ★★** 以对角化求 $A = \begin{pmatrix}0.6&0.2\\0&0.9\end{pmatrix}$、$B = (0, 1)^\top$、$C = (1, 0)$ 的 $K_j = CA^jB$ 封闭解，并找出 $K_j$ 达到峰值的时间差 $j$。

## 推导

**5.7 ★★ [不使用 AI]** 以归纳法证明矩阵封闭解 $h_t = A^th_0 + \sum_{k=1}^tA^{t-k}Bx_k$，并指出矩阵乘法的顺序在哪里重要。

**5.8 ★★** 对负数对角元素的对角 $A$，逐分量推导零阶保持离散化 $\bar A = e^{A\Delta}$、$\bar B = A^{-1}(e^{A\Delta}-I)B$，并验证当 $B_i = -a_i$ 时，每个分量都化为 $\alpha_i = e^{a_i\Delta}$ 的 EMA。

**5.9 ★★** 令 $A = \lambda I + N$，其中 $N^3 = 0$ 但 $N^2\ne0$。推导 $A^t$，并求 $\lambda = 0.9$ 时 $N^2$ 系数达到峰值的步数。与 §5.3.5 中 $N^2 = 0$ 的情况比较。

**5.10 ★★ [不使用 AI]** 对向量 RNN $h_t = \tanh(Wh_{t-1}+Ux_t+b)$、损失 $L = \sum_t L_t(h_t)$，推导 $\bar h_t$ 的 BPTT 递推与 $\partial L/\partial W$ 的表示式。接着以 NumPy 实现并用中心差分检查（单元 03 的[梯度检查](/zh-cn/atlas/nodes/method-gradient-check/)）。

**5.11 ★★** 对 §5.3.8 的 LSTM，推导 $\partial c_t/\partial c_{t-1}$，包括经由 $h_{t-1} = o_{t-1}\odot\tanh(c_{t-1})$ 进入时刻 $t$ 各个闸的路径。在什么条件下，直接项 $\mathrm{diag}(f_t)$ 占主导？

**5.12 ★★** 证明合成运算 $(a_1,b_1)\bullet(a_2,b_2) = (a_2a_1, a_2b_1+b_2)$ 满足结合律但不满足交换律。写出 $T = 8$ 时并行扫描的下扫，并计算每次扫描用了几次 $\bullet$ 运算。

**5.13 ★★** 对例题 5.3 的系统，推导转移函数 $G(z) = Cz(zI-A)^{-1}B$，找出其极点，并计算 $\omega = 0$ 与 $\omega = \pi$ 时的 $|G(e^{i\omega})|$。

**5.14 ★★** 在 §5.6.2 的“随机漫步加噪声”模型中，把稳态 $\alpha$ 表示成比值 $\rho = q/r$ 的函数。求 $\rho = 10^{-4}, 10^{-2}, 1$ 时的 $\alpha$，并解释 $\rho\to0$ 与 $\rho\to\infty$ 的极限。

## 证明

**5.15 ★★ [不使用 AI]** 证明定理 5.1（可对角化递推的稳定性），包括上界 $\|A^t\|\le\kappa(V)\rho(A)^t$。

**5.16 ★★** 证明当 $\|k\| = 1$ 时，一次 delta 规则更新 $S \leftarrow S + (v - Sk)k^\top$ 会使 $Sk = v$ 精确成立，且对每个 $k'\perp k$ 不改变 $Sk'$。由此推得：delta 规则只需一轮，就能以零干扰存储正交单位键。

**5.17 ★★★** 把定理 5.2 推广到“在均匀随机窗口上允许以概率 $\varepsilon$ 失败”的回忆。*提示：Fano 不等式给出状态熵的下界。*

**5.18 ★★★** 证明对任何切点 $c$，状态大小为 $N$ 的选择性 SSM，其矩阵 $M$ 的过去到未来区块秩至多为 $N$（§5.5.4）。再构造输入，使单一 softmax 注意力层的过去到未来区块秩为 $\min(c, T-c)$。

## 设计与实现

**5.19 ★★★** 设计一组 $N = 8$ 个 EMA，其半衰期在 1 到 $10^4$ 步之间以等比间隔分布。写出各个 $\alpha_i$。提出一个读出方式，能对该范围内任意 $W$ 估计“最近 $W$ 个输入的平均”，并分析其误差。这对初始化对角 SSM 的时间常数有什么启示？

**5.20 ★★（主干项目）** 完成实验 5.8：为主干模型实现 KV 缓存解码并附等价性测试，再加上回忆压力测试与其对照组。以一页报告：预测、结果、落差、解释。

## 部分解答

- **5.1** $h = (10, 10.6, 10.42, 16.294, 14.406)$。尖峰以权重 $(1-\alpha)\alpha = 0.21$ 进入 $h_5$；它比基线高出的 20 °C 对 $h_5$ 贡献 $0.21\cdot 20 = 4.2$（它对 $h_4$ 贡献 6，所以再过一步仍保留 70% 的效应）。
- **5.4** $\Delta t/\tau = 0.05$：精确值 $e^{-0.05} = 0.951229$；前向欧拉 $0.95$（误差 0.00123）；后向欧拉 $1/1.05 = 0.952381$（误差 0.00115）。两者都是一阶精度，并从两侧夹住精确值；这里后向欧拉略为接近，而且不像前向欧拉，它对任何步长都稳定。
- **5.5** $\lambda = 0.8\pm0.6i$，$|\lambda| = 1$：这个矩阵是纯旋转，角度 $\arctan(0.6/0.8)\approx0.6435$ 弧度，周期 $\approx 9.76$ 步。并非渐近稳定：脉冲响应永远振荡而不衰减（临界稳定，§5.3.4）。
- **5.14** $\alpha = 1 - K$，其中 $K = P/(P+1)$、$P = (\rho + \sqrt{\rho^2+4\rho})/2$（以 $r$ 为单位）。当 $\rho\to0$，$\alpha\to1$（永远取平均）；当 $\rho\to\infty$，$\alpha\to0$（只相信最新读数）。

## 延伸阅读

**基础。**
[Oppenheim & Schafer 2010](/zh-cn/references/#oppenheim-2010)，z 转换与 IIR 滤波器各章——§5.5.1 背后的经典理论。
[Trefethen & Embree 2005](/zh-cn/references/#trefethen-2005)——为什么只看特征值不够；瞬态增长的权威著作。
[Kalman 1960](/zh-cn/references/#kalman-1960)——状态空间估计的原始论文；请读其中“递推更新的充分统计量”这个想法。

**循环网络。**
[Bengio et al. 1994](/zh-cn/references/#bengio-1994) 与 [Pascanu et al. 2013](/zh-cn/references/#pascanu-2013)——梯度消失／爆炸的分析与梯度裁剪。
[Hochreiter & Schmidhuber 1997](/zh-cn/references/#hochreiter-1997)；[Gers et al. 2000](/zh-cn/references/#gers-2000)——LSTM 与遗忘闸的原始框架。

**状态空间模型与线性注意力。**
[Gu et al. 2020](/zh-cn/references/#gu-2020)（HiPPO）→ [Gu et al. 2022a](/zh-cn/references/#gu-2022)（S4）→ [Gu & Dao 2024](/zh-cn/references/#gu-2024)（Mamba）→ [Dao & Gu 2024](/zh-cn/references/#dao-2024)（状态空间对偶）：依此顺序阅读，可看到 §5.3.9–§5.3.10 与 §5.5.4 的想法如何发展。
[Katharopoulos et al. 2020](/zh-cn/references/#katharopoulos-2020)；[Schlag et al. 2021](/zh-cn/references/#schlag-2021)——线性注意力作为递推，以及作为快速权重记忆。

**极限。**
[Arora et al. 2024](/zh-cn/references/#arora-2024)；[Jelassi et al. 2024](/zh-cn/references/#jelassi-2024)——§5.7.1 背后的证据，也是如何检验记忆主张的模板。
