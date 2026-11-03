---
title: "1.10 习题与延伸阅读"
description: 二十道关于对数损失、最大似然、平滑、集中不等式、保留数据评估与类推理论的分级习题，附部分解答与附注释的阅读清单。
sidebar: { label: "1.10 习题", order: 10 }
---

难度：★ 常规 · ★★ 结合多个概念 · ★★★ 证明或开放式问题。**[不使用 AI]** 的题目用于独立练习与评估。

## 计算

**1.1 ★ [不使用 AI]** 一位预报员连续五天都预测降雨概率 0.8；其中三天下雨。计算平均对数损失与准确率（$q > 0.5$ 时就预测下雨）。对一位总是说 0.6 的预报员重做一次。两种评分各偏好哪一位？

**1.2 ★ [不使用 AI]** 以 $\alpha = 0.5$ 重做例题 1.2。保留数据损失现在低于 $\ln 3$ 了吗？

**1.3 ★** 把每字符 1.2 nats 的保留数据损失换算成每字符比特数与困惑度。困惑度 2 对应多少损失？

**1.4 ★** 对例题 1.1 的硬币，找出使 $q = 0.7$ 的期望测试损失等于其训练损失 0.6109 的真实 $p$。

**1.5 ★★** 一个二元语法在某段文本上拟合，其中字符 `z` 恰好出现一次，后面接 `e`。写出 $V = 50$ 时平滑后的行 $q_\alpha(\cdot\mid\texttt z)$。α 为多少时 $q_\alpha(\texttt e\mid\texttt z) = 0.5$？

**1.6 ★★** 利用 Hoeffding 不等式，要以 99% 的置信度把 0–1 错误率估计到 ±0.5 个百分点以内，需要多少保留样本？若有 1,000 个候选模型呢？

## 推导

**1.7 ★ [不使用 AI]** 以拉格朗日乘子推导 $V$ 种结果的类别分布的最大似然估计（命题 1.1）。

**1.8 ★★ [不使用 AI]** 证明 $H(p,q) = H(p) + \mathrm{KL}(p\,\Vert\,q)$ 与 Gibbs 不等式。举一个 $\mathrm{KL}(p\,\Vert\,q)\ne\mathrm{KL}(q\,\Vert\,p)$ 的例子。

**1.9 ★★** 证明在 Dirichlet($\alpha$) 先验下，类别参数的后验是 Dirichlet($c + \alpha$)，且结果 $b$ 的后验预测概率等于加 α 估计。

**1.10 ★★** 证明期望对数损失是*严格*恰当的评分：对任何 $q\ne p$，$\mathbb{E}_p[-\ln q] > \mathbb{E}_p[-\ln p]$。然后找出 $p$ 与 $q\ne p$，使 $q$ 的期望分数较低，以证明*绝对*评分 $1 - q(x)$ 不是恰当的。

**1.11 ★★** 证明以最大似然拟合的二元语法，其训练损失等于经验条件熵 $\sum_a \frac{c(a)}{n}H(\hat p(\cdot\mid a))$。

## 证明

**1.12 ★★ [不使用 AI]** 证明偏差–方差分解（命题 1.2），并指出每个假设在哪里被用到。

**1.13 ★★** 对允许随机化预测的学习者，证明没有免费午餐定理（定理 1.4）。

**1.14 ★★★** 证明 Hoeffding 引理：若 $\mathbb{E}X = 0$ 且 $X\in[a,b]$，则 $\mathbb{E}e^{sX}\le e^{s^2(b-a)^2/8}$。

**1.15 ★★★** 证明实数线上的门槛分类器族 $h_t(x) = \mathbb{1}[x\ge t]$ 的 VC 维度为 1，而区间 $\mathbb{1}[a\le x\le b]$ 的 VC 维度为 2。

**1.16 ★★★** 令 $\hat L_1,\dots,\hat L_K$ 为 $K$ 个模型的验证损失，其真实损失都等于 $L$，各自带有标准差 $\sigma$ 的独立高斯噪声。证明 $\mathbb{E}[\min_k\hat L_k] < L$，且当 $K$ 很大时，这个乐观程度像 $\sigma\sqrt{2\ln K}$ 一样成长。

## 实现与设计

**1.17 ★ [不使用 AI]** 不看数据、不使用 NumPy 的 `add.at`，实现 `fit_bigram` 与 `nll`。验证你的表每一行总和为 1，且 α → ∞ 时得到 $\ln V$。

**1.18 ★★** 在主干语料上实现带加 α 平滑的三元语法模型。比较它与二元语法在 2,000 与 385,274 个训练字符时的训练与保留数据损失。以 §1.7.7 解释结果。

**1.19 ★★** 实现*内插*平滑 $q(b\mid a) = \lambda\,\hat p(b\mid a) + (1-\lambda)\,\hat p(b)$，在验证故事上选择 λ，并与加 α 平滑比较。

**1.20 ★★★** 为一个将用于明年所写故事的语言模型设计评估流程。说明你所做的分布假设、你的切分如何检验它，以及什么结果会显示该假设是错的（单元 10 的预告）。

## 部分解答

- **1.1** 预报员 0.8：对数损失 $(3\times0.2231 + 2\times1.6094)/5 = 0.7777$，准确率 3/5。预报员 0.6：$(3\times0.5108 + 2\times0.9163)/5 = 0.6730$，准确率 3/5。准确率分不出两者；对数损失偏好概率（0.6）等于观察频率的那一位。
- **1.3** 1.2 nats = 1.731 比特；困惑度 $e^{1.2} = 3.32$。困惑度 2 ⇔ 损失 $\ln 2 = 0.693$ nats = 1 比特。
- **1.5** $q_\alpha(\texttt e\mid\texttt z) = (1+\alpha)/(1+50\alpha)$；令它等于 0.5，得 $2 + 2\alpha = 1 + 50\alpha$，即 $\alpha = 1/48$。
- **1.6** $n \ge \ln(2/\delta)/(2\varepsilon^2) = \ln 200/(2\times0.005^2) = 105{,}967$；若 $K = 1{,}000$，$n\ge(\ln 1000 + \ln 200)/(2\times 0.005^2) = 244{,}122$。
- **1.11** $n\hat L = -\sum_{a,b}c(a\to b)\ln\frac{c(a\to b)}{c(a)} = \sum_a c(a)\big[-\sum_b \hat p(b\mid a)\ln\hat p(b\mid a)\big]$。
- **1.16** $K$ 个 i.i.d. $N(0,\sigma^2)$ 变量的最小值，在 $K$ 很大时期望值约为 $-\sigma\sqrt{2\ln K}$；$K = 1000$ 时约为 $3.7\sigma$，这是一个上界，精确值（约 $3.24\sigma$）缓慢地趋近它。

## 延伸阅读

**基础。** [Shalev-Shwartz & Ben-David 2014](/zh-cn/references/#shalev-shwartz-2014)，第 2–6 章——关于 ERM、均匀收敛、VC 维度与没有免费午餐最清楚的现代论述。[Cover & Thomas 2006](/zh-cn/references/#cover-2006)，第 2 与 5 章——熵、KL 散度与编码。

**语言模型。** [Shannon 1951](/zh-cn/references/#shannon-1951)——至今仍是预测文本最好的入门。[Chen & Goodman 1999](/zh-cn/references/#chen-1999)——数十年平滑研究学到了什么。[Bengio et al. 2003](/zh-cn/references/#bengio-2003)——通往单元 02 的桥梁。

**评分规则。** [Gneiting & Raftery 2007](/zh-cn/references/#gneiting-2007)——为什么对数损失与 Brier 分数是恰当的，以及这带来什么好处。

**现代的谜题。** [Zhang et al. 2017](/zh-cn/references/#zhang-2017)、[Belkin et al. 2019](/zh-cn/references/#belkin-2019)、[Nakkiran et al. 2020](/zh-cn/references/#nakkiran-2020)、[Bartlett et al. 2020](/zh-cn/references/#bartlett-2020)——依此顺序阅读。

**评估实践。** [Recht et al. 2019](/zh-cn/references/#recht-2019) 与 [Dwork et al. 2015](/zh-cn/references/#dwork-2015)——被重复使用的测试集会发生什么事。
