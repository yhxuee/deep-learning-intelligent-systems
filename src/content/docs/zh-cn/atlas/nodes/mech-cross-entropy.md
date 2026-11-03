---
title: 交叉熵与似然
description: 默认的训练目标——观测数据的负对数似然——以及它应该从哪些数字开始。
atlas:
  id: mech-cross-entropy
  type: mechanism
  mastery: internalize
  core_units: [1, 3, 7, 10]
  prerequisites: [math-softmax]
  references: [goodfellow-2016, bengio-2003]
  last_reviewed: 2026-11-02
---

## 一段话说明

对模型分布 $q$ 与观测结果 $y$，损失 $-\log q(y)$ 是负对数似然（negative log-likelihood）。对数据平均后，它估计交叉熵（cross-entropy）$H(p, q) = -\mathbb{E}_{y\sim p}\log q(y) = H(p) + \mathrm{KL}(p\,\|\,q)$，所以最小化它就是最小化对数据分布的 KL 散度。语言模型以每个词元为单位报告它，单位为 nats（自然对数）或 bits（除以 $\ln 2$）；困惑度（perplexity）是每词元平均损失（nats）的 $\exp$。

## 不变量

- $-\log q(y) \ge 0$，等号只在 $q(y) = 1$ 时成立。
- 模型的交叉熵永远不低于数据的熵：$H(p,q)\ge H(p)$。
- 在 $V$ 个结果上均匀的模型，不论数据为何，得分恰好为 $\ln V$ nats。

## 已知数值

| 词表大小 $V$ | 均匀损失 $\ln V$（nats） | bits |
|---|---|---|
| 2 | 0.6931 | 1.0000 |
| 65（莎士比亚字符） | 4.1744 | 6.0224 |
| 50 257（GPT-2 BPE） | 10.8249 | 15.6170 |

刚初始化的语言模型应该从 $\ln V$ **附近**开始。高很多表示初始 logits 太大；低很多表示信息泄漏了（例如因果遮罩坏掉）。

## 常见错误

- 把填充词元也纳入平均。
- 在函数预期 logits 的地方传入概率（重复做两次 softmax）。
- 跨不同断词器比较损失：词元不同时，nats/词元无法比较；请改用每字节或每字符的 bits。
