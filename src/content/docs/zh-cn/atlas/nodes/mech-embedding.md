---
title: 嵌入表与绑定读出
description: 每个符号一个学习出的向量；查找是一个独热矩阵乘积，其梯度是散射加总；绑定用同一张表来读取与打分。
atlas:
  id: mech-embedding
  type: mechanism
  mastery: internalize
  core_units: [2, 4, 7]
  prerequisites: [math-chain-rule]
  related: [mech-cross-entropy, arch-transformer-block]
  references: [bengio-2003, press-2017, mikolov-2013]
  last_reviewed: 2026-11-02
---

## 一段话说明

嵌入表 $E\in\mathbb{R}^{V\times d}$ 为 $V$ 个符号各指定一个学习出的向量。查找符号 $a$ 等于以其独热向量做矩阵乘积 $E^{\top}\mathbf u_a$，所以对一批数据 $H = UE$，梯度为 $\partial L/\partial E = U^{\top}G$：符号的每次出现都把它的上游梯度**加到**该符号的行上。**绑定（tied）**的模型以同一行 $e_b$ 为输出 $b$ 打分，让嵌入参数减半。相似的符号收到相似的梯度并彼此靠拢，这就是嵌入共享统计强度的方式（单元 02，§2.2.4）。

## 不变量

- 查找梯度等于独热乘积的梯度：`np.add.at(dE, x, G)` $= U^{\top}G$（主干项目 `test_stage02_lookup_gradient_equals_one_hot_matmul_gradient`）。
- 在一批数据中未出现的符号，其行的梯度恰好为零（权重衰减之前）。
- 读出为零（或分解式二元语法中 $M$ 为零）时，初始损失恰好是 $\ln V$。
- 在每个上下文中都可互换的符号，会收敛到相同的嵌入（差模型的对称性）（主干项目 `test_stage02_interchangeable_symbols_get_the_same_embedding`）。

## 已知数值

| 量 | 值 |
|---|---|
| 查找梯度，$V = 3$、$x = (0, 2, 0)$、$G = [[1,2],[3,4],[5,6]]$ | 各行 $(6, 8)$、$(0, 0)$、$(3, 4)$ |
| 分解式二元语法参数量，$V = 50$、$d = 16$ | $Vd + d^2 + V = 1{,}106$ |
| 神经 4 元语法参数量，$d = 16$、$H = 128$ | $15{,}570$ |
| 主干模型中数字的最近邻余弦（$d = 16$） | 约 $0.97$–$0.98$ |

## 常见错误

- 用 `dE[x] = G` 取代散射加总：重复的符号会互相覆盖，而不是累加。
- 以过大的尺度初始化绑定表，使初始 logits 远离均匀，损失一开始就远高于 $\ln V$。
- 对罕见符号与常见符号的嵌入施加相同的权重衰减，却没注意到罕见的列在两次更新之间会缩向零。
- 逐坐标比较两次训练的嵌入：它们只在差一个可逆变换的意义下有定义（单元 02，习题 2.11）。
