---
title: MinHash 近似重复检测
description: 片段集合的 Jaccard 相似度、碰撞概率等于 Jaccard 相似度的 MinHash 签章，以及不必比较所有配对就能找出候选配对的分带法。
atlas:
  id: method-minhash-dedup
  type: method
  mastery: understand-and-verify
  core_units: [7, 10]
  prerequisites: []
  related: [method-contamination-check]
  references: [broder-1997, lee-2022]
  last_reviewed: 2026-11-02
---

## 一段话说明

以片段（shingle，词的 $n$ 元组）集合表示每份文档。相似程度是 Jaccard 相似度 $J = |S_1\cap S_2|/|S_1\cup S_2|$。MinHash 对片段的一个随机排序 $h$，保留最小值 $\min_{s\in S}h(s)$；两份文档的最小值相同的概率恰好是 $J$。把 $n$ 个最小值组成的签章分成 $b$ 个带、每带 $r$ 行，只要有任一带相同，两份文档就成为候选，概率为 $1 - (1 - J^r)^b$——一条在 $(1/b)^{1/r}$ 附近上升的 S 形曲线。候选再以精确的 Jaccard 相似度确认。

## 不变量

- $\Pr[\text{MinHash 相同}] = J$；用 $n$ 个哈希时，相同比例的标准误为 $\sqrt{J(1-J)/n}$。
- 相同的文档一定碰撞；没有共同片段的文档一定不碰撞。
- 候选概率对 $J$ 单调，且等于 $1 - (1-J^r)^b$（主干项目 `test_stage07_minhash_collision_probability_matches_theory`）。
- 在主干语料上，每一组植入的重复配对都被找到，且没有移除其他任何故事（主干项目 `test_stage07_finds_every_planted_duplicate_and_nothing_else`）。

## 已知数值

| 量 | 值 |
|---|---|
| 候选概率，$b = 16$、$r = 4$、$J = 0.3 / 0.5 / 0.7 / 0.8$ | 0.122 / 0.644 / 0.988 / 0.9998 |
| $b = 16$、$r = 4$ 时的陡峭门槛 $(1/b)^{1/r}$ | 0.5 |
| 主干语料：找到的植入配对（完全／近似） | 67 组中的 67 组（37／30），没有误报 |
| 主干语料：典型近似重复的 $J$；最相似的不同配对（前 400 个） | 0.775；0.245 |

## 常见错误

- 用 Python 内建的 `hash` 做 MinHash：它在每个行程中加了盐，签章在不同执行之间会不同。
- 从词片段改成字符片段（或反之）却没有重新调整门槛。
- 把重复配对的*两者*都移除，而不是保留其中一个。
- 把分带得到的候选直接当成重复，而没有以精确的 Jaccard 相似度确认。
