---
title: MinHash 近似重複偵測
description: 片段集合的 Jaccard 相似度、碰撞機率等於 Jaccard 相似度的 MinHash 簽章，以及不必比較所有配對就能找出候選配對的分帶法。
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

## 一段話說明

以片段（shingle，詞的 $n$ 元組）集合表示每份文件。相似程度是 Jaccard 相似度 $J = |S_1\cap S_2|/|S_1\cup S_2|$。MinHash 對片段的一個隨機排序 $h$，保留最小值 $\min_{s\in S}h(s)$；兩份文件的最小值相同的機率恰好是 $J$。把 $n$ 個最小值組成的簽章分成 $b$ 個帶、每帶 $r$ 列，只要有任一帶相同，兩份文件就成為候選，機率為 $1 - (1 - J^r)^b$——一條在 $(1/b)^{1/r}$ 附近上升的 S 形曲線。候選再以精確的 Jaccard 相似度確認。

## 不變量

- $\Pr[\text{MinHash 相同}] = J$；用 $n$ 個雜湊時，相同比例的標準誤為 $\sqrt{J(1-J)/n}$。
- 相同的文件一定碰撞；沒有共同片段的文件一定不碰撞。
- 候選機率對 $J$ 單調，且等於 $1 - (1-J^r)^b$（主幹專案 `test_stage07_minhash_collision_probability_matches_theory`）。
- 在主幹語料上，每一組植入的重複配對都被找到，且沒有移除其他任何故事（主幹專案 `test_stage07_finds_every_planted_duplicate_and_nothing_else`）。

## 已知數值

| 量 | 值 |
|---|---|
| 候選機率，$b = 16$、$r = 4$、$J = 0.3 / 0.5 / 0.7 / 0.8$ | 0.122 / 0.644 / 0.988 / 0.9998 |
| $b = 16$、$r = 4$ 時的陡峭門檻 $(1/b)^{1/r}$ | 0.5 |
| 主幹語料：找到的植入配對（完全／近似） | 67 組中的 67 組（37／30），沒有誤報 |
| 主幹語料：典型近似重複的 $J$；最相似的不同配對（前 400 個） | 0.775；0.245 |

## 常見錯誤

- 用 Python 內建的 `hash` 做 MinHash：它在每個行程中加了鹽，簽章在不同執行之間會不同。
- 從詞片段改成字元片段（或反之）卻沒有重新調整門檻。
- 把重複配對的*兩者*都移除，而不是保留其中一個。
- 把分帶得到的候選直接當成重複，而沒有以精確的 Jaccard 相似度確認。
