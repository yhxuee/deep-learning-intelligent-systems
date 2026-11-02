---
title: 嵌入表與綁定讀出
description: 每個符號一個學習出的向量；查找是一個獨熱矩陣乘積，其梯度是散射加總；綁定用同一張表來讀取與打分。
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

## 一段話說明

嵌入表 $E\in\mathbb{R}^{V\times d}$ 為 $V$ 個符號各指定一個學習出的向量。查找符號 $a$ 等於以其獨熱向量做矩陣乘積 $E^{\top}\mathbf u_a$，所以對一批資料 $H = UE$，梯度為 $\partial L/\partial E = U^{\top}G$：符號的每次出現都把它的上游梯度**加到**該符號的列上。**綁定（tied）**的模型以同一列 $e_b$ 為輸出 $b$ 打分，讓嵌入參數減半。相似的符號收到相似的梯度並彼此靠攏，這就是嵌入共享統計強度的方式（單元 02，§2.2.4）。

## 不變量

- 查找梯度等於獨熱乘積的梯度：`np.add.at(dE, x, G)` $= U^{\top}G$（主幹專案 `test_stage02_lookup_gradient_equals_one_hot_matmul_gradient`）。
- 在一批資料中未出現的符號，其列的梯度恰好為零（權重衰減之前）。
- 讀出為零（或分解式二元語法中 $M$ 為零）時，初始損失恰好是 $\ln V$。
- 在每個上下文中都可互換的符號，會收斂到相同的嵌入（差模型的對稱性）（主幹專案 `test_stage02_interchangeable_symbols_get_the_same_embedding`）。

## 已知數值

| 量 | 值 |
|---|---|
| 查找梯度，$V = 3$、$x = (0, 2, 0)$、$G = [[1,2],[3,4],[5,6]]$ | 各列 $(6, 8)$、$(0, 0)$、$(3, 4)$ |
| 分解式二元語法參數量，$V = 50$、$d = 16$ | $Vd + d^2 + V = 1{,}106$ |
| 神經 4 元語法參數量，$d = 16$、$H = 128$ | $15{,}570$ |
| 主幹模型中數字的最近鄰餘弦（$d = 16$） | 約 $0.97$–$0.98$ |

## 常見錯誤

- 用 `dE[x] = G` 取代散射加總：重複的符號會互相覆蓋，而不是累加。
- 以過大的尺度初始化綁定表，使初始 logits 遠離均勻，損失一開始就遠高於 $\ln V$。
- 對罕見符號與常見符號的嵌入施加相同的權重衰減，卻沒注意到罕見的列在兩次更新之間會縮向零。
- 逐座標比較兩次訓練的嵌入：它們只在差一個可逆變換的意義下有定義（單元 02，習題 2.11）。
