---
title: 交叉熵與概似
description: 預設的訓練目標——觀測資料的負對數概似——以及它應該從哪些數字開始。
atlas:
  id: mech-cross-entropy
  type: mechanism
  mastery: internalize
  core_units: [1, 3, 7, 10]
  prerequisites: [math-softmax]
  references: [goodfellow-2016, bengio-2003]
  last_reviewed: 2026-11-02
---

## 一段話說明

對模型分布 $q$ 與觀測結果 $y$，損失 $-\log q(y)$ 是負對數概似（negative log-likelihood）。對資料平均後，它估計交叉熵（cross-entropy）$H(p, q) = -\mathbb{E}_{y\sim p}\log q(y) = H(p) + \mathrm{KL}(p\,\|\,q)$，所以最小化它就是最小化對資料分布的 KL 散度。語言模型以每個詞元為單位報告它，單位為 nats（自然對數）或 bits（除以 $\ln 2$）；困惑度（perplexity）是每詞元平均損失（nats）的 $\exp$。

## 不變量

- $-\log q(y) \ge 0$，等號只在 $q(y) = 1$ 時成立。
- 模型的交叉熵永遠不低於資料的熵：$H(p,q)\ge H(p)$。
- 在 $V$ 個結果上均勻的模型，不論資料為何，得分恰好為 $\ln V$ nats。

## 已知數值

| 詞彙量 $V$ | 均勻損失 $\ln V$（nats） | bits |
|---|---|---|
| 2 | 0.6931 | 1.0000 |
| 65（莎士比亞字元） | 4.1744 | 6.0224 |
| 50 257（GPT-2 BPE） | 10.8249 | 15.6170 |

剛初始化的語言模型應該從 $\ln V$ **附近**開始。高很多表示初始 logits 太大；低很多表示資訊洩漏了（例如因果遮罩壞掉）。

## 常見錯誤

- 把填充詞元也納入平均。
- 在函式預期 logits 的地方傳入機率（重複做兩次 softmax）。
- 跨不同斷詞器比較損失：詞元不同時，nats/詞元無法比較；請改用每位元組或每字元的 bits。
