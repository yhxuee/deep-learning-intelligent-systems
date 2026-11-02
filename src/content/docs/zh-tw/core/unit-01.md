---
title: "單元 01 · 什麼是學習？"
description: "經驗風險、分布假設、泛化、基準與不確定性——以二元語法語言模型為基礎。"
sidebar: { label: "01 · 學習", order: 1 }
unit:
  number: 1
  part: 1
  question: "為什麼經驗能改善在未見過案例上的表現？"
  anchor: "以計數建立的二元語法（bigram）語言模型，以保留集負對數概似評估。"
  spine: "階段 01——在主幹語料上建立計數式二元語法；報告訓練與保留集的負對數概似，並與均勻基準 ln V 比較。"
  checkpoint: "構造一個「訓練準確率高卻不代表能泛化」的反例。"
  status: outline
  atlas: ["mech-cross-entropy"]
---

:::caution[規劃大綱——尚非教科書正文]
本頁記錄本單元的問題、錨定範例、學習成果與活動。它是一份課程大綱，而不是章節：它列出主題，卻沒有教它們。它將依[撰寫標準](/zh-tw/project/writing-standard/)改寫為十部分結構，就像單元 03–05 一樣（見[路線圖](/zh-tw/project/roadmap/)）。歡迎貢獻。
:::

## 問題

一個模型在 50,000 張**標籤被隨機打亂**的影像上達到 100% 的訓練準確率。它的測試準確率會是多少？這對「用訓練準確率——或參數數量——預測泛化」意味著什麼？

:::note[讀下去之前，先預測]
繼續之前，先寫下你的答案以及它所依賴的假設。
:::

## 核心內容

- 把學習看成估計：資料取自某個分布；**經驗風險（empirical risk）** 與**母體風險**；為什麼獨立同分布（i.i.d.）假設撐起整個論證。
- 以概似（likelihood）與交叉熵（cross-entropy）作為預設目標；**均勻基準** $\ln V$ 是每個模型都必須勝過的第一個數字。
- 訓練／驗證／測試的紀律；每個切分的*用途*；資料洩漏是違反假設，而不是文書錯誤。
- 古典的容量論證，以及它為什麼無法解釋過度參數化的網路：記住隨機標籤 [Zhang et al. 2017](/zh-tw/references/#zhang-2017)、雙重下降（double descent）[Belkin et al. 2019](/zh-tw/references/#belkin-2019)；[Nakkiran et al. 2020](/zh-tw/references/#nakkiran-2020)。
- 測試表現估計值的不確定性：要區分兩個模型需要多少保留樣本。

## 依掌握層次的學習成果

| 層次 | 學習成果 |
|---|---|
| <span class="tier tier-internalize">內化</span> | 經驗風險 vs. 母體風險；概似；保留集評估；均勻基準 |
| <span class="tier tier-verify">理解與驗證</span> | 雙重下降；隱式正則化的主張；泛化界 |
| <span class="tier tier-access">按需查閱</span> | 特定基準測試的流程與資料集授權 |

## 學習活動

**建構（不使用 AI）。** 實作加 α 平滑的計數式二元語法；以每詞元 nats 計算訓練與保留集的負對數概似；驗證 α→∞ 時恰好回到均勻模型（$\ln V$）。

**挑戰（評論者模式）。** 問 AI 為什麼大型網路能泛化。把它的每個主張分類為定理、實驗結果或推測，並找出一個講得比證據更強的主張。

**實驗（協作者模式）。** 讓訓練集大小變化兩個數量級；在執行前預測訓練／保留集落差的曲線形狀；擬合並報告不確定性。

**主幹專案進度。** 階段 01——在主幹語料上建立計數式二元語法；報告訓練與保留集的負對數概似，並與均勻基準 ln V 比較。

**遷移與答辯（不使用 AI）。** 給定一個新資料集的描述，指出最可能被違反的分布假設，並設計一個能偵測它的資料切分。

## 主要文獻

[zhang-2017](/zh-tw/references/#zhang-2017) · [belkin-2019](/zh-tw/references/#belkin-2019) · [nakkiran-2020](/zh-tw/references/#nakkiran-2020) · [bengio-2003](/zh-tw/references/#bengio-2003) · [goodfellow-2016](/zh-tw/references/#goodfellow-2016)
