---
title: "單元 02 · 什麼是表示？"
description: "向量空間、嵌入、歸納偏誤、不變性與等變性——以學習得到的詞元嵌入為錨定。"
sidebar: { label: "02 · 表示", order: 2 }
unit:
  number: 2
  part: 1
  question: "為什麼換一種表示，學習的難度就會改變？"
  anchor: "學習得到的詞元嵌入表，以及它所誘導的幾何結構。"
  spine: "階段 02——以學習的嵌入表（與輸出層綁定）取代 one-hot 輸入；探測最近鄰與線性結構。"
  checkpoint: "以一個構造的例子，說明何時某個表示丟掉了任務所需的資訊。"
  status: outline
  atlas: []
---

:::caution[規劃大綱——尚非教科書正文]
本頁記錄本單元的問題、錨定範例、學習成果與活動。它是一份課程大綱，而不是章節：它列出主題，卻沒有教它們。它將依[撰寫標準](/zh-tw/project/writing-standard/)改寫為十部分結構，就像單元 03–05 一樣（見[路線圖](/zh-tw/project/roadmap/)）。歡迎貢獻。
:::

## 問題

XOR 在 $(x_1,x_2)$ 中不是線性可分的，但在 $(x_1, x_2, x_1x_2)$ 中是。這說明了什麼一般原理？它與神經網路的隱藏層有什麼關係？

:::note[讀下去之前，先預測]
繼續之前，先寫下你的答案以及它所依賴的假設。
:::

## 核心內容

- 把表示看成座標系；線性可分；特徵就是學習得到的座標。
- 分散式表示與嵌入（embedding）[Bengio et al. 2003](/zh-tw/references/#bengio-2003)；[Mikolov et al. 2013](/zh-tw/references/#mikolov-2013)；綁定的輸入／輸出嵌入 [Press & Wolf 2017](/zh-tw/references/#press-2017)。
- 歸納偏誤（inductive bias）就是選擇讓什麼變得容易；**不變性（invariance）** vs. **等變性（equivariance）** [Bronstein et al. 2021](/zh-tw/references/#bronstein-2021)。
- 表示丟掉的資訊，下游無法找回——把資料處理不等式當作設計限制。
- 潛在表示的預告：自編碼器的瓶頸會迫使什麼發生（[單元 08](/zh-tw/core/unit-08/)）。

## 依掌握層次的學習成果

| 層次 | 學習成果 |
|---|---|
| <span class="tier tier-internalize">內化</span> | 線性可分；嵌入即查表加學習的座標；不變性 vs. 等變性 |
| <span class="tier tier-verify">理解與驗證</span> | 嵌入空間中的類比運算及其失效方式；表示相似度量測 |
| <span class="tier tier-access">按需查閱</span> | 特定斷詞器的實作與預訓練嵌入的發布版本 |

## 學習活動

**建構（不使用 AI）。** 實作嵌入查表及其梯度（分散累加）；驗證它等於 one-hot 矩陣乘以嵌入表。

**挑戰（評論者模式）。** AI 宣稱「king − man + woman ≈ queen 證明嵌入編碼了性別」。設計一個對照，檢驗這個結果是否只是最近鄰搜尋排除了輸入字詞所造成的假象。

**實驗（協作者模式）。** 以嵌入大小 4–256 訓練主幹模型；預測報酬遞減的位置；在每個大小下探測最近鄰。

**主幹專案進度。** 階段 02——以學習的嵌入表（與輸出層綁定）取代 one-hot 輸入；探測最近鄰與線性結構。

**遷移與答辯（不使用 AI）。** 給定一個新任務的對稱性（例如分子的旋轉），說明表示應該對它不變還是等變，以及原因。

## 主要文獻

[bengio-2003](/zh-tw/references/#bengio-2003) · [mikolov-2013](/zh-tw/references/#mikolov-2013) · [press-2017](/zh-tw/references/#press-2017) · [bronstein-2021](/zh-tw/references/#bronstein-2021)
