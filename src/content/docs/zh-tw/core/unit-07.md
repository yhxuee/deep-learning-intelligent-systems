---
title: "單元 07 · 模型如何不靠人工標籤學習？"
description: "概似、自迴歸、遮罩、對比式目標、資料品質與遷移。"
sidebar: { label: "07 · 自監督", order: 7 }
unit:
  number: 7
  part: 3
  question: "能從原始資料中建構出哪些有用的學習訊號？"
  anchor: "在小型語料上做下一個詞元預訓練，並附去重與資料污染檢查。"
  spine: "階段 07——在 TinyStories 子集上預訓練主幹模型；去重；驗證沒有任何保留集故事出現在訓練資料中。"
  checkpoint: "分析兩種自監督目標（自迴歸 vs. 遮罩）的歸納假設。"
  status: outline
  atlas: ["mech-cross-entropy", "method-contamination-check"]
---

:::caution[規劃大綱——尚非教科書正文]
本頁記錄本單元的問題、錨定範例、學習成果與活動。它是一份課程大綱，而不是章節：它列出主題，卻沒有教它們。它將依[撰寫標準](/zh-tw/project/writing-standard/)改寫為十部分結構，就像單元 03–05 一樣（見[路線圖](/zh-tw/project/roadmap/)）。歡迎貢獻。
:::

## 問題

下一個詞元預測與遮罩詞元預測使用相同的資料。各預測一項它們容易學會、以及一項難以學會的能力。

:::note[讀下去之前，先預測]
繼續之前，先寫下你的答案以及它所依賴的假設。
:::

## 核心內容

- 自監督（self-supervision）就是*從結構中建構標籤*：自迴歸、遮罩 [Devlin et al. 2019](/zh-tw/references/#devlin-2019)、對比式目標／InfoNCE [van den Oord et al. 2018](/zh-tw/references/#oord-2018)、跨模態配對 [Radford et al. 2021](/zh-tw/references/#radford-2021)。
- 斷詞是模型的一部分：位元組對編碼（BPE）[Sennrich et al. 2016](/zh-tw/references/#sennrich-2016)，以及它對算術與罕見字的影響。
- **資料是一級設計變數：** 去重 [Lee et al. 2022](/zh-tw/references/#lee-2022)、過濾、混合權重；小型模型的資料研究 [Eldan & Li 2023](/zh-tw/references/#eldan-2023)。
- 預訓練 → 遷移：為什麼概似目標能產生可重用的表示；微調改變了什麼。
- 資料污染：評估資料洩漏進預訓練資料 [Sainz et al. 2023](/zh-tw/references/#sainz-2023)。

## 依掌握層次的學習成果

| 層次 | 學習成果 |
|---|---|
| <span class="tier tier-internalize">內化</span> | 自迴歸分解；以 bits/詞元計的交叉熵；為什麼去重很重要 |
| <span class="tier tier-verify">理解與驗證</span> | 對比式目標；斷詞器的取捨；資料混合的效應 |
| <span class="tier tier-access">按需查閱</span> | 特定語料的處理流程與過濾規則 |

## 學習活動

**建構（不使用 AI）。** 實作精確比對與 n-gram 重疊的去重；在植入重複項的測試語料上，驗證對精確副本的召回率為 100%。

**挑戰（評論者模式）。** 請 AI 提出一個資料過濾器；構造一個它會刪掉、但你想保留的例子，以及一個它會保留、但你想刪掉的例子。

**實驗（協作者模式）。** 在固定計算量下，比較有無去重的預訓練；預測它對保留集損失與逐字記憶的影響。

**主幹專案進度。** 階段 07——在 TinyStories 子集上預訓練主幹模型；去重；驗證沒有任何保留集故事出現在訓練資料中。

**遷移與答辯（不使用 AI）。** 為一種新的模態（例如感測器時間序列）設計自監督目標，並說明它的歸納假設。

## 主要文獻

[devlin-2019](/zh-tw/references/#devlin-2019) · [oord-2018](/zh-tw/references/#oord-2018) · [radford-2021](/zh-tw/references/#radford-2021) · [sennrich-2016](/zh-tw/references/#sennrich-2016) · [lee-2022](/zh-tw/references/#lee-2022) · [eldan-2023](/zh-tw/references/#eldan-2023) · [sainz-2023](/zh-tw/references/#sainz-2023)
