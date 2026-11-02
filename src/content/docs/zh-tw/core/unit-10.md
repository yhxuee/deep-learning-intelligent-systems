---
title: "單元 10 · 模型如何失效？"
description: "效度、分布偏移、校準、穩健性、資料洩漏、統計推論與壓力測試。"
sidebar: { label: "10 · 失效", order: 10 }
unit:
  number: 10
  part: 4
  question: "為什麼基準測試分數不足以證明可靠性？"
  anchor: "在一個「好得不像真的」模型上量測校準並診斷資料洩漏。"
  spine: "階段 10——為主幹模型建立評估框架：保留集切分、校準、資料污染檢查與自助法信賴區間。"
  checkpoint: "診斷為什麼一個接近完美的驗證分數可能是假的。"
  status: outline
  atlas: ["method-contamination-check", "mech-cross-entropy"]
  answer: "分數只有在評估與部署情境相符時才是證據：分布偏移、洩漏與污染破壞了單元 01 的 i.i.d. 論證；校準、信賴區間，以及偏移或對抗性的測試集，量測了單一數字所隱藏的東西。"
  leaves: "行為測試能顯示模型「會」失敗，卻不能說明「為什麼」；要預見沒有人測過的失敗，就必須知道是哪個機制產生了這個行為。"
  builds_on:
    - { unit: 1, uses: "i.i.d. 論證與保留資料評估" }
    - { unit: 7, uses: "污染" }
    - { unit: 9, uses: "獎勵駭客" }
---

:::caution[規劃大綱——尚非教科書正文]
本頁記錄本單元的問題、錨定範例、學習成果與活動。它是一份課程大綱，而不是章節：它列出主題，卻沒有教它們。它將依[撰寫標準](/zh-tw/project/writing-standard/)改寫為十部分結構，就像單元 03–05 一樣（見[路線圖](/zh-tw/project/roadmap/)）。歡迎貢獻。
:::

## 問題

一個分類器在驗證集上得到 99.2%，在同一來源新收集的資料上只有 71%。列出四個假設，以及能區分它們的最便宜檢驗。

:::note[讀下去之前，先預測]
繼續之前，先寫下你的答案以及它所依賴的假設。
:::

## 核心內容

- 內部效度 vs. 外部效度；基準測試*量測*的是什麼 vs. 它*被用來宣稱*什麼。
- 分布偏移與重現：重新收集的測試集 [Recht et al. 2019](/zh-tw/references/#recht-2019)；對常見干擾的穩健性 [Hendrycks & Dietterich 2019](/zh-tw/references/#hendrycks-2019)。
- **校準（calibration）** [Guo et al. 2017](/zh-tw/references/#guo-2017)：有意義的信心。
- **資料洩漏**是不可重現結果的反覆來源 [Kapoor & Narayanan 2023](/zh-tw/references/#kapoor-2023)；預訓練模型的基準測試污染 [Sainz et al. 2023](/zh-tw/references/#sainz-2023)。
- 評估的統計：信賴區間、多重比較、亂數種子造成的變異。
- **實務技巧：** 分切片做錯誤分析；先讀失敗案例，再看總分。

## 依掌握層次的學習成果

| 層次 | 學習成果 |
|---|---|
| <span class="tier tier-internalize">內化</span> | 效度；資料洩漏；校準；準確率的信賴區間 |
| <span class="tier tier-verify">理解與驗證</span> | 穩健性基準；以 LLM 當評審的陷阱；資料污染偵測方法 |
| <span class="tier tier-access">按需查閱</span> | 目前的排行榜與基準測試版本 |

## 學習活動

**建構（不使用 AI）。** 實作期望校準誤差與自助法信賴區間；在已知校準程度的合成預測上驗證。

**挑戰（評論者模式）。** 給 AI 一份藏有洩漏的資料集描述（例如病人 ID 跨切分共用）。記錄它是否在沒有提示的情況下發現洩漏。

**實驗（協作者模式）。** 在三種受控的偏移下壓力測試主幹模型；先預測表現下降的排序。

**主幹專案進度。** 階段 10——為主幹模型建立評估框架：保留集切分、校準、資料污染檢查與自助法信賴區間。

**遷移與答辯（不使用 AI）。** 稽核一個已發表的結果（由教師指定）的效度威脅，並寫下它的證據能確立什麼、不能確立什麼。

## 主要文獻

[recht-2019](/zh-tw/references/#recht-2019) · [hendrycks-2019](/zh-tw/references/#hendrycks-2019) · [guo-2017](/zh-tw/references/#guo-2017) · [kapoor-2023](/zh-tw/references/#kapoor-2023) · [sainz-2023](/zh-tw/references/#sainz-2023)
