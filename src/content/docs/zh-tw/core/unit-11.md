---
title: "單元 11 · 模型實際上學到了什麼？"
description: "探測、因果介入、迴路、特徵，以及機制性主張的邏輯。"
sidebar: { label: "11 · 詮釋", order: 11 }
unit:
  number: 11
  part: 4
  question: "預測成功何時足以支持「關於機制或理解」的主張？"
  anchor: "以活化修補定位一個行為——在你自己的主幹模型中找出歸納頭。"
  spine: "階段 11——用注意力樣式加上活化修補，在主幹模型中定位歸納頭；以消融確認。"
  checkpoint: "設計一個能區分「同一模型行為的兩種解釋」的實驗。"
  status: outline
  atlas: ["mech-attention", "mech-residual-connection", "arch-transformer-block"]
---

:::caution[規劃大綱——尚非教科書正文]
本頁記錄本單元的問題、錨定範例、學習成果與活動。它是一份課程大綱，而不是章節：它列出主題，卻沒有教它們。它將依[撰寫標準](/zh-tw/project/writing-standard/)改寫為十部分結構，就像單元 03–05 一樣（見[路線圖](/zh-tw/project/roadmap/)）。歡迎貢獻。
:::

## 問題

一個線性探測器能從第 6 層以 95% 準確率還原詞性。第 6 層有「使用」詞性嗎？什麼實驗能說服你？

:::note[讀下去之前，先預測]
繼續之前，先寫下你的答案以及它所依賴的假設。
:::

## 核心內容

- 相關性證據（探測）vs. 因果證據（介入）[Belinkov 2022](/zh-tw/references/#belinkov-2022)。
- 殘差流觀點與迴路 [Elhage et al. 2021](/zh-tw/references/#elhage-2021)；以**歸納頭（induction head）** 為錨定 [Olsson et al. 2022](/zh-tw/references/#olsson-2022)。
- 活化修補／因果追蹤 [Meng et al. 2022](/zh-tw/references/#meng-2022)；小規模的迴路分析 [Wang et al. 2023](/zh-tw/references/#wang-2023)。
- 特徵與疊加；大規模的稀疏字典學習 [Templeton et al. 2024](/zh-tw/references/#templeton-2024)——Atlas 層次。
- 什麼算是解釋：必要性、充分性與忠實性。

## 依掌握層次的學習成果

| 層次 | 學習成果 |
|---|---|
| <span class="tier tier-internalize">內化</span> | 探測 vs. 介入；活化修補；歸納頭的機制 |
| <span class="tier tier-verify">理解與驗證</span> | 稀疏自編碼器；歸因方法及其失效方式 |
| <span class="tier tier-access">按需查閱</span> | 可解釋性函式庫與模型特定的掛鉤 |

## 學習活動

**建構（不使用 AI）。** 為主幹模型的一層實作活化修補；驗證以自己修補自己不改變結果，且修補所有位置會重現來源執行。

**挑戰（評論者模式）。** 請 AI 解釋你模型的某個行為。把它解釋中的每一句話轉成可否證的介入，並執行其中兩個。

**實驗（協作者模式）。** 追蹤歸納頭在訓練中的形成；預測它相對於損失曲線何時形成。

**主幹專案進度。** 階段 11——用注意力樣式加上活化修補，在主幹模型中定位歸納頭；以消融確認。

**遷移與答辯（不使用 AI）。** 對一個新的行為（例如括號配對），提出一個機制，以及能確認或推翻它的最小介入集合。

## 主要文獻

[belinkov-2022](/zh-tw/references/#belinkov-2022) · [elhage-2021](/zh-tw/references/#elhage-2021) · [olsson-2022](/zh-tw/references/#olsson-2022) · [meng-2022](/zh-tw/references/#meng-2022) · [wang-2023](/zh-tw/references/#wang-2023) · [templeton-2024](/zh-tw/references/#templeton-2024)
