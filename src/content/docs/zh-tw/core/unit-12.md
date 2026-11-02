---
title: "單元 12 · 我們如何創造可驗證的知識？"
description: "可否證的假設、對照、消融、重現、AI 輔助的發現、研究倫理與科學寫作。"
sidebar: { label: "12 · 發現", order: 12 }
unit:
  number: 12
  part: 4
  question: "一個看似合理的想法如何變成可重現的科學發現？"
  anchor: "一個附亂數種子、對照組與預先登記預測的消融研究，以及一次獨立重現。"
  spine: "總結專題——在 AI 協助下重現一個近期的小規模結果，找出一個失效模式，檢驗一個延伸，並以口試答辯。"
  checkpoint: "為一份完整的實驗計畫辯護，並精確說明其結果能確立什麼。"
  status: outline
  atlas: []
---

:::caution[規劃大綱——尚非教科書正文]
本頁記錄本單元的問題、錨定範例、學習成果與活動。它是一份課程大綱，而不是章節：它列出主題，卻沒有教它們。它將依[撰寫標準](/zh-tw/project/writing-standard/)改寫為十部分結構，就像單元 03–05 一樣（見[路線圖](/zh-tw/project/roadmap/)）。歡迎貢獻。
:::

## 問題

一個 AI 代理整夜跑了 400 個實驗，回報一個新的正規化層帶來 2% 的改善。在相信它之前，你需要知道什麼？

:::note[讀下去之前，先預測]
繼續之前，先寫下你的答案以及它所依賴的假設。
:::

## 核心內容

- 會失敗的假設；對照組與陰性對照；只孤立一個原因的消融。
- 亂數種子間的變異，以及為什麼單次執行會誤導 [Henderson et al. 2018](/zh-tw/references/#henderson-2018)；可重現性實務 [Pineau et al. 2021](/zh-tw/references/#pineau-2021)；學術寫作的失誤模式 [Lipton & Steinhardt 2019](/zh-tw/references/#lipton-2019)。
- **AI 輔助的發現：**「人建立評估器、AI 負責搜尋」的模式 [Romera-Paredes et al. 2024](/zh-tw/references/#romera-paredes-2024)；[Novikov et al. 2025](/zh-tw/references/#novikov-2025)；全自動研究流程的極限 [Lu et al. 2024](/zh-tw/references/#lu-2024)。
- 當 AI 讓實驗變便宜時的多重比較：以機器速度走進「分岔小徑花園」。
- 研究倫理、AI 協助的揭露，以及以證據所支持的強度寫出主張。

## 依掌握層次的學習成果

| 層次 | 學習成果 |
|---|---|
| <span class="tier tier-internalize">內化</span> | 對照；消融；亂數種子變異；結果能確立與不能確立什麼 |
| <span class="tier tier-verify">理解與驗證</span> | 預先登記；統計校正；自動化實驗流程 |
| <span class="tier tier-access">按需查閱</span> | 各研討會的可重現性檢核表 |

## 學習活動

**建構（不使用 AI）。** 在寫任何模型程式碼之前，先為總結專題寫好評估器：輸入、指標、對照，以及會否證你假設的結果。

**挑戰（評論者模式）。** 請 AI 評論者從干擾因素與分岔路徑攻擊你的計畫；以書面回應每一點。

**實驗（協作者模式）。** 執行總結專題：重現 → 挑戰 → 延伸，記錄每次執行並報告亂數種子。

**主幹專案進度。** 總結專題——在 AI 協助下重現一個近期的小規模結果，找出一個失效模式，檢驗一個延伸，並以口試答辯。

**遷移與答辯（不使用 AI）。** 口試答辯：主考官對你的主要結果提出另一種解釋；當場設計能區分兩者的實驗。

## 主要文獻

[henderson-2018](/zh-tw/references/#henderson-2018) · [pineau-2021](/zh-tw/references/#pineau-2021) · [lipton-2019](/zh-tw/references/#lipton-2019) · [romera-paredes-2024](/zh-tw/references/#romera-paredes-2024) · [novikov-2025](/zh-tw/references/#novikov-2025) · [lu-2024](/zh-tw/references/#lu-2024)
