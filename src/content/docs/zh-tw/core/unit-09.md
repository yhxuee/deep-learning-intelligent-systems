---
title: "單元 09 · 目標與回饋如何塑造行為？"
description: "馬可夫決策過程、Bellman 方程式、策略梯度、RLHF、DPO、可驗證獎勵強化學習、測試時計算與 AI 代理。"
sidebar: { label: "09 · 決策", order: 9 }
unit:
  number: 9
  part: 3
  question: "系統如何利用目標與回饋選擇行動——後訓練又如何改變語言模型？"
  anchor: "策略梯度估計量，以及套用在小型語言模型上的 DPO／可驗證獎勵強化學習。"
  spine: "階段 09——在答案可由程式檢查的任務（例如小型算術）上對主幹模型做後訓練；量測增益與獎勵駭入。"
  checkpoint: "找出一個會鼓勵不良行為的獎勵設定，並提出可量測的修正。"
  status: outline
  atlas: []
  answer: "當目標是獎勵而不是標準答案時，策略梯度估計量把試誤變成梯度步驟；RLHF、DPO 與可驗證獎勵強化學習把它應用到語言模型上，並以對預訓練模型的 KL 錨定限制偏移。"
  leaves: "最佳化一個被量測的獎勵，會暴露量測與意圖之間的每一個落差——獎勵駭客（reward hacking）——所以必須追問：分數什麼時候才算是可靠性的證據？"
  builds_on:
    - { unit: 3, uses: "隨機梯度及其變異數" }
    - { unit: 7, uses: "以預訓練模型作為初始策略" }
    - { unit: 8, uses: "抽樣與 KL 散度" }
---

:::caution[規劃大綱——尚非教科書正文]
本頁記錄本單元的問題、錨定範例、學習成果與活動。它是一份課程大綱，而不是章節：它列出主題，卻沒有教它們。它將依[撰寫標準](/zh-tw/project/writing-standard/)改寫為十部分結構，就像單元 03–05 一樣（見[路線圖](/zh-tw/project/roadmap/)）。歡迎貢獻。
:::

## 問題

一個模型因通過單元測試而得到獎勵。預測兩種它能提高獎勵、卻沒有變成更好程式設計師的方式。

:::note[讀下去之前，先預測]
繼續之前，先寫下你的答案以及它所依賴的假設。
:::

## 核心內容

- 馬可夫決策過程與 Bellman 關係 [Sutton & Barto 2018](/zh-tw/references/#sutton-2018)。
- **策略梯度（policy gradient）**（REINFORCE）估計量 [Williams 1992](/zh-tw/references/#williams-1992)、基準線與變異數；PPO [Schulman et al. 2017](/zh-tw/references/#schulman-2017) 為 Atlas 層次。
- **語言模型的後訓練：** RLHF [Ouyang et al. 2022](/zh-tw/references/#ouyang-2022)、DPO [Rafailov et al. 2023](/zh-tw/references/#rafailov-2023)、可驗證獎勵的強化學習與群組相對基準線 [Shao et al. 2024](/zh-tw/references/#shao-2024)；[DeepSeek-AI 2025](/zh-tw/references/#deepseek-2025)。
- 測試時計算：多次嘗試取樣、搜尋與驗證。
- **AI 代理：** 在迴圈中搭配工具與環境的語言模型；為什麼 AI 代理的評估是一個序列決策問題。
- 規格博弈與獎勵駭入，是對齊問題的縮影 [Amodei et al. 2016](/zh-tw/references/#amodei-2016)。

## 依掌握層次的學習成果

| 層次 | 學習成果 |
|---|---|
| <span class="tier tier-internalize">內化</span> | MDP；Bellman 方程式；策略梯度估計量；DPO 目標函數 |
| <span class="tier tier-verify">理解與驗證</span> | PPO 裁剪；對參考模型的 KL 正則化；獎勵模型的失效方式；AI 代理評估 |
| <span class="tier tier-access">按需查閱</span> | 特定的強化學習函式庫與 AI 代理框架 |

## 學習活動

**建構（不使用 AI）。** 在吃角子老虎問題上推導並實作帶基準線的 REINFORCE；以蒙地卡羅方法對照解析梯度，驗證估計量是不偏的。

**挑戰（評論者模式）。** 給 AI 一個你選的任務的獎勵函數，請它找漏洞；修補獎勵後再重複。報告哪些修補可以推廣。

**實驗（協作者模式）。** 以可驗證獎勵對主幹模型做後訓練；預測增益與相對基礎模型的 KL 漂移；檢查樣本中的獎勵駭入。

**主幹專案進度。** 階段 09——在答案可由程式檢查的任務（例如小型算術）上對主幹模型做後訓練；量測增益與獎勵駭入。

**遷移與答辯（不使用 AI）。** 對一個新的 AI 代理任務，說明獎勵、評估流程，以及一項防範規格博弈的措施。

## 主要文獻

[sutton-2018](/zh-tw/references/#sutton-2018) · [williams-1992](/zh-tw/references/#williams-1992) · [schulman-2017](/zh-tw/references/#schulman-2017) · [ouyang-2022](/zh-tw/references/#ouyang-2022) · [rafailov-2023](/zh-tw/references/#rafailov-2023) · [shao-2024](/zh-tw/references/#shao-2024) · [deepseek-2025](/zh-tw/references/#deepseek-2025) · [amodei-2016](/zh-tw/references/#amodei-2016)
