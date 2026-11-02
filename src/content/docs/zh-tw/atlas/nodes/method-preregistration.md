---
title: 預先登記的消融研究
description: 執行前先寫下假說、預測、指標、分析與決策規則；跨種子一次只改變一個因素；嚴格依登記內容分析；重現。
atlas:
  id: method-preregistration
  type: method
  mastery: internalize
  core_units: [12]
  prerequisites: [method-bootstrap-ci]
  related: [method-activation-patching, method-scaling-law-fit]
  references: [nosek-2018, henderson-2018, pineau-2021]
  last_reviewed: 2026-11-02
---

## 一段話說明

彈性的分析——在看到結果後才選擇指標、子集、種子或門檻——幾乎能讓任何假說看起來得到支持。**預先登記（pre-registration）**事先固定這些選擇並加上時間戳記。**消融**在其他一切固定的情況下一次只改變一個因素，跨多個種子進行，並以配對分析比較條件。只有當登記的決策規則這麼說、而且一次**獨立重現**——至少是新的種子，最好是另一份實作——也同意時，一個發現才算數。

## 不變量

- 相同種子、相同程式碼、相同資料 → 完全相同的結果（主幹專案 `test_stage12_same_seed_same_result`）。
- 編輯登記紀錄會被其摘要值偵測到（主幹專案 `test_stage12_preregistration_detects_tampering`）。
- 決策規則被機械式地套用：僅憑區間就給出「需要」「不需要」或「無定論」（主幹專案 `test_stage12_decision_rule`）。

## 已知數值

| 量 | 值 |
|---|---|
| 主幹總結專案中登記的門檻 | 0.02 nats／字元 |
| 主幹總結專案的結果 | 見單元 12，詳解例題 12.6 |

## 常見錯誤

- 登記了假說卻沒登記分析，事後再挑有效的分析。
- 在不同種子上比較條件（非配對），把雜訊讀成效應。
- 只回報多次執行中最好的一次。
