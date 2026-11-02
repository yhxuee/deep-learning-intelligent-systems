---
title: 資料污染與洩漏檢查
description: 偵測訓練資料中是否含有評估資料——精確與 n-gram 比對、正規化，以及「檢查乾淨」並不證明什麼。
atlas:
  id: method-contamination-check
  type: method
  mastery: understand-and-verify
  core_units: [7, 10]
  related: [mech-cross-entropy]
  references: [sainz-2023, kapoor-2023, lee-2022]
  last_reviewed: 2026-11-02
---

## 一段話說明

洩漏（leakage）——關於測試集的資訊進入了訓練——是機器學習結果無法重現的主要原因之一 [Kapoor & Narayanan 2023](/zh-tw/references/#kapoor-2023)。對預訓練語言模型而言，主要形式是基準測試**資料污染（contamination）** [Sainz et al. 2023](/zh-tw/references/#sainz-2023)。基本檢查：先正規化文字（大小寫、空白、標點），再在評估項目與訓練語料之間尋找精確比對與長 n-gram 重疊；並對語料本身去重 [Lee et al. 2022](/zh-tw/references/#lee-2022)。

## 不變量

- 植入的測試項目精確副本必須被偵測到（對植入重複項的召回率為 1.0）。
- 兩個語料必須使用相同的正規化。
- 以*實體*（使用者、病人、文件）切分資料時，不同切分不得共用實體，而不只是不共用資料列。

## 「檢查乾淨」並不證明什麼

沒有 n-gram 重疊，並不能排除改寫、翻譯或重新排版後的污染，對訓練資料不公開的模型也無法提供任何保證。請把方法及其盲點與結果一起報告。

## 常見錯誤

- 兩邊使用不同的斷詞或正規化。
- 檢查資料列而不是實體（例如同一位病人同時出現在訓練與測試中）。
- 在被污染的切分上調完超參數之後才做檢查。
