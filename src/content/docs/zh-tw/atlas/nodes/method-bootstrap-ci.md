---
title: 自助法信賴區間與配對比較
description: 重新抽樣獨立的單位，以估計任何統計量的不確定性；在相同項目上配對兩個模型，讓它們共有的難度抵銷。
atlas:
  id: method-bootstrap-ci
  type: method
  mastery: internalize
  core_units: [1, 10, 12]
  prerequisites: []
  related: [method-calibration, method-preregistration]
  references: [efron-1993, dietterich-1998]
  last_reviewed: 2026-11-02
---

## 一段話說明

要估計由 $n$ 個獨立單位計算出的統計量 $\hat\theta$ 的不確定性，就多次以放回方式抽出 $n$ 個單位、每次重算 $\hat\theta$，再取 2.5% 與 97.5% 分位數：**百分位自助法（percentile bootstrap）**區間。重新抽樣的必須是*獨立*的單位（故事、提示、使用者），絕不是它們內部彼此相依的片段（字元）。要比較兩個模型，就對每個項目的*差*做自助法：兩個模型共有的項目難度變化會抵銷，常讓區間縮小一個數量級。

## 不變量

- 在模擬中，95% 區間約有 95% 的時候涵蓋真實平均（主幹專案 `test_stage10_bootstrap_intervals_have_nominal_coverage`）。
- 對正相關的每項目分數，配對區間比兩個非配對區間合起來窄（主幹專案 `test_stage10_pairing_narrows_the_interval_for_correlated_models`）。
- 只有一個單位時，每次重抽都相同：區間寬度為零，沒有意義。

## 已知數值

| 量 | 值 |
|---|---|
| 主幹保留資料 NLL，階段 07，140 個故事 | 0.230 [0.222, 0.240] |
| 配對差 階段 04 − 階段 07 | +0.106 [+0.101, +0.110] |

## 常見錯誤

- 當抽樣單位是故事或文件時，卻重抽字元或詞元。
- 把對測試項目做的自助法區間，當成涵蓋了不同種子訓練之間的變異（它沒有；種子也要重抽）。
- 比較兩個模型的非配對區間，並把「重疊」說成「沒有差異」。
