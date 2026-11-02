---
title: 校準與期望校準誤差
description: 預測機率是否符合觀察頻率——可靠度表、期望校準誤差，以及溫度縮放如何修正它。
atlas:
  id: method-calibration
  type: method
  mastery: internalize
  core_units: [10]
  prerequisites: [mech-cross-entropy]
  related: [method-bootstrap-ci]
  references: [guo-2017, naeini-2015]
  last_reviewed: 2026-11-02
---

## 一段話說明

若在預測器說「機率 $p$」的情況中，事件恰好有比例 $p$ 發生，預測器就是校準的。依信心把預測分箱；**期望校準誤差**為 $\mathrm{ECE} = \sum_b\frac{n_b}{n}\big|\mathrm{acc}_b - \mathrm{conf}_b\big|$。對數損失會獎勵校準（它是恰當評分），但 ECE 把校準單獨分離出來。溫度縮放——把 logits 除以在驗證資料上擬合的單一 $\tau$——常能在不改變準確率的情況下修正現代網路的失準。

## 不變量

- 信心等於真實機率的預測器，隨 $n$ 增加 ECE → 0（主幹專案 `test_stage10_ece_of_calibrated_and_overconfident_predictors`）。
- 把校準預測器的每個信心都加上常數 $\delta$（在範圍內），ECE ≈ $\delta$。
- 溫度縮放不改變 argmax，因此不改變準確率。

## 已知數值

| 量 | 值 |
|---|---|
| 校準的合成預測器的 ECE（$n = 200{,}000$） | < 0.01 |
| 同一預測器加上 +0.04 的 ECE | ≈ 0.04 |
| 主幹階段 07 模型，下一個字元 | ECE ≈ 0.01 |

## 常見錯誤

- 以每箱極少的樣本（變異數高）或隱藏失準區域的分箱方式回報 ECE。
- 在測試集上做校準。
- 把低 ECE 當成準確的證據：永遠輸出基礎比例的預測器是完美校準的，卻毫無用處。
