---
title: 消融與激活修補
description: 檢驗模型元件是否被使用的介入——移除它（消融），或把它的激活換成來自另一個輸入的值（修補）——以及讓結果可以詮釋的對照組。
atlas:
  id: method-activation-patching
  type: method
  mastery: understand-and-verify
  core_units: [11]
  prerequisites: [mech-attention, arch-transformer-block]
  related: [method-preregistration]
  references: [meng-2022, wang-2023, zhang-nanda-2024]
  last_reviewed: 2026-11-02
---

## 一段話說明

觀察性證據——注意力模式、探測、歸因——顯示的是哪些資訊存在或相關。**介入**顯示的是哪些被使用。*消融*把元件的輸出換成零（或其平均），量測行為的變化。*激活修補*在「被破壞」的輸入上執行模型，並把某個元件的激活換成它在「乾淨」輸入上的值；如果乾淨的行為恢復了，該元件就承載了這個差異。兩者都需要對照組：消融不應該有影響的元件、以無關輸入的激活修補，並在機制已知的模型上驗證整個流程。

## 不變量

- 以一次執行自己的激活修補它，什麼都不會改變（主幹專案 `test_stage11_patching_a_run_with_its_own_activations_is_a_no_op`）。
- 把一個頭歸零消融，等於以零修補它（主幹專案 `test_ablating_a_head_equals_patching_it_with_zeros`）。
- 在手工建構的歸納電路上，消融任一電路頭都會破壞複製，消融未使用的頭則不會（主幹專案 `test_stage11_hand_built_circuit_is_found_used_and_needed`）。

## 已知數值

| 量 | 值 |
|---|---|
| 手工電路在重複後半的損失：完整／消融電路頭／消融未使用的頭 | < 0.01 ／ > 1 ／ < 0.01 nats |

## 常見錯誤

- 只憑注意力模式就斷言「這個頭實作了 X」。
- 歸零消融把激活推到遠離分布之處，量到的效應反映的是衝擊而不是機制；應與平均消融比較。
- 用來修補的被破壞輸入與乾淨輸入之間，不只差了感興趣的那個變數。
