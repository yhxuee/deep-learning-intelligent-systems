---
title: 反向模式自動微分
description: 把反向傳播看成依反向拓撲排序套用的連鎖律——以前向成本的小常數倍取得純量的完整梯度。
atlas:
  id: mech-reverse-mode-autodiff
  type: mechanism
  mastery: internalize
  core_units: [3]
  prerequisites: [math-chain-rule]
  related: [method-gradient-check]
  lab: labs/unit-03-autodiff
  references: [rumelhart-1986, griewank-2008, baydin-2018, karpathy-micrograd]
  last_reviewed: 2026-11-02
---

## 一段話說明

把前向計算記錄成一張圖。把輸出的伴隨值設為 $\bar L = 1$，再以反向拓撲排序走訪節點，把 $\bar v_j\,\partial v_j/\partial v_i$ 推進每個父節點的伴隨值中。對純量損失，這一次反向掃描就得到每個偏導數，成本是前向掃描的小常數倍；代價是要儲存中間值。

## 不變量（在實驗室中測試）

1. **詳解範例：** [單元 03 §3.3.3](/zh-tw/core/unit-03/construction/) 的單一神經元網路給出 $\bar w_1 = 0.098678$、$\bar w_2 = 0.197356$、$\bar b = 0.098678$。
2. **扇出要累加：** 對 $y = a\cdot a$，$\bar a = 2a$；對 $y = a + a$，$\bar a = 2$。
3. **與中央差分一致：** 在隨機運算式圖上，相對誤差 $<10^{-6}$（float64）。
4. **線性性：** $\alpha f + \beta g$ 的梯度為 $\alpha\nabla f + \beta\nabla g$。
5. **葉節點的梯度跨圖累加：** 共用一個葉節點的兩張不同圖，各做一次反向傳遞，其貢獻會相加——累加是定義好的行為，所以歸零是呼叫者的責任。（不把中間節點歸零就在*同一張*圖上重跑 `backward()`，會得到錯誤的梯度，而不是加倍的梯度。）

## 參考實作

`labs/unit-03-autodiff/reference.py`（約 90 行，純量 `Value` 類別）。請先在 `engine.py` 寫你自己的版本；同一組測試對兩者都執行。

## 常見錯誤

- 累加到父節點時用 `=` 而不是 `+=`（不變量 2 失敗）。
- 建立拓撲排序時沒有「已走訪」集合（在共用子圖上變成指數時間）。
- `tanh` 的局部導數寫成 $1 - \tanh(x)$ 而不是 $1 - \tanh^2(x)$。
- 最佳化步驟之間忘了把梯度歸零。
