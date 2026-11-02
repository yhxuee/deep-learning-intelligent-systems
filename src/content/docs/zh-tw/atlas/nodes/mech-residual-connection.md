---
title: 殘差連接
description: x + F(x)——讓訊號與梯度都有一條恆等路徑，使深層堆疊可以訓練；以及 Transformer 的殘差流觀點。
atlas:
  id: mech-residual-connection
  type: mechanism
  mastery: internalize
  core_units: [3, 4, 11]
  prerequisites: [math-chain-rule]
  related: [arch-transformer-block]
  references: [he-2016, radford-2019, elhage-2021]
  last_reviewed: 2026-11-02
---

## 一段話說明

殘差區塊計算 $x + F(x)$。它的雅可比矩陣為 $I + \partial F/\partial x$，所以即使 $\partial F$ 很小，梯度仍有一條回到前面各層的恆等路徑 [He et al. 2016](/zh-tw/references/#he-2016)。在 Transformer 中，每個子層都從同一個逐位置向量——*殘差流（residual stream）*——讀取並加回去，因此各子層的輸出可以當作可加的貢獻來分析 [Elhage et al. 2021](/zh-tw/references/#elhage-2021)。

## 不變量

- 若 $F \equiv 0$，區塊恰好是恆等映射。
- 在前置正規化的堆疊中，殘差流是各子層輸出的**總和**，所以若不縮放輸出投影，它的變異數會隨深度增長；GPT-2 在初始化時把殘差層的權重乘上 $1/\sqrt{N}$，$N$ 為殘差層數 [Radford et al. 2019](/zh-tw/references/#radford-2019)。

## 常見錯誤

- 參考實作是前置正規化（pre-LN），卻正規化了殘差流本身（post-LN），或反之——兩者都能訓練，但在深度上行為不同。
- 把 dropout 套用在跳接路徑上，而不是 $F(x)$ 上。
- $x$ 與 $F(x)$ 的形狀不符，卻被廣播（broadcasting）掩蓋。
