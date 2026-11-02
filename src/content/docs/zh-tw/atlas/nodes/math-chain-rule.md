---
title: 多變數連鎖律
description: 雅可比矩陣以矩陣乘法合成；沿所有路徑的導數相加。
atlas:
  id: math-chain-rule
  type: mathematics
  mastery: internalize
  core_units: [3]
  related: [mech-reverse-mode-autodiff]
  references: [griewank-2008, baydin-2018]
  last_reviewed: 2026-11-02
---

## 一段話說明

對 $g:\mathbb{R}^n\to\mathbb{R}^m$ 與 $f:\mathbb{R}^m\to\mathbb{R}^k$，合成函數的雅可比矩陣（Jacobian）是雅可比矩陣的乘積：$J_{f\circ g}(x) = J_f(g(x))\,J_g(x)$。當一個變數沿多條路徑影響輸出時，各路徑的貢獻**相加**。反向模式微分就是對純量輸出由右往左計算這條規則。

## 不變量

- **形狀。** $J_f(g(x))$ 為 $k\times m$，$J_g(x)$ 為 $m\times n$；乘積為 $k\times n$。形狀無法合成的推導就是錯的。
- **路徑和。** 若 $y = f(u(x), v(x))$，則 $\frac{dy}{dx} = \frac{\partial f}{\partial u}u'(x) + \frac{\partial f}{\partial v}v'(x)$。
- **順序。** 矩陣型的雅可比矩陣不可交換；除非形狀強制，否則顛倒順序會改變答案。

## 已知數值

- $\frac{d}{dx}\sin(x^2) = 2x\cos(x^2)$；在 $x = 1$ 時為 $1.0806046$。
- $y = x\cdot x$ 經兩條路徑：$\frac{dy}{dx} = x + x = 2x$。

## 常見錯誤

- 變數被重複使用時漏掉一條路徑（這就是自動微分中 `=` 與 `+=` 錯誤的根源）。
- 手推矩陣梯度時雅可比矩陣轉置錯誤——用形狀規則抓出來。
