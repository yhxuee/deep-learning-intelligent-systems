---
title: 位置編碼
description: 順序如何進入一個置換等變的模型——正弦、學習式與旋轉式（RoPE）編碼及其不變量。
atlas:
  id: mech-positional-encoding
  type: mechanism
  mastery: understand-and-verify
  core_units: [4]
  prerequisites: [mech-attention]
  related: [arch-transformer-block]
  references: [vaswani-2017, su-2024]
  last_reviewed: 2026-11-02
---

## 一段話說明

沒有位置資訊的自注意力把輸入當成集合。絕對式方法為每個位置**加上**一個向量（正弦式 [Vaswani et al. 2017](/zh-tw/references/#vaswani-2017) 或學習式）。旋轉式嵌入把查詢／鍵座標的每一個二維對**旋轉**一個與位置成正比的角度，使查詢–鍵內積只依賴相對位移 [Su et al. 2024](/zh-tw/references/#su-2024)。

## 不變量

- 正弦式：$PE(p, 2i) = \sin(p/10000^{2i/d})$、$PE(p, 2i+1) = \cos(p/10000^{2i/d})$；每一對的範數為 1。
- RoPE：旋轉保持範數；$\langle R_m q, R_n k\rangle$ 只依賴 $m-n$；$R_0 = I$。
- RoPE 作用在查詢與鍵上，**不**作用在值上。

## 常見錯誤

- 配對慣例不一致（交錯 $(0,1),(2,3)$ vs. 對半 $(0, d/2)$）：載入以另一種慣例訓練的權重時，輸出會變差卻不會當掉。
- 快取解碼時沒有遞增位置索引（見 [KV 快取](/zh-tw/atlas/nodes/mech-kv-cache/)）。
- 使用學習式絕對嵌入卻超出訓練的上下文長度（索引超出範圍，或悄悄地繞回）。
