---
title: 選擇性狀態空間模型
description: 依輸入而變的（選擇性）遞迴，如 Mamba——選擇性帶來什麼、代價是什麼，以及如何檢查實作。
atlas:
  id: arch-selective-ssm
  type: architecture
  mastery: understand-and-verify
  core_units: [5]
  prerequisites: [mech-linear-recurrence]
  predecessors: [mech-linear-recurrence]
  alternatives: [mech-attention, arch-transformer-block]
  references: [gu-2022, gu-2024, arora-2024, jelassi-2024]
  last_reviewed: 2026-11-02
---

## 一段話說明

選擇性 SSM 以依輸入而變的步長 $\Delta_t$，以及依輸入而變的 $B_t, C_t$，離散化一個連續時間線性系統：$h_t = \bar A_t h_{t-1} + \bar B_t x_t$、$y_t = C_t h_t$，其中 $\bar A_t = \exp(\Delta_t A)$ [Gu & Dao 2024](/zh-tw/references/#gu-2024)。選擇性讓模型決定要寫入什麼、何時遺忘——像注意力一樣依內容決定——同時保持固定大小的狀態與對序列長度線性的成本。它放棄了非時變的卷積形式，所以訓練時使用平行掃描。

## 不變量

- 若 $\Delta, B, C$ 與輸入無關，它就化為非時變 SSM（[線性遞迴](/zh-tw/atlas/nodes/mech-linear-recurrence/)）。
- 對負實數對角 $A$：$\Delta_t\to0$ 得 $\bar A_t\to I$（保留狀態）；$\Delta_t\to\infty$ 得 $\bar A_t\to0$（重設）。
- 掃描的輸出等於循序迴圈的輸出。

## 已知的限制（證據）

在相同規模下，固定狀態的模型在聯想回憶與複製任務上落後於注意力 [Arora et al. 2024](/zh-tw/references/#arora-2024)；[Jelassi et al. 2024](/zh-tw/references/#jelassi-2024)，這與[單元 05](/zh-tw/core/unit-05/) 的容量論證一致。混合注意力層與 SSM 層的架構是活躍的研究領域——除非引用了對照比較，否則相關主張應歸類為*開放假說*。
