---
title: 选择性状态空间模型
description: 依输入而变的（选择性）递推，如 Mamba——选择性带来什么、代价是什么，以及如何检查实现。
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

## 一段话说明

选择性 SSM 以依输入而变的步长 $\Delta_t$，以及依输入而变的 $B_t, C_t$，离散化一个连续时间线性系统：$h_t = \bar A_t h_{t-1} + \bar B_t x_t$、$y_t = C_t h_t$，其中 $\bar A_t = \exp(\Delta_t A)$ [Gu & Dao 2024](/zh-cn/references/#gu-2024)。选择性让模型决定要写入什么、何时遗忘——像注意力一样依内容决定——同时保持固定大小的状态与对序列长度线性的成本。它放弃了非时变的卷积形式，所以训练时使用并行扫描。

## 不变量

- 若 $\Delta, B, C$ 与输入无关，它就化为非时变 SSM（[线性递推](/zh-cn/atlas/nodes/mech-linear-recurrence/)）。
- 对负实数对角 $A$：$\Delta_t\to0$ 得 $\bar A_t\to I$（保留状态）；$\Delta_t\to\infty$ 得 $\bar A_t\to0$（重设）。
- 扫描的输出等于循序循环的输出。

## 已知的限制（证据）

在相同规模下，固定状态的模型在联想回忆与复制任务上落后于注意力 [Arora et al. 2024](/zh-cn/references/#arora-2024)；[Jelassi et al. 2024](/zh-cn/references/#jelassi-2024)，这与[单元 05](/zh-cn/core/unit-05/) 的容量论证一致。混合注意力层与 SSM 层的架构是活跃的研究领域——除非引用了对照比较，否则相关主张应归类为*开放假说*。
