---
title: 预注册的消融研究
description: 执行前先写下假说、预测、指标、分析与决策规则；跨种子一次只改变一个因素；严格依注册内容分析；复现。
atlas:
  id: method-preregistration
  type: method
  mastery: internalize
  core_units: [12]
  prerequisites: [method-bootstrap-ci]
  related: [method-activation-patching, method-scaling-law-fit]
  references: [nosek-2018, henderson-2018, pineau-2021]
  last_reviewed: 2026-11-02
---

## 一段话说明

弹性的分析——在看到结果后才选择指标、子集、种子或阈值——几乎能让任何假说看起来得到支持。**预注册（pre-registration）**事先固定这些选择并加上时间戳记。**消融**在其他一切固定的情况下一次只改变一个因素，跨多个种子进行，并以配对分析比较条件。只有当注册的决策规则这么说、而且一次**独立复现**——至少是新的种子，最好是另一份实现——也同意时，一个发现才算数。

## 不变量

- 相同种子、相同代码、相同数据 → 完全相同的结果（主干项目 `test_stage12_same_seed_same_result`）。
- 编辑注册记录会被其摘要值检测到（主干项目 `test_stage12_preregistration_detects_tampering`）。
- 决策规则被机械式地应用：仅凭区间就给出“需要”“不需要”或“无定论”（主干项目 `test_stage12_decision_rule`）。

## 已知数值

| 量 | 值 |
|---|---|
| 主干综合项目中注册的阈值 | 0.02 nats／字符 |
| 主干综合项目的结果 | 见单元 12，例题 12.6 |

## 常见错误

- 注册了假说却没注册分析，事后再挑有效的分析。
- 在不同种子上比较条件（非配对），把噪声读成效应。
- 只回报多次执行中最好的一次。
