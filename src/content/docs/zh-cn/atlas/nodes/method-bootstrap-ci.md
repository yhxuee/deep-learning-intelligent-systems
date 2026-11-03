---
title: 自助法置信区间与配对比较
description: 重新采样独立的单位，以估计任何统计量的不确定性；在相同项目上配对两个模型，让它们共有的难度抵销。
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

## 一段话说明

要估计由 $n$ 个独立单位计算出的统计量 $\hat\theta$ 的不确定性，就多次以放回方式抽出 $n$ 个单位、每次重算 $\hat\theta$，再取 2.5% 与 97.5% 分位数：**百分位自助法（percentile bootstrap）**区间。重新采样的必须是*独立*的单位（故事、提示、用户），绝不是它们内部彼此依赖的片段（字符）。要比较两个模型，就对每个项目的*差*做自助法：两个模型共有的项目难度变化会抵销，常让区间缩小一个数量级。

## 不变量

- 在模拟中，95% 区间约有 95% 的时候涵盖真实平均（主干项目 `test_stage10_bootstrap_intervals_have_nominal_coverage`）。
- 对正相关的每项目分数，配对区间比两个非配对区间合起来窄（主干项目 `test_stage10_pairing_narrows_the_interval_for_correlated_models`）。
- 只有一个单位时，每次重采样都相同：区间宽度为零，没有意义。

## 已知数值

| 量 | 值 |
|---|---|
| 主干保留数据 NLL，阶段 07，140 个故事 | 0.230 [0.222, 0.240] |
| 配对差 阶段 04 − 阶段 07 | +0.106 [+0.101, +0.110] |

## 常见错误

- 当采样单位是故事或文档时，却重采样字符或词元。
- 把对测试项目做的自助法区间，当成涵盖了不同种子训练之间的差异（它没有；种子也要重采样）。
- 比较两个模型的非配对区间，并把“重叠”说成“没有差异”。
