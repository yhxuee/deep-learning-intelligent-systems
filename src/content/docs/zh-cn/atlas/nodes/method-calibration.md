---
title: 校准与期望校准误差
description: 预测概率是否符合观察频率——可靠性表、期望校准误差，以及温度缩放如何修正它。
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

## 一段话说明

若在预测器说“概率 $p$”的情况中，事件恰好有比例 $p$ 发生，预测器就是校准的。依置信度把预测分箱；**期望校准误差**为 $\mathrm{ECE} = \sum_b\frac{n_b}{n}\big|\mathrm{acc}_b - \mathrm{conf}_b\big|$。对数损失会奖励校准（它是恰当评分），但 ECE 把校准单独分离出来。温度缩放——把 logits 除以在验证数据上拟合的单一 $\tau$——常能在不改变准确率的情况下修正现代网络的失准。

## 不变量

- 置信度等于真实概率的预测器，随 $n$ 增加 ECE → 0（主干项目 `test_stage10_ece_of_calibrated_and_overconfident_predictors`）。
- 把校准预测器的每个置信度都加上常数 $\delta$（在范围内），ECE ≈ $\delta$。
- 温度缩放不改变 argmax，因此不改变准确率。

## 已知数值

| 量 | 值 |
|---|---|
| 校准的合成预测器的 ECE（$n = 200{,}000$） | < 0.01 |
| 同一预测器加上 +0.04 的 ECE | ≈ 0.04 |
| 主干阶段 07 模型，下一个字符 | ECE ≈ 0.01 |

## 常见错误

- 以每箱极少的样本（方差高）或隐藏失准区域的分箱方式回报 ECE。
- 在测试集上做校准。
- 把低 ECE 当成准确的证据：永远输出基础比例的预测器是完美校准的，却毫无用处。
