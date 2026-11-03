---
title: 数据污染与泄漏检查
description: 检测训练数据中是否含有评估数据——精确与 n-gram 比对、归一化，以及“检查干净”并不证明什么。
atlas:
  id: method-contamination-check
  type: method
  mastery: understand-and-verify
  core_units: [7, 10]
  related: [mech-cross-entropy]
  references: [sainz-2023, kapoor-2023, lee-2022]
  last_reviewed: 2026-11-02
---

## 一段话说明

泄漏（leakage）——关于测试集的信息进入了训练——是机器学习结果无法复现的主要原因之一 [Kapoor & Narayanan 2023](/zh-cn/references/#kapoor-2023)。对预训练语言模型而言，主要形式是基准测试**数据污染（contamination）** [Sainz et al. 2023](/zh-cn/references/#sainz-2023)。基本检查：先归一化文本（大小写、空白、标点），再在评估项目与训练语料之间寻找精确比对与长 n-gram 重叠；并对语料本身去重 [Lee et al. 2022](/zh-cn/references/#lee-2022)。

## 不变量

- 植入的测试项目精确副本必须被检测到（对植入重复项的召回率为 1.0）。
- 两个语料必须使用相同的归一化。
- 以*实体*（用户、病人、文档）切分数据时，不同切分不得共用实体，而不只是不共用数据行。

## “检查干净”并不证明什么

没有 n-gram 重叠，并不能排除改写、翻译或重新排版后的污染，对训练数据不公开的模型也无法提供任何保证。请把方法及其盲点与结果一起报告。

## 常见错误

- 两边使用不同的断词或归一化。
- 检查数据行而不是实体（例如同一位病人同时出现在训练与测试中）。
- 在被污染的切分上调完超参数之后才做检查。
