---
title: 实验室
description: 笔记本电脑规模、附不变量测试的实验——Atlas 验证工具包中可执行的那一半。
sidebar: { label: 实验室总览, order: 1 }
---

实验把数学概念转成预测与度量。你不需要一次就写出正确实现。每个实验都从一个**不使用 AI 的构建**文件开始：里面有函数签名，但没有解答；另有一组取自 Atlas 不变量的测试。同一组测试在 CI 中也会对参考实现执行，因此刚开始的失败是在告诉你还需要构建什么，不是在怀疑测试能否成立。

目标不只是让进度行变成绿色，而是把四件事连起来：数学主张、事前预测、程序行为，以及能把预期机制与合理错误区分开来的测试。

| 实验 | 单元 | 你要实现（不使用 AI） | 测试检查 |
|---|---|---|---|
| [`labs/unit-03-autodiff`](https://github.com/yhxuee/deep-learning-intelligent-systems/tree/main/labs/unit-03-autodiff) | [03](/zh-cn/core/unit-03/) | 标量反向模式自动微分 `Value` | 例题、扇出累加、随机图上的有限差分、线性性 |
| [`labs/unit-04-attention`](https://github.com/yhxuee/deep-learning-intelligent-systems/tree/main/labs/unit-04-attention) | [04](/zh-cn/core/unit-04/) | 以 NumPy 实现因果多头自注意力 | 行总和、因果性、置换等变性、KV 缓存等价性、√d 缩放、GPT-2 参数量；以及五个植入错误的**调试挑战** |
| [`labs/unit-05-memory`](https://github.com/yhxuee/deep-learning-intelligent-systems/tree/main/labs/unit-05-memory) | [05](/zh-cn/core/unit-05/) | 线性递推（循环 + 封闭解）、线性注意力记忆 | 封闭解、谱衰减、非正规瞬态增长、$d_k$ 以内精确回忆 vs. 超过后的干扰 |
| [`spine/`](https://github.com/yhxuee/deep-learning-intelligent-systems/tree/main/spine) | 01–12 | 主干项目的语言模型 | 各阶段的验证目标 |
| [`labs/test_worked_examples.py`](https://github.com/yhxuee/deep-learning-intelligent-systems/blob/main/labs/test_worked_examples.py) | 03–05 | — | 重新计算各单元例题中引用的每个数字，让正文与计算不会脱节 |

另外，[交互式图表](/zh-cn/labs/interactive/)页让你在写程序之前，先在浏览器中探索本书的关键结果。

## 执行实验

需求：Python ≥ 3.10、NumPy、pytest。

```bash
pip install -r labs/requirements.txt
cd labs/unit-04-attention
python -m pytest                       # 测试你的实现（attention.py）
LAB_IMPL=reference python -m pytest    # 参考实现通过同一组测试
python -m pytest test_bug_hunt.py -v   # 哪个不变量抓到哪个植入的错误？
```

刚取得的代码中，`engine.py`／`attention.py`／`memory.py` 只有函数签名与 `NotImplementedError`，所以在你实现之前测试会失败。这是刻意的设计。

## 第一次做实验：逐步流程

如果你以前没有写过数值程序，先做单元 03 的自动微分。

1. 阅读对应的例题，手算预期输出。
2. 打开起始档，找出最小的缺失运算；先不要阅读完整参考答案。
3. 执行一个具名测试，例如 `python -m pytest -k add -v`，由左到右读断言：输入、观测值、预期性质。
4. 只实现那个运算。重跑同一个测试，再跑会用到扇出或随机图的相邻测试。
5. 测试失败时，保留最小失败案例；在询问 AI 或修改代码前，先用文本说明机制。
6. 全部通过后，再与参考实现比较，并写下一种现有测试仍可能漏掉的情况。

完成独立尝试后，可以请 AI 提示、提供另一种推导，或设计带敌意的测试。不要把生成的解答直接贴进起始档，再把测试通过当成理解。即使没有工具在场，你也应能解释数据形状、预测简单输出，并指出哪个不变量会抓到某个假想错误。

## 解读测试结果

- **只有一个聚焦测试失败：**定位该运算并缩小输入。
- **前向数值正确但有限差分失败：**怀疑微分规则、扇出处的梯度累加，或原地修改。
- **参考实现与学习者实现都失败：**先检查环境与版本；接着把它视为可能的测试或文档缺陷，并保留证据。
- **全部通过：**列出的不变量在受测范围内成立。性能、极端数值下的行为，以及对新任务是否有用，仍是不同问题。

## 可复现性记录

你报告的每个实验都要记录：假设与预测；环境（`python --version`、套件版本）；完整指令与代码版本；随机数种子；计算预算；评估方式；带不确定性的结果；负面结果；以及结论的效度边界。精简的即时记录，比事后重建的漂亮故事更有价值。
