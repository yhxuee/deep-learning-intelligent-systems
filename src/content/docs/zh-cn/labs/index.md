---
title: 实验室
description: 笔记本电脑规模、附不变量测试的实验——Atlas 验证工具包中可执行的那一半。
sidebar: { label: 实验室总览, order: 1 }
---

实验把数学概念转成预测与度量。每个实验都有一个你自己写的**不使用 AI 的构建**文件，以及取自 Atlas 不变量的测试组。同一组测试在 CI 中也对参考实现执行，所以测试本身也经过验证。

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

## 可复现性记录

你报告的每个实验都要记录：假设与预测；环境（`python --version`、套件版本）；随机数种子；计算预算；评估方式；以及负面结果。
