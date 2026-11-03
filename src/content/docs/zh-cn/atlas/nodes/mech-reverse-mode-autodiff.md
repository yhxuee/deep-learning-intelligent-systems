---
title: 反向模式自动微分
description: 把反向传播看成依反向拓扑排序应用的链式法则——以前向成本的小常数倍取得标量的完整梯度。
atlas:
  id: mech-reverse-mode-autodiff
  type: mechanism
  mastery: internalize
  core_units: [3]
  prerequisites: [math-chain-rule]
  related: [method-gradient-check]
  lab: labs/unit-03-autodiff
  references: [rumelhart-1986, griewank-2008, baydin-2018, karpathy-micrograd]
  last_reviewed: 2026-11-02
---

## 一段话说明

把前向计算记录成一张图。把输出的伴随值设为 $\bar L = 1$，再以反向拓扑排序走访节点，把 $\bar v_j\,\partial v_j/\partial v_i$ 推进每个父节点的伴随值中。对标量损失，这一次反向扫描就得到每个偏导数，成本是前向扫描的小常数倍；代价是要存储中间值。

## 不变量（在实验室中测试）

1. **详解范例：** [单元 03 §3.3.3](/zh-cn/core/unit-03/construction/) 的单一神经元网络给出 $\bar w_1 = 0.098678$、$\bar w_2 = 0.197356$、$\bar b = 0.098678$。
2. **扇出要累加：** 对 $y = a\cdot a$，$\bar a = 2a$；对 $y = a + a$，$\bar a = 2$。
3. **与中心差分一致：** 在随机表达式图上，相对误差 $<10^{-6}$（float64）。
4. **线性性：** $\alpha f + \beta g$ 的梯度为 $\alpha\nabla f + \beta\nabla g$。
5. **叶节点的梯度跨图累加：** 共用一个叶节点的两张不同图，各做一次反向传递，其贡献会相加——累加是定义好的行为，所以归零是调用者的责任。（不把中间节点归零就在*同一张*图上重跑 `backward()`，会得到错误的梯度，而不是加倍的梯度。）

## 参考实现

`labs/unit-03-autodiff/reference.py`（约 90 行，标量 `Value` 类别）。请先在 `engine.py` 写你自己的版本；同一组测试对两者都执行。

## 常见错误

- 累加到父节点时用 `=` 而不是 `+=`（不变量 2 失败）。
- 建立拓扑排序时没有“已走访”集合（在共用子图上变成指数时间）。
- `tanh` 的局部导数写成 $1 - \tanh(x)$ 而不是 $1 - \tanh^2(x)$。
- 优化步骤之间忘了把梯度归零。
