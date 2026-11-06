---
title: 消融与激活修补
description: 检验模型组件是否被使用的干预——移除它（消融），或把它的激活换成来自另一个输入的值（修补）——以及让结果可以诠释的对照组。
atlas:
  id: method-activation-patching
  type: method
  mastery: understand-and-verify
  core_units: [11]
  prerequisites: [mech-attention, arch-transformer-block]
  related: [method-preregistration]
  references: [meng-2022, wang-2023, zhang-nanda-2024]
  last_reviewed: 2026-11-02
---

## 一段话说明

观察性证据——注意力模式、探测、归因——显示的是哪些信息存在或相关。**干预**显示的是哪些被使用。*消融*把组件的输出换成零（或其平均），度量行为的变化。*激活修补*在“被破坏”的输入上执行模型，并把某个组件的激活换成它在“干净”输入上的值；如果干净的行为恢复了，该组件就承载了这个差异。两者都需要对照组：消融不应该有影响的组件、以无关输入的激活修补，并在机制已知的模型上验证整个流程。

## 不变量

- 以一次执行自己的激活修补它，什么都不会改变（主干项目 `test_stage11_patching_a_run_with_its_own_activations_is_a_no_op`）。
- 把一个头归零消融，等于以零修补它（主干项目 `test_ablating_a_head_equals_patching_it_with_zeros`）。
- 在手工构建的归纳电路上，消融任一电路头都会破坏复制，消融未使用的头则不会（主干项目 `test_stage11_hand_built_circuit_is_found_used_and_needed`）。

## 已知数值

| 量 | 值 |
|---|---|
| 手工电路在重复后半的损失：完整／消融电路头／消融未使用的头 | < 0.01 ／ > 1 ／ < 0.01 nats |

## 常见错误

- 只凭注意力模式就断言“这个头实现了 X”。
- 归零消融把激活推到远离分布之处，量到的效应反映的是冲击而不是机制；应与平均消融比较。
- 用来修补的被破坏输入与干净输入之间，不只差了感兴趣的那个变量。
