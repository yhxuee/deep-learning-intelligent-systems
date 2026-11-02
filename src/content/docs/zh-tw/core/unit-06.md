---
title: "單元 06 · 計算限制如何形塑智慧？"
description: "複雜度、算術強度、記憶體階層、尺度定律與計算最佳化訓練。"
sidebar: { label: "06 · 計算", order: 6 }
unit:
  number: 6
  part: 2
  question: "何時值得用資料、記憶體、能源與延遲去換取更大的模型容量？"
  anchor: "Transformer 的 FLOP 與記憶體估算，以及在五種模型大小上擬合尺度定律。"
  spine: "階段 06——在固定詞元預算下訓練五種大小的主幹模型；擬合損失對計算量；在訓練之前預測第六種大小。"
  checkpoint: "以算術強度說明為什麼理論 FLOPs 可能與實測執行時間不一致。"
  status: outline
  atlas: ["method-scaling-law-fit", "arch-transformer-block", "mech-kv-cache", "opt-adamw"]
---

:::caution[規劃大綱——尚非教科書正文]
本頁記錄本單元的問題、錨定範例、學習成果與活動。它是一份課程大綱，而不是章節：它列出主題，卻沒有教它們。它將依[撰寫標準](/zh-tw/project/writing-standard/)改寫為十部分結構，就像單元 03–05 一樣（見[路線圖](/zh-tw/project/roadmap/)）。歡迎貢獻。
:::

## 問題

兩個注意力核心執行相同數量的 FLOPs，其中一個快了 3 倍。預測是哪一項硬體資源造成差異，以及你會如何量測它。

:::note[讀下去之前，先預測]
繼續之前，先寫下你的答案以及它所依賴的假設。
:::

## 核心內容

- 計數：$N$ 個參數、$D$ 個詞元的訓練約需 $6ND$ FLOPs；權重、梯度、最佳化器狀態與活化值的記憶體。
- 算術強度與 **roofline 模型** [Williams et al. 2009](/zh-tw/references/#williams-2009)；為什麼考量 IO 的核心會勝出 [Dao et al. 2022](/zh-tw/references/#dao-2022)。
- 混合精度 [Micikevicius et al. 2018](/zh-tw/references/#micikevicius-2018) 與平行化策略（資料、張量、管線）[Narayanan et al. 2021](/zh-tw/references/#narayanan-2021)——Atlas 層次。
- 經驗性的**尺度定律（scaling law）** [Kaplan et al. 2020](/zh-tw/references/#kaplan-2020)，以及在參數與資料之間的計算最佳分配 [Hoffmann et al. 2022](/zh-tw/references/#hoffmann-2022)；冪律擬合能外插什麼、不能外插什麼。
- 推論經濟學：KV 快取記憶體、批次處理、延遲 vs. 吞吐量 [Pope et al. 2023](/zh-tw/references/#pope-2023)。
- **實務技巧：** 在不同規模下判讀損失曲線；一次訓練是「壞了」還是只是「太小」。

## 依掌握層次的學習成果

| 層次 | 學習成果 |
|---|---|
| <span class="tier tier-internalize">內化</span> | 6ND；一次訓練的記憶體預算；算術強度；冪律擬合是什麼 |
| <span class="tier tier-verify">理解與驗證</span> | 計算最佳比例；平行化的取捨；量化誤差 |
| <span class="tier tier-access">按需查閱</span> | GPU 規格；框架特定的切分 API |

## 學習活動

**建構（不使用 AI）。** 寫一個函式，對給定的 Transformer 設定回傳參數量、訓練 FLOPs 與訓練峰值記憶體；以 GPT-2 small 的已知數值檢查它。

**挑戰（評論者模式）。** AI 用 3 個點擬合的尺度定律外插 4 個數量級。找出這個外插需要的每一個假設。

**實驗（協作者模式）。** 在五種大小的主幹模型上擬合損失對計算量；預先登記對第六種大小的預測；報告預測區間與誤差。

**主幹專案進度。** 階段 06——在固定詞元預算下訓練五種大小的主幹模型；擬合損失對計算量；在訓練之前預測第六種大小。

**遷移與答辯（不使用 AI）。** 給定裝置端模型的延遲預算與記憶體上限，選擇深度、寬度與上下文長度，並以數字說明取捨。

## 主要文獻

[kaplan-2020](/zh-tw/references/#kaplan-2020) · [hoffmann-2022](/zh-tw/references/#hoffmann-2022) · [williams-2009](/zh-tw/references/#williams-2009) · [dao-2022](/zh-tw/references/#dao-2022) · [micikevicius-2018](/zh-tw/references/#micikevicius-2018) · [narayanan-2021](/zh-tw/references/#narayanan-2021) · [pope-2023](/zh-tw/references/#pope-2023)
