---
title: 實驗室
description: 筆電規模、附不變量測試的實驗——Atlas 驗證工具組中可執行的那一半。
sidebar: { label: 實驗室總覽, order: 1 }
---

實驗把數學概念轉成預測與量測。每個實驗都有一個你自己寫的**不使用 AI 的建構**檔案，以及取自 Atlas 不變量的測試組。同一組測試在 CI 中也對參考實作執行，所以測試本身也經過驗證。

| 實驗 | 單元 | 你要實作（不使用 AI） | 測試檢查 |
|---|---|---|---|
| [`labs/unit-03-autodiff`](https://github.com/yhxuee/deep-learning-intelligent-systems/tree/main/labs/unit-03-autodiff) | [03](/zh-tw/core/unit-03/) | 純量反向模式自動微分 `Value` | 詳解例題、扇出累加、隨機圖上的有限差分、線性性 |
| [`labs/unit-04-attention`](https://github.com/yhxuee/deep-learning-intelligent-systems/tree/main/labs/unit-04-attention) | [04](/zh-tw/core/unit-04/) | 以 NumPy 實作因果多頭自注意力 | 列總和、因果性、置換等變性、KV 快取等價性、√d 縮放、GPT-2 參數量；以及五個植入錯誤的**除錯挑戰** |
| [`labs/unit-05-memory`](https://github.com/yhxuee/deep-learning-intelligent-systems/tree/main/labs/unit-05-memory) | [05](/zh-tw/core/unit-05/) | 線性遞迴（迴圈 + 封閉解）、線性注意力記憶 | 封閉解、譜衰減、非正規暫態增長、$d_k$ 以內精確回憶 vs. 超過後的干擾 |
| [`spine/`](https://github.com/yhxuee/deep-learning-intelligent-systems/tree/main/spine) | 01–12 | 主幹專案的語言模型 | 各階段的驗證目標 |
| [`labs/test_worked_examples.py`](https://github.com/yhxuee/deep-learning-intelligent-systems/blob/main/labs/test_worked_examples.py) | 03–05 | — | 重新計算各單元詳解例題中引用的每個數字，讓正文與計算不會脫節 |

另外，[互動圖表](/zh-tw/labs/interactive/)頁讓你在寫程式之前，先在瀏覽器中探索本書的關鍵結果。

## 執行實驗

需求：Python ≥ 3.10、NumPy、pytest。

```bash
pip install -r labs/requirements.txt
cd labs/unit-04-attention
python -m pytest                       # 測試你的實作（attention.py）
LAB_IMPL=reference python -m pytest    # 參考實作通過同一組測試
python -m pytest test_bug_hunt.py -v   # 哪個不變量抓到哪個植入的錯誤？
```

剛取得的程式碼中，`engine.py`／`attention.py`／`memory.py` 只有函式簽名與 `NotImplementedError`，所以在你實作之前測試會失敗。這是刻意的設計。

## 可重現性紀錄

你報告的每個實驗都要記錄：假設與預測；環境（`python --version`、套件版本）；亂數種子；計算預算；評估方式；以及負面結果。
