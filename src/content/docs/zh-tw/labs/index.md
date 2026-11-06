---
title: 實驗室
description: 筆電規模、附不變量測試的實驗——Atlas 驗證工具組中可執行的那一半。
sidebar: { label: 實驗室總覽, order: 1 }
---

實驗把數學概念轉成預測與量測。你不需要一次就寫出正確實作。每個實驗都從一個**不使用 AI 的建構**檔案開始：裡面有函式簽名，但沒有解答；另有一組取自 Atlas 不變量的測試。同一組測試在 CI 中也會對參考實作執行，因此剛開始的失敗是在告訴你還需要建構什麼，不是在懷疑測試能否成立。

目標不只是讓進度列變成綠色，而是把四件事連起來：數學主張、事前預測、程式行為，以及能把預期機制與合理錯誤區分開來的測試。

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

## 第一次做實驗：逐步流程

如果你以前沒有寫過數值程式，先做單元 03 的自動微分。

1. 閱讀對應的詳解例題，手算預期輸出。
2. 打開起始檔，找出最小的缺失運算；先不要閱讀完整參考答案。
3. 執行一個具名測試，例如 `python -m pytest -k add -v`，由左到右讀斷言：輸入、觀測值、預期性質。
4. 只實作那個運算。重跑同一個測試，再跑會用到扇出或隨機圖的相鄰測試。
5. 測試失敗時，保留最小失敗案例；在詢問 AI 或修改程式碼前，先用文字說明機制。
6. 全部通過後，再與參考實作比較，並寫下一種現有測試仍可能漏掉的情況。

完成獨立嘗試後，可以請 AI 提示、提供另一種推導，或設計帶敵意的測試。不要把生成的解答直接貼進起始檔，再把測試通過當成理解。即使沒有工具在場，你也應能解釋資料形狀、預測簡單輸出，並指出哪個不變量會抓到某個假想錯誤。

## 解讀測試結果

- **只有一個聚焦測試失敗：**定位該運算並縮小輸入。
- **前向數值正確但有限差分失敗：**懷疑微分規則、扇出處的梯度累加，或原地修改。
- **參考實作與學習者實作都失敗：**先檢查環境與版本；接著把它視為可能的測試或文件缺陷，並保留證據。
- **全部通過：**列出的不變量在受測範圍內成立。效能、極端數值下的行為，以及對新任務是否有用，仍是不同問題。

## 可重現性紀錄

你報告的每個實驗都要記錄：假設與預測；環境（`python --version`、套件版本）；完整指令與程式碼版本；亂數種子；計算預算；評估方式；帶不確定性的結果；負面結果；以及結論的效度邊界。精簡的即時紀錄，比事後重建的漂亮故事更有價值。
