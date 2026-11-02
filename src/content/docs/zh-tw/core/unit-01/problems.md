---
title: "1.10 習題與延伸閱讀"
description: 二十道關於對數損失、最大概似、平滑、集中不等式、保留資料評估與類推理論的分級習題，附部分解答與附註解的閱讀清單。
sidebar: { label: "1.10 習題", order: 10 }
---

難度：★ 例行 · ★★ 結合多個概念 · ★★★ 證明或開放式問題。**[不使用 AI]** 的題目用於獨立練習與評量。

## 計算

**1.1 ★ [不使用 AI]** 一位預報員連續五天都預測降雨機率 0.8；其中三天下雨。計算平均對數損失與準確率（$q > 0.5$ 時就預測下雨）。對一位總是說 0.6 的預報員重做一次。兩種評分各偏好哪一位？

**1.2 ★ [不使用 AI]** 以 $\alpha = 0.5$ 重做詳解例題 1.2。保留資料損失現在低於 $\ln 3$ 了嗎？

**1.3 ★** 把每字元 1.2 nats 的保留資料損失換算成每字元位元數與困惑度。困惑度 2 對應多少損失？

**1.4 ★** 對詳解例題 1.1 的硬幣，找出使 $q = 0.7$ 的期望測試損失等於其訓練損失 0.6109 的真實 $p$。

**1.5 ★★** 一個二元語法在某段文本上擬合，其中字元 `z` 恰好出現一次，後面接 `e`。寫出 $V = 50$ 時平滑後的列 $q_\alpha(\cdot\mid\texttt z)$。α 為多少時 $q_\alpha(\texttt e\mid\texttt z) = 0.5$？

**1.6 ★★** 利用 Hoeffding 不等式，要以 99% 的信心把 0–1 錯誤率估計到 ±0.5 個百分點以內，需要多少保留樣本？若有 1,000 個候選模型呢？

## 推導

**1.7 ★ [不使用 AI]** 以拉格朗日乘數推導 $V$ 種結果的類別分布的最大概似估計（命題 1.1）。

**1.8 ★★ [不使用 AI]** 證明 $H(p,q) = H(p) + \mathrm{KL}(p\,\Vert\,q)$ 與 Gibbs 不等式。舉一個 $\mathrm{KL}(p\,\Vert\,q)\ne\mathrm{KL}(q\,\Vert\,p)$ 的例子。

**1.9 ★★** 證明在 Dirichlet($\alpha$) 先驗下，類別參數的後驗是 Dirichlet($c + \alpha$)，且結果 $b$ 的後驗預測機率等於加 α 估計。

**1.10 ★★** 證明期望對數損失是*嚴格*恰當的評分：對任何 $q\ne p$，$\mathbb{E}_p[-\ln q] > \mathbb{E}_p[-\ln p]$。然後找出 $p$ 與 $q\ne p$，使 $q$ 的期望分數較低，以證明*絕對*評分 $1 - q(x)$ 不是恰當的。

**1.11 ★★** 證明以最大概似擬合的二元語法，其訓練損失等於經驗條件熵 $\sum_a \frac{c(a)}{n}H(\hat p(\cdot\mid a))$。

## 證明

**1.12 ★★ [不使用 AI]** 證明偏差–變異數分解（命題 1.2），並指出每個假設在哪裡被用到。

**1.13 ★★** 對允許隨機化預測的學習者，證明沒有免費午餐定理（定理 1.4）。

**1.14 ★★★** 證明 Hoeffding 引理：若 $\mathbb{E}X = 0$ 且 $X\in[a,b]$，則 $\mathbb{E}e^{sX}\le e^{s^2(b-a)^2/8}$。

**1.15 ★★★** 證明實數線上的門檻分類器族 $h_t(x) = \mathbb{1}[x\ge t]$ 的 VC 維度為 1，而區間 $\mathbb{1}[a\le x\le b]$ 的 VC 維度為 2。

**1.16 ★★★** 令 $\hat L_1,\dots,\hat L_K$ 為 $K$ 個模型的驗證損失，其真實損失都等於 $L$，各自帶有標準差 $\sigma$ 的獨立高斯雜訊。證明 $\mathbb{E}[\min_k\hat L_k] < L$，且當 $K$ 很大時，這個樂觀程度像 $\sigma\sqrt{2\ln K}$ 一樣成長。

## 實作與設計

**1.17 ★ [不使用 AI]** 不看資料、不使用 NumPy 的 `add.at`，實作 `fit_bigram` 與 `nll`。驗證你的表每一列總和為 1，且 α → ∞ 時得到 $\ln V$。

**1.18 ★★** 在主幹語料上實作帶加 α 平滑的三元語法模型。比較它與二元語法在 2,000 與 385,274 個訓練字元時的訓練與保留資料損失。以 §1.7.7 解釋結果。

**1.19 ★★** 實作*內插*平滑 $q(b\mid a) = \lambda\,\hat p(b\mid a) + (1-\lambda)\,\hat p(b)$，在驗證故事上選擇 λ，並與加 α 平滑比較。

**1.20 ★★★** 為一個將用於明年所寫故事的語言模型設計評估流程。說明你所做的分布假設、你的切分如何檢驗它，以及什麼結果會顯示該假設是錯的（單元 10 的預告）。

## 部分解答

- **1.1** 預報員 0.8：對數損失 $(3\times0.2231 + 2\times1.6094)/5 = 0.7777$，準確率 3/5。預報員 0.6：$(3\times0.5108 + 2\times0.9163)/5 = 0.6730$，準確率 3/5。準確率分不出兩者；對數損失偏好機率（0.6）等於觀察頻率的那一位。
- **1.3** 1.2 nats = 1.731 位元；困惑度 $e^{1.2} = 3.32$。困惑度 2 ⇔ 損失 $\ln 2 = 0.693$ nats = 1 位元。
- **1.5** $q_\alpha(\texttt e\mid\texttt z) = (1+\alpha)/(1+50\alpha)$；令它等於 0.5，得 $2 + 2\alpha = 1 + 50\alpha$，即 $\alpha = 1/48$。
- **1.6** $n \ge \ln(2/\delta)/(2\varepsilon^2) = \ln 200/(2\times0.005^2) = 105{,}967$；若 $K = 1{,}000$，$n\ge(\ln 1000 + \ln 200)/(2\times 0.005^2) = 244{,}122$。
- **1.11** $n\hat L = -\sum_{a,b}c(a\to b)\ln\frac{c(a\to b)}{c(a)} = \sum_a c(a)\big[-\sum_b \hat p(b\mid a)\ln\hat p(b\mid a)\big]$。
- **1.16** $K$ 個 i.i.d. $N(0,\sigma^2)$ 變數的最小值，在 $K$ 很大時期望值約為 $-\sigma\sqrt{2\ln K}$；$K = 1000$ 時約為 $3.7\sigma$，這是一個上界，精確值（約 $3.24\sigma$）緩慢地趨近它。

## 延伸閱讀

**基礎。** [Shalev-Shwartz & Ben-David 2014](/zh-tw/references/#shalev-shwartz-2014)，第 2–6 章——關於 ERM、均勻收斂、VC 維度與沒有免費午餐最清楚的現代論述。[Cover & Thomas 2006](/zh-tw/references/#cover-2006)，第 2 與 5 章——熵、KL 散度與編碼。

**語言模型。** [Shannon 1951](/zh-tw/references/#shannon-1951)——至今仍是預測文本最好的入門。[Chen & Goodman 1999](/zh-tw/references/#chen-1999)——數十年平滑研究學到了什麼。[Bengio et al. 2003](/zh-tw/references/#bengio-2003)——通往單元 02 的橋樑。

**評分規則。** [Gneiting & Raftery 2007](/zh-tw/references/#gneiting-2007)——為什麼對數損失與 Brier 分數是恰當的，以及這帶來什麼好處。

**現代的謎題。** [Zhang et al. 2017](/zh-tw/references/#zhang-2017)、[Belkin et al. 2019](/zh-tw/references/#belkin-2019)、[Nakkiran et al. 2020](/zh-tw/references/#nakkiran-2020)、[Bartlett et al. 2020](/zh-tw/references/#bartlett-2020)——依此順序閱讀。

**評估實務。** [Recht et al. 2019](/zh-tw/references/#recht-2019) 與 [Dwork et al. 2015](/zh-tw/references/#dwork-2015)——被重複使用的測試集會發生什麼事。
