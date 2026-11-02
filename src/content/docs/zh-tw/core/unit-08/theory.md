---
title: "8.5 一般理論"
description: 分布之間的最佳輸運與直線路徑為何重要、分數匹配及其與去雜訊的等價、抽樣中的離散化誤差、生成模型總覽表，以及概似量測與沒量測的東西。
sidebar: { label: "8.5 一般理論", order: 5 }
---

## 8.5.1 分布之間的輸運

§8.3.2–§8.3.6 的每個方法，都建構一個映射——或一個流——把簡單的分布推到資料分布上。這樣的映射有很多。**最佳輸運（optimal transport）**挑出移動質量最少的那一個：在 $p_0$ 與 $p_1$ 的所有配對中，最小化 $\mathbb{E}\lVert x_1 - x_0\rVert^2$（[Villani 2009](/zh-tw/references/#villani-2009)）。它的解讓每個粒子以等速沿直線移動，而且粒子的路徑不會交叉。

以*獨立*配對做的流匹配（§8.3.5）使用直線的條件路徑，但它們會交叉，所以邊際場 $v^\star$ 是彎曲的，常微分方程需要許多步。理論給出兩種補救：選擇更接近最佳輸運的配對（例如在小批次內解一個小型指派問題來配對雜訊與資料），或**整流（rectify）**——以學到的流產生配對 $(x_0, x(1))$，再以它們訓練新的流，這可證明不會增加輸運成本，並讓路徑變直（[Liu et al. 2023](/zh-tw/references/#liu-2023)）**[D]**。更直的路徑允許更少的積分步數（§8.5.3）。

## 8.5.2 分數匹配與去雜訊

密度的**分數**是 $s(x) = \nabla_x\ln p(x)$。它不依賴正規化常數，所以能為未正規化的模型學習（[Hyvärinen 2005](/zh-tw/references/#hyvarinen-2005)）。從樣本學習它似乎需要 $\nabla\ln p$，而我們沒有；技巧在於加雜訊。

**命題 8.3（去雜訊分數匹配；[Vincent 2011](/zh-tw/references/#vincent-2011)）。** 令 $\tilde x = x + \sigma\varepsilon$、$\varepsilon\sim N(0, I)$，$p_\sigma$ 為 $\tilde x$ 的密度。則
$$
\mathbb{E}\Big\lVert s_\theta(\tilde x) + \frac{\varepsilon}{\sigma}\Big\rVert^2 = \mathbb{E}\big\lVert s_\theta(\tilde x) - \nabla\ln p_\sigma(\tilde x)\big\rVert^2 + \text{常數}.
$$

*證明概要。* 展開兩個平方；它們只在交叉項上不同。利用 $\nabla\ln p_\sigma(\tilde x) = \mathbb{E}[-\varepsilon/\sigma\mid\tilde x]$（在積分號下對 $p_\sigma(\tilde x) = \int p(x)N(\tilde x; x, \sigma^2I)\,dx$ 微分），得 $\mathbb{E}[s_\theta(\tilde x)^{\top}\varepsilon/\sigma] = -\mathbb{E}[s_\theta(\tilde x)^{\top}\nabla\ln p_\sigma(\tilde x)]$，所以交叉項相同。$\blacksquare$

所以預測加入的雜訊，在差一個常數的意義下，就等於估計帶雜訊資料的分數——這是 §8.3.4 背後的恆等式。在許多雜訊等級上的分數模型，結合 Langevin 動力學或反向時間 SDE，就是擴散的分數觀點（[Song & Ermon 2019](/zh-tw/references/#song-2019)；[Song et al. 2021](/zh-tw/references/#song-2021)）；擴散、分數式與流匹配模型，是同一族輸運的不同參數化（[Albergo & Vanden-Eijnden 2023](/zh-tw/references/#albergo-2023)）**[D]**。

## 8.5.3 抽樣誤差如何依賴步長

以步長 $h$ 的 Euler 步積分 $\dot x = v(x, t)$，每步的局部誤差量級為 $h^2\lVert\partial_tv + (\nabla v)v\rVert$，終點的整體誤差量級為 $h$，其常數與*路徑的曲率*成正比 **[D]**。若場產生等速的直線路徑，Euler 一步就精確。這就是為什麼「變直」（§8.5.1）與少步數抽樣是同一個研究問題。高階求解器降低誤差的冪次；單元 05 的穩定性條件（§5.3.4）也適用，因為在某些雜訊排程下，常微分方程在 $t = 0$ 或 $t = 1$ 附近可能是剛性的（[Karras et al. 2022](/zh-tw/references/#karras-2022)）**[E]**。

## 8.5.4 生成模型總覽

| 家族 | 抽樣 | 概似 | 訓練訊號 | 主要成本 |
|---|---|---|---|---|
| 自迴歸 | 精確，每詞元一步 | 精確 | 交叉熵 | 循序生成 |
| 正規化流 | 一次傳遞 | 精確 | 最大概似 | 可逆架構 |
| VAE | 一次傳遞 | 下界（ELBO） | 重建 + KL | 模糊的樣本、後驗崩塌 |
| GAN（[Goodfellow et al. 2014](/zh-tw/references/#goodfellow-2014)） | 一次傳遞 | 無 | 對抗賽局 | 訓練不穩、遺漏眾數 |
| 擴散／分數 | 許多步（SDE 或 ODE） | 透過 ODE（昂貴） | 去雜訊迴歸 | 許多次網路評估 |
| 流匹配 | 許多步（ODE） | 透過 ODE（昂貴） | 速度迴歸 | 許多次網路評估 |

沒有任何家族在每個面向都勝出。文本使用自迴歸，因為精確且便宜的每詞元概似很重要，而且詞元是離散的；影像使用擴散與流匹配，因為迴歸訓練穩定、樣本品質高，而許多步的成本則由 §8.8 的方法處理。

## 8.5.5 概似量測了什麼

保留資料的對數概似（單元 01）仍是密度模型有原則的評分。對生成器而言，它在兩個方向上都不完整（[Theis et al. 2016](/zh-tw/references/#theis-2016)）**[D]/[E]**：模型可以有極佳的概似卻有差的樣本（它把一點質量分散到垃圾上，同時很好地涵蓋了所有資料），也可以有極佳的樣本卻有差或無定義的概似（它完美涵蓋少數眾數，卻漏掉其餘）。以樣本為基礎的指標，例如真實與生成影像特徵統計之間的 Fréchet 距離（[Heusel et al. 2017](/zh-tw/references/#heusel-2017)），量測品質與涵蓋度，代價是依賴一個特徵擷取器。單元 10 把指標的選擇當成評估設計的決定來處理。
