---
title: "8.5 一般理論"
description: 分布之間的最佳輸運與直線路徑為何重要、分數匹配及其與去雜訊的等價、抽樣中的離散化誤差、生成模型總覽表，以及概似量測與沒量測的東西。
sidebar: { label: "8.5 一般理論", order: 5 }
---

## 8.5.1 分布之間的幾何輸運：最佳輸運與直線流

§8.3.2–§8.3.6 的連續生成方法，本質上都是在構建一個時變速度場向量 $v(x, t)$，將簡單的先驗高斯雜訊分佈 $p_0$ 沿著軌跡平滑推動至複雜的真實數據分佈 $p_1$。在所有可能將 $p_0$ 變形為 $p_1$ 的連續速度場中，**最佳輸運理論（Optimal Transport, OT）**挑選出了能量消耗最少的那一個：在 $p_0$ 與 $p_1$ 的所有聯合分佈 $\pi(x_0, x_1)$ 中，最小化二階 Wasserstein 距離 $\mathbb{E}_{(x_0, x_1)\sim\pi}\lVert x_1 - x_0\rVert^2$（[Villani 2009](/zh-tw/references/#villani-2009)）。

**最佳輸運的幾何性質：**
OT 的理論最優解（位移插值，Displacement Interpolation）保證每個粒子以恆定速度沿**嚴格不相交的直線路徑**勻速行進：$\dot{x}_t = x_1 - x_0$。

**獨立流匹配的彎曲困境與整流（Rectification）：**
然而，基礎的流匹配方法（§8.3.5）採用了獨立耦合先驗 $q(x_0, x_1) = p_0(x_0) p_1(x_1)$。雖然對任一單獨配對而言條件路徑 $x_t = (1-t)x_0 + t x_1$ 是一條直線，但**來自不同雜訊與數據對的直線軌跡在高維空間中頻繁交叉**！
- 當兩條軌跡在 $(x, t)$ 處相交時，邊際速度場 $v^\star(x, t) = \mathbb{E}[x_1 - x_0 \mid x_t = x]$ 被迫對截然不同的速度向量進行平均；
- 這種平均效應使得宏觀速度場發生劇烈旋轉與彎曲，導致數值常微分方程（ODE）求解器在採樣時必須採用數十步甚至上百步微小步長才能避免軌道偏離。

理論給出兩種優雅的解法：
1. **OT 迷你批次配對：** 在訓練批次內透過求解小型線性指派問題（Hungarian 演算法或 Sinkhorn 迭代），將幾何距離最近的雜訊與數據點進行配對；
2. **整流演算法（Reflow / Rectified Flow，[Liu et al. 2023](/zh-tw/references/#liu-2023)）：** 利用已訓練的第一代流模型從 $x_0$ 數值模擬生成端點 $x(1)$，構建全新的確定性配對 $(x_0, x(1))$。理論證明，以這種自生成配對重新訓練第二代流，可以在嚴格不增加輸運成本的前提下**徹底解開交叉路徑，將軌跡強力拉直（Straightening the Flow）** **[D]**。直化的速度場使得 Euler 單步或二步採樣即可達到頂級生成品質（§8.5.3）。

## 8.5.2 分數匹配、能量爬坡與去雜訊等價性

連續分佈的**分數函數（Score Function）**定義為對數概率密度的梯度向量：
$s(x) = \nabla_x\ln p(x) = \frac{\nabla_x p(x)}{p(x)}$。
分數向量直觀上就是**指向概率密度更高區域的局部爬坡方向**。

**分數的核心數學優勢：**
若密度由能量模型定義 $p(x) = \frac{1}{Z} e^{-E(x)}$，對其求空間梯度：
$\nabla_x \ln p(x) = \nabla_x \big(-E(x) - \ln Z\big) = -\nabla_x E(x)$。
極其昂貴且通常在數學上無法解析積分的配分函數常數 $Z = \int e^{-E(x)}dx$ 在梯度運算下完全消失！這使得我們能夠為任意未歸一化的複雜能量模型學習梯度場（[Hyvärinen 2005](/zh-tw/references/#hyvarinen-2005)）。

但實踐困境在於：我們只有離散的數據樣本，不知道真實連續分佈的 $\nabla\ln p(x)$。**去雜訊分數匹配（Denoising Score Matching）**透過人工加噪巧妙解決了這一僵局。

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
