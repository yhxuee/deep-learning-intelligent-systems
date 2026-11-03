---
title: "2.5 一般理論"
description: 普遍逼近定理及其限制、Cover 函數計數定理、隨機投影與 Johnson–Lindenstrauss 引理、以群平均得到不變性與等變線性映射，以及 Eckart–Young 定理的證明。
sidebar: { label: "2.5 一般理論", order: 5 }
---

## 2.5.1 普遍逼近——以及它沒有保證的事

**定理 2.2（普遍逼近；[Cybenko 1989](/zh-tw/references/#cybenko-1989)、[Hornik et al. 1989](/zh-tw/references/#hornik-1989)）。** 令 $\sigma$ 為連續的 S 形函數。對緊緻集 $K\subset\mathbb{R}^p$ 上的每個連續函數 $f$ 與每個 $\varepsilon>0$，存在 $H$ 與參數，使單隱藏層網路 $g(x) = \sum_{j=1}^{H}a_j\sigma(w_j^{\top}x + b_j)$ 滿足 $\sup_{x\in K}|f(x) - g(x)|<\varepsilon$。

這個定理說：夠寬的隱藏層基本上能表示任何函數。它沒有說**需要多少**隱藏單元（對某些函數，需要的數量隨 $p$ 指數成長）、需要多少**資料**才能找到正確的參數，也沒有說**梯度下降法**能否找到它們。表示是學習的必要條件；定理只保證它在原則上不是瓶頸。本單元的每個實務問題——共享、樣本效率、不變性——都存在於這個定理留下的空隙中。以 XOR 來說：隱藏層*能*計算 $x_1x_2$；問題在於資料與最佳化器是否會引導它去計算。

## 2.5.2 Cover 計數定理：為什麼維度有幫助

**定理 2.3（[Cover 1965](/zh-tw/references/#cover-1965)）。** 令 $\mathbb{R}^p$ 中的 $N$ 個點處於一般位置（其中任意至多 $p$ 個點都線性獨立）。能被通過原點的超平面分開的標註數為
$$
C(N, p) = 2\sum_{k=0}^{p-1}\binom{N-1}{k}.
$$
對仿射分類器 $w^{\top}x + b$，附加一個常數座標，並使用 $C(N, p+1)$。

*證明概要。* 加入第 $N{+}1$ 個點。前 $N$ 個點的一種可分標註，若沒有分隔超平面通過新點，就只能以一種方式延伸到新點；若有，就能以兩種方式延伸；後者對應於這 $N$ 個點投影到 $(p-1)$ 維空間後的可分標註。由此得到遞迴 $C(N+1, p) = C(N, p) + C(N, p-1)$，且 $C(1, p) = 2$，其解就是上式（它是巴斯卡法則的累加）。$\blacksquare$

當 $N\le p$ 時，$C(N, p) = 2^N$：**每一種**標註都可分。當 $N = 2p$ 時，恰好一半的標註可分。所以線性讀出的容量大約是每個維度兩個點。以固定的非線性特徵——隨機特徵（[Rahimi & Recht 2007](/zh-tw/references/#rahimi-2007)）、核函數或寬的隨機隱藏層——把輸入映射到更高維空間，幾乎任何標註都能線性達到，這既是它有效的原因，也是它會記住雜訊的原因。學習的表示追求更好的東西：讓*真正的*標註變得簡單的少數維度。

## 2.5.3 隨機投影保留幾何

一個嵌入需要多少維度，才能讓 $n$ 個不同項目彼此保持距離？遠少於 $n$。

**定理 2.4（Johnson–Lindenstrauss；[Johnson & Lindenstrauss 1984](/zh-tw/references/#johnson-1984)）。** 對任何 $0<\varepsilon<1$ 與任何 $n$ 個點 $x_1,\dots,x_n\in\mathbb{R}^D$，若 $d \ge 8\ln n/\varepsilon^2$，則元素獨立服從 $N(0, 1/d)$ 的隨機線性映射 $R:\mathbb{R}^D\to\mathbb{R}^d$ 以正的機率滿足
$$
(1-\varepsilon)\lVert x_i - x_j\rVert^2 \le \lVert Rx_i - Rx_j\rVert^2 \le (1+\varepsilon)\lVert x_i - x_j\rVert^2 \quad\text{對所有 } i, j .
$$

*證明想法（[Dasgupta & Gupta 2003](/zh-tw/references/#dasgupta-2003)）。* 對固定的單位向量 $u$，$d\lVert Ru\rVert^2$ 是 $d$ 個獨立標準常態平方的和，即 $\chi^2_d$。Chernoff 界給出 $\Pr(|\lVert Ru\rVert^2 - 1| > \varepsilon) \le 2e^{-d(\varepsilon^2/2 - \varepsilon^3/3)/2}$。把它套用到 $\binom n2$ 個差向量並取聯集界；給定的 $d$ 讓總失敗機率小於 1。$\blacksquare$

維度只依賴 $\ln n$，不依賴 $n$ 或 $D$。一個系理對單元 11 很重要：在 $d$ 維中，可以放入指數多個*近似*正交（兩兩餘弦至多 $\varepsilon$）的向量。所以只要同時只有少數特徵活躍，網路能表示的特徵就可以遠多於它的維度——這就是**疊加（superposition）**假說（[Elhage et al. 2022](/zh-tw/references/#elhage-2022)）。

## 2.5.4 以平均得到不變性；等變映射

令有限群 $\mathcal{G}$ 作用在輸入上（例如 $T$ 個位置的 $T!$ 種置換，或長度 $T$ 訊號的 $T$ 種循環位移）。

**命題 2.2（群平均）。** 對任何函數 $\phi$，平均 $\bar\phi(x) = \frac{1}{|\mathcal G|}\sum_{g\in\mathcal G}\phi(g\cdot x)$ 是不變的：對每個 $h\in\mathcal G$，$\bar\phi(h\cdot x) = \bar\phi(x)$。

*證明。* $\bar\phi(h\cdot x) = \frac{1}{|\mathcal G|}\sum_g\phi(gh\cdot x)$，而當 $g$ 跑遍 $\mathcal G$ 時，$gh$ 也跑遍 $\mathcal G$。$\blacksquare$

把詞嵌入加總成「詞袋（bag of words）」就是置換群的平均構造：結果對詞序不變。這對主題分類是好的表示，對「狗咬人」與「人咬狗」則是災難（§2.7.1）。

若對所有 $g$ 都有 $F(g\cdot x) = g\cdot F(x)$，線性映射 $F$ 就是**等變的**。等變線性映射構成的空間比所有線性映射小得多，這就是它們樣本效率的來源。書中兩種情況一再出現。對長度 $T$ 訊號的循環位移，等變線性映射恰好是循環卷積——$T$ 個參數而不是 $T^2$ 個（單元 04，定理 4.2）。對單一通道（$X\in\mathbb{R}^{T}$）時 $T$ 個位置的置換，它們恰好是 $X\mapsto \lambda X + \gamma\mathbf 1\mathbf 1^{\top}X$——兩個參數（單元 04，定理 4.3）。有 $d_{\text{in}}$ 個輸入通道與 $d_{\text{out}}$ 個輸出通道（$X\in\mathbb{R}^{T\times d_{\text{in}}}$）時，它們恰好是 $X\mapsto XA + \mathbf 1\mathbf 1^{\top}XB$，其中 $A, B\in\mathbb{R}^{d_{\text{in}}\times d_{\text{out}}}$（等價地可用 $\frac1T\mathbf 1\mathbf 1^{\top}$，即平均而非總和）——共 $2d_{\text{in}}d_{\text{out}}$ 個參數，可以任意混合通道（[Zaheer et al. 2017](/zh-tw/references/#zaheer-2017)）。仍然貧乏的是*位置*結構：每個位置只看到它自己與所有位置的彙總，透過兩種彙總方式。注意力就是擺脫這種貧乏的非線性出路。由對稱性建構網路的一般理論是幾何深度學習的主題（[Bronstein et al. 2021](/zh-tw/references/#bronstein-2021)）。

## 2.5.5 ★ 進階：Eckart–Young 定理的證明

令 $A = \sum_i\sigma_iu_iv_i^{\top}$，$B$ 的秩至多為 $d$。$B$ 的零空間維度至少為 $n - d$，所以它與 $v_1,\dots,v_{d+1}$ 張成的 $(d+1)$ 維空間交於一個非零向量 $z$，將其歸一化。於是 $Bz = 0$，且
$$
\lVert A - B\rVert_2^2 \ge \lVert(A - B)z\rVert^2 = \lVert Az\rVert^2 = \sum_{i\le d+1}\sigma_i^2(v_i^{\top}z)^2 \ge \sigma_{d+1}^2 .
$$
所以在譜範數下，沒有秩 $d$ 的矩陣能勝過 $\sigma_{d+1}$，而 $A_d$ 恰好達到。對 Frobenius 範數，利用 Weyl 不等式 $\sigma_{i+j-1}(X+Y)\le\sigma_i(X)+\sigma_j(Y)$，取 $X = A - B$、$Y = B$、$j = d+1$：$\sigma_i(A - B)\ge\sigma_{i+d}(A)$，因此 $\lVert A - B\rVert_F^2 = \sum_i\sigma_i(A-B)^2\ge\sum_{i>d}\sigma_i(A)^2$。$\blacksquare$

## 2.5.6 摘要

| 結果 | 說了什麼 | 對表示的意義 |
|---|---|---|
| 普遍逼近（定理 2.2） | 單隱藏層能表示任何連續函數 | 原則上表達能力不是瓶頸；樣本與最佳化效率才是 |
| Cover（定理 2.3） | $p$ 維中約可打散 $2p$ 個點 | 更高維的特徵讓更多標註變成線性——包括雜訊 |
| Johnson–Lindenstrauss（定理 2.4） | $O(\ln n/\varepsilon^2)$ 維即可保留 $n$ 個點的距離 | 嵌入可以很小；疊加是可能的 |
| 群平均（命題 2.2） | 平均產生不變性 | 嵌入袋對順序不變——依任務而定，可能有用也可能致命 |
| Eckart–Young（定理 2.1） | 截斷 SVD 是最佳低秩近似 | 把嵌入看成共現統計的分解 |
