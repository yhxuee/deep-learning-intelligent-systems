---
title: "2.5 一般理論"
description: 普遍逼近定理及其限制、Cover 函數計數定理、隨機投影與 Johnson–Lindenstrauss 引理、以群平均得到不變性與等變線性映射，以及 Eckart–Young 定理的證明。
sidebar: { label: "2.5 一般理論", order: 5 }
---

本頁各項結果回答的是不同問題。普遍逼近問網路*能表示什麼*；Cover 定理計算線性讀出*能分開哪些標註*；Johnson–Lindenstrauss 問投影*能保留多少距離*；群平均建構指定的對稱性；Eckart–Young 則精確解出一項矩陣近似問題。沒有任何一項結果能單獨保證：梯度下降法會從有限資料學到我們想要的表示。

## 2.5.1 普遍逼近——以及它沒有保證的事

**定理 2.2（普遍逼近；[Cybenko 1989](/zh-tw/references/#cybenko-1989)、[Hornik et al. 1989](/zh-tw/references/#hornik-1989)）。** 令 $\sigma$ 為連續的 S 形函數。對緊緻集 $K\subset\mathbb{R}^p$ 上的每個連續函數 $f$ 與每個 $\varepsilon>0$，存在 $H$ 與參數，使單隱藏層網路 $g(x) = \sum_{j=1}^{H}a_j\sigma(w_j^{\top}x + b_j)$ 滿足 $\sup_{x\in K}|f(x) - g(x)|<\varepsilon$。

這個定理談的是緊緻定義域上的連續函數，以及任意指定的近似精度。它沒有說**需要多少**隱藏單元（對指定精度，有些目標函數族所需的單元數會隨 $p$ 指數增加）、需要多少**資料**，也沒有說**梯度下降法**能否找到參數。它只排除一種可能障礙——在無限寬極限下表達能力不足——卻不提供實際學習保證。共享、樣本效率與不變性等問題，全都存在於這項空白中。對 XOR 而言，隱藏層*能*建立分隔特徵；資料與最佳化器是否會引導它找到，則是另一回事。

## 2.5.2 Cover 計數定理：為什麼維度有幫助

**定理 2.3（[Cover 1965](/zh-tw/references/#cover-1965)）。** 令 $\mathbb{R}^p$ 中的 $N$ 個點處於一般位置（其中任意至多 $p$ 個點都線性獨立）。能被通過原點的超平面分開的標註數為
$$
C(N, p) = 2\sum_{k=0}^{p-1}\binom{N-1}{k}.
$$
對仿射分類器 $w^{\top}x + b$，附加一個常數座標，並使用 $C(N, p+1)$。

*證明概要。* 加入第 $N{+}1$ 個點。前 $N$ 個點的一種可分標註，若沒有分隔超平面通過新點，就只能以一種方式延伸到新點；若有，就能以兩種方式延伸；後者對應於這 $N$ 個點投影到 $(p-1)$ 維空間後的可分標註。由此得到遞迴 $C(N+1, p) = C(N, p) + C(N, p-1)$，且 $C(1, p) = 2$，其解就是上式（它是巴斯卡法則的累加）。$\blacksquare$

當 $N\le p$ 時，$C(N,p)=2^N$：一般位置上的每一種標註，都可由通過原點的超平面分開。當 $N=2p$ 時，全部標註中恰好有一半可分。這**不代表** $2p$ 個點可被擊碎。擊碎要求*每一種*標註都能實現；通過原點的線性分隔器之 VC 維度是 $p$，仿射分隔器則是 $p+1$。$2p$ 這個數字代表的是 Cover 可分標註比例的中點。

透過固定的非線性特徵——隨機特徵（[Rahimi & Recht 2007](/zh-tw/references/#rahimi-2007)）、核函數或寬的隨機隱藏層——把輸入映射到更高維空間，可以讓更多標註變得線性可達。這會增加表達容量，也包括擬合雜訊的容量；它本身不能解釋泛化。學習表示追求的是更具選擇性的結果：讓任務相關的標註變簡單，而不是讓每項差異都同樣容易表達。

## 2.5.3 隨機投影保留幾何

一個嵌入需要多少維度，才能讓 $n$ 個不同項目彼此保持距離？遠少於 $n$。

**定理 2.4（Johnson–Lindenstrauss；[Johnson & Lindenstrauss 1984](/zh-tw/references/#johnson-1984)）。** 對任何 $0<\varepsilon<1$ 與任何 $n$ 個點 $x_1,\dots,x_n\in\mathbb R^D$，令隨機線性映射 $R:\mathbb R^D\to\mathbb R^d$ 的元素獨立服從 $N(0,1/d)$。以下不等式以正機率同時成立的一項充分條件是

$$
d>\frac{4\ln n}{\varepsilon^2/2-\varepsilon^3/3}=O(\varepsilon^{-2}\ln n).
$$

$$
(1-\varepsilon)\lVert x_i - x_j\rVert^2 \le \lVert Rx_i - Rx_j\rVert^2 \le (1+\varepsilon)\lVert x_i - x_j\rVert^2 \quad\text{對所有 } i, j .
$$

*證明想法（[Dasgupta & Gupta 2003](/zh-tw/references/#dasgupta-2003)）。* 對固定的單位向量 $u$，$d\lVert Ru\rVert^2$ 服從 $\chi_d^2$。Chernoff 界給出

$$
\Pr\!\left(\left|\lVert Ru\rVert^2-1\right|>\varepsilon\right)
\le2\exp\!\left[-\frac d2\left(\frac{\varepsilon^2}{2}-\frac{\varepsilon^3}{3}\right)\right].
$$

把它套用到少於 $n^2/2$ 個非零的兩兩差向量。根據上述 $d$ 條件，聯集界小於一，所以至少存在一個投影能保留所有配對。零差向量本來就滿足結論。$\blacksquare$

所需維度只隨 $n$ 的對數增加，而且不依賴原始維度 $D$。把這個想法用在許多彼此正交的基底向量上，可以看出：$d$ 維空間能容納指數多個正規化向量，而它們兩兩內積的絕對值都很小。這項幾何事實啟發了**疊加（superposition）**假說：若同時活躍的特徵很少，網路或許能容忍干擾，編碼比維度更多的稀疏特徵（[Elhage et al. 2022](/zh-tw/references/#elhage-2022)）。Johnson–Lindenstrauss 證明的是幾何上的可能性，不是訓練後的網路一定會找到這種編碼，也不保證它能無誤使用。

## 2.5.4 以平均得到不變性；等變映射

令有限群 $\mathcal{G}$ 作用在輸入上（例如 $T$ 個位置的 $T!$ 種置換，或長度 $T$ 訊號的 $T$ 種循環位移）。

**命題 2.2（群平均）。** 對任何函數 $\phi$，平均 $\bar\phi(x) = \frac{1}{|\mathcal G|}\sum_{g\in\mathcal G}\phi(g\cdot x)$ 是不變的：對每個 $h\in\mathcal G$，$\bar\phi(h\cdot x) = \bar\phi(x)$。

*證明。* $\bar\phi(h\cdot x) = \frac{1}{|\mathcal G|}\sum_g\phi(gh\cdot x)$，而當 $g$ 跑遍 $\mathcal G$ 時，$gh$ 也跑遍 $\mathcal G$。$\blacksquare$

把詞嵌入加總成「詞袋（bag of words）」是一種簡單的置換不變聚合，與平均構造密切相關：重新排列輸入不會改變總和。它可能適合主題分類；若詞序決定答案，例如「狗咬人」與「人咬狗」，就會造成災難（§2.7.1）。

若對所有 $g$ 都有 $F(g\cdot x)=g\cdot F(x)$，線性映射 $F$ 就是**等變的**。把模型限制在等變映射中，可以大幅減少參數量與假設類別；當對稱性假設正確時，這可能改善樣本效率，但改善不會自動發生。書中有兩個反覆出現的例子。對長度 $T$ 訊號的循環位移，等變線性映射恰好是循環卷積——$T$ 個參數，而非 $T^2$ 個（單元 04，定理 4.2）。對 $T\ge2$ 個位置、單一通道的置換（$X\in\mathbb R^T$），它們恰好是 $X\mapsto\lambda X+\gamma\mathbf1\mathbf1^{\top}X$——兩個參數（單元 04，定理 4.3）。若有 $d_{\text{in}}$ 個輸入通道與 $d_{\text{out}}$ 個輸出通道，則形式為 $X\mapsto XA+\mathbf1\mathbf1^{\top}XB$，其中 $A,B\in\mathbb R^{d_{\text{in}}\times d_{\text{out}}}$，共 $2d_{\text{in}}d_{\text{out}}$ 個參數（[Zaheer et al. 2017](/zh-tw/references/#zaheer-2017)）。把總和改成平均，只會重新縮放 $B$。

這個置換等變線性函數族的位置結構仍很貧乏：每個位置只看見自己，以及所有位置的一項彙總。注意力會提供非線性、依內容而變的聚合方式。從對稱性建構網路的更廣泛理論，就是幾何深度學習（[Bronstein et al. 2021](/zh-tw/references/#bronstein-2021)）。

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
| Cover（定理 2.3） | $N=2p$ 時，一般位置標註中有一半可分；VC 維度仍是 $p$ | 更高維的特徵讓更多標註變成線性——包括雜訊——但不能擊碎 $2p$ 個點 |
| Johnson–Lindenstrauss（定理 2.4） | $O(\ln n/\varepsilon^2)$ 維足以保留 $n$ 個點的兩兩距離 | 小型嵌入可以保留有限幾何；這會啟發疊加，但不能證明學習出的疊加 |
| 群平均（命題 2.2） | 平均產生不變性 | 嵌入袋對順序不變——依任務而定，可能有用也可能致命 |
| Eckart–Young（定理 2.1） | 截斷 SVD 是 Frobenius 範數下的最佳低秩近似 | 精確理解低秩表格近似的視角，不代表每種嵌入目標 |
