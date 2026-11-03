---
title: "1.5 一般理論"
description: 有限模型族的均勻收斂與 VC 維度、Hoeffding 不等式的證明、偏差–變異數分解、沒有免費午餐定理，以及古典容量論證為何無法解釋現代網路。
sidebar: { label: "1.5 一般理論", order: 5 }
---

§1.3 一次只分析一個模型。學習理論問一個更強的問題：如果我們在整個模型族中選出訓練損失最小的模型，它的保留資料損失什麼時候會接近訓練損失？本節給出古典的答案、證明其中關鍵的部分，然後說明它們在哪裡不再能解釋實務。

## 1.5.1 均勻收斂

令 $\mathcal{H}$ 為一個預測器族，$\ell\in[0,1]$ 為有界損失。ERM 回傳 $\hat h = \arg\min_{h\in\mathcal H}\hat L(h)$。因為 $\hat h$ 取決於訓練樣本，Hoeffding 不等式不能直接套用在它身上。它可以套用在每一個*固定的* $h$ 上；如果這個界對所有 $h\in\mathcal H$ **同時**成立，那它對 $\hat h$ 也成立。

**定理 1.3（有限模型族）。** 若 $|\mathcal H| = K$，則對大小為 $n$ 的 i.i.d. 樣本，以至少 $1-\delta$ 的機率，對值域在 $[0, 1]$ 的損失，每個 $h\in\mathcal H$ 都滿足
$$
\big|L(h) - \hat L(h)\big| \le \varepsilon,\qquad \varepsilon = \sqrt{\frac{\ln K + \ln(2/\delta)}{2n}} = \sqrt{\frac{\ln(2K/\delta)}{2n}} .
$$

*證明。* 對每個固定的 $h$，雙尾 Hoeffding 不等式（定理 1.2）給出 $\Pr(|L(h) - \hat L(h)|\ge\varepsilon)\le 2e^{-2n\varepsilon^2} = \delta/K$。對 $K$ 個成員取聯集界（§1.3.6），使*任一*偏差達到 $\varepsilon$ 的機率至多為 $\delta$。$\blacksquare$

下面兩個方向都需要：上偏差界定所選 $\hat h$ 的風險，下偏差界定比較對象 $h^\star$ 的經驗風險。（單尾陳述只需 $\ln(K/\delta)$，但無法支持這個系理。）

**系理 1.2。** 令 $h^\star$ 為 $\mathcal H$ 中母體風險最小者。以至少 $1-\delta$ 的機率，$L(\hat h) \le L(h^\star) + 2\sqrt{(\ln K + \ln(2/\delta))/(2n)}$。

*證明。* $L(\hat h)\le \hat L(\hat h) + \varepsilon \le \hat L(h^\star) + \varepsilon \le L(h^\star) + 2\varepsilon$，在兩個方向上各使用一次雙尾均勻界，並使用 $\hat L(\hat h)\le\hat L(h^\star)$（ERM）。$\blacksquare$

有兩個特點值得強調。額外風險只透過 $\ln K$——一種**容量（capacity）**量度——依賴模型族，並以 $1/\sqrt{n}$ 縮小。而且論證完全沒有用到 ERM 選了*哪一個* $h$；它對族中最差的成員也成立。這是它的優點（假設很少），也是它的弱點，§1.5.4 會說明。

**無限模型族。** 二元語法表的參數是連續的，所以 $K$ 是無限大。古典的補救方法是以組合方式量測容量。對二元分類，$\mathcal H$ 的 **VC 維度** $d_{\mathrm{VC}}$ 是 $\mathcal H$ 能以全部 $2^m$ 種方式標註（「打散，shatter」）的最大點數。$\mathbb{R}^k$ 中的線性分類器有 $d_{\mathrm{VC}} = k+1$。統計學習的基本定理說：一個模型族可由 ERM 學習，若且唯若其 VC 維度有限，且額外風險的量級為 $\sqrt{(d_{\mathrm{VC}} + \ln(1/\delta))/n}$（[Vapnik 1998](/zh-tw/references/#vapnik-1998)；[Shalev-Shwartz & Ben-David 2014](/zh-tw/references/#shalev-shwartz-2014)，第 6 章）。上面有限族的證明就是範本：把 $\ln K$ 換成計算該族在 $n$ 個點上*實際不同*行為數量的量。

### ★ 進階：Hoeffding 不等式的證明

令 $Z_i\in[a,b]$ 彼此獨立，$\mu_i = \mathbb{E}Z_i$，$S = \sum_i(Z_i - \mu_i)$。對任何 $s>0$，把 Markov 不等式套用在 $e^{sS}$ 上，得到
$$
\Pr(S\ge n\varepsilon) \le e^{-sn\varepsilon}\,\mathbb{E}[e^{sS}] = e^{-sn\varepsilon}\prod_i\mathbb{E}[e^{s(Z_i-\mu_i)}].
$$
**Hoeffding 引理**為每個因子給出上界：平均為零的變數 $X\in[a',b']$ 滿足 $\mathbb{E}[e^{sX}]\le e^{s^2(b'-a')^2/8}$。（證明概要：由 $e^{sx}$ 的凸性，$e^{sX}\le\frac{b'-X}{b'-a'}e^{sa'} + \frac{X-a'}{b'-a'}e^{sb'}$；取期望值，把結果寫成 $e^{\varphi(u)}$，其中 $u = s(b'-a')$，並檢查 $\varphi(0) = \varphi'(0) = 0$、$\varphi''\le 1/4$，由泰勒定理得 $\varphi(u)\le u^2/8$。）因此 $\Pr(S\ge n\varepsilon)\le\exp(-sn\varepsilon + ns^2(b-a)^2/8)$。在 $s = 4\varepsilon/(b-a)^2$ 處對 $s$ 取最小值，得到 $\exp(-2n\varepsilon^2/(b-a)^2)$。下尾完全相同；因子 2 來自兩尾相加。$\blacksquare$

## 1.5.2 偏差–變異數分解

對平方損失，一個學習*程序*的期望誤差（對訓練集取平均）有精確的分解。令 $y = f(x) + \varepsilon$，$\mathbb{E}\varepsilon = 0$、$\mathrm{Var}\,\varepsilon = \sigma^2$，並令 $\hat f_{\mathcal D}$ 為在隨機訓練集 $\mathcal{D}$ 上訓練出的預測器。在固定的測試點 $x$，記 $\bar f(x) = \mathbb{E}_{\mathcal D}[\hat f_{\mathcal D}(x)]$。

**命題 1.2。**
$$
\mathbb{E}_{\mathcal D,\varepsilon}\big[(y - \hat f_{\mathcal D}(x))^2\big] = \underbrace{\sigma^2}_{\text{雜訊}} + \underbrace{(f(x) - \bar f(x))^2}_{\text{偏差}^2} + \underbrace{\mathbb{E}_{\mathcal D}\big[(\hat f_{\mathcal D}(x) - \bar f(x))^2\big]}_{\text{變異數}} .
$$

*證明。* 寫成 $y - \hat f = \varepsilon + (f - \bar f) + (\bar f - \hat f)$，平方後取期望值。交叉項消失：$\varepsilon$ 與 $\mathcal D$ 獨立且平均為零；$f - \bar f$ 是常數；而由 $\bar f$ 的定義，$\mathbb{E}_{\mathcal D}[\bar f - \hat f] = 0$。$\blacksquare$

**雜訊**項是不可約減的下限（相當於系理 1.1 中的熵）。**偏差**量測*平均*訓練出的模型離真相多遠——模型族越靈活，偏差越小。**變異數**量測訓練出的模型隨訓練集不同而改變多少——它隨靈活度增加而變大，隨資料增加而變小。§1.2.7 的平滑就是一種偏差–變異數取捨：α → ∞ 時變異數為零（均勻模型忽略資料）而偏差很大；α → 0 時在大量資料的極限下偏差為零，但在小樣本上變異數很高。由這個圖像得出的古典預測是：測試誤差隨容量增加呈 U 形（[Geman et al. 1992](/zh-tw/references/#geman-1992)；[Hastie et al. 2009](/zh-tw/references/#hastie-2009)，§7.3）。

## 1.5.3 沒有免費午餐

有沒有某種學習程序在每個問題上都是最好的？沒有。

**定理 1.4（沒有免費午餐，有限版本）。** 令 $\mathcal{X}$ 為有限集合，標籤為二元，並固定任一學習者。對*全部* $2^{|\mathcal X|}$ 個目標函數 $f:\mathcal X\to\{0,1\}$ 取平均，訓練集以 $f$ 標註。則在任何不在訓練集中的點上，學習者的期望準確率恰好是 $1/2$。

*證明。* 固定訓練輸入集 $S$ 與測試點 $x\notin S$。學習者在 $x$ 的預測只取決於 $S$ 的標籤。對 $S$ 的每一種標註，與之相容的目標函數中，恰好一半有 $f(x) = 1$、一半有 $f(x) = 0$，因為 $x$ 上的值不受限制。所以無論學習者預測什麼，它恰好對一半的目標函數是對的。$\blacksquare$

這個定理並不是說學習不可能；它說的是，**若不對哪些目標較可能做出假設**，學習就不可能（[Wolpert 1996](/zh-tw/references/#wolpert-1996)）。每個成功的學習者，都是在某些問題上獲勝，代價是在它隱含地認為不太可能的其他問題上落敗。平滑性、局部性、稀疏性、對稱性，以及「相似的輸入有相似的輸出」，都是這樣的假設——**歸納偏差**。單元 02 談如何透過表示來選擇它們；單元 04 由對稱性論證推導出其中之一（注意力的置換等變性）。

## 1.5.4 古典理論停在哪裡

把定理 1.3 的邏輯套用在現代網路上。一個有數百萬參數的網路能完全擬合隨機標籤（§1.1），所以它在 $n = 50{,}000$ 個點上的有效容量，至少等於任意標籤的資訊量；任何「訓練誤差 + 容量項」形式的界於是都變得空洞——它超過 100% 的錯誤率。然而同一個網路以真實標籤訓練時類推得很好。過去十年的三個觀察讓這個謎題更加尖銳：

- **記住隨機標籤**（[Zhang et al. 2017](/zh-tw/references/#zhang-2017)）：標準架構與最佳化器能擬合隨機標籤與隨機像素；顯式正則化對類推既非必要也非充分。
- **雙重下降（double descent）**（[Belkin et al. 2019](/zh-tw/references/#belkin-2019)；[Nakkiran et al. 2020](/zh-tw/references/#nakkiran-2020)）：當模型大小增加、越過恰好能內插訓練資料的那一點時，測試誤差先上升（古典的 U 形），然後*再次下降*，常常低於古典的最小值。同樣的形狀也出現在以訓練時間或資料量為橫軸的情況。
- **良性過擬合（benign overfitting）**（[Bartlett et al. 2020](/zh-tw/references/#bartlett-2020)）：在參數遠多於樣本的線性迴歸中，最小範數的內插解完全擬合了雜訊，卻仍幾乎最佳地預測——前提是資料的共變異數有許多小方向，能無害地吸收雜訊。

共通的線索是：最壞情況的容量是錯誤的量。重要的是訓練程序找到的是**哪一個**內插解。從小初始值出發的梯度下降法，對簡單的解有*隱式偏差（implicit bias）*（在線性模型中是最小範數；單元 03 §3.5 與單元 06 會回到這一點），而真實資料的結構讓簡單的解存在。類推是資料、架構與最佳化器的共同性質——而不只是參數量的性質。精確的理論仍是開放的研究領域（§1.8）；*不*開放的，是本單元賴以建立的實務規則：**在來自同一來源、從未用於任何決策的保留資料上量測。**

## 1.5.5 理論摘要

| 結果 | 說了什麼 | 假設 |
|---|---|---|
| 系理 1.1 | 對數損失 = 不可約減的熵 + KL 額外損失 | — |
| Hoeffding（定理 1.2） | 保留資料估計誤差為 $O(1/\sqrt n)$ | i.i.d.、有界損失、模型事先固定 |
| 均勻收斂（定理 1.3） | ERM 的額外風險為 $O(\sqrt{\ln K / n})$ | i.i.d.、有界損失、有限或 VC 維度有限的模型族 |
| 偏差–變異數（命題 1.2） | 誤差 = 雜訊 + 偏差² + 變異數 | 平方損失 |
| 沒有免費午餐（定理 1.4） | 沒有學習者在所有目標上都最好 | 對目標均勻平均 |
| 現代證據（§1.5.4） | 容量界對大型網路是空洞的；最佳化器找到什麼才重要 | 實驗性，部分理論 |
