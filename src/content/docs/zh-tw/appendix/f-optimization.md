---
title: "F · 最佳化"
description: 凸集與凸函數及其一階與二階刻畫；平滑性、強凸性與 Polyak–Łojasiewicz 條件；梯度下降、動量與隨機梯度的收斂速率，並證明凸情形的速率；牛頓法與預條件；拉格朗日乘數、KKT 條件與對偶；最大熵與吉布斯變分原理。
sidebar: { label: "F · 最佳化", order: 6 }
---

單元 03 從第一原理推導梯度下降法並證明它的主要保證：下降引理、不需凸性也能收斂到駐點、二次函數上的條件數速率、強凸性下的線性收斂、隨機梯度的雜訊下限，以及動量的速率（§3.5.2–§3.5.7）。本附錄提供這些證明所假設的凸分析背景，把各種速率集中在一張表中，證明單元 03 沒有證明的那一個速率，並補上受限最佳化——拉格朗日乘數、KKT 條件與對偶——本書用它們推導最大概似估計、softmax 輸出，以及 KL 正則化後訓練的最佳解。例題中的數字由 `labs/test_appendices.py` 檢查。

| 小節 | 用於 |
|---|---|
| F.1–F.2 凸性、平滑性 | 01（對數損失）、03（下降引理）、06（計算最佳分配） |
| F.3 收斂速率 | 03、05（穩定遞迴）、06 |
| F.4 牛頓法、預條件 | 03（Adam）、06（二階方法） |
| F.5 拉格朗日與 KKT | 01（MLE 二元語法）、06（Chinchilla 分配）、09（KL 最佳解） |
| F.6 最大熵、吉布斯 | 01、04（softmax）、09（傾斜的參考分布） |

## F.1 凸集與凸函數

若集合 $C$ 包含任兩點之間的線段，它就是**凸的**。凸集上的函數 $f$ 若滿足
$$
f(\lambda x + (1 - \lambda)y)\le\lambda f(x) + (1 - \lambda)f(y)\qquad\text{對所有 }x, y\text{ 與 }\lambda\in[0, 1],
$$
它就是**凸函數**。

**命題 F.1（刻畫）。** 對開凸集上的可微函數 $f$，下列等價：(i) $f$ 是凸的；(ii) 對所有 $x, y$，$f(y)\ge f(x) + \nabla f(x)^{\top}(y - x)$——函數位於它的切平面之上；(iii) $(\nabla f(x) - \nabla f(y))^{\top}(x - y)\ge0$——梯度是單調的。若 $f$ 二次可微，它們也等價於 (iv) 處處 $\nabla^2f(x)\succeq0$。

*(i)⇒(ii) 的證明。* 由凸性，$f(x + \lambda(y - x))\le f(x) + \lambda(f(y) - f(x))$。減去 $f(x)$、除以 $\lambda$ 並令 $\lambda\to0$：$\nabla f(x)^{\top}(y - x)\le f(y) - f(x)$。其他蘊含關係類似（[Boyd & Vandenberghe 2004](/zh-tw/references/#boyd-2004)，§3.1）。$\blacksquare$

(ii) 最有用的推論：**凸函數的每個駐點都是全域極小**，因為 $\nabla f(x^\star) = 0$ 給出對所有 $y$ 都有 $f(y)\ge f(x^\star)$。

**保持凸性的運算。** 非負和；與仿射映射的合成 $f(Ax + b)$；凸函數的逐點最大值；以及 $\mathrm{LSE}(z) = \ln\sum_je^{z_j}$（附錄 B，命題 B.3）。所以線性 softmax 模型的交叉熵在其權重上是凸的，而二元語法的最大概似問題（單元 01，§1.3.3）是有唯一解的凸問題。深層網路在其參數上不是凸的：凸損失與非線性網路的合成破壞了凸性，這就是為什麼單元 03 在沒有凸性時的保證比較弱。

## F.2 平滑性、強凸性與 PL 條件

- 若 $\lVert\nabla f(x) - \nabla f(y)\rVert\le L\lVert x - y\rVert$，$f$ 就是 **$L$-平滑的**。對二次可微的 $f$，這等於 $-LI\preceq\nabla^2f\preceq LI$，並蘊含二次上界 $f(y)\le f(x) + \nabla f(x)^{\top}(y - x) + \frac L2\lVert y - x\rVert^2$（附錄 B.3）。
- 若 $f(y)\ge f(x) + \nabla f(x)^{\top}(y - x) + \frac\mu2\lVert y - x\rVert^2$，$f$ 就是 **$\mu$-強凸的**；對二次可微的 $f$，即 $\nabla^2f\succeq\mu I$。強凸函數有唯一的極小點，並滿足 $f(x) - f^\star\le\frac1{2\mu}\lVert\nabla f(x)\rVert^2$。
- 最後這個不等式，**Polyak–Łojasiewicz（PL）條件**，可以在*沒有*凸性的情況下成立，而它正是梯度下降法線性收斂證明所需要的全部。在插值解的流形附近，過參數化模型常常滿足局部的 PL 條件，這是它們儘管不凸、訓練卻收斂得很快的一種解釋。

比值 $\kappa = L/\mu$ 是問題的**條件數**；對二次函數 $\frac12x^{\top}Hx$，它就是 $H$ 的條件數（附錄 A.7）。

## F.3 收斂速率

對梯度下降法 $x_{k+1} = x_k - \eta\nabla f(x_k)$，$f^\star = \inf f$：

| 假設 | 步長 | $k$ 步之後的保證 | 證明 |
|---|---|---|---|
| $L$-平滑 | $1/L$ | $\min_{j<k}\lVert\nabla f(x_j)\rVert^2\le2L(f(x_0) - f^\star)/k$ | 單元 03，§3.5.3 |
| $L$-平滑、凸 | $1/L$ | $f(x_k) - f^\star\le L\lVert x_0 - x^\star\rVert^2/(2k)$ | 定理 F.2 |
| $L$-平滑、$\mu$-強凸（或 PL） | $1/L$ | $f(x_k) - f^\star\le(1 - 1/\kappa)^k(f(x_0) - f^\star)$ | 單元 03，§3.5.5 |
| 二次函數，特徵值在 $[\mu, L]$ | $2/(L + \mu)$ | $\lVert x_k - x^\star\rVert\le\big(\frac{\kappa - 1}{\kappa + 1}\big)^k\lVert x_0 - x^\star\rVert$ | 單元 03，§3.5.4 |
| 二次函數，調好的重球動量 | — | 每步速率 $\frac{\sqrt\kappa - 1}{\sqrt\kappa + 1}$（漸近） | 單元 03，§3.5.7 |
| 隨機梯度，變異數 $\sigma^2$，強凸 | $\eta_k\propto1/k$ | $\mathbb{E}f(x_k) - f^\star = O(1/k)$ | [Bottou et al. 2018](/zh-tw/references/#bottou-2018) |

**定理 F.2（凸情形的速率）。** 若 $f$ 是凸的且 $L$-平滑，極小點為 $x^\star$，則步長 $\eta = 1/L$ 的梯度下降法滿足 $f(x_k) - f^\star\le\frac{L\lVert x_0 - x^\star\rVert^2}{2k}$。

*證明。* 記 $g_j = \nabla f(x_j)$。下降引理給出 $f(x_{j+1})\le f(x_j) - \frac1{2L}\lVert g_j\rVert^2$，凸性給出 $f(x_j)\le f^\star + g_j^{\top}(x_j - x^\star)$。合起來，
$$
f(x_{j+1}) - f^\star\le g_j^{\top}(x_j - x^\star) - \tfrac1{2L}\lVert g_j\rVert^2 = \tfrac L2\big(\lVert x_j - x^\star\rVert^2 - \lVert x_{j+1} - x^\star\rVert^2\big),
$$
其中等式來自展開 $\lVert x_{j+1} - x^\star\rVert^2 = \lVert x_j - x^\star - g_j/L\rVert^2$。對 $j = 0,\dots,k-1$ 求和，右邊伸縮相消，至多為 $\frac L2\lVert x_0 - x^\star\rVert^2$；由於 $f(x_j)$ 不增，左邊至少為 $k(f(x_k) - f^\star)$。$\blacksquare$

**例題 F.1。** 對 $f(x, y) = \frac12(x^2 + 10y^2)$，$\mu = 1$、$L = 10$、$\kappa = 10$。每步誤差的收縮倍數：$\eta = 1/L$ 時為 $1 - 1/\kappa = 0.9$；最佳固定步長 $2/11$ 時為 $(\kappa - 1)/(\kappa + 1) = 0.818$；調好的重球動量為 $(\sqrt{10} - 1)/(\sqrt{10} + 1) = 0.519$。要把誤差縮小 $10^6$ 倍，分別約需 131、69 與 21 步。在 $\kappa = 10^4$ 時，前兩者約需 140 000 與 69 000 步，動量約需 690 步——這就是動量與預條件對病態條件問題之所以重要的原因（單元 03，§3.7.3）。

**隨機梯度。** 以變異數為 $\sigma^2$ 的不偏梯度估計與固定步長 $\eta$，梯度下降法收斂到極小值的一個鄰域，其大小隨 $\eta\sigma^2$ 增長，而不是收斂到極小值本身（單元 03，§3.5.6）。要精確收斂，需要滿足 **Robbins–Monro 條件** $\sum_k\eta_k = \infty$、$\sum_k\eta_k^2 < \infty$ 的遞減步長（[Robbins & Monro 1951](/zh-tw/references/#robbins-1951)）——前者讓迭代能走任意遠的距離，後者讓累積的雜訊有有限的變異數。實務上的學習率排程（暖身後再以餘弦或線性衰減；單元 03，§3.8.3）是在這些要求與有限預算之間的工程折衷。

## F.4 牛頓法與預條件

**牛頓法**精確地最小化局部二次模型：$x_{k+1} = x_k - \nabla^2f(x_k)^{-1}\nabla f(x_k)$。在二次函數上，無論條件如何，它一步就收斂。在黑塞矩陣正定且 Lipschitz 的極小點附近，它**二次收斂**：誤差大致每步平方一次。

**例題 F.2。** 最小化 $f(x) = x - \ln x$（極小點在 $x = 1$）。牛頓更新為 $x_{k+1} = 2x_k - x_k^2$，從 $x_0 = 0.5$ 出發，誤差 $1 - x_k$ 依序為 $0.25,\ 0.0625,\ 0.0039,\ 1.5\times10^{-5},\ 2.3\times10^{-10}$——每一個都是前一個的平方。

對有數百萬參數的網路，牛頓法太昂貴（$O(N^2)$ 記憶體、$O(N^3)$ 求解），而且會被吸向黑塞矩陣不定的鞍點。**預條件梯度下降** $x_{k+1} = x_k - \eta P^{-1}\nabla f(x_k)$ 以較低成本保留這個想法：若 $P$ 近似黑塞矩陣，有效條件數就變成 $P^{-1/2}\nabla^2fP^{-1/2}$ 的條件數。Adam 除以每個梯度座標的移動均方根，就是一個**對角**預條件器（單元 03，§3.3.8 與 §3.5.8）；Shampoo 之類的結構化預條件器則逐塊近似黑塞矩陣（單元 03，§3.8.6）。

## F.5 受限最佳化：拉格朗日乘數、KKT 與對偶

**等式限制。** 要在 $g_i(x) = 0$ 的限制下最小化 $f(x)$，構造**拉格朗日函數** $\mathcal L(x, \lambda) = f(x) + \sum_i\lambda_ig_i(x)$。在限制梯度線性獨立的局部極小點，存在乘數使
$$
\nabla f(x^\star) + \sum_i\lambda_i\nabla g_i(x^\star) = 0 .
$$
*理由。* 任何在一階上保持限制成立的方向 $d$，對所有 $i$ 都有 $\nabla g_i^{\top}d = 0$；在極小點，對所有這樣的 $d$ 都有 $\nabla f^{\top}d = 0$；所以 $\nabla f$ 落在 $\nabla g_i$ 張成的空間中。$\blacksquare$

*例子：最大概似類別分布。* 在 $\sum_kp_k = 1$ 的限制下最大化 $\sum_kc_k\ln p_k$。駐點條件給出 $c_k/p_k = \lambda$，所以 $p_k\propto c_k$，而限制確定 $p_k = c_k/n$——即單元 01 命題 1.1。

**不等式限制（KKT）。** 要在 $g_i(x)\le0$ 與 $h_j(x) = 0$ 的限制下最小化 $f(x)$，**Karush–Kuhn–Tucker 條件**為：駐點條件 $\nabla f + \sum_i\lambda_i\nabla g_i + \sum_j\nu_j\nabla h_j = 0$；原始可行性；對偶可行性 $\lambda_i\ge0$；以及**互補鬆弛** $\lambda_ig_i(x) = 0$——不緊的限制其乘數為零。對滿足某個限制條件資格（例如存在嚴格可行點——Slater 條件）的凸問題，KKT 條件是全域極小的必要且充分條件。

**對偶。** **對偶函數** $q(\lambda, \nu) = \inf_x\mathcal L(x, \lambda, \nu)$ 是凹的，而且對每個 $\lambda\ge0$，它都是最佳值的下界（**弱對偶**）：對可行的 $x$，$\mathcal L(x, \lambda, \nu)\le f(x)$。在 Slater 條件下，最佳下界是緊的（**強對偶**）。對偶把單元 06 的計算最佳分配——在 $6ND = C$ 的限制下最小化損失 $L(N, D)$——變成一行計算：在最佳點，參數與資料每 FLOP 的邊際損失下降相等（單元 06，命題 6.2）。

## F.6 最大熵與吉布斯變分原理

**最大熵。** 在有限集合上、期望值被規定為 $\mathbb{E}_p[f_j(x)] = m_j$ 的分布中，熵最大者的形式為
$$
p(x) = \frac{\exp\big(\sum_j\lambda_jf_j(x)\big)}{Z(\lambda)},
$$
乘數 $\lambda_j$ 被選來滿足限制。*推導。* $-\sum p\ln p$ 加上這些限制與 $\sum p = 1$ 的拉格朗日函數，在 $-\ln p(x) - 1 + \sum_j\lambda_jf_j(x) + \nu = 0$ 時為駐點。$\blacksquare$ 沒有限制時這是均勻分布；對變異數固定的實數變數加上 $\mathbb{E}[x]$ 的限制，就是高斯分布（附錄 D.4）。

**例題 F.3。** 已知一顆六面骰的平均是 4.5 而不是 3.5。最大熵分布為 $p_k\propto e^{\lambda k}$，$\lambda = 0.371$：$(0.054, 0.079, 0.114, 0.165, 0.240, 0.348)$，熵為 1.614 nats——與限制相容、承諾最少的分布，略低於均勻骰子的 $\ln6 = 1.792$。

**吉布斯變分原理。** 對任何分數 $r(x)$ 與溫度 $\tau > 0$，
$$
\max_p\ \Big(\mathbb{E}_p[r(x)] + \tau H(p)\Big) = \tau\ln\sum_xe^{r(x)/\tau},\qquad\text{在 }p^\star(x) = \frac{e^{r(x)/\tau}}{\sum_ye^{r(y)/\tau}}\text{ 處達到。}
$$
*證明。* $\mathbb{E}_p[r] + \tau H(p) = \tau\ln Z - \tau\,\mathrm{KL}(p\,\Vert\,p^\star)$，而 KL 在 $p = p^\star$ 時最小為零。$\blacksquare$

本書三個核心公式都是這個原理的化身。**Softmax** 是 logits 的熵正則化 argmax，它的對數正規化常數就是 LSE（附錄 B.5）；溫度就是 $\tau$。**注意力**權重是查詢–鍵分數上的 softmax（單元 04）。而把熵換成對參考分布的 KL 散度，$\mathbb{E}_p[r] - \beta\,\mathrm{KL}(p\,\Vert\,\pi_{\text{ref}})$，就得到 KL 正則化後訓練的傾斜最佳解 $\pi^\star\propto\pi_{\text{ref}}e^{r/\beta}$（單元 09，命題 9.1）——證明相同，只是以 $\pi_{\text{ref}}$ 取代均勻分布。

## 習題

**F.1** 證明 $f(x) = \lVert Ax - b\rVert^2$ 是凸的，並以 $A$ 的奇異值表示 $L$ 與 $\mu$。

**F.2** 證明凸且 $L$-平滑的函數滿足 $f(x) - f^\star\ge\frac1{2L}\lVert\nabla f(x)\rVert^2$。

**F.3** 對例題 F.1 以 $\eta = 1/L$ 從 $(1, 1)$ 開始做梯度下降。寫出 $x_k$ 與 $y_k$ 的封閉形式。哪一個座標限制了收斂？

**F.4** 以拉格朗日乘數求在 $a^{\top}x = 1$ 限制下 $\frac12\lVert x\rVert^2$ 的最小值，並把它詮釋為投影。

**F.5** 證明牛頓法在可逆線性變數變換 $x = Ty$ 下不變，而梯度下降法不是。

**F.6** ★ 以拉格朗日乘數推導 $L(N, D) = E + AN^{-\alpha} + BD^{-\beta}$ 在 $6ND = C$ 限制下的計算最佳分配，並與單元 06 命題 6.2 核對。

**解答。** **F.1** $L = 2\sigma_{\max}^2$、$\mu = 2\sigma_{\min}^2$（當 $A$ 行滿秩時）。**F.3** $x_k = 0.9^k$，一步之後 $y_k = 0$：平坦方向 $x$ 限制了收斂。**F.4** $x^\star = a/\lVert a\rVert^2$，超平面上離原點最近的點。

## 延伸閱讀

[Boyd & Vandenberghe 2004](/zh-tw/references/#boyd-2004) 講凸性與對偶；[Nesterov 2018](/zh-tw/references/#nesterov-2018) 講收斂速率與下界；[Bottou et al. 2018](/zh-tw/references/#bottou-2018) 講機器學習中的隨機最佳化；[Deisenroth et al. 2020](/zh-tw/references/#deisenroth-2020) 第 7 章為入門。
