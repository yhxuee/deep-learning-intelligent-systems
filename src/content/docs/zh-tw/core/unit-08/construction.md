---
title: "8.3 數學建構"
description: 祖先抽樣的精確性、核心抽樣、變數變換公式與正規化流、潛變數模型與證據下界、把擴散看成去雜訊，以及流匹配與連續方程式和精確的高斯速度場。
sidebar: { label: "8.3 數學建構", order: 3 }
---

## 8.3.1 祖先抽樣及其修改

**命題 8.1。** 以條件分布 $q(x_t\mid x_{<t})$ 做祖先抽樣，回傳序列 $x_{1:T}$ 的機率為 $q(x_{1:T}) = \prod_t q(x_t\mid x_{<t})$。

*證明。* 各步是只依賴目前為止序列的抽樣；某個特定序列的機率是各步機率的乘積，由連鎖律它等於 $q(x_{1:T})$。$\blacksquare$

溫度、top-$k$ 與核心抽樣把每個條件分布換成修改過的 $\tilde q(x_t\mid x_{<t})$，因此恰好從*另一個*序列分布 $\prod_t\tilde q(x_t\mid x_{<t})$ 抽樣。門檻 $p$ 的核心是 $S_p = \{k_1,\dots,k_m\}$，即依機率遞減排列的最可能符號，$m$ 是使 $\sum_{i\le m}q(k_i)\ge p$ 的最小索引；截斷後的分布在 $S_p$ 上為 $q(k)/\sum_{j\in S_p}q(j)$，其餘為零。注意像 $\tau < 1$ 這樣的逐步修改，*不等於*從重新正規化的 $q(x_{1:T})^{1/\tau}$ 抽樣序列；前者是局部的，後者是全域的。

## 8.3.2 變數變換與正規化流

令 $\mathbb{R}^n$ 中的 $z\sim p_Z$，且 $x = g(z)$，$g$ 可逆且可微，$f = g^{-1}$。則

$$
p_X(x) = p_Z(f(x))\,\big|\det J_f(x)\big| .
$$

一維中：若 $x = \mu + \sigma z$、$z\sim N(0,1)$，則 $p_X(x) = \frac{1}{\sigma}\phi\big(\frac{x-\mu}{\sigma}\big)$——$1/\sigma$ 就是雅可比，拉伸空間會讓密度變稀。**正規化流**組合許多雅可比行列式很便宜的簡單可逆映射——例如耦合層：以一個依另一半座標決定參數的仿射映射轉換一半座標，使雅可比為三角矩陣（[Dinh et al. 2017](/zh-tw/references/#dinh-2017)；[Rezende & Mohamed 2015](/zh-tw/references/#rezende-2015)）。對數概似 $\ln p_X(x) = \ln p_Z(f(x)) + \sum_\ell\ln|\det J_{f_\ell}|$ 於是是精確的，可以直接最大化——單元 01 的原則用在連續密度上。代價在架構上：每一層都必須可逆，且行列式容易計算。

## 8.3.3 潛變數與證據下界

**潛變數模型**先抽 $z\sim p(z)$ 再抽 $x\sim p_\theta(x\mid z)$，所以 $p_\theta(x) = \int p_\theta(x\mid z)p(z)\,dz$——一個我們算不出來的積分。對任何分布 $q_\phi(z\mid x)$，

$$
\ln p_\theta(x) = \underbrace{\mathbb{E}_{q_\phi(z\mid x)}\big[\ln p_\theta(x\mid z)\big] - \mathrm{KL}\big(q_\phi(z\mid x)\,\Vert\,p(z)\big)}_{\text{ELBO}(x)} + \mathrm{KL}\big(q_\phi(z\mid x)\,\Vert\,p_\theta(z\mid x)\big).
$$

*推導。* 寫成 $\ln p_\theta(x) = \mathbb{E}_{q}[\ln p_\theta(x)] = \mathbb{E}_q\big[\ln\frac{p_\theta(x,z)}{p_\theta(z\mid x)}\big] = \mathbb{E}_q\big[\ln\frac{p_\theta(x,z)}{q(z\mid x)}\big] + \mathbb{E}_q\big[\ln\frac{q(z\mid x)}{p_\theta(z\mid x)}\big]$，再展開 $p_\theta(x,z) = p_\theta(x\mid z)p(z)$。$\blacksquare$

最後一項是 KL 散度，所以非負：ELBO 是對數概似的下界，當 $q_\phi$ 等於真正的後驗時取等號。**變分自編碼器**以最大化 ELBO 同時訓練編碼器 $q_\phi(z\mid x)$ 與解碼器 $p_\theta(x\mid z)$，並使用*重新參數化* $z = \mu_\phi(x) + \sigma_\phi(x)\odot\varepsilon$、$\varepsilon\sim N(0, I)$，讓梯度能穿過抽樣步驟（[Kingma & Welling 2014](/zh-tw/references/#kingma-2014)）。第一項獎勵重建，第二項讓編碼靠近先驗，使得抽 $z\sim p(z)$ 再解碼能得到合理的資料。

## 8.3.4 擴散：學會消除雜訊

固定一個逐步破壞資料的**前向過程**：$x_t = \alpha_tx_0 + \sigma_t\varepsilon$、$\varepsilon\sim N(0, I)$，$\alpha_t$ 從 1 降到接近 0、$\sigma_t$ 上升，使 $x_T$ 基本上是純雜訊（[Sohl-Dickstein et al. 2015](/zh-tw/references/#sohl-dickstein-2015)；[Ho et al. 2020](/zh-tw/references/#ho-2020)）。訓練網路從帶雜訊的樣本預測雜訊：

$$
\mathcal L(\theta) = \mathbb{E}_{t, x_0, \varepsilon}\big\lVert\varepsilon_\theta(x_t, t) - \varepsilon\big\rVert^2 .
$$

這又是迴歸。它的最佳解是 $\mathbb{E}[\varepsilon\mid x_t]$，而由一個古典恆等式（Tweedie 公式），這個條件平均與帶雜訊資料分布的**分數（score）** $\nabla_x\ln p_t(x)$ 成正比：$\mathbb{E}[\varepsilon\mid x_t] = -\sigma_t\nabla\ln p_t(x_t)$。所以去雜訊器*就是*分數估計器（[Vincent 2011](/zh-tw/references/#vincent-2011)）。生成時反轉這個過程，可以是隨機的（反向時間 SDE），也可以是確定的（邊際相同的 ODE），兩者都由學到的分數驅動（[Song et al. 2021](/zh-tw/references/#song-2021)）。

## 8.3.5 流匹配

選擇雜訊與資料的一種配對——這裡是獨立的：$x_0\sim N(0, I)$、$x_1\sim p_{\text{data}}$——以及直線路徑 $x_t = (1-t)x_0 + tx_1$。它的**條件速度**是 $\dot x_t = x_1 - x_0$。定義**邊際速度場**

$$
v^\star(x, t) = \mathbb{E}\big[x_1 - x_0\ \big|\ x_t = x\big].
$$

**命題 8.2（流匹配；[Lipman et al. 2023](/zh-tw/references/#lipman-2023)、[Liu et al. 2023](/zh-tw/references/#liu-2023)）。**
(a) 迴歸損失 $\mathcal L(\theta) = \mathbb{E}_{t, x_0, x_1}\lVert v_\theta(x_t, t) - (x_1 - x_0)\rVert^2$ 在 $v_\theta = v^\star$ 時最小。
(b) 若 $x(0)\sim N(0, I)$ 且 $\dot x = v^\star(x, t)$，則對每個 $t$，$x(t)$ 與 $x_t$ 同分布；特別是 $x(1)\sim p_{\text{data}}$。

*(a) 的證明。* 對固定的 $(x, t)$，損失是在給定 $x_t = x$ 下預測隨機向量 $x_1 - x_0$ 的期望平方誤差；期望平方誤差的最小解就是條件平均。$\blacksquare$

*(b) 的證明概要。* 令 $p_t$ 為 $x_t$ 的密度。機率沿著路徑守恆，所以 $p_t$ 滿足**連續方程式（continuity equation）** $\partial_tp_t + \nabla\cdot(p_tv^\star) = 0$，其中正是平均速度 $v^\star$（對配對取平均條件連續方程式）。常微分方程 $\dot x = v^\star(x, t)$ 依同一方程式輸運密度；初始密度相同，由解的唯一性得到相同的 $p_t$。$\blacksquare$

(a) 部分說明了訓練為什麼不需要知道 $v^\star$：以*條件*目標 $x_1 - x_0$ 為迴歸目標（每個抽到的配對都有），在期望上就得到*邊際*場。最小損失不是零：它等於給定 $x_t$ 時 $x_1 - x_0$ 的平均條件變異數——路徑會交叉，在交叉處網路必須取平均。主幹專案的二維模型把損失從 5.27 降到 3.81，而不是 0（詳解例題 8.8）。

## 8.3.6 高斯資料的精確速度場

令 $x_1\sim N(\mu, \sigma^2I)$。則 $x_t\sim N(t\mu, s_t^2I)$，$s_t^2 = (1-t)^2 + t^2\sigma^2$，且 $x_1 - x_0$ 與 $x_t$ 聯合高斯，$\mathrm{Cov}(x_1 - x_0, x_t) = (t\sigma^2 - (1-t))I$。高斯的條件平均是線性的：

$$
v^\star(x, t) = \mu + \frac{t\sigma^2 - (1-t)}{s_t^2}\,(x - t\mu).
$$

這個常微分方程有明確解 $x(t) = t\mu + s_tx_0$：微分得 $\dot x = \mu + \dot s_t x_0 = \mu + \frac{\dot s_t}{s_t}(x - t\mu)$，而 $\dot s_t/s_t = (t\sigma^2 - (1-t))/s_t^2$。在 $t = 1$ 時，$x(1) = \mu + \sigma x_0\sim N(\mu, \sigma^2I)$ 恰好成立。`stage08_sampling.py` 實作了這個場（`gaussian_velocity`），主幹專案的測試從 20,000 個雜訊樣本積分它，並檢查結果的平均與標準差。

## 8.3.7 抽樣就是求解常微分方程

有了學到的 $v_\theta$，生成就是數值積分。$n$ 步、步長 $h = 1/n$ 的 Euler 方法更新 $x\leftarrow x + h\,v_\theta(x, t)$。它每單位時間的誤差是 $O(h)$（單元 05，§5.3.9），而且誤差是系統性的，不是隨機的：對高斯場，一步 Euler 就把*每個*雜訊樣本送到 $\mu$——樣本完全沒有分散——因為 $v^\star(x, 0) = \mu - x$ 直指平均（詳解例題 8.7）。步數少而大，抽樣就快，樣本就模糊、多樣性較低。高階求解器、更直的學習路徑（§8.8.1）以及蒸餾成少步數抽樣器（§8.8.3），都在處理這個取捨。
