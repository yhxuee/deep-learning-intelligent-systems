---
title: "8.3 數學建構"
description: 從「這個程序究竟生成哪個分布」出發，推導祖先抽樣、變數變換、ELBO、去雜訊分數、流匹配、精確高斯速度場與數值抽樣。
sidebar: { label: "8.3 數學建構", order: 3 }
---

上一節把三件事分開：學到的分布、從中取樣的規則，以及最後實際出現的樣本。現在我們把這個區分寫成數學。每一步都反覆問同一個問題：

> 假如這個程序能精確執行，它究竟會生成哪個分布？

這個問題能避免一種常見混淆：模型定義的分布可能沒有問題，但近似抽樣器會扭曲它；反過來，抽樣器也可能很精確，只是它抽的是經過刻意修改的分布，而不是訓練時的原分布。

## 8.3.1 祖先抽樣及其修改

假設自回歸模型在每個位置 $t$ 都給出正規化的條件分布 $q_t(\,\cdot\mid x_{<t})$。**祖先抽樣（ancestral sampling）**從空前綴開始，先抽 $X_1\sim q_1$，把結果接到前綴，再抽 $X_2\sim q_2(\cdot\mid X_1)$，如此繼續。

**命題 8.1。** 對固定長度 $T$，祖先抽樣產生特定序列 $x_{1:T}$ 的機率為

$$
\Pr(X_{1:T}=x_{1:T})
=\prod_{t=1}^{T}q_t(x_t\mid x_{<t})
=q(x_{1:T}).
$$

*證明。* 「演算法回傳 $x_{1:T}$」表示第 1 步抽到 $x_1$，第 2 步在看過 $x_1$ 後抽到 $x_2$，依此類推。條件機率的乘法法則給出

$$
\Pr(X_{1:T}=x_{1:T})
=\Pr(X_1=x_1)\prod_{t=2}^{T}
\Pr(X_t=x_t\mid X_{<t}=x_{<t}).
$$

這些因子正是模型的條件分布；它們的乘積就是 $q(x_{1:T})$ 的連鎖律分解。$\blacksquare$

若序列長度可變，必須加入序列結束符號 EOS。此時一個完整結果是 $(x_1,\ldots,x_L,\mathrm{EOS})$，其機率也包含最後「停止」的機率。若略去 EOS，樣本空間就被悄悄改變了：程序雖然指定了符號，卻沒有指定序列何時結束。

**修改解碼規則時，什麼改變了？** 令某個前綴下的 logits 為 $a_k$。溫度 $\tau>0$ 產生新的局部條件分布

$$
\widetilde q_\tau(k\mid x_{<t})
=\frac{\exp(a_k/\tau)}
{\sum_j\exp(a_j/\tau)}.
$$

當 $\tau\to0^+$，機率集中到最大 logit；若最大值不只一個，極限機率會由並列最大者共享，除非另外指定破除平手的規則。當 $\tau\to\infty$，有限詞彙表會趨近均勻分布。$\tau=1$ 則還原原本的條件分布。

Top-$k$ 保留機率最高的 $k$ 個符號。**核心抽樣（nucleus sampling）**或 top-$p$ 則先排列符號，使 $q(k_1)\ge q(k_2)\ge\cdots$，再選擇滿足下式的最小 $m$：

$$
\sum_{i=1}^{m}q(k_i)\ge p.
$$

令 $S_p=\{k_1,\ldots,k_m\}$，抽樣分布為

$$
\widetilde q_p(k)
=
\begin{cases}
\displaystyle\frac{q(k)}{\sum_{j\in S_p}q(j)}, & k\in S_p,\\[6pt]
0, & k\notin S_p.
\end{cases}
$$

分母不可省略：截斷刪掉了一些結果，重新正規化才會讓剩餘數值再次成為機率。

在每個前綴套用上述任一規則，仍然是在做精確的祖先抽樣——但抽的是*新*序列分布

$$
\widetilde q(x_{1:T})
=\prod_{t=1}^{T}\widetilde q_t(x_t\mid x_{<t}),
$$

一般不再是原本的 $q$。局部溫度也不同於把完整序列機率整體取次方：

$$
\prod_t
\frac{q(x_t\mid x_{<t})^{1/\tau}}
{\sum_k q(k\mid x_{<t})^{1/\tau}}
\ne
\frac{q(x_{1:T})^{1/\tau}}
{\sum_{y_{1:T}}q(y_{1:T})^{1/\tau}}.
$$

左式每一步都有依前綴而變的正規化常數；右式只有一個對所有完整序列計算的正規化常數。貪婪解碼走得更遠：它把每個條件分布換成集中在某個最大值上的點質量。指定平手規則後，它是確定性的，而且不保證找到全域機率最高的完整序列（§8.2）。

## 8.3.2 變數變換與正規化流

自回歸模型一次建構一個條件分布。**正規化流（normalizing flow）**採取另一條路：先抽一個簡單隨機向量 $Z$，再以可逆映射一次變換整個向量。

令 $\mathbb R^n$ 中的 $Z\sim p_Z$、$X=g(Z)$。假設 $g$ 是可微雙射，逆映射 $f=g^{-1}$ 也可微，而且雅可比行列式幾乎處處非零。機率必須守恆：$x$ 周圍的一小塊區域，與其在 $z=f(x)$ 周圍的原像，承載相同機率：

$$
p_X(x)\,d x\approx p_Z(f(x))\,d z.
$$

在局部，逆映射把體積改變為 $d z=\left|\det J_f(x)\right|d x$。代入便得到**變數變換公式**

$$
\boxed{
p_X(x)=p_Z(f(x))\left|\det J_f(x)\right|
}.
$$

行列式不是裝飾性的修正項。密度是「每單位體積的機率」；如果映射把一小塊區域拉伸成兩倍體積，同一份機率就會被攤薄成一半。

**一維檢查。** 令 $Z\sim N(0,1)$，且 $X=\mu+\sigma Z$、$\sigma>0$。此時 $f(x)=(x-\mu)/\sigma$，且 $|f'(x)|=1/\sigma$，所以

$$
p_X(x)=\frac{1}{\sigma}
\phi\left(\frac{x-\mu}{\sigma}\right).
$$

熟悉的高斯縮放因子，正是一個雅可比。

實用的流會組合多個映射，$g=g_L\circ\cdots\circ g_1$。行列式在組合下相乘，因此取對數後相加。若把逆映射逐層執行得到 $z=f(x)$，則

$$
\ln p_X(x)
=\ln p_Z(z)
+\sum_{\ell=1}^{L}
\ln\left|\det J_{f_\ell}\right|.
$$

這同時提供精確密度計算與直接的最大概似訓練。真正困難的是架構：任意神經網路通常既不可逆，行列式也不便宜。耦合層的解法是讓一組座標保持不變，再用第一組座標算出的參數去變換另一組。它的雅可比是三角矩陣，行列式只需把對角元素相乘（[Dinh et al. 2017](/zh-tw/references/#dinh-2017)；[Rezende & Mohamed 2015](/zh-tw/references/#rezende-2015)）。

這裡的「精確概似」以數學模型與數值計算都正確為前提：映射必須真的可逆、行列式必須正確求值，而且基底密度必須已知。這些限制換來可處理性，也約束了架構。同維度的平滑雙射還會保留較多拓撲性質；後面的章節會說明這為何重要。

## 8.3.3 潛變數與證據下界

**潛變數模型**先抽取未觀察到的編碼 $Z\sim p(z)$，再由解碼器抽取資料 $X\sim p_\theta(x\mid Z)$。觀察資料的密度要把所有可能解釋平均起來：

$$
p_\theta(x)=\int p_\theta(x\mid z)p(z)\,d z.
$$

這個積分通常難以處理——不是依定義必然無法計算，而是對神經解碼器而言，往往成本太高或沒有封閉形式。後驗

$$
p_\theta(z\mid x)
=\frac{p_\theta(x\mid z)p(z)}{p_\theta(x)}
$$

含有同一個麻煩的正規化常數。變分自編碼器引入可處理的編碼器 $q_\phi(z\mid x)$ 來近似它（[Kingma & Welling 2014](/zh-tw/references/#kingma-2014)）。

為了推導訓練目標，在聯合密度中乘上再除以編碼器：

$$
\begin{aligned}
\ln p_\theta(x)
&=\ln\mathbb E_{q_\phi(z\mid x)}
\left[
\frac{p_\theta(x,z)}
{q_\phi(z\mid x)}
\right]\\
&\ge
\mathbb E_{q_\phi(z\mid x)}
\left[
\ln p_\theta(x,z)-\ln q_\phi(z\mid x)
\right],
\end{aligned}
$$

第二行使用 Jensen 不等式。再展開 $p_\theta(x,z)=p_\theta(x\mid z)p(z)$，便得到**證據下界（evidence lower bound, ELBO）**

$$
\boxed{
\mathcal L_{\mathrm{ELBO}}(x)
=
\mathbb E_{q_\phi(z\mid x)}
\left[\ln p_\theta(x\mid z)\right]
-
\mathrm{KL}
\left(q_\phi(z\mid x)\,\Vert\,p(z)\right)
}.
$$

名稱是字面意義：$\mathcal L_{\mathrm{ELBO}}(x)\le\ln p_\theta(x)$。下列精確恆等式揭示差距有多大：

$$
\ln p_\theta(x)
=\mathcal L_{\mathrm{ELBO}}(x)
+\mathrm{KL}
\left(q_\phi(z\mid x)\,\Vert\,p_\theta(z\mid x)\right).
$$

因此，當且僅當編碼器幾乎處處等於模型的真正後驗時，這個界才取等號。

ELBO 的兩項各有精確含義。

- 期望對數概似要求解碼器在拿到由資料推斷的編碼後，給該筆觀察高機率。只有在觀察模型是固定變異數的高斯分布時，它才會化成平方誤差重建項。Bernoulli、類別、離散化 logistic 或可學習變異數的解碼器，會產生不同損失。
- KL 項抑制每筆觀察的編碼器分布偏離所選先驗太遠。這有助於讓先驗樣本可供使用，卻不能單獨保證生成結果逼真：解碼器、變分分布族、最佳化與泛化仍然都會影響結果。

如何對隨機抽樣微分？對角高斯編碼器可寫成

$$
q_\phi(z\mid x)
=N\!\left(
\mu_\phi(x),
\operatorname{diag}(\sigma_\phi^2(x))
\right).
$$

再令

$$
\varepsilon\sim N(0,I),
\qquad
Z=\mu_\phi(x)+\sigma_\phi(x)\odot\varepsilon.
$$

隨機性現在由不含參數的 $\varepsilon$ 注入，而 $Z$ 是 $\phi$ 的可微函數。這個**重新參數化技巧（reparameterization trick）**讓我們能以一般反向傳播，對期望值的 Monte Carlo 估計求梯度。它改變的是梯度估計方式，不是模型分布。

## 8.3.4 擴散：學會消除雜訊

擴散從一個容易執行、卻難以逆轉的操作開始：不斷把雜訊加入資料，直到幾乎沒有結構。為避免符號衝突，本節把乾淨樣本寫成 $X_{\mathrm{data}}$；到了 §8.3.5，$X_0$ 會改指雜訊，$X_1$ 才指資料。

在雜訊等級 $t$，考慮高斯破壞過程

$$
X_t=\alpha_t X_{\mathrm{data}}+\sigma_t\varepsilon,
\qquad
\varepsilon\sim N(0,I).
$$

排程從接近 $(\alpha_0,\sigma_0)=(1,0)$ 開始，終點的訊號則足夠小，使 $X_T$ 接近一個已知雜訊分布。離散擴散模型以許多 Markov 轉移實現這件事；上式總結了選定時間的邊際破壞分布（[Sohl-Dickstein et al. 2015](/zh-tw/references/#sohl-dickstein-2015)；[Ho et al. 2020](/zh-tw/references/#ho-2020)）。

訓練網路預測當次實際加入的雜訊：

$$
\mathcal L(\theta)
=\mathbb E_{t,X_{\mathrm{data}},\varepsilon}
\left[
\left\|
\varepsilon_\theta(X_t,t)-\varepsilon
\right\|^2
\right].
$$

平方誤差迴歸告訴我們，母體最佳解是 $\varepsilon_\theta^\star(x,t)=\mathbb E[\varepsilon\mid X_t=x]$。但生成時不會有人告訴模型真正的 $\varepsilon$，這個量為何有用？固定乾淨資料時，破壞分布是高斯分布，所以

$$
\nabla_x\ln p_t(x\mid X_{\mathrm{data}})
=-\frac{x-\alpha_tX_{\mathrm{data}}}{\sigma_t^2}
=-\frac{\varepsilon}{\sigma_t}.
$$

在積分號下對邊際分布 $p_t(x)=\int p_t(x\mid x_{\mathrm{data}})p_{\mathrm{data}}(x_{\mathrm{data}})\,d x_{\mathrm{data}}$ 微分，再除以 $p_t(x)$。在通常的正則條件下，

$$
\begin{aligned}
\nabla_x\ln p_t(x)
&=
\mathbb E\left[
\nabla_x\ln p_t(x\mid X_{\mathrm{data}})
\mid X_t=x
\right]\\
&=-\frac{1}{\sigma_t}
\mathbb E[\varepsilon\mid X_t=x].
\end{aligned}
$$

因此

$$
\boxed{
\mathbb E[\varepsilon\mid X_t=x]
=-\sigma_t\nabla_x\ln p_t(x)
}.
$$

向量 $\nabla_x\ln p_t(x)$ 稱為**分數（score）**：它指向帶雜訊資料的對數密度增加最快的方向。因此，對這個高斯破壞過程與雜訊預測參數化而言，最佳去雜訊器與分數含有相同資訊（[Vincent 2011](/zh-tw/references/#vincent-2011)）。這是一個有條件的恆等式，並不是宣稱所有雜訊模型下的每個去雜訊器，字面上都是分數網路。

在所有相關雜訊等級上都準確的分數，配合已指定的排程，會定義反向時間隨機微分方程。相關的 probability-flow ODE 在理想連續時間模型中具有相同的單一時間邊際分布（[Song et al. 2021](/zh-tw/references/#song-2021)）。實務上，網路誤差、端點近似、時間離散化與數值求解器，都會使生成樣本偏離這個理想。

## 8.3.5 流匹配

擴散藉由學習分數來掌握密度如何變化。**流匹配（flow matching）**則直接學習速度場。

獨立抽取雜訊與資料：

$$
X_0\sim N(0,I),
\qquad
X_1\sim p_{\mathrm{data}},
$$

再以直線條件路徑連接每一組配對：

$$
X_t=(1-t)X_0+tX_1,
\qquad
\dot X_t=X_1-X_0.
$$

同一個位置 $X_t=x$ 可能落在許多抽樣直線上，每一條線的速度不同。確定性的 ODE 在 $(x,t)$ 只能有一個速度，因此定義**邊際速度場**

$$
v^\star(x,t)=\mathbb E[X_1-X_0\mid X_t=x].
$$

**命題 8.2（流匹配；[Lipman et al. 2023](/zh-tw/references/#lipman-2023)、[Liu et al. 2023](/zh-tw/references/#liu-2023)）。** 假設相關密度與速度場足夠正則，使條件期望、連續方程式與 ODE 流都存在，而且密度演化具有唯一解。

1. 母體損失

   $$
   \mathcal L(\theta)
   =
   \mathbb E
   \left[
   \left\|
   v_\theta(X_t,t)-(X_1-X_0)
   \right\|^2
   \right]
   $$

   在 $v_\theta(x,t)=v^\star(x,t)$ 幾乎處處成立時最小。
2. 若 $Y(0)\sim N(0,I)$ 且 $\dot Y(t)=v^\star(Y(t),t)$，則 $Y(t)$ 與 $X_t$ 的邊際分布相同。特別是 $Y(1)\sim p_{\mathrm{data}}$。

*第 1 部分的證明。* 令 $V=X_1-X_0$，且 $m(X_t,t)=\mathbb E[V\mid X_t,t]$。加上再減去 $m$：

$$
\begin{aligned}
\mathbb E\|V-v_\theta\|^2
&=\mathbb E\|V-m\|^2
+\mathbb E\|m-v_\theta\|^2\\
&\quad+
2\mathbb E[(V-m)^\top(m-v_\theta)].
\end{aligned}
$$

因為 $\mathbb E[V-m\mid X_t,t]=0$，交叉項為零。第一項不依賴 $\theta$；第二項非負，而且只在 $v_\theta=m$ 幾乎處處成立時為零。$\blacksquare$

這個分解也說明為什麼最佳可達的迴歸損失不必等於零：

$$
\mathcal L_{\min}
=\mathbb E
\left[
\|V-\mathbb E[V\mid X_t,t]\|^2
\right].
$$

獨立配對會產生彼此交叉的條件路徑。即使已知 $(X_t,t)$，交叉處的目標速度仍有不確定性；網路學到的是它們的條件平均。

*第 2 部分的證明概要。* 令 $p_t$ 為 $X_t$ 的密度。條件路徑所承載的機率遵守守恆律；在每個位置平均其速度可得

$$
\partial_t p_t(x)
+\nabla\cdot
\left(p_t(x)v^\star(x,t)\right)
=0,
$$

這就是**連續方程式（continuity equation）**。由 ODE $\dot Y=v^\star(Y,t)$ 輸運的密度滿足同一方程式，而且從同一個 $p_0$ 出發。在上述唯一性條件下，兩條密度路徑因此都是 $p_t$。$\blacksquare$

這個結果有一個細微後果：訓練時使用從抽樣配對直接取得的簡單目標 $X_1-X_0$，但生成時走的是平均後的邊際場 $v^\star$。個別生成 ODE 軌跡通常**不會**重走某一組訓練配對的直線。

## 8.3.6 高斯資料的精確速度場

高斯目標讓我們能逐項檢查命題 8.2。令

$$
X_0\sim N(0,I),
\qquad
X_1\sim N(\mu,\sigma^2I),
$$

兩者獨立，且 $\sigma>0$。沿著直線插值，$X_t=(1-t)X_0+tX_1$。平均會線性相加，所以 $\mathbb E[X_t]=t\mu$。由於獨立，協方差項也直接相加：

$$
\operatorname{Var}(X_t)
=\left((1-t)^2+t^2\sigma^2\right)I
=s_t^2I,
\qquad
s_t=\sqrt{(1-t)^2+t^2\sigma^2}.
$$

現在令 $V=X_1-X_0$。其平均為 $\mu$，而

$$
\begin{aligned}
\operatorname{Cov}(V,X_t)
&=
\operatorname{Cov}
\left(
X_1-X_0,\,
(1-t)X_0+tX_1
\right)\\
&=\left(t\sigma^2-(1-t)\right)I.
\end{aligned}
$$

由於 $(V,X_t)$ 聯合高斯，其條件平均是仿射函數：

$$
\boxed{
v^\star(x,t)
=\mu+
\frac{t\sigma^2-(1-t)}{s_t^2}
(x-t\mu)
}.
$$

這個 ODE 不需數值方法就能求解。給定初始值 $Y(0)=y_0$，猜測 $Y(t)=t\mu+s_ty_0$。由於

$$
\frac{\dot s_t}{s_t}
=\frac{t\sigma^2-(1-t)}{s_t^2},
$$

對猜測路徑微分可得

$$
\dot Y(t)
=\mu+
\frac{t\sigma^2-(1-t)}{s_t^2}
\left(Y(t)-t\mu\right)
=v^\star(Y(t),t).
$$

因為 $s_0=1$，它具有正確初始值。終點則有 $s_1=\sigma$，所以 $Y(1)=\mu+\sigma y_0$。若 $y_0\sim N(0,I)$，便精確得到 $Y(1)\sim N(\mu,\sigma^2I)$。注意這個解保留了什麼：每個初始標準化座標 $y_0$ 都還在，只是隨 $s_t$ 連續縮放。這個流改變點雲的平均與分散程度，並不會把每個樣本都送到平均。

\`stage08_sampling.py\` 中的 \`gaussian_velocity\` 實作這個精確場。相應測試會積分許多初始雜訊樣本，再檢查終點平均與標準差。通過測試表示「速度場加上求解器」在選定容差下通過檢查；它不是對每一種求解器設定的證明。

## 8.3.7 抽樣就是求解常微分方程

速度網路訓練完成後，生成需要求解

$$
\dot x(t)=v_\theta(x(t),t),
\qquad
x(0)\sim p_0.
$$

Euler 方法把 $[0,1]$ 分成 $n$ 步，步長 $h=1/n$：

$$
x_{j+1}=x_j+h\,v_\theta(x_j,t_j).
$$

在標準的平滑與 Lipschitz 條件下，Euler 每一步的局部截斷誤差為 $O(h^2)$，在固定時間區間累積後的**全域誤差為 $O(h)$**。「一階」描述的是步長縮小時最終誤差如何下降，不是每單位時間隨機加入多少誤差。

精確高斯場揭示一個劇烈的一步失敗。在 $t=0$，$v^\star(x,0)=\mu-x$。因此，用步長 $h=1$ 做一次 Euler 更新會得到

$$
x_{\mathrm{Euler}}(1)=x(0)+\mu-x(0)=\mu.
$$

每個初始樣本都塌縮到平均，儘管精確 ODE 解的終點是 $\mu+\sigma x(0)$。速度場是正確的；錯的是數值近似。面對其他場時，大步長也可能過衝、移動平均、製造瑕疵或遺漏模態。因此，步數較少並不普遍只意味著「更模糊」或只意味著「多樣性更低」。

最好把兩種誤差分開：

$$
\text{生成誤差}
\quad\longleftarrow\quad
\underbrace{v_\theta-v^\star}_{\text{學到的速度場誤差}}
\quad+\quad
\underbrace{\text{數值積分誤差}}_{\text{求解器與步長排程}}.
$$

增加求解器步數只會降低第二項，而且這還假設方法穩定且收斂；它無法修復有偏的學習速度場。高階方法在每一步多次評估速度場，以消去低階誤差；自適應方法則把評估花在軌跡較難的區域。真正相關的成本通常是神經網路函數評估次數，而不只是時間區間被切成幾段。更直的路徑（§8.8.1）、更好的求解器，以及蒸餾成少步生成器（§8.8.3），處理的是這個精度與計算量問題的不同部分。
