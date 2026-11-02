---
title: "3.3 數學建構：從連鎖律到反向傳播與現代最佳化器"
description: 計算圖、前向與反向模式微分及其成本分析、矩陣運算與 softmax 的向量–雅可比乘積、MLP 的反向傳播、隨機梯度、動量、Adam 與 AdamW，以及自動微分引擎的設計。
sidebar: { label: "3.3 數學建構", order: 3 }
---

## 3.3.1 把程式看成計算圖

任何從參數算出損失的程式，都執行一連串**基本運算**——加法、乘法、$\tanh$、$\exp$、矩陣乘法——每一個都從先前的值產生一個新的中間值。按順序寫下這些值，就得到一條**求值軌跡（evaluation trace）**。

以單一神經元搭配平方誤差損失為例，它是本節的貫穿範例：

$$
z = w_1x_1 + w_2x_2 + b,\qquad a = \tanh z,\qquad L = \tfrac12(a - t)^2 .
$$

它的軌跡，每個中間值都有名字：

| 步驟 | 運算 | 在 $x = (1,2)$、$w = (0.5,-0.25)$、$b = 0.1$、$t = 0$ 時的值 |
|---|---|---|
| $v_1$ | $w_1\cdot x_1$ | 0.5 |
| $v_2$ | $w_2\cdot x_2$ | −0.5 |
| $v_3$ | $v_1 + v_2$ | 0.0 |
| $z$ | $v_3 + b$ | 0.1 |
| $a$ | $\tanh z$ | 0.099668 |
| $d$ | $a - t$ | 0.099668 |
| $L$ | $\tfrac12 d^2$ | 0.0049669 |

從每個值畫一個箭頭，指向每個由它算出來的值，就得到一個**有向無環圖**（directed acyclic graph, DAG）：**計算圖（computation graph）**。輸入與參數是它的起點；損失是終點。每個箭頭 $p\to c$ 都帶著一個**局部導數** $\partial c/\partial p$，只依賴該基本運算與它的輸入：對 $c = p\cdot q$ 是 $q$；對 $c = \tanh p$ 是 $1-\tanh^2 p$；對 $c = \tfrac12p^2$ 是 $p$。

由 §3.2.6 的連鎖律，$\partial L/\partial w_1$ 是從 $w_1$ 到 $L$ 的**所有路徑**上**局部導數乘積的總和**。這裡只有一條路徑 $w_1\to v_1\to v_3\to z\to a\to d\to L$，乘積為 $x_1\cdot1\cdot1\cdot(1-a^2)\cdot1\cdot d$。真實網路的路徑多到天文數字。以下兩種演算法組織這個總和，使它永遠不必被逐一列舉。

## 3.3.2 前向模式：讓導數跟著值往前推

固定一個輸入方向——例如我們要對 $w_1$ 的導數。對每個中間值 $v$，定義它的**切向量（tangent）** $\dot v = \partial v/\partial w_1$。設 $\dot w_1 = 1$，其他起點 $\dot u = 0$。由連鎖律，每個值的切向量在計算該值的同時，由其父節點的切向量算出：

$$
\dot v_c = \sum_{p\,\in\,\text{parents}(c)} \frac{\partial v_c}{\partial v_p}\;\dot v_p .
$$

在軌跡上：$\dot v_1 = x_1\dot w_1 = 1$、$\dot v_2 = 0$、$\dot v_3 = 1$、$\dot z = 1$、$\dot a = (1-a^2)\dot z = 0.990066$、$\dot d = 0.990066$、$\dot L = d\,\dot d = 0.098678$。這就是 $\partial L/\partial w_1$。

前向模式（forward mode）簡單、不需要額外記憶體，但一次傳遞只得到對**一個**輸入方向的導數。要得到 $\partial L/\partial w_2$ 與 $\partial L/\partial b$，就得換起始值再跑一次：**$n$ 個參數要 $n$ 次傳遞**，次數上並不比有限差分好（雖然是精確的）。

## 3.3.3 反向模式：從損失把導數往回拉

把問題倒過來。對每個中間值 $v$，定義它的**伴隨值（adjoint）**

$$
\bar v = \frac{\partial L}{\partial v},
$$

即最終損失對該中間值的敏感度。

**遞迴式的推導。** 從 $v_p$ 到 $L$ 的每一條路徑，都以一條從 $v_p$ 到它某個子節點 $v_c$ 的邊開始。依照這第一條邊把路徑分組。以 $p\to c$ 開頭的路徑，貢獻為 $\frac{\partial v_c}{\partial v_p}$ 乘上從 $v_c$ 到 $L$ 所有路徑的總和，而那正是 $\bar v_c$。對所有子節點加總：

$$
\boxed{\;\bar v_p = \sum_{c\,\in\,\text{children}(p)} \bar v_c\;\frac{\partial v_c}{\partial v_p}\;}
\qquad\text{且}\qquad \bar L = 1 .
$$

要計算它，在處理父節點之前必須先有所有子節點的伴隨值：以**反向拓撲排序（topological order）** 走訪節點——從損失往參數走。走到起點時，我們就一次得到了**每個**參數的 $\bar\theta_j = \partial L/\partial\theta_j$。

<div class="dlis-widget not-content" data-widget="backprop-neuron">互動圖需要 JavaScript；下方的表格包含相同的資訊。</div>

**在軌跡上**（局部導數使用儲存的前向值）：

| 伴隨值 | 計算方式 | 值 |
|---|---|---|
| $\bar L$ | 起始值 | 1 |
| $\bar d$ | $\bar L\cdot d$ | 0.099668 |
| $\bar a$ | $\bar d\cdot1$ | 0.099668 |
| $\bar z$ | $\bar a\,(1-a^2)$ | 0.098678 |
| $\bar v_3,\ \bar b$ | $\bar z\cdot 1$ | 0.098678 |
| $\bar v_1,\ \bar v_2$ | $\bar v_3\cdot1$ | 0.098678 |
| $\bar w_1$ | $\bar v_1\,x_1$ | 0.098678 |
| $\bar w_2$ | $\bar v_2\,x_2$ | 0.197356 |

一次反向掃描就得到全部三個參數的梯度。注意每個實作都必須遵守的兩個特點：

- **累加。** 如果一個值有多個子節點，它的伴隨值是*總和*，每個子節點貢獻一份。所以實作會把伴隨值初始化為零，並把每份貢獻**加上去**（$\mathrel{+}=$）。若寫成賦值（$=$），就只留下最後一個子節點的貢獻——這是最常見的自動微分錯誤。
- **儲存中間值。** 局部導數（$d$、$1-a^2$、$x_1$、$x_2$）要用到前向傳遞的值，這些值必須保留到反向傳遞用到它們為止。

**演算法（反向模式微分）。**

1. *前向：* 執行程式，記錄每個運算、它的輸入與輸出。
2. 把所有伴隨值設為 0；設 $\bar L = 1$。
3. *反向：* 對每個記錄的運算 $c = \varphi(p_1,\dots,p_r)$，依相反順序：對每個輸入，$\bar p_i \mathrel{+}= \bar c\cdot\partial\varphi/\partial p_i$。
4. 讀出參數的 $\bar\theta$。

:::note[歷史]
反向模式微分最早在數值分析中發展——見於 Linnainmaa 1970 年的碩士論文與他 1976 年關於捨入誤差傳播的論文（[Linnainmaa 1976](/zh-tw/references/#linnainmaa-1976)）——後來由 [Rumelhart、Hinton & Williams 1986](/zh-tw/references/#rumelhart-1986) 以「反向傳播（back-propagation）」之名推廣到多層神經網路的訓練。一般理論見 [Griewank & Walther 2008](/zh-tw/references/#griewank-2008)。
:::

## 3.3.4 反向模式的成本

**時間。** 反向傳遞中，每個記錄的運算只走訪一次，每條輸入邊計算一個局部導數、做一次乘加。基本運算的局部導數成本至多是該運算本身的小常數倍（往往更少：$\tanh'$ 直接重用儲存的 $\tanh$）。對所有運算加總：

$$
\text{cost}(\nabla L) \;\le\; c\cdot\text{cost}(L),
$$

其中小常數 $c$ **與參數數量 $n$ 無關**。這就是*廉價梯度原理（cheap gradient principle）*；[Griewank & Walther（2008）](/zh-tw/references/#griewank-2008)在標準的運算計數模型下給出並證明了明確的上界。這回答了 §3.1 的預測：純量損失的精確梯度只要幾次前向傳遞的成本，不論模型有十個參數還是一千億個。

**記憶體。** 代價是儲存：反向傳遞需要的前向值。對神經網路而言，主要是批次中每個樣本在每一層的活化值——往往是訓練中最大的記憶體開銷。檢查點重算（§3.7.6）以重新計算換取記憶體。

**一般情況下的前向 vs. 反向。** 對 $f:\mathbb{R}^n\to\mathbb{R}^m$，前向模式每次傳遞給出雅可比矩陣的一行（column；全部要 $n$ 次）；反向模式每次給出一列（row；$m$ 次）。訓練損失的 $m = 1$，所以反向模式快了 $n$ 倍。對一個輸入、多個輸出的函數，則是前向模式勝出。詳解例題 3.8 計算兩者的次數。

## 3.3.5 向量與矩陣：雅可比矩陣與向量–雅可比乘積

真實網路處理的是向量與矩陣，實作也在這個層次上進行，而不是純量。對 $y = f(x)$，$x\in\mathbb{R}^n$、$y\in\mathbb{R}^m$，**雅可比矩陣（Jacobian）** $J\in\mathbb{R}^{m\times n}$ 的元素為 $J_{ij} = \partial y_i/\partial x_j$，合成函數的連鎖律則變成矩陣乘法 $J_{f\circ g} = J_fJ_g$。

對一個向量值運算，伴隨值遞迴逐元素寫出為 $\bar x_j = \sum_i\bar y_i\,\partial y_i/\partial x_j$，即

$$
\bar x = J^{\top}\bar y .
$$

這是一個**向量–雅可比乘積（vector–Jacobian product, VJP）**。反向模式從不建立 $J$（它可能極大）；每個運算只需要一條有效計算 $J^{\top}\bar y$ 的規則。以下四條規則涵蓋了本書中你要驗證的大部分情況。

**(a) 線性映射。** $y = Wx$，$W\in\mathbb{R}^{m\times n}$，即 $y_i = \sum_jW_{ij}x_j$。則 $\partial y_i/\partial x_j = W_{ij}$ 且 $\partial y_i/\partial W_{ij} = x_j$（$W_{ij}$ 不影響其他輸出）。因此

$$
\bar x_j = \sum_i\bar y_iW_{ij}\;\Longrightarrow\;\bar x = W^{\top}\bar y,
\qquad
\bar W_{ij} = \bar y_i\,x_j\;\Longrightarrow\;\bar W = \bar y\,x^{\top},
\qquad
\bar b = \bar y \text{（對 } y = Wx + b\text{）}.
$$

對以列儲存的批次，$Y = XW$，$X\in\mathbb{R}^{B\times n}$、$W\in\mathbb{R}^{n\times m}$，同樣的推理得到

$$
\bar W = X^{\top}\bar Y,\qquad \bar X = \bar Y\,W^{\top}.
$$

**形狀規則：** 一個張量的伴隨值與該張量形狀相同。$\bar W$ 必須是 $n\times m$；$X^\top$（$n\times B$）與 $\bar Y$（$B\times m$）唯一能得到這個形狀的乘積就是 $X^\top\bar Y$。在算任何數值之前先檢查形狀，就能抓到大部分的轉置錯誤。

**(b) 逐元素非線性。** $y_i = \varphi(x_i)$ 的雅可比矩陣是對角的，所以 $\bar x = \bar y\odot\varphi'(x)$。常用導數：$\tanh' = 1-\tanh^2$；logistic 函數 $\sigma' = \sigma(1-\sigma)$；$\mathrm{ReLU}'(x) = \mathbb 1[x>0]$（在 $0$ 處依慣例取值；§3.7.1）。

**(c) softmax 加交叉熵。** 對分數 $s\in\mathbb{R}^V$ 與目標類別 $y$，$p = \mathrm{softmax}(s)$、$L = -\log p_y$。把 $L$ 寫成容易微分的形式：

$$
L = -s_y + \log\sum_{j}e^{s_j}.
$$

則 $\partial L/\partial s_k = -\mathbb 1[k = y] + e^{s_k}/\sum_je^{s_j} = p_k - \mathbb 1[k=y]$，即

$$
\boxed{\;\bar s = p - e_y\;}
$$

其中 $e_y$ 是目標的 one-hot 向量。梯度就是「預測機率減去真相」——有界、便宜、數值上溫和。（單獨的 softmax 雅可比矩陣為 $\mathrm{diag}(p) - pp^{\top}$；單元 04 的注意力機制會用到它。）

**(d) 複製與加總。** 如果一個向量被用在多處（扇出），它的伴隨值是各處伴隨值的總和——又是累加規則。反過來，對 $y = \sum_ix_i$，每個 $\bar x_i = \bar y$。兩者互為轉置：前向傳遞中的廣播，在反向傳遞中變成加總。

## 3.3.6 多層網路中的反向傳播

一個有 $L$ 個隱藏層的多層感知器（multilayer perceptron, MLP）計算

$$
h_0 = x,\qquad z_\ell = W_\ell\,h_{\ell-1} + b_\ell,\quad h_\ell = \varphi(z_\ell)\ \ (\ell = 1,\dots,L),\qquad \hat y = W_{L+1}h_L + b_{L+1},
$$

接著計算損失 $\ell(\hat y, y)$。由上往下套用規則 (a)、(b)、(c)：

**演算法（MLP 的反向傳播，單一樣本）。**

1. 前向傳遞；儲存每個 $z_\ell$ 與 $h_\ell$。
2. $\bar{\hat y} = \partial\ell/\partial\hat y$（例如平方誤差 $\tfrac12\|\hat y - y\|^2$ 時為 $\hat y - y$；softmax 交叉熵時為 $p - e_y$）。
3. 輸出層：$\bar W_{L+1} = \bar{\hat y}\,h_L^{\top}$、$\bar b_{L+1} = \bar{\hat y}$、$\bar h_L = W_{L+1}^{\top}\bar{\hat y}$。
4. 對 $\ell = L, L-1, \dots, 1$：
   $\bar z_\ell = \bar h_\ell\odot\varphi'(z_\ell)$；$\ \bar W_\ell = \bar z_\ell\,h_{\ell-1}^{\top}$；$\ \bar b_\ell = \bar z_\ell$；$\ \bar h_{\ell-1} = W_\ell^{\top}\bar z_\ell$。

對一個批次，參數在樣本間共用，所以它們的梯度是各樣本梯度的**總和**（或平均）——由批次規則 $\bar W = X^{\top}\bar Y$ 一次算出。詳解例題 3.3 對一個 2–2–1 網路手算這個演算法，並以有限差分檢查其中一個元素。

看第 4 步：反向訊號 $\bar h_{\ell-1} = W_\ell^{\top}\bigl(\bar h_\ell\odot\varphi'(z_\ell)\bigr)$ 每經過一層，就乘上一個轉置的權重矩陣與一個導數的對角矩陣。經過許多層，這個矩陣乘積可能以等比方式縮小或放大。這個觀察推動了 §3.7.4 與 §3.8，並在單元 05 以「隨*時間*傳遞的梯度」再次出現。

## 3.3.7 隨機梯度

訓練損失是對全部 $N$ 個樣本的平均，所以它的梯度要掃過整個資料集。取而代之，從中均勻隨機地（放回地）抽取 $B$ 個索引 $i_1,\dots,i_B$ 組成一個**小批次（minibatch）**，並使用

$$
g_B = \frac1B\sum_{b=1}^{B}\nabla\ell_{i_b}(\theta).
$$

**它是不偏的（unbiased）。** 每個抽到的索引在 $\{1,\dots,N\}$ 上均勻，所以 $\mathbb E[\nabla\ell_{i_b}] = \frac1N\sum_i\nabla\ell_i = \nabla L$，由線性性得 $\mathbb E[g_B] = \nabla L$。

**它的雜訊以 $1/\sqrt B$ 縮小。** 令 $\Sigma = \mathrm{Cov}(\nabla\ell_i)$ 為單一抽樣梯度的共變異數。$B$ 個樣本彼此獨立，而 $B$ 個獨立複本之平均的共變異數為 $\Sigma/B$：

$$
\mathrm{Cov}(g_B) = \frac{\Sigma}{B},\qquad \mathbb E\|g_B - \nabla L\|^2 = \frac{\mathrm{tr}\,\Sigma}{B}.
$$

批次變為四倍，典型誤差減半，成本則變為四倍。**隨機梯度下降法（stochastic gradient descent, SGD）** 在更新中以 $g_B$ 取代 $\nabla L$。每一步便宜而有雜訊；§3.5.6 會說明雜訊為常數步長的 SGD 能多接近最小值設下了底限，以及為什麼遞減的步長能消除它（[Robbins & Monro 1951](/zh-tw/references/#robbins-1951)；[Bottou et al. 2018](/zh-tw/references/#bottou-2018)）。實務上，資料在每個**訓練輪次（epoch）** 打亂一次後依序取用（不放回抽樣），行為類似。

## 3.3.8 動量與自適應步長

§3.2.3 說明安全步長受*最大*曲率限制。如果不同方向的曲率差異很大，對陡峭方向安全的步長，對平坦方向就太小，於是梯度下降法會在陡峭方向上來回鋸齒，在平坦方向上緩慢爬行（圖 3.2，以及 §3.5.4 的精確分析）。由此產生兩類補救方法。

**動量（momentum；重球法）。** 維持一個累積過去梯度的「速度」（[Polyak 1964](/zh-tw/references/#polyak-1964)）：

$$
v_{k+1} = \beta v_k - \eta\nabla L(\theta_k),\qquad \theta_{k+1} = \theta_k + v_{k+1},\qquad 0\le\beta<1 .
$$

展開速度得到 $v_{k+1} = -\eta\sum_{j\le k}\beta^{k-j}\nabla L(\theta_j)$——過去梯度的指數加權和，記憶約 $1/(1-\beta)$ 步（與單元 05 的移動平均是同一個物件）。符號保持一致的梯度分量會累積；符號交替的分量（如鋸齒）會互相抵消。物理上，它是一顆在損失地形中受摩擦力滾動的有質量的球。在條件數為 $\kappa$ 的二次函數上，調好的動量把收斂率從 $(\kappa-1)/(\kappa+1)$ 改善到 $(\sqrt\kappa - 1)/(\sqrt\kappa+1)$（§3.5.7）——當 $\kappa = 10$，每步從 0.818 降到 0.519。

![在曲率分別為 1 與 10 的二次函數上，從 (1, 1) 出發的梯度下降法與重球動量路徑：梯度下降法在陡峭方向上鋸齒前進；動量用較少步數到達最小值。](/figures/u03-quadratic-paths.svg)

*圖 3.2——病態條件（$\kappa = 10$）：使用最佳固定步長的梯度下降法 vs. 調好參數的動量。詳解例題 3.5 給出數字。*

<div class="dlis-widget not-content" data-widget="gd-2d">互動圖需要 JavaScript；上方的靜態圖與表格包含相同的資訊。</div>

**逐座標的自適應步長：Adam。** 第二個想法是給每個座標各自的步長，依它的梯度通常有多大來縮放。Adam（[Kingma & Ba 2015](/zh-tw/references/#kingma-2015)）維護梯度及其逐元素平方的指數移動平均：

$$
\begin{aligned}
m_k &= \beta_1m_{k-1} + (1-\beta_1)g_k, & v_k &= \beta_2v_{k-1} + (1-\beta_2)g_k^{2},\\
\hat m_k &= \frac{m_k}{1-\beta_1^{k}}, & \hat v_k &= \frac{v_k}{1-\beta_2^{k}}, \qquad
\theta_{k} = \theta_{k-1} - \eta\,\frac{\hat m_k}{\sqrt{\hat v_k}+\epsilon}.
\end{aligned}
$$

*為什麼要偏差校正（bias correction）？* 由 $m_0 = 0$，$m_k = (1-\beta_1)\sum_{j=1}^{k}\beta_1^{k-j}g_j$。若梯度為常數 $g_j = g$，這個和等於 $(1-\beta_1^{k})g$——早期太小，因為平均是從零開始的。除以 $1-\beta_1^k$ 正好消除這個偏差；$v_k$ 亦同。

*這個更新在做什麼。* 比值 $\hat m/\sqrt{\hat v}$ 沒有單位：把每個梯度乘上常數 $c>0$，比值不變（除了 $\epsilon$）。第一步時 $\hat m_1 = g_1$、$\hat v_1 = g_1^2$，所以每個座標都恰好移動 $\eta\,\mathrm{sign}(g_1)$，不論梯度大小（詳解例題 3.7）。因此 Adam 的行為像逐座標、類似符號的方法，在梯度雜訊大的地方（$|\hat m|\ll\sqrt{\hat v}$）步長縮小。

**AdamW：解耦的權重衰減。** 常見的正則化（regularization）是在損失加上 $\tfrac\lambda2\|\theta\|^2$，也就是在梯度加上 $\lambda\theta$。在 Adam 內部，這一項會被 $\sqrt{\hat v}$ 除，所以梯度大的參數反而被正則化得*較少*——這不是懲罰項的本意。AdamW 把權重衰減（weight decay）直接作用在參數上，放在自適應縮放之外（[Loshchilov & Hutter 2019](/zh-tw/references/#loshchilov-2019)）：

$$
\theta_k = \theta_{k-1} - \eta\Bigl(\frac{\hat m_k}{\sqrt{\hat v_k}+\epsilon} + \lambda\,\theta_{k-1}\Bigr).
$$

它是 Transformer 與主幹專案的預設最佳化器。[AdamW Atlas 節點](/zh-tw/atlas/nodes/opt-adamw/)列出了它的不變量與常見實作錯誤。

## 3.3.9 建構引擎

實驗室要求你為純量實作反向模式。設計直接由 §3.3.3 而來：

- **一種節點型別**（`Value`）儲存它的數值（`data`）、它的伴隨值（`grad`，初始為 0）、對父節點的參照，以及一個小函式：給定自己的伴隨值，依規則 $\bar p \mathrel{+}= \bar c\,\partial c/\partial p$ 把貢獻加到父節點的伴隨值上。
- **每個運算**（`+`、`*`、`tanh`……）建立一個新節點，並附上正確的局部導數函式，同時捕捉它需要的值（例如 $\tanh$ 捕捉輸出 $t$，使用 $1-t^2$）。
- **`backward()`** 建立圖的拓撲排序（以迭代方式——遞迴的深度優先搜尋在 10,000 個節點深的圖上會使呼叫堆疊溢位），把輸出的伴隨值設為 1，再以相反順序呼叫每個節點的局部函式。
- **歸零是呼叫者的責任。** 共用葉節點的不同圖之間，伴隨值會在多次反向傳遞中累加；最佳化迴圈在每一步之前把它們設為零。

[實驗室](https://github.com/yhxuee/deep-learning-intelligent-systems/tree/main/labs/unit-03-autodiff)正是測試這些性質：詳解例題的數字、扇出下的累加（`a*a`、`a+a`）、在隨機運算式圖上與中央差分一致、線性性、深層圖，以及跨圖的累加。一旦你的引擎通過這些測試，你就**贏得了使用抽象的資格**：PyTorch 與 JAX 等框架在張量上實作同一個演算法，每個運算一條 VJP 規則；從單元 04 起，主幹專案就改用它們。
