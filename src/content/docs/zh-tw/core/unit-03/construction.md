---
title: "3.3 數學建構：從連鎖律到反向傳播與現代最佳化器"
description: 計算圖、前向與反向模式微分及其成本分析、矩陣運算與 softmax 的向量–雅可比乘積、MLP 的反向傳播、隨機梯度、動量、Adam 與 AdamW，以及自動微分引擎的設計。
sidebar: { label: "3.3 數學建構", order: 3 }
---

## 3.3.1 把程式看成計算圖

任何從參數算出損失的程式，都會執行一連串**基本運算**——加法、乘法、$\tanh$、$\exp$、矩陣乘法——每一個都從先前的值產生中間值。記錄實際執行的運算與其數值，就得到一條**求值軌跡（evaluation trace）**。迴圈會展開成重複運算；分支則記錄實際走過的路徑。分支邊界是否可微，是另一個問題（§3.7.1）。

同一條軌跡會有兩種讀法。**前向傳遞**問：「這個節點算出什麼值？」**反向傳遞**問：「若這個節點改變一點，最終損失會怎麼改變？」分開這兩個問題，會讓以下符號容易理解許多。

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

從每個值畫一個箭頭，指向每個直接使用它的後續值，就得到這次執行的**有向無環圖**（directed acyclic graph, DAG），也就是**計算圖（computation graph）**。輸入與參數是起點；純量損失是終點。每條邊 $p\to c$ 都帶有一個由單一基本運算決定的**局部導數** $\partial c/\partial p$：若 $c=pq$，固定 $q$ 後就是 $q$；若 $c=\tanh p$，就是 $1-\tanh^2p$；若 $c=\tfrac12p^2$，就是 $p$。

由 §3.2.6 的連鎖律，$\partial L/\partial w_1$ 是從 $w_1$ 到 $L$ 的**所有路徑**上**局部導數乘積的總和**。這裡只有一條路徑 $w_1\to v_1\to v_3\to z\to a\to d\to L$，乘積為 $x_1\cdot1\cdot1\cdot(1-a^2)\cdot1\cdot d$。真實網路的路徑多到天文數字。以下兩種演算法組織這個總和，使它永遠不必被逐一列舉。

## 3.3.2 前向模式：讓導數跟著值往前推

先固定起點空間中的一個方向——第一個例子只讓 $w_1$ 增加一單位，其他起點不變。對每個中間值 $v$，定義其**切向量（tangent）** $\dot v=\partial v/\partial w_1$。設 $\dot w_1=1$，其他起點 $\dot u=0$。連鎖律會在計算數值的同時，從父節點的切向量算出每個新切向量：

$$
\dot v_c = \sum_{p\,\in\,\text{parents}(c)} \frac{\partial v_c}{\partial v_p}\;\dot v_p .
$$

在軌跡上：$\dot v_1 = x_1\dot w_1 = 1$、$\dot v_2 = 0$、$\dot v_3 = 1$、$\dot z = 1$、$\dot a = (1-a^2)\dot z = 0.990066$、$\dot d = 0.990066$、$\dot L = d\,\dot d = 0.098678$。這就是 $\partial L/\partial w_1$。

更一般地說，任意起點切向量會指定方向 $r$，一次傳遞計算雅可比–向量乘積 $Jr$。前向模式可以串流數值與切向量，不需儲存完整的反向軌跡，但仍要攜帶切向資訊。一次純量方向傳遞只得到一個方向導數。若要分別取得 $\partial L/\partial w_2$ 與 $\partial L/\partial b$，就得換座標起始值重跑：全部 $n$ 個參數偏導數需要 **$n$ 次傳遞**。這是依連鎖律得到的精確微分，但當純量損失依賴許多參數時，傳遞次數仍不理想。

## 3.3.3 反向模式：從損失把導數往回拉

反向模式把問題倒過來。它不問一個起點方向如何改變所有後續值，而是問最終純量如何隨每個中間值改變。定義**伴隨值（adjoint）**

$$
\bar v = \frac{\partial L}{\partial v},
$$

即最終損失對該中間值的敏感度。上方的橫線不是平均符號；它標記反向模式的導數。

**遞迴式的推導。** 從 $v_p$ 到 $L$ 的每一條路徑，都以一條從 $v_p$ 到它某個子節點 $v_c$ 的邊開始。依照這第一條邊把路徑分組。以 $p\to c$ 開頭的路徑，貢獻為 $\frac{\partial v_c}{\partial v_p}$ 乘上從 $v_c$ 到 $L$ 所有路徑的總和，而那正是 $\bar v_c$。對所有子節點加總：

$$
\boxed{\;\bar v_p = \sum_{c\,\in\,\text{children}(p)} \bar v_c\;\frac{\partial v_c}{\partial v_p}\;}
\qquad\text{且}\qquad \bar L = 1 .
$$

要計算這個遞迴，某節點的所有子節點都必須先送回貢獻，該節點才能繼續往前傳遞伴隨值。因此要依**反向拓撲排序（topological order）**走訪——從損失往參數走。抵達起點時，這次執行所使用的每個參數都有 $\bar\theta_j=\partial L/\partial\theta_j$。「一次得到」是指一次反向走訪，不是單一常數時間的機器指令。

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

**時間。** 反向傳遞會走訪每個記錄的運算一次，並對每條輸入邊套用局部 VJP 規則、累加一項貢獻。在標準成本模型中，若每項基本運算的導數規則成本至多是原運算的固定倍數（通常更少：$\tanh'$ 會重用儲存的輸出），對整條軌跡加總可得

$$
\text{cost}(\nabla L) \;\le\; c\cdot\text{cost}(L),
$$

其中 $\operatorname{cost}(\nabla L)$ 包含前向與反向工作，而 $c$ 不會額外多出參數數量這個因子。這就是**廉價梯度原理（cheap gradient principle）**；[Griewank & Walther（2008）](/zh-tw/references/#griewank-2008)對指定的基本運算集合與成本模型給出明確上界。這項陳述談的是比例：較大模型的前向傳遞較昂貴，梯度當然也較昂貴；反向模式省掉的是「為每個參數各跑一次前向計算」。

**記憶體。** 代價是儲存：反向傳遞需要的前向值。對神經網路而言，主要是批次中每個樣本在每一層的活化值——往往是訓練中最大的記憶體開銷。檢查點重算（§3.7.6）以重新計算換取記憶體。

**一般情況下的前向 vs. 反向。** 對 $f:\mathbb{R}^n\to\mathbb{R}^m$，以座標為起始方向的前向模式，每次會給出雅可比矩陣的一行（column）；以座標為起始伴隨值的反向模式，每次給出一列（row）。所以建立*完整*雅可比矩陣，需要 $n$ 次前向掃描或 $m$ 次反向掃描。訓練損失有 $m=1$，反向模式一次就能取得完整梯度；若函數輸入很少、輸出很多，前向模式可能更合適。通常我們只需要乘積 $Jr$ 或 $J^\top v$，不會真的建立完整雅可比矩陣。詳解例題 3.8 會計算掃描次數。

## 3.3.5 向量與矩陣：雅可比矩陣與向量–雅可比乘積

真實網路處理的是向量與矩陣，實作也在這個層次上進行，而不是純量。對 $y = f(x)$，$x\in\mathbb{R}^n$、$y\in\mathbb{R}^m$，**雅可比矩陣（Jacobian）** $J\in\mathbb{R}^{m\times n}$ 的元素為 $J_{ij} = \partial y_i/\partial x_j$，合成函數的連鎖律則變成矩陣乘法 $J_{f\circ g} = J_fJ_g$。

對一個向量值運算，伴隨值遞迴逐元素寫出為 $\bar x_j = \sum_i\bar y_i\,\partial y_i/\partial x_j$，即

$$
\bar x = J^{\top}\bar y .
$$

這是**向量–雅可比乘積（vector–Jacobian product, VJP）**的欄向量寫法：較傳統的列記號寫成 $\bar y^\top J$，其轉置就是 $J^\top\bar y$。反向模式不必建立可能極大的 $J$；每個運算只需提供一條把輸入伴隨值 $\bar y$ 映成 $J^\top\bar y$ 的規則。以下四條規則涵蓋本書驗證的大部分情況。

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

其中 $e_y$ 是目標的 one-hot 向量。每個分數梯度都在 $[-1,1]$ 內，而且各分量總和為零，因為對所有分數加上同一常數不會改變 softmax。這條簡潔規則計算便宜，但前向損失仍必須使用穩定的 log-sum-exp；直接計算 $e^{s_j}$ 可能溢位。單獨的 softmax 雅可比矩陣為 $\mathrm{diag}(p)-pp^\top$；單元 04 會在注意力中使用其 VJP。

**(d) 複製、廣播與加總。** 如果一個值被複製到多個用途（扇出），它的伴隨值就是各用途貢獻的總和。若 $y=\sum_i x_i$，則每個 $\bar x_i=\bar y$。這些運算彼此轉置：前向的廣播或重複使用，會在反向變成加總；前向的歸約，則在反向變成廣播。這就是陣列形式的累加規則。

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

**在上述抽樣方式下，它是不偏的（unbiased）。** 固定 $\theta$ 時，每個放回抽樣的索引都在 $\{1,\dots,N\}$ 上均勻，所以 $\mathbb E[\nabla\ell_{i_b}]=N^{-1}\sum_i\nabla\ell_i=\nabla L$，由線性性得 $\mathbb E[g_B]=\nabla L$。「固定 $\theta$」很重要：更新後，參數就改變了。

**它的雜訊以 $1/\sqrt B$ 縮小。** 令 $\Sigma = \mathrm{Cov}(\nabla\ell_i)$ 為單一抽樣梯度的共變異數。$B$ 個樣本彼此獨立，而 $B$ 個獨立複本之平均的共變異數為 $\Sigma/B$：

$$
\mathrm{Cov}(g_B) = \frac{\Sigma}{B},\qquad \mathbb E\|g_B - \nabla L\|^2 = \frac{\mathrm{tr}\,\Sigma}{B}.
$$

把 $B$ 增為四倍，這項均方根抽樣誤差減半，算術工作則增為四倍；平行硬體可能改變實際耗時的比例。**隨機梯度下降法（stochastic gradient descent, SGD）**在更新中以 $g_B$ 取代 $\nabla L$。每次更新都比全資料梯度便宜，卻帶有雜訊。在 §3.5.6 明確列出的假設下，這種雜訊會為固定步長造成誤差底限，而遞減步長可以移除它（[Robbins & Monro 1951](/zh-tw/references/#robbins-1951)；[Bottou et al. 2018](/zh-tw/references/#bottou-2018)）。

實際的資料載入器常在每個**訓練輪次（epoch）**打亂一次，再以不放回方式取用樣本。若固定 $\theta$，完整一輪的平均就是全資料梯度；但訓練時，相鄰小批次彼此相依，而且參數不斷改變。因此，上面的 $\Sigma/B$ 證明精確適用於彼此獨立的放回抽樣，不能原封不動套到每一種資料載入實作。

## 3.3.8 動量與自適應步長

§3.2.3 說明安全步長受*最大*曲率限制。如果不同方向的曲率差異很大，對陡峭方向安全的步長，對平坦方向就太小，於是梯度下降法會在陡峭方向上來回鋸齒，在平坦方向上緩慢爬行（圖 3.2，以及 §3.5.4 的精確分析）。由此產生兩類補救方法。

**動量（momentum；重球法）。** 維持一個累積過去梯度的「速度」（[Polyak 1964](/zh-tw/references/#polyak-1964)）：

$$
v_{k+1} = \beta v_k - \eta\nabla L(\theta_k),\qquad \theta_{k+1} = \theta_k + v_{k+1},\qquad 0\le\beta<1 .
$$

在 $v_0=0$ 時展開速度，可得 $v_{k+1}=-\eta\sum_{j=0}^{k}\beta^{k-j}\nabla L(\theta_j)$，也就是過去梯度的指數加權和，有效視窗約為 $1/(1-\beta)$ 步（與單元 05 的移動平均是同一個數學物件）。符號持續一致的梯度分量會累積；交替的分量則可能抵銷。「重球」名稱來自慣性與摩擦的物理類比，但真正的定義仍是上方的離散更新。對特徵值落在已知區間內的正定二次函數，調好的動量會把最壞情況收縮因子從 $(\kappa-1)/(\kappa+1)$ 改善到 $(\sqrt\kappa-1)/(\sqrt\kappa+1)$（§3.5.7）——當 $\kappa=10$，從 0.818 降到 0.519。

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

*這個更新在做什麼。* 忽略 $\epsilon$ 時，比值 $\hat m/\sqrt{\hat v}$ 沒有單位：把所有梯度乘上常數 $c>0$，比值不變。第一步有 $\hat m_1=g_1$、$\hat v_1=g_1^2$，所以非零座標會移動

$$
-\eta\frac{g_1}{|g_1|+\epsilon},
$$

只有在 $|g_1|\gg\epsilon$ 時，才近似 $-\eta\operatorname{sign}(g_1)$（詳解例題 3.7）。之後若梯度方向持續一致，$|\hat m|$ 可以接近 $\sqrt{\hat v}$；符號改變或變動很大時，這個比值會縮小。這是一種實用的逐座標正規化啟發法，不是反曲率的精確估計，也不是普遍的收斂保證。

**AdamW：解耦的權重衰減。** 在損失加入 $\tfrac\lambda2\|\theta\|^2$，會在梯度加入 $\lambda\theta$。Adam 接著把這一項混入兩個動差估計，並對它做自適應縮放，所以所得更新不等同於普通 SGD 權重衰減所產生的簡單乘法收縮。AdamW 改為在自適應梯度變換之外施加收縮（[Loshchilov & Hutter 2019](/zh-tw/references/#loshchilov-2019)）：

$$
\theta_k = \theta_{k-1} - \eta\Bigl(\frac{\hat m_k}{\sqrt{\hat v_k}+\epsilon} + \lambda\,\theta_{k-1}\Bigr).
$$

等價地說，顯示的公式會先把舊參數乘上 $1-\eta\lambda$，再套用自適應更新。AdamW 是 Transformer 常用的最佳化器，也是主幹專案相關階段採用的選擇；流行的工程慣例不等於它對每種模型都最佳。[AdamW Atlas 節點](/zh-tw/atlas/nodes/opt-adamw/)列出了它的不變量與常見實作錯誤。

## 3.3.9 建構引擎

實驗室要求你為純量實作反向模式。設計直接由 §3.3.3 而來：

- **一種節點型別**（`Value`）儲存它的數值（`data`）、它的伴隨值（`grad`，初始為 0）、對父節點的參照，以及一個小函式：給定自己的伴隨值，依規則 $\bar p \mathrel{+}= \bar c\,\partial c/\partial p$ 把貢獻加到父節點的伴隨值上。
- **每個運算**（`+`、`*`、`tanh`……）建立一個新節點，並附上正確的局部導數函式，同時捕捉它需要的值（例如 $\tanh$ 捕捉輸出 $t$，使用 $1-t^2$）。
- **`backward()`** 對輸出可達的計算圖建立拓撲排序，把輸出的伴隨值設為 1，再以相反順序呼叫每個節點的局部函式。實驗室要求使用迭代走訪，因為遞迴深度優先搜尋可能在深達 10,000 個節點的圖上使語言呼叫堆疊溢位。
- **在這個引擎中，歸零是呼叫者的責任。** 伴隨值會刻意累加，不論是在同一張扇出圖內，還是在共用葉節點的多次反向呼叫之間。因此，除非本來就想跨批次累積梯度，最佳化迴圈必須在計算下一次更新前重設參數伴隨值。

[實驗室](https://github.com/yhxuee/deep-learning-intelligent-systems/tree/main/labs/unit-03-autodiff)正是測試這些性質：詳解例題的數字、扇出下的累加（`a*a`、`a+a`）、在隨機運算式圖上與中央差分一致、線性性、深層圖，以及跨圖的累加。通過測試，驗證的是這個有明確範圍的純量實作。PyTorch 與 JAX 等張量框架採用相同的反向模式想法，每個基本運算有一條 VJP 規則，此外還要處理廣播、裝置、突變規則、控制流程，以及大量效能最佳化。從單元 04 起，主幹專案就使用這些框架。
