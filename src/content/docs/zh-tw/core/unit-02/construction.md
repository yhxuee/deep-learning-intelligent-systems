---
title: "2.3 數學建構"
description: 線性可分性、把嵌入查找看成矩陣乘積及其散射加總梯度、秩 d 分解式二元語法及其梯度、神經 n 元語法、最佳低秩近似，以及不變性與等變性的精確定義。
sidebar: { label: "2.3 數學建構", order: 3 }
---

## 2.3.1 線性讀出與可分性

特徵 $\phi(x)\in\mathbb{R}^d$ 上的二元線性分類器預測 $\mathrm{sign}(w^{\top}\phi(x) + b)$。若某個 $(w, b)$ 能完美分類一組有標籤的資料，該資料在 $\phi$ 中就是**線性可分（linearly separable）**的。對機率輸出，對應的是 **logistic**（兩類）或 **softmax**（多類）讀出：

$$
q(y\mid x) = \frac{\exp(w_y^{\top}\phi(x) + b_y)}{\sum_{y'}\exp(w_{y'}^{\top}\phi(x) + b_{y'})}.
$$

任兩類之間的對數勝算 $\ln q(y\mid x) - \ln q(y'\mid x) = (w_y - w_{y'})^{\top}\phi(x) + (b_y - b_{y'})$ **對特徵是線性的**。所以「目標在這個表示中很簡單」對本書每個模型都有精確的意義：任務所需的對數勝算（接近）是抵達讀出層之表示的線性函數。改變 $\phi$，就改變了哪些對數勝算函數可以達到。

## 2.3.2 嵌入查找是一個矩陣乘積

令 $E\in\mathbb{R}^{V\times d}$，$e_a\in\mathbb{R}^{V}$ 為符號 $a$ 的獨熱向量（可能與嵌入 $e_a$ 混淆時，記為 $\mathbf{u}_a$）。則

$$
E^{\top}\mathbf{u}_a = E[a] \qquad(E \text{ 的第 } a \text{ 列}).
$$

對一批索引 $x_1,\dots,x_N$，把獨熱向量疊成 $U\in\{0,1\}^{N\times V}$ 的各列；查出的嵌入就是 $H = UE\in\mathbb{R}^{N\times d}$。實作中從不真的形成 $U$——索引便宜得多——但這個恆等式告訴我們梯度是什麼。

**命題 2.1（查找的梯度）。** 若純量損失 $L$ 只透過 $H = UE$ 依賴 $E$，且 $G = \partial L/\partial H$，則
$$
\frac{\partial L}{\partial E} = U^{\top}G,\qquad\text{亦即}\qquad \frac{\partial L}{\partial E[a]} = \sum_{i:\,x_i = a} G[i].
$$

*證明。* $H_{ik} = \sum_a U_{ia}E_{ak}$，所以 $\partial H_{ik}/\partial E_{ak} = U_{ia}$，由連鎖律 $\partial L/\partial E_{ak} = \sum_i G_{ik}U_{ia} = (U^{\top}G)_{ak}$。$U^{\top}G$ 的第 $a$ 列，就是把符號 $a$ 出現位置上的 $G$ 各列加總。$\blacksquare$

在程式中這是一個**散射加總（scatter-add）**：`np.add.at(dE, x, G)`。改用單純指定（`dE[x] = G`）是經典錯誤：當一個符號在同一批次中出現兩次，第二個梯度會覆蓋第一個，而不是加上去。主幹專案階段 02 的測試把散射加總與明確的 $U^{\top}G$ 相比較（詳解例題 2.7）。

## 2.3.3 分解式二元語法及其梯度

以下式為配對「$a$ 接著 $b$」打分：

$$
z_{ab} = e_a^{\top}Me_b + c_b,\qquad q(b\mid a) = \mathrm{softmax}_b(z_{a\cdot}),
$$

兩側使用**同一張**表 $E$（綁定），加上一個 $d\times d$ 矩陣 $M$ 與輸出偏置 $c$。以矩陣形式寫，對查出列為 $H = E[x]$（$N\times d$）的一批資料，分數為 $Z = HME^{\top} + \mathbf 1c^{\top}$（$N\times V$）。因為 $HME^{\top}$ 的秩至多為 $d$，這是**二元語法對數機率表的秩 $d$ 模型**（差一個逐列正規化）。

令 $\Delta = \partial L/\partial Z = (Q - Y)/N$，其中 $Q$ 為預測機率、$Y$ 為獨熱目標（softmax–交叉熵的梯度，於單元 03 §3.3.5 推導）。對 $Z = HME^{\top} + \mathbf 1c^{\top}$ 微分：

$$
\frac{\partial L}{\partial c} = \Delta^{\top}\mathbf 1,\qquad
\frac{\partial L}{\partial M} = H^{\top}\Delta E,\qquad
\frac{\partial L}{\partial E}\Big|_{\text{輸出側}} = \Delta^{\top}HM,\qquad
\frac{\partial L}{\partial H} = \Delta EM^{\top}.
$$

最後一項透過查找抵達 $E$（命題 2.1），所以綁定表的總梯度為

$$
\frac{\partial L}{\partial E} = \Delta^{\top}HM + U^{\top}(\Delta EM^{\top}).
$$

每一項都來自規則「對 $Y = AXB$，$\partial L/\partial X = A^{\top}(\partial L/\partial Y)B^{\top}$」，也就是對每個元素兩次套用單變數連鎖律（習題 2.8）。`spine/stage02_embeddings.py` 正是實作這四行，並以有限差分檢查。

**初始化與基準。** 當 $M = 0$、$c = 0$ 時，每個分數都是零、每個預測都是均勻的，所以初始損失恰好是 $\ln V$——單元 01 的基準，現在用來檢查程式碼。

**最佳值受計數所限。** 每個分解式二元語法都是一個二元語法，所以由命題 1.1，它的訓練損失永遠不可能低於計數式最大概似表。當 $d$ 增加到 $V$ 時，分解可以表示任何表，這個界就能達到。

## 2.3.4 神經 n 元語法

對 $k$ 個符號的上下文，把它們的嵌入串接成 $u = [e_{x_{t-k}};\dots;e_{x_{t-1}}]\in\mathbb{R}^{kd}$，並計算

$$
a = W_1^{\top}u + b_1,\quad h = \tanh(a),\quad z = W_2^{\top}h + b_2,\quad q = \mathrm{softmax}(z),
$$

其中 $W_1\in\mathbb{R}^{kd\times H}$、$W_2\in\mathbb{R}^{H\times V}$。其參數量為 $Vd + kdH + H + HV + V$：當 $V = 50$、$d = 16$、$k = 4$、$H = 128$ 時，為 $800 + 8{,}192 + 128 + 6{,}400 + 50 = 15{,}570$。相同上下文的計數表最多需要 $V^k = 6{,}250{,}000$ 列、每列 $V$ 個數。

單一樣本的反向傳遞，記 $\delta_z = q - \mathbf{u}_y$：

$$
\frac{\partial L}{\partial W_2} = h\,\delta_z^{\top},\quad \frac{\partial L}{\partial b_2} = \delta_z,\quad
\delta_a = (W_2\delta_z)\odot(1 - h\odot h),\quad
\frac{\partial L}{\partial W_1} = u\,\delta_a^{\top},\quad
\frac{\partial L}{\partial u} = W_1\delta_a .
$$

梯度 $\partial L/\partial u$ 切成 $k$ 段、每段長 $d$，第 $j$ 段散射加總到上下文第 $j$ 個位置符號所對應的列。在同一個上下文中出現兩次的符號（`aa`）會收到兩段。因子 $1 - h\odot h$ 是 $\tanh$ 的導數。

[Bengio et al. (2003)](/zh-tw/references/#bengio-2003) 在詞層級語料上使用這個架構（另加從 $u$ 到 $z$ 的直接線性連接），報告的測試困惑度大幅低於當時最好的平滑 n 元語法模型。我們的字元層級重現（詳解例題 2.6）顯示，在小資料量時有同樣的效果，而在大資料量時效果消失——原論文不需要展示這一點，但它闡明了嵌入*為什麼*有幫助。

## 2.3.5 最佳低秩近似

為什麼 $V\times V$ 表的秩 $d$ 模型會好？奇異值分解給出答案。任何實矩陣 $A\in\mathbb{R}^{m\times n}$ 都可寫成 $A = \sum_{i=1}^{r}\sigma_iu_iv_i^{\top}$，其中 $u_i$ 兩兩正交歸一、$v_i$ 兩兩正交歸一，且 $\sigma_1\ge\sigma_2\ge\dots\ge\sigma_r>0$。

**定理 2.1（Eckart–Young，[Eckart & Young 1936](/zh-tw/references/#eckart-1936)）。** 對每個 $d < r$，截斷 $A_d = \sum_{i=1}^{d}\sigma_iu_iv_i^{\top}$ 在所有秩至多為 $d$ 的矩陣 $B$ 中使 $\lVert A - B\rVert_F$ 最小，最小值為 $\sqrt{\sum_{i>d}\sigma_i^2}$。

（證明見 §2.5.5 ★。）如果共現統計矩陣的奇異值迅速衰減——少數強模式、許多弱模式——低秩表示就能捕捉大部分，而被捨棄的方向大多是有限計數造成的雜訊。古典的*潛在語意分析（latent semantic analysis）*正是對詞–文件矩陣計算這種截斷（[Deerwester et al. 1990](/zh-tw/references/#deerwester-1990)）。學習的嵌入並不計算 SVD，但 [Levy & Goldberg (2014)](/zh-tw/references/#levy-2014) 證明，流行的 skip-gram 嵌入目標在最佳點上，是對位移後的逐點互資訊矩陣的隱式分解。嵌入學習與低秩分解，是從兩個側面看到的同一個想法。

## 2.3.6 不變性與等變性

表示讓學習變容易的第二種方式，是**捨棄任務不依賴的東西**。令 $g$ 為輸入的變換——平移一張影像、置換一個集合、替故事中的朋友換名字。

- 若 $\phi(g\cdot x) = \phi(x)$，表示 $\phi$ 對 $g$ 是**不變的（invariant）**：輸入被變換時，表示不改變。
- 若 $f(g\cdot x) = g\cdot f(x)$，映射 $f$ 對 $g$ 是**等變的（equivariant）**：變換輸入，會以對應的方式變換輸出。

若目標是不變的（影像平移時類別不變），不變的表示就移除了學習者原本必須從資料中學到的干擾——每張訓練影像的每個平移版本。若目標是等變的（每個像素的標籤隨影像平移），等變的層就能保留結構，同時在各位置之間共享參數。卷積是對平移等變的線性映射；單元 04 證明了這一點（定理 4.2），並證明沒有位置資訊的注意力對位置置換是等變的（命題 4.2）。§2.5.4 給出一般的構造：對群取平均就產生不變性。

硬幣的另一面是 §2.7.1：**捨棄了任務所需資訊的表示，下游任何學習者都無法修補**。不變性是一場賭注，賭的是被捨棄的資訊無關緊要。
