---
title: "2.3 數學建構"
description: 線性可分性、把嵌入查找看成矩陣乘積及其散射加總梯度、秩 d 分解式二元語法及其梯度、神經 n 元語法、最佳低秩近似，以及不變性與等變性的精確定義。
sidebar: { label: "2.3 數學建構", order: 3 }
---

本節會把 §2.2 的圖像轉成矩陣方程式。每個式子都請按照同一順序閱讀：

1. 說出每個符號代表的物件；
2. 檢查每個因子的形狀；
3. 找出哪些參數被重複使用；
4. 最後才沿著導數往回走。

反覆出現的維度包括：$V$ 個詞彙符號、嵌入寬度 $d$、批次大小 $N$、上下文長度 $k$，以及隱藏寬度 $H$。形狀不符不只是記號上的小問題；它通常會顯示出錯誤的轉置，或送到錯誤位置的梯度。

## 2.3.1 線性讀出與可分性

特徵 $\phi(x)\in\mathbb R^d$ 上的二元線性分類器，依 $w^{\top}\phi(x)+b$ 的正負做預測。若至少有一組 $(w,b)$ 能把每個樣本都分對，這組有標籤資料就是**在 $\phi$ 中線性可分（linearly separable）**的。請注意，可分性屬於「某個特定表示中的資料」；同一批有標籤物件，在一種 $\phi$ 下可能可分，換一種則不可分。

若需要機率輸出，兩類使用 **logistic** 讀出，多類則使用 **softmax** 讀出：

$$
q(y\mid x) = \frac{\exp(w_y^{\top}\phi(x) + b_y)}{\sum_{y'}\exp(w_{y'}^{\top}\phi(x) + b_{y'})}.
$$

Softmax 明明含有指數函數，為什麼仍說它是線性讀出？比較兩個類別時，共同分母會消掉：

$$
\ln\frac{q(y\mid x)}{q(y'\mid x)}
=(w_y-w_{y'})^{\top}\phi(x)+(b_y-b_{y'}).
$$

因此，每一對類別的**對數勝算**都對特徵呈線性。「目標在這個表示中很簡單」現在有了精確版本：任務需要的對數勝算，應該接近抵達讀出層之表示的線性函數。改變 $\phi$ 會改變哪些對數勝算函數容易表達，但不會自行保證訓練一定找得到它們。

## 2.3.2 嵌入查找是一個矩陣乘積

令 $E\in\mathbb R^{V\times d}$ 為嵌入表。為了避免用 $e_a$ 表示兩種不同物件，我們以 $\mathbf u_a\in\mathbb R^V$ 表示符號 $a$ 的獨熱向量，並以 $e_a=E[a]\in\mathbb R^d$ 表示它學習出的嵌入。則

$$
E^{\top}\mathbf u_a=e_a \qquad(E\text{ 的第 }a\text{ 列，以行向量寫出}).
$$

對批次中的索引 $x_1,\dots,x_N$，把它們的獨熱向量疊成 $U\in\{0,1\}^{N\times V}$ 的各列。所有查出的嵌入就是

$$
H=UE\in\mathbb R^{N\times d}.
$$

形狀檢查為 $(N\times V)(V\times d)=N\times d$。真正的實作會使用 `E[x]`，不會建立幾乎全為零的矩陣 $U$；但把等價的矩陣乘積寫出來，能讓反向傳遞變得可見。

**命題 2.1（查找的梯度）。** 若純量損失 $L$ 只透過 $H=UE$ 依賴 $E$，且 $G=\partial L/\partial H\in\mathbb R^{N\times d}$，則
$$
\frac{\partial L}{\partial E} = U^{\top}G,\qquad\text{亦即}\qquad \frac{\partial L}{\partial E[a]} = \sum_{i:\,x_i = a} G[i].
$$

證明之前，先用文字讀結果：符號 $a$ 每次出現時，都查找了同一列參數 $E[a]$，所以這些位置的梯度貢獻都必須回到該列，並且加總起來。

*證明。* 因為 $H_{ik}=\sum_aU_{ia}E_{ak}$，所以 $\partial H_{ik}/\partial E_{ak}=U_{ia}$。連鎖律給出 $\partial L/\partial E_{ak}=\sum_iG_{ik}U_{ia}=(U^{\top}G)_{ak}$。因此，$U^{\top}G$ 的第 $a$ 列，恰好加總了符號 $a$ 出現位置上的 $G$ 各列。$\blacksquare$

在程式中這是一個**散射加總（scatter-add）**：`np.add.at(dE, x, G)`。改用單純指定（`dE[x] = G`）是經典錯誤：當一個符號在同一批次中出現兩次，第二個梯度會覆蓋第一個，而不是加上去。主幹專案階段 02 的測試把散射加總與明確的 $U^{\top}G$ 相比較（詳解例題 2.7）。

## 2.3.3 分解式二元語法及其梯度

以下式為配對「$a$ 接著 $b$」打分：

$$
z_{ab} = e_a^{\top}Me_b + c_b,\qquad q(b\mid a) = \mathrm{softmax}_b(z_{a\cdot}),
$$

同一張表 $E$ 會出現兩次：$e_a$ 表示觀察到的輸入，$e_b$ 表示一個候選輸出。這就是權重綁定。矩陣 $M\in\mathbb R^{d\times d}$ 讓輸入與輸出座標交互作用，$c_b$ 則是所有輸入共用的「下一個符號」偏置。

對一個批次，$H=E[x]\in\mathbb R^{N\times d}$，且

$$
Z=HME^{\top}+\mathbf1c^{\top}\in\mathbb R^{N\times V}.
$$

檢查主要路徑：$(N\times d)(d\times d)(d\times V)=N\times V$。交互作用項 $HME^{\top}$ 的秩至多為 $d$。更精確地說，這是**輸入–輸出交互作用 logit 矩陣**的低秩模型；加入的偏置 $\mathbf1c^{\top}$，以及 log-softmax 產生的逐列常數，都是額外的秩一項。直接把整張對數機率表稱為「秩 $d$」，會隱藏這些限定。

令 $\Delta=\partial L/\partial Z=(Q-Y)/N$，其中 $Q$ 是預測機率，$Y$ 是獨熱目標。這項 softmax–交叉熵結果會在單元 03 §3.3.5 推導；此處先把 $\Delta$ 每一列讀成「預測機率減去目標機率」。對 $Z=HME^{\top}+\mathbf1c^{\top}$ 微分，得到

$$
\frac{\partial L}{\partial c} = \Delta^{\top}\mathbf 1,\qquad
\frac{\partial L}{\partial M} = H^{\top}\Delta E,\qquad
\frac{\partial L}{\partial E}\Big|_{\text{輸出側}} = \Delta^{\top}HM,\qquad
\frac{\partial L}{\partial H} = \Delta EM^{\top}.
$$

綁定表 $E$ 會從兩條路徑收到梯度：

- **輸出路徑：** 每個候選向量都出現在 $E^{\top}$ 中；
- **輸入路徑：** 觀察到的符號先經查找形成 $H=UE$。

上一個式子中的最後一項，是抵達 $H$ 的梯度。命題 2.1 會把它從輸入路徑散射回去，因此兩條路徑必須相加：

$$
\frac{\partial L}{\partial E} = \Delta^{\top}HM + U^{\top}(\Delta EM^{\top}).
$$

每個式子都來自規則「對 $Y=AXB$，$\partial L/\partial X=A^{\top}(\partial L/\partial Y)B^{\top}$」；它本身只是把純量連鎖律逐元素套用（習題 2.8）。也可以用形狀逐項稽核：例如 $H^{\top}\Delta E$ 的形狀是 $(d\times N)(N\times V)(V\times d)=d\times d$，正好與 $M$ 相同。`spine/stage02_embeddings.py` 實作了這些式子，並以有限差分檢查。

**初始化與基準。** 當 $M = 0$、$c = 0$ 時，每個分數都是零、每個預測都是均勻的，所以初始損失恰好是 $\ln V$——單元 01 的基準，現在用來檢查程式碼。

**最佳值受計數所限。** 每個分解式二元語法都是一個二元語法，所以由命題 1.1，它的訓練損失永遠不可能低於計數式最大概似表。當 $d$ 增加到 $V$ 時，分解可以表示任何表，這個界就能達到。

## 2.3.4 神經 n 元語法

對 $k$ 個符號的上下文，把它們的嵌入串接成 $u = [e_{x_{t-k}};\dots;e_{x_{t-1}}]\in\mathbb{R}^{kd}$，並計算

$$
a = W_1^{\top}u + b_1,\quad h = \tanh(a),\quad z = W_2^{\top}h + b_2,\quad q = \mathrm{softmax}(z),
$$

其中 $W_1\in\mathbb{R}^{kd\times H}$、$W_2\in\mathbb{R}^{H\times V}$。其參數量為 $Vd + kdH + H + HV + V$：當 $V = 50$、$d = 16$、$k = 4$、$H = 128$ 時，為 $800 + 8{,}192 + 128 + 6{,}400 + 50 = 15{,}570$。相同上下文的計數表最多需要 $V^k = 6{,}250{,}000$ 列、每列 $V$ 個數。

這五項依序計算嵌入表、第一層權重、第一層偏置、輸出權重與輸出偏置。因此，計數表最壞情況的儲存量是 $V^{k+1}$ 個數值；不過，稀疏實作只會儲存觀察到的上下文。神經模型之所以較小，是因為許多上下文都必須通過同一組 $W_1$ 與 $W_2$；這項共享既是它的優點，也是限制。

對單一樣本，從 $\delta_z=q-\mathbf u_y$ 開始反向傳遞，再以與前向方程式相反的順序往回走：

$$
\frac{\partial L}{\partial W_2} = h\,\delta_z^{\top},\quad \frac{\partial L}{\partial b_2} = \delta_z,\quad
\delta_a = (W_2\delta_z)\odot(1 - h\odot h),\quad
\frac{\partial L}{\partial W_1} = u\,\delta_a^{\top},\quad
\frac{\partial L}{\partial u} = W_1\delta_a .
$$

因子 $1-h\odot h$ 是 $\tanh$ 的逐元素導數。最後，把 $\partial L/\partial u$ 切成 $k$ 段，每段長 $d$。第 $j$ 段屬於上下文位置 $j$ 所查找的嵌入，因此要散射加總到該符號的列。如果同一符號在一個上下文中出現兩次，例如 `aa`，兩段都必須加上；這與命題 2.1 的重複索引問題完全相同。

[Bengio et al. (2003)](/zh-tw/references/#bengio-2003) 在詞級語料上使用這個架構（另加從 $u$ 到 $z$ 的直接線性連接），報告的測試困惑度大幅低於當時最好的平滑 n 元語法模型。我們的字元級重現（詳解例題 2.6）顯示，在小資料量時有同樣的效果，而在大資料量時效果消失——原論文不需要展示這一點，但它闡明了嵌入*為什麼*有幫助。

## 2.3.5 最佳低秩近似

低秩模型在什麼情況下能妥善近似一張大表？奇異值分解給出精確的診斷方式。任何秩為 $r$ 的實矩陣 $A\in\mathbb R^{m\times n}$ 都可寫成

$$
A=\sum_{i=1}^{r}\sigma_i u_i v_i^{\top},
$$

其中 $u_i$ 彼此正交歸一，$v_i$ 也彼此正交歸一，而且 $\sigma_1\ge\sigma_2\ge\dots\ge\sigma_r>0$。每一項都是一個秩一模式，$\sigma_i$ 則衡量其強度。

**定理 2.1（Eckart–Young，[Eckart & Young 1936](/zh-tw/references/#eckart-1936)）。** 對每個 $d < r$，截斷 $A_d = \sum_{i=1}^{d}\sigma_iu_iv_i^{\top}$ 在所有秩至多為 $d$ 的矩陣 $B$ 中使 $\lVert A - B\rVert_F$ 最小，最小值為 $\sqrt{\sum_{i>d}\sigma_i^2}$。

（證明見 §2.5.5 ★。）若共現矩陣的奇異值迅速衰減，定理便告訴我們：低秩近似的 Frobenius 誤差很小。但它**沒有**說明尾部為什麼小，也沒有證明被捨棄的方向就是雜訊；這種解讀還需要抽樣模型或保留資料證據。古典的*潛在語意分析（latent semantic analysis）*會對詞–文件矩陣計算這項截斷（[Deerwester et al. 1990](/zh-tw/references/#deerwester-1990)）。學習出的嵌入一般不會直接計算 SVD。對一個重要特例，[Levy & Goldberg (2014)](/zh-tw/references/#levy-2014) 證明，skip-gram 目標在最佳點會隱式分解一個位移後的逐點互資訊矩陣。因此，低秩分解是理解某些嵌入目標的有用視角，不是涵蓋所有學習表示的恆等關係。

## 2.3.6 不變性與等變性

表示可能讓學習變容易的第二種方式，是**捨棄任務不依賴的東西**。令 $g$ 為輸入的變換——平移一張影像、置換一個集合，或一致地替故事中的朋友換名字。

- 若 $\phi(g\cdot x) = \phi(x)$，表示 $\phi$ 對 $g$ 是**不變的（invariant）**：輸入被變換時，表示不改變。
- 若 $f(g\cdot x) = g\cdot f(x)$，映射 $f$ 對 $g$ 是**等變的（equivariant）**：變換輸入，會以對應的方式變換輸出。

兩者的差異在於理想輸出應該如何變化。影像分類時，平移輸入不應改變類別，因此需要不變性。逐像素分割時，輸出遮罩應該跟著影像一起平移，因此需要等變性。在單元 04 明確採用的邊界慣例下，卷積是對平移等變的線性映射（定理 4.2）；該單元也證明，沒有位置資訊的注意力對位置置換是等變的（命題 4.2）。§2.5.4 則透過對有限群取平均，給出一般的不變構造。

硬幣的另一面是 §2.7.1：**捨棄了任務所需資訊的表示，下游任何學習者都無法修補**。因此，不變性是一項與任務相關的賭注。讓數字辨識器對小幅平移不變可能有幫助；若讓它對所有旋轉都不變，卻可能把 `6` 與 `9` 視為相同，直接破壞答案。
