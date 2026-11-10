---
title: "2.3 數學建構"
description: 線性可分性、把嵌入查找看成矩陣乘積及其散射加總梯度、秩 d 分解式二元語法及其梯度、神經 n 元語法、最佳低秩近似，以及不變性與等變性的精確定義。
sidebar: { label: "2.3 數學建構", order: 3 }
---

本節將 §2.2 的幾何直覺精確轉化為現代深度學習的張量代數系統。在閱讀每一個矩陣方程式時，請始終遵循一套四步思維協議：

1. **命名本體：** 說出每一個符號在客觀世界中所代表的物理或語義對象；
2. **形狀校驗（Shape Check）：** 嚴格檢查每個因子的張量維度（Dimensions）；
3. **參數複用識別：** 找出哪些底層權重在正向或反向傳播的多個路徑中被同時複用；
4. **沿著計算圖回溯：** 搞清楚梯度反向流動的每一條分支與匯聚點。

全書反覆出現的關鍵維度符號約定如下：詞彙表大小 $V$、嵌入向量維度 $d$、批次大小 $N$、上下文長度 $k$，以及神經網絡隱藏層寬度 $H$。維度不匹配絕非粗心的書寫瑕疵；在神經網絡系統中，它往往直接揭示出致命的張量轉置錯誤，或是把梯度誤送到了計算圖的錯誤節點。

## 2.3.1 線性讀出與可分性

令輸入對象 $x$ 映射至高維特徵空間中的向量 $\phi(x) \in \mathbb{R}^d$。一個定義在該表示上的二元線性分類器，其決策邊界由超平面參數 $(w, b)$ 決定，根據判別式 $w^{\top}\phi(x) + b$ 的代數符號進行正負類別預測。

若存在至少一組參數 $(w, b) \in \mathbb{R}^d \times \mathbb{R}$，能夠使給定數據集中的每一個樣本均被正確劃分，則稱該標註數據集**在表示 $\phi$ 下是線性可分的（Linearly Separable）**。
請敏銳地注意：**可分性並不是原始數據對象自身的固有屬性，而是數據在特定幾何表示 $\phi$ 下的相互關係。** 同一組複雜的客觀數據（例如 XOR 異或邏輯），在原始原始坐標下可能徹底不可分，但在經過某個非線性變換 $\phi$ 投影到新幾何空間後，卻可能變得輕而易舉地線性可分。

若任務需要輸出校準的類別機率分佈，二元分類使用 **Logistic（Sigmoid）** 讀出層；多類別分類（如預測詞彙表中 $V$ 個候選詞）則使用 **Softmax** 讀出層：

$$
q(y\mid x) = \frac{\exp(w_y^{\top}\phi(x) + b_y)}{\sum_{y'}\exp(w_{y'}^{\top}\phi(x) + b_{y'})}.
$$

### 為什麼含有指數函數的 Softmax 依然被稱為「線性讀出」？
初學者常感到困惑：Softmax 運算中顯然包含非線性的指數運算 $\exp(\cdot)$ 與歸一化分母，為何學術界堅定地將其歸類為線性模型？
考察任意兩個候選類別 $y$ 與 $y'$ 的機率比值，它們共同的歸一化分母在相除時優雅地完全對消：

$$
\ln\frac{q(y\mid x)}{q(y'\mid x)}
=(w_y-w_{y'})^{\top}\phi(x)+(b_y-b_{y'}).
$$

取自然對數後，**任意一對類別之間的對數勝算（Log-Odds / Logit Difference），嚴格等於特徵向量 $\phi(x)$ 的線性仿射函數！**
這賦予了「任務在該表示下很簡單」一個精確的幾何與代數定義：一個優秀的表徵 $\phi(x)$，應當將輸入流形解開，使得下游任務所需的對數勝算曲面，恰好與該表示空間中的超平面高度平行。

## 2.3.2 嵌入查找是一個矩陣乘積

令 $E \in \mathbb{R}^{V\times d}$ 為模型的詞嵌入矩陣（Embedding Table），其第 $a$ 列儲存了詞彙符號 $a$ 所對應的 $d$ 維連續幾何向量。
為了避免符號混淆，我們用標準基向量 $\mathbf{u}_a \in \{0, 1\}^V$ 表示符號 $a$ 的 One-Hot（獨熱）編碼，並用 $e_a = E[a] \in \mathbb{R}^d$ 表示其稠密嵌入向量。則符號查找可以嚴格寫作轉置矩陣與獨熱向量的矩陣乘積：

$$
E^{\top}\mathbf u_a=e_a \qquad(E\text{ 的第 }a\text{ 列，以行向量寫出}).
$$

在一個包含 $N$ 個樣本的批次中，將各樣本輸入符號 $x_1, \dots, x_N$ 的獨熱行向量堆疊為大矩陣 $U \in \{0, 1\}^{N\times V}$。則整批樣本的查表過程在代數上等價於純粹的矩陣乘法：

$$
H=UE\in\mathbb R^{N\times d}.
$$

維度校驗：$(N\times V) \times (V\times d) = N\times d$。在工業級底層實作中，GPU 核心直接執行記憶體指針尋址（即 `E[x]`），絕不會在內存中真正分配幾乎全為 0 的稀疏矩陣 $U$。但將其寫作等價的矩陣乘法，能使反向傳播的鏈式法則推導一覽無餘。

**命題 2.1（查找操作的梯度傳播）。** 若純量標量損失 $L$ 僅透過批次表示 $H=UE$ 依賴於嵌入矩陣 $E$，且抵達 $H$ 的上游梯度記為 $G = \partial L/\partial H \in \mathbb{R}^{N\times d}$，則損失對底層嵌入矩陣 $E$ 的梯度嚴格為：

$$
\frac{\partial L}{\partial E} = U^{\top}G,\qquad\text{亦即}\qquad \frac{\partial L}{\partial E[a]} = \sum_{i:\,x_i = a} G[i].
$$

在閱讀數學推導之前，請先理解其物理本質：在同一個批次中，若符號 $a$ 在多個不同位置（如第 2 個與第 5 個樣本）被重複查閱，這些位置所獲取的都是同一套參數 $E[a]$。根據多元微積分的鏈式法則，**所有引用了該參數的下游節點所傳回的梯度分量，必須全部匯流並累加到該參數行之上！**

*證明。* 由前向定義，$H_{ik} = \sum_{v=1}^V U_{iv} E_{vk}$。對特定的參數元素求偏導：$\partial H_{ik} / \partial E_{vk} = U_{iv}$。
由鏈式法則，損失對 $E_{vk}$ 的總偏導為所有依賴路徑之和：
$\frac{\partial L}{\partial E_{vk}} = \sum_{i=1}^N \frac{\partial L}{\partial H_{ik}} \frac{\partial H_{ik}}{\partial E_{vk}} = \sum_{i=1}^N G_{ik} U_{iv} = \sum_{i=1}^N (U^{\top})_{vi} G_{ik} = (U^{\top}G)_{vk}$。
將此矩陣形式展開到第 $a$ 列：由於 $U_{ia} = 1$ 當且僅當第 $i$ 個樣本的輸入符號為 $a$（其餘為 0），因此矩陣乘積 $U^{\top}G$ 的第 $a$ 列正好等於所有輸入等於 $a$ 的樣本索引 $i$ 對應的梯度行 $G[i]$ 之和。$\blacksquare$

### 工程陷阱：散射加總（Scatter-Add）與並發競態
在深度學習框架底層（NumPy / PyTorch），這項累加運算被稱為 **Scatter-Add（散射加總）**：`np.add.at(dE, x, G)` 或 `dE.index_add_(0, x, G)`。
初學者常犯的經典重大 bug 是使用普通賦值語句：`dE[x] = G`。如果同一個詞在批次中出現了多次，普通的索引賦值會導致後出現的梯度直接覆蓋掉先前的梯度，造成致命的梯度遺失與模型發散！在 GPU 多線程並行架構中，多個線程同時向同一行 $E[a]$ 寫入梯度時，必須仰賴硬體級別的原子加法指令（`atomicAdd`）以消除競態條件（Race Condition）。

## 2.3.3 分解式二元語法及其梯度

現在，我們用連續嵌入向量徹底重構單元 01 的二元語法模型。我們為任意「前驅字元 $a$ 接續後繼字元 $b$」的轉移打分，分數記為未歸一化的 logit $z_{ab}$：

$$
z_{ab} = e_a^{\top}Me_b + c_b,\qquad q(b\mid a) = \mathrm{softmax}_b(z_{a\cdot}),
$$

這項雙線性打分公式蘊含著優美的幾何幾何含義：
1. **權重綁定（Weight Tying）：** 同一張嵌入矩陣 $E \in \mathbb{R}^{V\times d}$ 在輸入端與輸出端被同時複用——$e_a = E[a]$ 代表觀察到的輸入實體，$e_b = E[b]$ 代表候選預測對象；
2. **語意躍遷矩陣 $M \in \mathbb{R}^{d\times d}$：** 試想，如果沒有矩陣 $M$（即 $M = I$），則打分退化為內積 $z_{ab} = e_a^\top e_b + c_b$。為了讓轉移機率最大化，模型會被迫將頻繁相鄰的詞（如 `q` 與 `u`）的嵌入向量拉到完全重合的方向。但這在語義上是災難性的，因為相鄰的詞在語義上往往截然不同。**矩陣 $M$ 扮演了嵌入空間中的語義旋轉與剪切算子**：它將前驅向量 $e_a$ 投影為「對下一個詞的期望特徵特徵向量」 $M^\top e_a$，再與後繼候選 $e_b$ 進行內積比對；
3. **下一個符號偏置 $c_b$：** 捕捉符號 $b$ 的邊際獨立出現頻率（類似一元語法）。

在批次矩陣計算中，令輸入索引查出的特徵矩陣為 $H = E[x] \in \mathbb{R}^{N\times d}$，則整批樣本在全部 $V$ 個候選詞上的 logit 矩陣為：

$$
Z=HME^{\top}+\mathbf1c^{\top}\in\mathbb R^{N\times V}.
$$

張量維度核驗：$(N\times d) \times (d\times d) \times (d\times V) + (N\times 1)(1\times V) = N\times V$。
核心交互項 $HME^{\top}$ 的代數秩最高不超過 $d$。我們用一個僅包含 $Vd + d^2 + V$ 個參數的低秩幾何系統，成功逼近了原本包含 $V^2$ 個參數的完整離散大矩陣！

令交叉熵損失傳回的 logit 誤差矩陣為 $\Delta = \partial L/\partial Z = (Q - Y)/N \in \mathbb{R}^{N\times V}$（此處 $Q$ 為 Softmax 預測機率矩陣，$Y$ 為標籤的 One-Hot 矩陣，完整證明見單元 03 §3.3.5）。對矩陣方程 $Z = HME^{\top} + \mathbf{1}c^{\top}$ 求各參數的偏導數：

$$
\frac{\partial L}{\partial c} = \Delta^{\top}\mathbf 1,\qquad
\frac{\partial L}{\partial M} = H^{\top}\Delta E,\qquad
\frac{\partial L}{\partial E}\Big|_{\text{輸出側}} = \Delta^{\top}HM,\qquad
\frac{\partial L}{\partial H} = \Delta EM^{\top}.
$$

### 雙路徑梯度匯聚
由於我們對輸入與輸出採用了權重綁定，嵌入表 $E$ 同時承擔了兩個角色，因而會沿著計算圖的兩條不同分支同時收到反向梯度：
- **輸出路徑：** 每個候選向量作為矩陣 $E^\top$ 的一列參與了分類比對，直接貢獻梯度 $\Delta^\top H M$；
- **輸入路徑：** 觀測到的前驅符號查表形成了 $H$，抵達 $H$ 的梯度為 $\Delta E M^\top$，根據命題 2.1，該梯度必須透過 $U^\top$ 散射加總回輸入行。

兩條路徑在代數上完美累加：

$$
\frac{\partial L}{\partial E} = \Delta^{\top}HM + U^{\top}(\Delta EM^{\top}).
$$

每個推導均完美遵循矩陣微分的通用法則：「若 $Y = AXB$，則 $\partial L/\partial X = A^\top (\partial L/\partial Y) B^\top$」。讀者可用維度進行自檢：例如 $H^\top \Delta E$ 的維度為 $(d\times N) \times (N\times V) \times (V\times d) = d\times d$，與矩陣 $M$ 的尺寸嚴格吻合。

## 2.3.4 神經 n 元語法

如果上下文包含 $k$ 個連續歷史符號 $(x_{t-k}, \dots, x_{t-1})$，神經模型如何超越離散表格的組合爆炸？
經典的 Bengio 神經語言模型（[Bengio et al. 2003](/zh-tw/references/#bengio-2003)）將這 $k$ 個符號的嵌入向量拼接為長向量 $u = [e_{x_{t-k}}; \dots; e_{x_{t-1}}] \in \mathbb{R}^{kd}$，並透過單隱藏層多層感知機（MLP）進行非線性特徵合成：

$$
a = W_1^{\top}u + b_1,\quad h = \tanh(a),\quad z = W_2^{\top}h + b_2,\quad q = \mathrm{softmax}(z),
$$

其中第一層投影權重為 $W_1 \in \mathbb{R}^{kd\times H}$，輸出層權重為 $W_2 \in \mathbb{R}^{H\times V}$。
模型的總參數量為五項之和：$Vd + kdH + H + HV + V$。
以典型設定 $V = 50, d = 16, k = 4, H = 128$ 為例：
總參數量為 $800 + 8{,}192 + 128 + 6{,}400 + 50 = \mathbf{15{,}570}$ 個參數。
對比之下，同等上下文長度的傳統離散計數表，其最壞情況需要 $V^k = 50^4 = \mathbf{6{,}250{,}000}$ 列、共計 3.125 億個單元格！神經網絡將參數規模壓縮了整整兩萬倍，使長上下文學習成為現實。

對單一樣本，令輸出誤差向量為 $\delta_z = q - \mathbf{u}_y \in \mathbb{R}^V$。沿著前向計算圖依序逆向回溯：

$$
\frac{\partial L}{\partial W_2} = h\,\delta_z^{\top},\quad \frac{\partial L}{\partial b_2} = \delta_z,\quad
\delta_a = (W_2\delta_z)\odot(1 - h\odot h),\quad
\frac{\partial L}{\partial W_1} = u\,\delta_a^{\top},\quad
\frac{\partial L}{\partial u} = W_1\delta_a .
$$

其中 $\odot$ 代表 Hadamard 逐元素相乘，項 $1 - h\odot h$ 是雙曲正切激活函數 $\tanh'(a) = 1 - \tanh^2(a)$ 的精確導數。
最後，將梯度向量 $\partial L/\partial u \in \mathbb{R}^{kd}$ 均勻切分為 $k$ 個長度為 $d$ 的子切片。第 $j$ 個切片對應上下文歷史中第 $j$ 個位置所查閱的嵌入行，再次透過 Scatter-Add 散射累加回該符號的嵌入參數中。

## 2.3.5 最佳低秩近似與奇異值分解

低秩矩陣分解在什麼情況下能夠高保真地逼近一個龐大的離散表格？線性代數中的奇異值分解（Singular Value Decomposition, SVD）為我們提供了最嚴格的分析診斷工具。

任何秩為 $r$ 的實矩陣 $A \in \mathbb{R}^{m\times n}$ 均可精確分解為一組正交秩一外積矩陣的加權和：

$$
A=\sum_{i=1}^{r}\sigma_i u_i v_i^{\top},
$$

其中左奇異向量 $\{u_i\}$ 與右奇異向量 $\{v_i\}$ 分別構成正交歸一基底，且奇異值按降序排列：$\sigma_1 \ge \sigma_2 \ge \dots \ge \sigma_r > 0$。

**定理 2.1（Eckart–Young 最佳低秩逼近定理，[Eckart & Young 1936](/zh-tw/references/#eckart-1936)）。** 對於任意指定的正整數 $d < r$，在所有秩不超過 $d$ 的實矩陣集合 $\mathcal{M}_d = \{B \in \mathbb{R}^{m\times n} : \mathrm{rank}(B) \le d\}$ 中，截斷奇異值矩陣 $A_d = \sum_{i=1}^{d}\sigma_iu_iv_i^{\top}$ 是在 Frobenius 範數意義下距離 $A$ 最近的唯一最優逼近，其最小逼近誤差精確等於被捨棄的奇異值平方和的開方：

$$\min_{\mathrm{rank}(B)\le d} \lVert A - B\rVert_F = \lVert A - A_d \rVert_F = \sqrt{\sum_{i=d+1}^{r}\sigma_i^2}.$$

（完整證明見 §2.5.5 ★。）
若一個語言共現矩陣的奇異值光譜呈現出急劇的指數級衰減（即前 $d$ 個奇異值主導了整體能量，而後續奇異值迅速趨於零），Eckart-Young 定理保證了將維度壓縮至 $d$ 維只會損失微不足道的訊號。現代嵌入演算法（如 Word2Vec 與 GloVe）在數學極限下本質上是在隱式分解某種經過重整化的點互資訊（PMI）矩陣（[Levy & Goldberg 2014](/zh-tw/references/#levy-2014)）。

## 2.3.6 不變性與等變性：幾何先驗的設計藝術

表徵使得深度學習高效的另一條終極途徑，是**透過架構設計主動濾除任務無關的干擾資訊**。令 $g$ 代表輸入空間上的某種對稱群變換操作（例如圖像的空間平移、點雲的旋轉、符號序列的置換）：

- **不變性（Invariance）：** 若對任意群操作 $g$，特徵映射均滿足 $\phi(g\cdot x) = \phi(x)$，則稱表徵 $\phi$ 對變換 $g$ 具有不變性。變換輸入，特徵輸出紋絲不動；
- **等變性（Equivariance）：** 若映射函數滿足 $f(g\cdot x) = g\cdot f(x)$，則稱 $f$ 對群操作 $g$ 具有等變性。變換輸入，輸出以完全一致的方式同步變換。

兩者的選擇完全取決於下游任務的語義本質：
- 在圖像分類任務中，物體出現在畫面的左側還是右側不應改變「這是一隻貓」的結論，因此分類頭需要**平移不變性**；
- 在像素級圖像分割或物體檢測中，物體在畫面中平移，其分割掩碼必須同步平移，因此主幹網絡需要**平移等變性**（如卷積神經網絡的平移等變性定理 4.2）；
- 在單元 04 中我們將看到：未注入位置編碼的自注意力機制，對輸入序列的順序置換天然具有嚴格的**置換等變性（Permutation Equivariance）**（命題 4.2）。

請銘記幾何深度學習的雙刃劍鐵律（§2.7.1）：**一旦表徵在前期主動丟棄了某項特徵資訊，下游的任何神經網絡都不可能無中生有地將其挽回。** 不變性是一場嚴肅的工程賭注：賦予手寫數字識別網絡輕微的平移不變性是大有裨益的；但若草率地賦予其 180 度旋轉不變性，模型將永遠無法區分數字 `6` 與數字 `9`！
