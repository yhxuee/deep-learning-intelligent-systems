---
title: "4.3 數學建構：從注意力到 Transformer 區塊"
description: 矩陣形式的縮放點積注意力；因果遮罩與置換等變性及其證明；位置編碼（含 RoPE）；多頭注意力；反向傳遞；MLP、殘差流與層正規化；僅解碼器語言模型；以 KV 快取生成。
sidebar: { label: "4.3 數學建構", order: 3 }
---

## 4.3.1 矩陣形式的注意力

把所有位置的查詢、鍵與值堆成列。在矩陣形式中，把投影矩陣存成 §4.2.7 的轉置（程式碼也這樣做）會比較方便：

$$
Q = XW_Q\in\mathbb{R}^{T\times d_k},\qquad K = XW_K\in\mathbb{R}^{T\times d_k},\qquad V = XW_V\in\mathbb{R}^{T\times d_v},
$$

其中 $W_Q, W_K\in\mathbb{R}^{d\times d_k}$、$W_V\in\mathbb{R}^{d\times d_v}$。全部 $T^2$ 個分數是一次矩陣乘法，整層就是

$$
\boxed{\;\mathrm{Attn}(Q,K,V) = \mathrm{softmax}\!\Bigl(\frac{QK^{\top}}{\sqrt{d_k}}\Bigr)V\;}
$$

其中 softmax 對**每一列**分別作用。分步驟來看：

1. **分數** $S = QK^{\top}/\sqrt{d_k}\in\mathbb{R}^{T\times T}$：$S_{ij}$ 是查詢 $i$ 與鍵 $j$ 的匹配程度。
2. **權重** $A = \mathrm{softmax}_{\text{列}}(S)$：$A$ 的每一列都是位置上的機率分布。
3. **輸出** $Y = AV$：$Y$ 的第 $i$ 列是值向量以 $A_{i\cdot}$ 加權的平均。

每一步都值得檢查形狀：$T\times d_k$ 乘 $d_k\times T$ 得 $T\times T$；$T\times T$ 乘 $T\times d_v$ 得 $T\times d_v$。softmax 必須對**鍵**的索引 $j$（沿每一列）作用：對查詢索引正規化，是 [Atlas](/zh-tw/atlas/nodes/mech-attention/) 中五個經典注意力錯誤的第一個。

當 $Q$、$K$、$V$ 都來自同一個序列時，這一層是**自注意力（self-attention）**。當查詢來自一個序列、鍵與值來自另一個序列時（翻譯解碼器讀取編碼後的原文，或文字模型讀取影像特徵），則是**交叉注意力（cross-attention）**。數學完全相同。

## 4.3.2 因果遮罩

語言模型被訓練來從詞元 $1,\dots,t$ 預測詞元 $t+1$。如果位置 $t$ 能注意位置 $t+1$，模型就能直接複製答案，訓練訊號也就毫無意義。修正方法是**因果遮罩（causal mask）**：在 softmax 之前，把矩陣 $M$ 加到分數上，其中 $j\le i$ 時 $M_{ij} = 0$、$j>i$ 時 $M_{ij} = -\infty$：

$$
A = \mathrm{softmax}_{\text{列}}(S + M).
$$

因為 $e^{-\infty} = 0$，被遮罩的元素權重恰好為 0，每一列在剩下的位置上仍然總和為 1。

**命題 4.1（因果性）。** 有因果遮罩時，輸出 $y_i$ 只依賴 $x_1,\dots,x_i$。

*證明。* $q_i$ 只由 $x_i$ 算出，$k_j, v_j$ 只由 $x_j$ 算出。$A$ 的第 $i$ 列只在 $j\le i$ 處非零，每一個都是 $\exp(q_i^{\top}k_j/\sqrt{d_k})$ 除以對 $j'\le i$ 的總和——所以只涉及 $x_1,\dots,x_i$。於是 $y_i = \sum_{j\le i}A_{ij}v_j$。$\blacksquare$

因果性讓僅解碼器模型能**在一次前向傳遞中訓練序列的所有位置**：對每個 $t$，位置 $t$ 同時預測詞元 $t+1$，而沒有任何一個能看到自己的答案。它也是一個可以精確測試的不變量（實驗室會擾動未來的輸入，檢查過去的輸出是否不變）。

兩個陷阱：在 softmax **之後**才套用遮罩（把權重設為零卻不重新正規化），會同時破壞正規化*與*因果性，因為正規化的總和已經包含了未來的分數；而差一位的遮罩（$j < i$ 而非 $j\le i$）不是遮住當前詞元，就是往另一邊偏移、洩漏下一個詞元。

:::note[主張檢核]
- **主張。** 加上因果遮罩後，輸出 $y_i$ 只依賴 $x_1,\dots,x_i$（命題 4.1）。
- **假設。** 遮罩是在 softmax 之前加到分數上的；查詢、鍵與值都是逐位置計算的；模型中沒有其他運算會混合不同位置。
- **檢驗。** 擾動一個未來的輸入，檢查每一個較早的輸出在浮點精度內都不變——這正是實驗室所測試的不變量（實驗 4.4）。
- **限制。** 證明涵蓋的是如上所寫的注意力。任何其他混合位置的運算——沿序列做的正規化、錯位的快取——都可能在能執行、也能訓練的程式碼中破壞因果性，而閱讀程式碼未必看得出來（§4.7.7）。
:::


## 4.3.3 注意力把輸入當成集合

**置換矩陣（permutation matrix）** $P\in\{0,1\}^{T\times T}$ 的每一列與每一行恰好有一個 1；$PX$ 重新排列 $X$ 的各列。置換矩陣是正交的：$P^{\top}P = I$。

**命題 4.2（置換等變性）。** 沒有遮罩、也沒有位置資訊時，自注意力對每個置換 $P$ 滿足 $\mathrm{Attn}(PX) = P\,\mathrm{Attn}(X)$。

*證明。* 重新排列 $X$ 的列，就重新排列了 $Q, K, V$ 的列：$Q' = PQ$、$K' = PK$、$V' = PV$。分數變成 $S' = PQK^{\top}P^{\top}/\sqrt{d_k} = PSP^{\top}$：$S$ 的列**與**行以同一個置換重新排列。逐列的 softmax 與此可交換——重排列會重排輸出的列，而在一列之內重排元素，該列的 softmax 也以相同方式重排——所以 $A' = PAP^{\top}$。最後 $Y' = A'V' = PAP^{\top}PV = PAV = PY$。$\blacksquare$

所以，沒有額外資訊時，自注意力對「狗咬人」與「人咬狗」算出*相同*的輸出，只是列出的順序不同：它看到的是詞元的**集合**，而不是序列。對集合（點雲中的點、分子中的原子、場景中的物體）而言，這正是正確的對稱性（§4.5.3）。對語言而言，則必須加入順序。

## 4.3.4 加入位置

**絕對位置嵌入。** 為每個位置 $t$（直到最大長度）學習一個向量 $p_t\in\mathbb{R}^d$，加到詞元嵌入上：$x_t\leftarrow x_t + p_t$（如 GPT-2，[Radford et al. 2019](/zh-tw/references/#radford-2019)）。簡單有效；超過最大長度的位置則完全沒有嵌入。

**正弦位置編碼（positional encoding）。** 使用固定向量，其座標位於等比間隔的頻率 $\omega_i = 10000^{-2i/d}$（[Vaswani et al. 2017](/zh-tw/references/#vaswani-2017)）：

$$
p_{t,2i} = \sin(\omega_it),\qquad p_{t,2i+1} = \cos(\omega_it).
$$

每一對 $(\sin\omega_it,\cos\omega_it)$ 都是單位圓上的一點，從 $t$ 移到 $t+k$ 就把它**旋轉**角度 $\omega_ik$：由和角公式，

$$
\begin{pmatrix}\sin\omega_i(t+k)\\\cos\omega_i(t+k)\end{pmatrix} = \begin{pmatrix}\cos\omega_ik & \sin\omega_ik\\-\sin\omega_ik & \cos\omega_ik\end{pmatrix}\begin{pmatrix}\sin\omega_it\\\cos\omega_it\end{pmatrix}.
$$

一個固定的線性映射就能把 $t$ 的編碼變成 $t+k$ 的編碼，讓線性層容易讀出相對位移。

**旋轉位置嵌入（RoPE）。** 不把位置加到輸入上，而是旋轉查詢與鍵本身（[Su et al. 2024](/zh-tw/references/#su-2024)）。把 $q$ 與 $k$ 分成二維的一對對；在位置 $m$，把查詢的第 $i$ 對以旋轉 $R(m\theta_i)$ 轉 $m\theta_i$ 角；鍵在其位置 $n$ 也同樣處理。對其中一對，

$$
\bigl(R(m\theta)q\bigr)^{\top}\bigl(R(n\theta)k\bigr) = q^{\top}R(m\theta)^{\top}R(n\theta)\,k = q^{\top}R\bigl((n-m)\theta\bigr)\,k ,
$$

這裡用到 $R(a)^{\top} = R(-a)$ 與 $R(-a)R(b) = R(b-a)$。**分數只透過兩個位置的差 $n - m$ 依賴位置。** 詳解例題 4.5 以數值檢查這點。RoPE 作用在查詢與鍵上，不作用在值上；旋轉保持範數，所以它改變的是*哪些*鍵匹配，而不是分數的尺度。

## 4.3.5 多個注意力頭

一層注意力對每個查詢只計算**一個**位置上的機率分布。一個位置往往同時需要好幾種資訊：前一個詞元、句子的主詞、對應的左括號。單一分布可以把權重分給多個位置，但那樣得到的是它們值的混合，而且單一的雙線性分數 $q^{\top}k$ 必須同時編碼所有關係。

**多頭注意力（multi-head attention）** 平行執行 $H$ 個注意力運算，每個都有自己投影到維度 $d_h = d/H$ 之較小空間的投影，再把輸出組合起來：

$$
\mathrm{head}_h = \mathrm{Attn}\bigl(XW_Q^{(h)}, XW_K^{(h)}, XW_V^{(h)}\bigr),\qquad
\mathrm{MHA}(X) = \bigl[\mathrm{head}_1,\dots,\mathrm{head}_H\bigr]\,W_O ,
$$

其中 $W_Q^{(h)}, W_K^{(h)}, W_V^{(h)}\in\mathbb{R}^{d\times d_h}$、$W_O\in\mathbb{R}^{d\times d}$。每個注意力頭可以學一種自己的關係，例如「注意前一個詞元」或「注意與我相符的詞元」——單元 11 會在主幹模型中找出這樣的頭。

**成本。** 實作上，各頭的投影就是三個 $d\times d$ 矩陣的行區塊，所以參數量為 $4d^2$（加上偏置），**與 $H$ 無關**；算術量也和一個全寬的頭差不多。改變的是*表達能力*：每個查詢有 $H$ 個獨立的分布，而不是一個。

## 4.3.6 穿過注意力的反向傳遞

訓練需要損失對 $Q$、$K$、$V$ 的梯度（再由單元 03 §3.3.5 的線性映射規則，得到對 $W_Q, W_K, W_V$ 的梯度）。給定 $\bar Y = \partial L/\partial Y$，沿著 §4.3.1 的三個步驟往回走。

**輸出，$Y = AV$。** 線性映射規則給出

$$
\bar V = A^{\top}\bar Y,\qquad \bar A = \bar YV^{\top}.
$$

**逐列 softmax，$A_{i\cdot} = \mathrm{softmax}(S_{i\cdot})$。** 每一列都是獨立的 softmax，雅可比矩陣為 $\mathrm{diag}(a) - aa^{\top}$（單元 03，§3.3.5）。第 $i$ 列的 VJP 為

$$
\bar S_{i\cdot} = \bigl(\mathrm{diag}(a_i) - a_ia_i^{\top}\bigr)\bar A_{i\cdot} = a_i\odot\bigl(\bar A_{i\cdot} - (a_i^{\top}\bar A_{i\cdot})\,\mathbf 1\bigr).
$$

寫成矩陣形式，$\odot$ 為逐元素相乘，列總和在每一列上廣播：

$$
\bar S = A\odot\Bigl(\bar A - \mathrm{rowsum}\bigl(A\odot\bar A\bigr)\Bigr).
$$

被遮罩的元素 $A_{ij} = 0$，所以自動有 $\bar S_{ij} = 0$：沒有梯度流向未來。

**分數，$S = QK^{\top}/\sqrt{d_k}$。** 再對兩個因子套用線性映射規則：

$$
\bar Q = \frac{\bar SK}{\sqrt{d_k}},\qquad \bar K = \frac{\bar S^{\top}Q}{\sqrt{d_k}} .
$$

詳解例題 4.4 在一個三個詞元的例子上完成每一步，並以有限差分檢查其中一個元素。有兩個觀察之後會用到。反向傳遞需要 $A$（或能重算它），一個 $T\times T$ 矩陣——這正是考量 IO 的核心要解決的記憶體問題（§4.8.1）。而當 $A$ 的某一列幾乎是 one-hot 時，該列的 $\bar S$ 幾乎為零——這就是促成 $\sqrt{d_k}$ 縮放的梯度消失。

## 4.3.7 為什麼需要逐位置的 MLP

給定權重後，注意力的輸出 $y_i = \sum_jA_{ij}v_j$ 是值向量的**線性**組合；它在位置之間*搬運*資訊，對資訊本身卻幾乎不做計算。所以每個 Transformer 區塊在注意力之後，都接一個**逐位置前饋網路（position-wise feed-forward network）**，以共用的權重分別作用在每個位置上：

$$
\mathrm{MLP}(u) = W_2\,\varphi(W_1u + b_1) + b_2,\qquad W_1\in\mathbb{R}^{4d\times d},\ W_2\in\mathbb{R}^{d\times4d},
$$

其中 $\varphi$ 是 GELU 之類的非線性函數。隱藏寬度 $4d$ 是原始 Transformer 與 GPT-2 的慣例。這種分工是區塊的核心設計概念：**注意力在位置之間傳遞資訊；MLP 在一個位置之內轉換資訊。**

MLP 不是可有可無的。對只由注意力組成、沒有 MLP 與跳接的網路，[Dong et al.（2021）](/zh-tw/references/#dong-2021)證明其輸出隨深度以*雙重指數*速度收斂到秩一矩陣——每個位置最後都是同一個向量——並顯示跳接與 MLP 能抵抗這種崩塌。

## 4.3.8 殘差流與層正規化

各子層都包在**殘差連接**中（單元 03，§3.8.2；[He et al. 2016](/zh-tw/references/#he-2016)）：每個子層計算一個*加到*其輸入上的更新。因此每個位置都帶著一個向量——**殘差流（residual stream）**——每個注意力層與 MLP 都以累加方式從中讀取、向其寫入。這個觀點由 [Elhage et al.（2021）](/zh-tw/references/#elhage-2021) 為了可解釋性研究而發展，讓每個子層的貢獻能被分開研究（單元 11）。

**層正規化（layer normalization；[Ba et al. 2016](/zh-tw/references/#ba-2016)）** 把每個詞元的向量在其 $d$ 個特徵上重新縮放為平均 0、變異數 1，再套用可學習的增益 $\gamma$ 與偏置 $\beta$：

$$
\mathrm{LN}(u) = \gamma\odot\frac{u - \mu(u)}{\sqrt{\sigma^2(u)+\epsilon}} + \beta .
$$

放在哪裡是真正的設計選擇。原始 Transformer 在加上殘差*之後*才正規化（**後置正規化，post-LN**：$x\leftarrow\mathrm{LN}(x + F(x))$）。GPT-2 與大多數現代模型正規化每個子層的*輸入*（**前置正規化，pre-LN**：$x\leftarrow x + F(\mathrm{LN}(x))$），讓殘差路徑保持為未正規化的恆等映射。[Xiong et al.（2020）](/zh-tw/references/#xiong-2020)顯示後置正規化在初始化時會在輸出附近產生很大的梯度，因此需要學習率暖身，而前置正規化不需要暖身也表現良好。

## 4.3.9 Transformer 區塊與僅解碼器語言模型

**區塊**（前置正規化、僅解碼器，如 GPT-2）：

$$
\begin{aligned}
X &\leftarrow X + \mathrm{MHA}_{\text{causal}}\bigl(\mathrm{LN}_1(X)\bigr),\\
X &\leftarrow X + \mathrm{MLP}\bigl(\mathrm{LN}_2(X)\bigr).
\end{aligned}
$$

**語言模型。** 對詞元序列 $t_1,\dots,t_T$：

1. **嵌入：** $X^{(0)}_i = E[t_i] + P[i]$，其中詞元嵌入表 $E\in\mathbb{R}^{V\times d}$，位置表 $P$（或在注意力內改用 RoPE 而不用 $P$）。
2. **處理：** 對 $\ell = 1,\dots,L$，$X^{(\ell)} = \mathrm{Block}_\ell\bigl(X^{(\ell-1)}\bigr)$。
3. **反嵌入：** logits $Z = \mathrm{LN}_f\bigl(X^{(L)}\bigr)E^{\top}\in\mathbb{R}^{T\times V}$，重用（「綁定」）嵌入表（[Press & Wolf 2017](/zh-tw/references/#press-2017)）。
4. **損失：** 由 $Z$ 的第 $i$ 列預測 $t_{i+1}$ 的平均交叉熵，對 $i = 1,\dots,T-1$——在每個位置套用單元 03 的 softmax 交叉熵。

多虧因果遮罩（命題 4.1），一次前向傳遞就算出全部 $T-1$ 個預測，每一個都只用到自己的前綴。**參數量：** 每個區塊 $12d^2 + 13d$（含偏置與層正規化），加上綁定嵌入的 $Vd$、學習的位置的 $T_{\max}d$，以及最後層正規化的 $2d$。詳解例題 4.7 推導這些項並重現 GPT-2 small 的 124,439,808 個參數。

## 4.3.10 生成與 KV 快取

**生成**時，模型預測下一個詞元的分布，取樣（或選出）一個，接到序列後面，再重複。每一步都對越來越長的前綴重算整個前向傳遞，在長度 $t$ 時每個新詞元要花 $O(t^2d)$ 的注意力計算。

因果性讓其中大部分工作是多餘的。詞元 $1,\dots,t$ 的鍵與值在詞元 $t+1$ 到來時不會改變（它們只依賴自己的前綴）。所以每一層都維護一個存放所有過去鍵與值的 **KV 快取（KV cache）**。每一步只為新詞元：計算 $q, k, v$；把 $k, v$ 接到快取後面；以 $q$ 對整個快取做注意力；繼續到下一層。這樣每一步每層只需 $O(t\,d)$ 的注意力計算，而不是 $O(t^2d)$。

正確實作由兩個不變量定義：**快取解碼產生的 logits 與對同一前綴做完整前向傳遞的結果完全相同**（在浮點捨入內），以及**新詞元的位置索引是 $t+1$**，不是 1——忘記遞增它是經典的 KV 快取錯誤（[KV 快取](/zh-tw/atlas/nodes/mech-kv-cache/)）。快取的記憶體——每條序列 $2\times L\times t\times d$ 個數——隨上下文增長；單元 05 把它當作未壓縮的記憶來研究，單元 06 則當作推論成本來研究。
