---
title: "9.2 第一原理"
description: 動作、獎勵與策略；為什麼抽樣擋住了反向傳播，以及分數函數技巧如何繞過它；基準；連續決策；對預訓練模型的 KL 錨定；以及以偏好取代獎勵來學習。
sidebar: { label: "9.2 第一原理", order: 2 }
---

## 9.2.1 從目標值到行動後果

監督式學習會在預測之前提供理想輸出 $y$。若模型給它機率 $p_\theta(y\mid x)$，每個樣本的交叉熵損失就是 $-\log p_\theta(y\mid x)$。目標值直接告訴學習器該提高哪個輸出。

獎勵回饋的介面不同：學習器先選擇，再觀察這個選擇值多少。

| 符號 | 名稱 | 主幹算術範例 |
|---|---|---|
| $s$ | 狀態或脈絡 | 以「Now Ben has 」結尾的提示 |
| $a$ | 動作 | 抽出的一個下一字元 |
| $\pi_\theta(a\mid s)$ | 策略 | 模型在各字元上的分布 |
| $r(s,a)$ | 獎勵 | 字元是 $x+y$ 的數字時為 $1$，否則為 $0$ |

先考慮最簡單情況：提示來自固定分布 $d(s)$，一個動作就結束一回合。目標函數為

$$
J(\theta)
=\mathbb E_{s\sim d}
  \mathbb E_{a\sim\pi_\theta(\cdot\mid s)}
  [r(s,a)].
$$

兩層期望都很重要。外層決定哪些提示算數；內層則平均目前策略可能做出的行動。策略在某次幸運抽樣拿到獎勵 1，不代表它已經達成 $J=1$。

獎勵不必可微。它可以包含相等測試、執行編譯器、玩一局遊戲，或請人比較兩個答案。困難在於：$\theta$ 改變的是*選到某動作的機率*，但實際觀察到的獎勵附著在已抽出的離散動作上。

:::tip[不要硬造出不存在的差別]
若一支程式能在訓練前便宜地產生唯一正確答案，它也能提供監督式目標值。這個玩具例子仍採用獎勵學習，是為了讓估計量可稽核。更強的動機來自結果延遲、許多動作序列都能成功，或回饋只提供排序的情況。
:::

## 9.2.2 為什麼反向傳播停在樣本上

要以梯度上升改善 $J$，我們需要 $\nabla_\theta J$。一個看似自然的計算圖是

$$
\theta\longrightarrow \pi_\theta
\longrightarrow \text{抽取 }a
\longrightarrow r(s,a).
$$

一般反向傳播無法穿過離散抽樣決策。參數微小改變時，抽到的符號通常完全不變，偶爾又會突然換成另一個。獎勵程式本身也可能不可微。把抽到的詞元 ID 當成平滑數值，會憑空捏造問題原本沒有的導數。

出路是對*每個可能動作的機率*微分。先固定一個狀態，並假設有限動作集合與獎勵函數本身不依賴 $\theta$：

$$
\begin{aligned}
\nabla_\theta J_s
&=\nabla_\theta\sum_a\pi_\theta(a\mid s)r(s,a)\\
&=\sum_a r(s,a)\nabla_\theta\pi_\theta(a\mid s)\\
&=\sum_a\pi_\theta(a\mid s)r(s,a)
   \nabla_\theta\log\pi_\theta(a\mid s)\\
&=\mathbb E_{a\sim\pi_\theta}
\left[r(s,a)\nabla_\theta\log\pi_\theta(a\mid s)\right].
\end{aligned}
$$

第三行使用

$$
\nabla_\theta \pi_\theta
=\pi_\theta\nabla_\theta\log\pi_\theta,
$$

它直接來自 $p>0$ 時的 $\nabla\log p=(\nabla p)/p$。

這就是**分數函數（score-function）**或 **REINFORCE** 恆等式（[Williams 1992](/zh-tw/references/#williams-1992)）。這裡的「score」是機率分數 $\nabla_\theta\log\pi_\theta(a\mid s)$，**不是**獎勵分數。單一樣本梯度

$$
\widehat g
=r(s,a)\nabla_\theta\log\pi_\theta(a\mid s)
$$

雖然嘈雜，期望值卻等於精確梯度。平均多個獨立樣本可以降低蒙地卡羅雜訊。

### 三動作的完整計算

令 $\pi=(1/3,1/3,1/3)$，獎勵為 $r=(1,0,0)$，則 $J=1/3$。若策略是 logit $z$ 上的 softmax，單元 03 給出

$$
\frac{\partial J}{\partial z_i}
=\pi_i(r_i-J),
$$

因此

$$
\nabla_z J
=\left(\frac{2}{9},-\frac{1}{9},-\frac{1}{9}\right).
$$

提高受獎動作的 logit 時，其他動作必須下降，因為機率總和固定為一。REINFORCE 不需要列舉每個動作，也能估計這個向量。面對巨大詞彙表或長軌跡時，避免完整列舉正是關鍵。

把 REINFORCE 讀成**對策略自身樣本的獎勵加權模仿**很有用，但不完整：負優勢會明確降低抽到動作的機率；連續決策中的資料會依賴策略；off-policy 樣本還需要校正。

## 9.2.3 基準：和預期相比

估計量可以無偏，卻仍然嘈雜到無法有效學習。假設每個獎勵都接近 100，而動作之間真正有意義的差別只有 0.1。把分數向量乘上大約 100，會加入巨大波動，卻沒有多少資訊能指出哪個動作較好。

減去一個可以依賴狀態、但不能依賴所抽動作的**基準（baseline）** $b(s)$：

$$
\nabla_\theta J_s
=\mathbb E_{a\sim\pi_\theta}
\left[(r(s,a)-b(s))
\nabla_\theta\log\pi_\theta(a\mid s)\right].
$$

為什麼期望值不變？固定 $s$ 後，

$$
\begin{aligned}
\mathbb E_a\left[
b(s)\nabla_\theta\log\pi_\theta(a\mid s)
\right]
&=b(s)\sum_a\pi_\theta(a\mid s)
  \nabla_\theta\log\pi_\theta(a\mid s)\\
&=b(s)\nabla_\theta\sum_a\pi_\theta(a\mid s)\\
&=b(s)\nabla_\theta 1=0.
\end{aligned}
$$

因此，基準改變的是估計量變異數，不是理想期望值。差值 $r-b$ 是一個**優勢估計（advantage estimate）**：正值表示比比較點好，負值表示較差。

回到三動作例子。當 $b=0$ 時，第一個 logit 的單樣本梯度變異數約為 $0.0988$；改成 $b=1/3$ 後，平均仍為 $2/9$，變異數卻降到 $0.0247$——只剩四分之一。詳解例題 9.2 會列出每一種結果。

以下三項提醒可以避免常見錯誤：

1. **動作獨立性是條件式的。**學到的價值 $V(s)$ 可以使用，因為它只看狀態；偷看已抽動作的基準卻可能改變平均。
2. **價值很方便，卻不一定讓變異數最小。**最佳常數基準還取決於分數的平方範數；§9.5 會推導。
3. **批次平均可能包含樣本本身。**批次大小為 $B$ 時，減去普通獎勵批次平均，會把期望梯度縮成 $(B-1)/B$。主幹專案改為讓每個樣本使用其餘 $B-1$ 個樣本的平均——也就是留一法基準。群組相對方法也可以比較同一提示的多個答案（[Shao et al. 2024](/zh-tw/references/#shao-2024)）。

:::note[合理性檢查]
減去基準不是「把壞獎勵改成負獎勵」，而是把梯度估計量置中。底層目標函數 $J$ 仍然使用原本的獎勵。
:::

## 9.2.4 連續決策

語言模型的答案通常包含許多詞元。會使用工具的代理也會改變下一個狀態：搜尋查詢決定觀察到哪個網頁，而這項觀察又影響下一個動作。一條**軌跡（trajectory）**

$$
\tau=(s_0,a_0,r_0,s_1,a_1,r_1,\ldots)
$$

的機率為

$$
p_\theta(\tau)
=p(s_0)\prod_t
\pi_\theta(a_t\mid s_t)
P(s_{t+1}\mid s_t,a_t).
$$

假設環境轉移 $P$ 不依賴 $\theta$。對數導數中只有策略因子會留下來：

$$
\nabla_\theta\log p_\theta(\tau)
=\sum_t\nabla_\theta\log\pi_\theta(a_t\mid s_t).
$$

對折扣回報

$$
R(\tau)=\sum_{t=0}^{T-1}\gamma^t r_t,
\qquad 0\le\gamma\le1,
$$

分數函數恆等式先得到 $\mathbb E[R\sum_t\nabla\log\pi_t]$。發生在動作 $a_t$ 之前的獎勵不可能由該動作造成，其期望分數貢獻為零。移除這些項後，得到從當下開始的回報形式：

$$
\nabla_\theta J
=\mathbb E_\tau\left[
\sum_{t=0}^{T-1}
\gamma^t
\nabla_\theta\log\pi_\theta(a_t\mid s_t)
\bigl(G_t-b(s_t)\bigr)
\right],
$$

其中

$$
G_t=\sum_{k=t}^{T-1}\gamma^{k-t}r_k.
$$

對這裡寫出的目標函數，外面的 $\gamma^t$ 不能漏掉；漏掉就會改變後期決策的權重。有些教材把它吸收到折扣狀態造訪分布中；兩種慣例都可以，但必須前後一致。

若一個 200 詞元的答案只在結尾收到一個獎勵，同一個純量結果起初會乘上許多詞元的分數向量。正確答案該歸功於哪個詞元？失敗又該怪哪個？這就是**功勞分配問題（credit-assignment problem）**。價值與優勢利用未來回報的預測，讓比較更局部（§9.3.3），卻不會讓因果歧義憑空消失。

主幹專案的答案只有一個動作，因此刻意拿掉軌跡中的功勞分配問題。這項簡化讓我們先隔離估計量與獎勵設計錯誤，再把它們和長期視野組合。

## 9.2.5 錨定在模型已知的東西上

只為量測獎勵最佳化的策略可能遠離有用的預訓練行為、塌縮到狹窄的輸出集合，或利用評估器怪癖。一種常見目標函數會懲罰策略偏離固定的**參考策略** $\pi_{\text{ref}}$：

$$
J_\beta(\theta)
=\mathbb E_{s\sim d,\;a\sim\pi_\theta}[r(s,a)]
-\beta\,
\mathbb E_{s\sim d}
\left[
\mathrm{KL}\!\left(
\pi_\theta(\cdot\mid s)
\Vert
\pi_{\mathrm{ref}}(\cdot\mid s)
\right)
\right].
$$

固定一個狀態，並允許任意策略時，§9.3.4 會證明最佳解是

$$
\pi^\star(a\mid s)
=\frac{
\pi_{\mathrm{ref}}(a\mid s)
\exp(r(s,a)/\beta)}
{\sum_{a'}
\pi_{\mathrm{ref}}(a'\mid s)
\exp(r(s,a')/\beta)}.
$$

獎勵以指數方式讓參考分布**傾斜（tilt）**：

- 小 $\beta$ 讓獎勵差異主導；
- 大 $\beta$ 讓策略更靠近參考；
- 所有獎勵相等時，重現參考分布；
- 若參考策略給某動作的機率恰為零，這個前向 KL 解也不能給它正機率。

這是一個取捨參數，不是普遍安全保證。小 KL 仍可能掩蓋罕見但重要提示上的嚴重改變；大 KL 也可能阻止必要修正。方向同樣重要：$\mathrm{KL}(\pi_\theta\Vert\pi_{\mathrm{ref}})$ 不能隨意換成反方向。

主幹專案在完整條件分布上計算 KL，不只看抽到的單一動作。因此，對這個玩具詞彙表，KL 梯度是精確的。大型語言模型系統常以樣本或詞元對數比估計序列層級 KL，因而加入額外變異數與實作選擇。

## 9.2.6 沒有獎勵時：偏好

有些任務有明確驗證器：單元測試通過、定理檢查器接受，或算術答案相符。然而，實用、文體與無害等性質通常沒有一支完整的程式測試。一種常見介面改用比較：

> 對提示 $x$，回應 $y_w$ 優於回應 $y_l$。

比較不會自動可靠。標註者可能意見不同、誤解提示、採用不同價值，或受長度與呈現方式影響。偏好資料是一種需要規程與留出稽核的量測，不是直接讀取人類意圖。

第一條路以 **Bradley–Terry** 概似擬合純量獎勵模型：

$$
P(y_w\succ y_l\mid x)
=\sigma\!\left(
r_\phi(x,y_w)-r_\phi(x,y_l)
\right),
$$

其中 $\sigma(u)=1/(1+e^{-u})$（[Bradley & Terry 1952](/zh-tw/references/#bradley-1952)）。接著針對這個學到的代理獎勵最佳化策略，形成通常稱為 RLHF 的「獎勵建模加強化學習」管線（[Christiano et al. 2017](/zh-tw/references/#christiano-2017)；[Ouyang et al. 2022](/zh-tw/references/#ouyang-2022)）。

第二條路利用 KL 正則化最佳解，把顯式獎勵模型從策略訓練目標中消去。**直接偏好最佳化（direct preference optimization，DPO）**比較策略相對於參考模型，將贏家提高得比輸家多多少（[Rafailov et al. 2023](/zh-tw/references/#rafailov-2023)）。它沒有讓獎勵假設消失：推導內仍含 Bradley–Terry 偏好模型與 KL 正則化策略關係。§9.3.6 會寫出每一步代數。

這帶出一個重要區分：

| 回饋物件 | 觀察到什麼？ | 推斷什麼？ |
|---|---|---|
| 純量獎勵 | 一次抽樣結果的一個數字 | 哪些動作該變得更可能 |
| 成對偏好 | 贏家與輸家 | 潛在效用差或直接策略比值 |
| 監督式目標 | 理想輸出本身 | 不需要比較模型 |

## 9.2.7 目前為止

| 訊號 | 更新時問的問題 | 主幹觀察 |
|---|---|---|
| 目標字元 | 「提高這個指定字元」 | 預訓練：貪婪正確率 $0.3175$ |
| 精確程式化獎勵 | 「提高抽到而且通過的動作」 | REINFORCE：留出貪婪正確率 $0.82$ |
| 正確 $\succ$ 錯誤配對 | 「相對於輸家與參考，提高贏家」 | DPO：留出貪婪正確率 $0.78$ |
| 有錯的獎勵 $\mathbf1\{\text{答案}\ge x+y\}$ | 「提高任何通過錯誤規則的抽樣數字」 | 訓練獎勵 $0.99$，真實正確率 $0.09$ |

演算法沒有自己發現理想目標，而是轉換供給它的回饋。因此，一份後訓練報告至少要分開四條曲線：最佳化損失、量測訓練獎勵、相對於參考的漂移，以及不重用訓練獎勵實作的留出任務指標。

**下一節：**[§9.3 數學構造](/zh-tw/core/unit-09/construction/)把單一動作恆等式推廣到 MDP，推導價值與貝爾曼方程式，再在不跳過假設的情況下建構 PPO 與 DPO。
