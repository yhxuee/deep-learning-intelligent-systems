---
title: "9.3 數學建構"
description: 馬可夫決策過程、帶基準與優勢的策略梯度定理、價值函數與貝爾曼方程式、KL 正則化獎勵最大化的封閉形式最佳解、由偏好得到的獎勵模型、PPO 的截斷目標函數、DPO 的推導，以及可驗證獎勵與群組基準。
sidebar: { label: "9.3 數學建構", order: 3 }
---

## 9.3.1 馬可夫決策過程

**馬可夫決策過程（Markov decision process，MDP）**形式化「動作會影響下一步」的決策問題（[Sutton & Barto 2018](/zh-tw/references/#sutton-2018)）：

$$
(\mathcal S,\mathcal A,P,r,\gamma).
$$

| 元件 | 意義 | 文字生成實例 |
|---|---|---|
| 狀態 $s_t$ | 行動前可用的資訊 | 提示加上已生成前綴 |
| 動作 $a_t$ | 一次選擇 | 下一個詞元 |
| 轉移 $P(s_{t+1}\mid s_t,a_t)$ | 世界如何改變 | 接上詞元，或執行工具 |
| 獎勵 $r_t$ | 量測到的結果 | 驗證器分數、偏好模型分數、任務結果 |
| 折扣 $\gamma$ | 後期獎勵的相對權重 | 短有限答案可用 $1$；持續任務常小於 $1$ |

**馬可夫性質**表示：狀態含有足夠資訊，使下一狀態的分布在給定 $(s_t,a_t)$ 後，不必再查更早歷史。它不是說世界沒有記憶。若歷史很重要，有效狀態就必須摘要它；語言模型前綴直接包含先前所有詞元，因此做到這點。

策略 $\pi_\theta(a\mid s)$ 與轉移規則共同產生軌跡。從時間 $t$ 起的回報為

$$
G_t
=r_t+\gamma r_{t+1}+\gamma^2r_{t+2}+\cdots
=\sum_{k=0}^{T-t-1}\gamma^k r_{t+k},
$$

回合式目標函數則是

$$
J(\theta)
=\mathbb E_{\tau\sim p_\theta(\tau)}[G_0].
$$

這個期望同時涵蓋策略與環境的隨機性。對有限回合，數學上可以取 $\gamma=1$；對無限視野，$\gamma<1$ 能讓有界獎勵的總和收斂，也表達時間偏好。

文字生成的詞元轉移是確定的：接上所選詞元，就固定了下一個前綴。工具使用較不簡單，因為同一查詢可能回傳隨機或隨時間改變的觀察。主幹算術任務在一個詞元後結束，沒有動作會影響後續決策。因此它是 **情境式吃角子老虎機（contextual bandit）**，也就是 MDP 的單步邊界情況。

:::note[吃角子老虎機，還是 MDP？]
若每個提示都獨立抽出，一個答案只得到一個分數，使用吃角子老虎機模型即可。若動作會改變下一個觀察或之後可選的選項，連續 MDP 結構才不可省略。把所有後訓練問題都稱為「強化學習」，不會讓這個差別消失。
:::

## 9.3.2 策略梯度定理

微分之前，先辨認哪些物件依賴 $\theta$：

- 初始狀態分布固定；
- 環境轉移與獎勵不直接依賴 $\theta$；
- 策略可微，而且在使用對數機率之處分配正機率；
- 微分可與有限軌跡總和或積分交換。

這些假設在最後的估計量中看不見，卻決定了它的意義。若獎勵模型一起更新、環境會隨策略改變，或資料來自另一個策略，就可能需要額外項或校正。

**定理 9.1（策略梯度，回合式折扣目標）。** 令 $J(\theta) = \mathbb{E}_{\tau\sim\pi_\theta}[G_0]$，其中對有限長度的回合 $G_t = \sum_{k\ge0}\gamma^kr_{t+k}$。對任何不依賴 $a_t$ 的基準 $b(s_t)$，
$$
\nabla_\theta J(\theta) = \mathbb{E}_{\tau\sim\pi_\theta}\Big[\sum_{t}\gamma^t\,\nabla_\theta\ln\pi_\theta(a_t\mid s_t)\,\big(G_t - b(s_t)\big)\Big].
$$

*證明，第一步：對軌跡微分。* 軌跡機率是

$$
p_\theta(\tau)
=p(s_0)\prod_t
\pi_\theta(a_t\mid s_t)
P(s_{t+1}\mid s_t,a_t).
$$

只有策略因子依賴 $\theta$，所以

$$
\nabla_\theta\log p_\theta(\tau)
=\sum_t\nabla_\theta
\log\pi_\theta(a_t\mid s_t).
$$

分數函數恆等式於是給出

$$
\nabla_\theta J
=\mathbb E_\tau\left[
G_0\sum_t\nabla_\theta
\log\pi_\theta(a_t\mid s_t)
\right].
$$

*第二步：移除過去的獎勵。* 時間 $t$ 之前收到的獎勵，在抽出 $a_t$ 前就已確定。固定 $s_t$ 後，它與動作分數之乘積的期望為零。$G_0=\sum_k\gamma^k r_k$ 在第 $t$ 步留下的是

$$
\sum_{k\ge t}\gamma^k r_k
=\gamma^tG_t.
$$

這就是因果性或從當下開始的回報步驟。

*第三步：減去基準。* 固定 $s_t$ 後，

$$
\mathbb E_{a_t}\left[
b(s_t)\nabla_\theta\log\pi_\theta(a_t\mid s_t)
\right]
=b(s_t)\nabla_\theta\sum_a\pi_\theta(a\mid s_t)
=0.
$$

乘上 $\gamma^t$ 仍然是零，所以可以減去與動作無關的基準。$\blacksquare$

這條定理是精確的期望恆等式。REINFORCE 以抽樣軌跡代替期望。因此，定理在其假設下建立無偏性，卻不保證低變異數、安全探索或正確獎勵。

**因子 $\gamma^t$。** 它很容易被漏掉，而許多實作是刻意省略它：$\sum_t\nabla\ln\pi(a_t\mid s_t)(G_t - b)$ 是*另一個*目標函數的梯度——它把每個造訪到的狀態都當成回合的起點來加權。$\gamma = 1$ 時兩者一致。*例子：*一個兩步的回合，$t = 0$ 只有一個動作；$t = 1$ 時是 logit 為 $\theta = 0$ 的伯努利動作，若為 1 則得到獎勵 1。以 $\gamma = 0.5$，$J = 0.5\,\sigma(\theta)$，$\partial J/\partial\theta = 0.125$。定理給出 0.125；省略 $\gamma^t$ 的式子給出 0.25。主幹專案的單步任務只有 $t = 0$，此時 $\gamma^0 = 1$，因此無法揭露這個差異；`test_stage09_discounted_policy_gradient_carries_gamma_t` 檢查了兩步的情況。

對有限動作集上 logits 為 $z$ 的 softmax 策略，$\nabla_z\ln\pi(a) = e_a - \pi$（單元 03，§3.3.5）。因此其梯度即為 REINFORCE 估計的單樣本損失是 $\ell = -(G - b)\ln\pi(a)$，且 $\partial\ell/\partial z = (G - b)(\pi - e_a)$——朝抽到之動作的交叉熵，以其優勢縮放。`spine/stage09_post_training.py` 正好實作了這一行，並以有限差分穿過整個 Transformer 檢查它。

以梯度**下降**實作時，損失帶有上面的負號；描述對期望回報做梯度**上升**時，估計量則是正的。混用這兩種正負號慣例，常會寫出穩定學到相反策略的程式。

## 9.3.3 價值函數與貝爾曼方程式

在 $\pi$ 下一個狀態的**價值（value）**是 $V^\pi(s) = \mathbb{E}_\pi[G_t\mid s_t = s]$，**動作價值（action value）**是 $Q^\pi(s, a) = \mathbb{E}_\pi[G_t\mid s_t = s, a_t = a]$。它們滿足**貝爾曼方程式（Bellman equation）**（[Bellman 1957](/zh-tw/references/#bellman-1957)）：

$$
V^\pi(s) = \sum_a\pi(a\mid s)\Big[r(s, a) + \gamma\sum_{s'}P(s'\mid s, a)V^\pi(s')\Big],
$$

對固定策略，這是一個線性方程組。拆開一次遞迴後，意思會更清楚：

$$
\text{目前價值}
=\mathbb E[
\text{立即獎勵}
+\gamma\,\text{下一狀態價值}
].
$$

例如，狀態 $s$ 只有一個動作，沒有立即獎勵，接著確定地移到價值為 1 的終止前狀態 $u$。若 $\gamma=0.9$，

$$
V(s)=0+0.9V(u)=0.9.
$$

反覆代入就會把遞迴展開回期望折扣獎勵總和。對固定 $\pi$，這是**策略評估（policy evaluation）**。把策略加權平均換成 $\max_a$，是在問最佳動作，並得到貝爾曼最佳性方程式；那是**控制（control）**。悄悄混用兩條方程式，就改變了問題。

$V^\pi(s_t)$ 是最常用的狀態相依基準，但一般而言*不是*使變異數最小的那一個。§9.5.1 推導出的最佳解，是以分數平方加權的動作價值平均，可能與 $V^\pi$ 相去甚遠。對 $p=0.8$、動作 1 得到獎勵 1 的伯努利策略，$V=0.8$，最佳常數基準卻是 0.2，並使這個一維估計量的變異數為零（[Greensmith et al. 2004](/zh-tw/references/#greensmith-2004)）。

**優勢**

$$
A^\pi(s,a)=Q^\pi(s,a)-V^\pi(s)
$$

衡量一個動作比策略在該狀態下的平均好多少。行動者–評論者（actor–critic）方法以**評論者**學習價值近似，再更新策略**行動者**。時間差分殘差

$$
\delta_t
=r_t+\gamma V_\phi(s_{t+1})-V_\phi(s_t)
$$

在 $V_\phi=V^\pi$ 時，條件平均等於正確優勢；使用近似評論者時則可能有偏。廣義優勢估計以參數 $\lambda$ 組合多種時間差分視野，在變異數與偏差間取捨（[Schulman et al. 2016](/zh-tw/references/#schulman-2016)）；它不會普遍地只「加入一點」固定大小的偏差。

## 9.3.4 KL 正則化目標函數及其最佳解

固定一個狀態，把 $\pi(\cdot\mid s)$ 寫成 $\pi$。考慮

$$
J_\beta(\pi) = \sum_a\pi(a)r(a) - \beta\sum_a\pi(a)\ln\frac{\pi(a)}{\pi_{\text{ref}}(a)} .
$$

假設 $\beta>0$、獎勵有限，而且最佳化只在 $\pi_{\text{ref}}$ 的支撐集上進行。否則，只要 $\pi$ 在參考機率為零之處分配正質量，前向 KL 就是無限大。

**命題 9.1。** 在有支撐的分布中，$J_\beta$ 的最大化者為

$$
\pi^\star(a)
=\frac{\pi_{\text{ref}}(a)e^{r(a)/\beta}}{Z},
\qquad
Z=\sum_a\pi_{\text{ref}}(a)e^{r(a)/\beta},
$$

最大值為 $\beta\ln Z$。

*證明。* 因為 $\pi_{\text{ref}}(a)e^{r(a)/\beta}=Z\pi^\star(a)$，

$$
\begin{aligned}
J_\beta(\pi)
&=-\beta\sum_a\pi(a)
\log\frac{\pi(a)}
{\pi_{\text{ref}}(a)e^{r(a)/\beta}}\\
&=-\beta\sum_a\pi(a)
\log\frac{\pi(a)}
{Z\pi^\star(a)}\\
&=-\beta\,\mathrm{KL}(\pi\Vert\pi^\star)
+\beta\log Z.
\end{aligned}
$$

由吉布斯不等式（單元 01，定理 1.1），KL 非負，且恰在 $\pi=\pi^\star$ 時為零。$\blacksquare$

正規化常數不只是簿記。它讓機率總和為一，也給出最佳目標函數值。勝算形式通常最好解讀：

$$
\frac{\pi^\star(a)}{\pi^\star(b)}
=\frac{\pi_{\text{ref}}(a)}
       {\pi_{\text{ref}}(b)}
\exp\!\left(
\frac{r(a)-r(b)}{\beta}
\right).
$$

獎勵相等的動作保留參考勝算；獎勵優勢 $\Delta r$ 則把勝算乘上 $e^{\Delta r/\beta}$。若 $\Delta r=1$ 且 $\beta=0.5$，乘數是 $e^2\approx7.39$。這不表示較好動作的機率變成 1，因為參考勝算與所有其他動作仍然存在。

主幹專案的測試檢查對 $J_\beta$ 做梯度上升會收斂到 $\pi^\star$，精確到小數點後六位。把這個關係反過來，就得到 DPO 需要的那一步：

$$
r(a) = \beta\ln\frac{\pi^\star(a)}{\pi_{\text{ref}}(a)} + \beta\ln Z .
$$

加法項 $\beta\log Z$ 依賴狀態，卻不依賴動作。比較同一提示的兩個回應時，它會相消。

## 9.3.5 獎勵模型與 PPO

經典語言模型 RLHF 管線分開三種資料與目標函數：

| 階段 | 資料 | 目標函數 |
|---|---|---|
| 監督式微調 | 提示—回應示範 | 模仿有用的回應分布 |
| 獎勵建模 | 提示加上偏好與非偏好回應 | 預測成對偏好 |
| 策略最佳化 | 不斷改變的策略所產生的樣本 | 提高學到的獎勵，同時控制漂移 |

分開各階段，才能看見各自的失敗模式。獎勵模型在驗證集表現好，不代表最佳化它之後，在分布外仍能保留人類偏好。

**獎勵模型。** 收集提示 $x$、回應 $(y_w,y_l)$，以及人類或 AI 對 $y_w$ 較佳的標註。以最小化下式擬合 $r_\phi$：

$$
\mathcal L_{\text{RM}}
=-\mathbb E\left[
\log\sigma\!\left(
r_\phi(x,y_w)-r_\phi(x,y_l)
\right)
\right].
$$

這就是對獎勵差做邏輯斯迴歸（[Christiano et al. 2017](/zh-tw/references/#christiano-2017)；[Stiennon et al. 2020](/zh-tw/references/#stiennon-2020)）。只有差值能被辨識：對所有回應獎勵加入同一個依狀態常數，不會改變比較。獎勵尺度也會影響策略最佳化，因為它改變了獎勵與 KL 之間的取捨。

**PPO。** 假設一批樣本由舊策略 $\pi_{\text{old}}$ 產生，但最佳化已把目前策略移到 $\pi_\theta$。每個動作的重要性比值

$$
\rho_t(\theta)
=\frac{\pi_\theta(a_t\mid s_t)}
       {\pi_{\text{old}}(a_t\mid s_t)}
$$

校正這項分布改變的一階代理目標。近端策略最佳化（Proximal Policy Optimization，PPO）使用

$$
L_{\text{clip}}(\theta)
=\mathbb E\left[
\min\!\left(
\rho_t A_t,\;
\operatorname{clip}(\rho_t,1-\epsilon,1+\epsilon)A_t
\right)
\right]
$$

（[Schulman et al. 2017](/zh-tw/references/#schulman-2017)）。

解讀最小值之前，要先看 $A_t$ 的正負：

- 若 $A_t>0$，把抽到動作的比值提高到 $1+\epsilon$ 以上，不再增加代理收益；
- 若 $A_t<0$，把比值降到 $1-\epsilon$ 以下，也不再增加代理收益。

截斷移除的是抽樣代理中的一項*誘因*，不是對參數、未抽到動作或總 KL 的硬限制。共享神經網路的一次更新可以移動許多其他機率。因此，實作通常還會監控 KL 估計，在它過大時提早結束該輪，並另加參考策略 KL 懲罰。

監督式微調、獎勵模型與帶 KL 的 PPO 管線曾用於具影響力的指令遵循系統（[Ouyang et al. 2022](/zh-tw/references/#ouyang-2022)）**[E]**。這項歷史結果支持管線可行，不表示每個偏好學習問題都必須使用 PPO。

## 9.3.6 直接偏好最佳化

對回應序列 $y=(y_1,\ldots,y_T)$，

$$
\log\pi_\theta(y\mid x)
=\sum_{t=1}^{T}
\log\pi_\theta(y_t\mid x,y_{<t}).
$$

DPO 使用這個完整序列對數機率，通常會用注意力遮罩排除提示與填補詞元。它不是只比較最後一個詞元。

現在把 §9.3.4 反解出的最佳解代入 Bradley–Terry 模型。對同一個提示的兩個回應而言，$\beta\log Z(x)$ 項相等而相消：

$$
P(y_w\succ y_l) = \sigma\Big(\beta\ln\frac{\pi(y_w)}{\pi_{\text{ref}}(y_w)} - \beta\ln\frac{\pi(y_l)}{\pi_{\text{ref}}(y_l)}\Big).
$$

在此模型下最大化觀測到之偏好的概似，就得到 **DPO 損失**（[Rafailov et al. 2023](/zh-tw/references/#rafailov-2023)）：

$$
\mathcal L_{\text{DPO}} = -\mathbb{E}\Big[\ln\sigma\big(u\big)\Big],\qquad u = \beta\Big[\big(\ln\pi(y_w) - \ln\pi_{\text{ref}}(y_w)\big) - \big(\ln\pi(y_l) - \ln\pi_{\text{ref}}(y_l)\big)\Big].
$$

這裡沒有另行訓練的獎勵模型、每個最佳化步驟內不需要 on-policy 抽樣，也沒有 PPO：更新是在已收集成對資料上的分類損失。它對對數機率的梯度是 $-\beta\sigma(-u)\big(\nabla\ln\pi(y_w) - \nabla\ln\pi(y_l)\big)$——提高偏好的回應、降低不偏好的回應，權重是目前模型把兩者排錯的程度。對單一詞元答案，兩個對數機率梯度中的 softmax 項相消，在 logits 上只剩 $-\beta\sigma(-u)(e_{y_w} - e_{y_l})$；主幹專案以有限差分檢查它。

初始化時 $\pi=\pi_{\text{ref}}$，所以 $u=0$、損失為 $\log2\approx0.6931$、邏輯斯權重為 $\sigma(0)=1/2$。若目前策略已經提高贏家的相對對數勝算，$u>0$，該配對權重就變小；若排序方向錯誤，$u<0$，權重便增大。

這裡的「沒有獎勵模型」是指策略最佳化期間，不必執行一個另行訓練的純量網路。推導仍假設：

1. 對某個潛在獎勵而言，偏好符合 Bradley–Terry 形式；
2. 目標策略與固定參考之間滿足 KL 正則化最佳解的關係；
3. 贏家與輸家使用相同提示，因此配分函數項能相消；
4. 離線偏好配對涵蓋想改善的行為。

違反這些假設，不一定會讓數值最佳化失敗；它會改變我們能把損失解讀成正在估計什麼。DPO 對固定資料集而言也是離線的，因此不會自動為更新後策略產生的新行為收集回饋。

## 9.3.7 可驗證獎勵與群組基準

當正確性能由程式檢查——算術、單元測試、證明檢查器——獎勵就不需要人，也不需要學習出來的模型。對同一個提示抽樣 $G$ 個答案，以它們的平均獎勵作為基準，得到優勢 $A_i = r_i - \bar r$（可再除以該群組的標準差），不需要價值網路（[Shao et al. 2024](/zh-tw/references/#shao-2024)）。因為 $\bar r$ 包含 $r_i$ 本身，這個基準並不獨立於 $a_i$：期望上它把梯度縮放了 $(G-1)/G$。**留一法（leave-one-out）**基準 $b_i = \frac{1}{G-1}\sum_{j\ne i}r_j$ 獨立於 $a_i$，使估計量保持不偏；它與群組平均只差優勢上的一個因子 $G/(G-1)$。除以群組標準差則引入另一個依資料而定的縮放。以這種方式在數學與程式上訓練大型模型，並讓它在作答前寫出長串推理，帶來了強大的推理表現（[DeepSeek-AI 2025](/zh-tw/references/#deepseek-2025)）**[E]**。主幹專案的階段 09 是這個配方的單詞元版本，以批次上的留一法作為基準；它用有錯檢查器的那次執行，則是警示版本（§9.7.1）。因為一個批次中的 64 個提示各不相同，這個基準是常數，而不是狀態的函數——不偏，但降低變異數的效果不如逐提示的基準。（v2.1 之前的版本使用單純的批次平均，使獎勵梯度向下偏差 1/64；`test_stage09_batch_mean_baseline_is_biased_and_leave_one_out_is_not` 檢查兩者。）

不用機率論也能直接看出縮放。因為

$$
\bar r=\frac1G\left(r_i+\sum_{j\ne i}r_j\right),
$$

群組平均優勢為

$$
r_i-\bar r
=\frac{G-1}{G}
\left(
r_i-\frac1{G-1}\sum_{j\ne i}r_j
\right)
=\frac{G-1}{G}(r_i-b_i).
$$

因此，對固定群組而言，群組平均與留一法優勢指向同一方向，卻相差已知因子。在最佳化器中，尺度會與學習率、截斷及正規化互動；「方向相同」不會讓每次訓練都完全一樣。

群組也必須含有資訊：

- 若所有抽樣回應都失敗，二元獎勵全是 0，置中優勢也全是 0；
- 若所有回應都通過，獎勵全是 1，置中優勢仍全是 0；
- 除非加入另一個訊號，只有群組內獎勵有差異時才會學習。

若還要把優勢除以群組標準差，全相等群組的分母也是零；實作必須明訂慣例，例如跳過正規化或加入 $\varepsilon$ 下限。這造成探索需求：只有策略抽到好與壞兩種結果後，驗證器才能區分它們。若正確解的機率幾乎為零，相對基準無法憑空發明解答。

:::note[主張檢核]
- **主張。** 留一法基準讓策略梯度估計量保持不偏。
- **假設。** $G$ 個樣本是獨立抽出的，因此 $b_i$ 不依賴 $a_i$。
- **檢驗。** 在一個小問題上精確計算期望梯度，再與各種基準下估計量的期望值比較（`test_stage09_batch_mean_baseline_is_biased_and_leave_one_out_is_not`）。
- **限制。** 不偏不等於變異數最小：使變異數最小的基準是以分數平方加權的平均獎勵（命題 9.2），一般而言它既不同於群組平均，也不同於 $V^\pi$。
:::

