---
title: "H · 馬可夫鏈與決策過程"
description: 馬可夫鏈、平穩分布與混合；馬可夫決策過程、價值函數與作為線性方程組的貝爾曼方程式；壓縮映射與價值迭代；折扣佔用分布與策略梯度定理的回合形式與佔用形式；基準及其偏差與變異數；熵與 KL 正則化的決策問題。
sidebar: { label: "H · 馬可夫決策過程", order: 8 }
---

單元 09 把後訓練當作強化學習來處理，而且主要處理單步問題，也就是答案只是一個詞元。本附錄以一般形式給出多步理論：馬可夫結構、貝爾曼方程式、價值迭代為什麼收斂、折扣因子放在正確位置的策略梯度定理，以及基準在什麼條件下不會讓梯度產生偏差。例題中的數字由 `labs/test_appendices.py` 檢查，策略梯度的部分則由 `spine/test_spine.py` 檢查。

| 小節 | 用於 |
|---|---|
| H.1 馬可夫鏈 | 01（n-gram 模型）、07（序列模型）、08（祖先抽樣） |
| H.2–H.3 MDP、貝爾曼方程式、價值迭代 | 09（§9.3.1、§9.3.3） |
| H.4 策略梯度定理 | 09（§9.2.2、§9.3.2） |
| H.5 基準 | 09（§9.2.3、§9.3.7、§9.5.1） |
| H.6 正則化決策問題 | 09（§9.3.4、§9.5.2） |

## H.1 馬可夫鏈

若有限集合中的狀態序列 $s_0, s_1, \dots$ 滿足 $\Pr(s_{t+1}\mid s_0,\dots,s_t) = \Pr(s_{t+1}\mid s_t)$，它就是**馬可夫鏈**。時間齊次的鏈由列隨機的**轉移矩陣** $P_{ij} = \Pr(s_{t+1} = j\mid s_t = i)$ 描述。狀態上的分布寫成列向量 $\mu_t$，依 $\mu_{t+1} = \mu_tP$ 演化。

二元語法模型是字元上的馬可夫鏈；$n$-gram 模型是 $(n-1)$ 元組上的馬可夫鏈；而自迴歸 Transformer 是以整個前綴為狀態的馬可夫鏈。祖先抽樣（單元 08，§8.3.1）就是執行這條鏈。

**平穩分布**滿足 $\pi = \pi P$。若鏈是**不可約的**（每個狀態都能到達其他每個狀態）且**非週期的**（回到某狀態的時間不限於某個 $d > 1$ 的倍數），則 $\pi$ 唯一，且從任何初始分布出發都有 $\mu_t\to\pi$（[Levin & Peres 2017](/zh-tw/references/#levin-2017)）。速度由 $P$ 第二大的特徵值模決定：到 $\pi$ 的距離大致像 $|\lambda_2|^t$ 那樣縮小。

**例題 H.1。** 兩狀態鏈 $P = \begin{pmatrix}0.9 & 0.1\\0.3 & 0.7\end{pmatrix}$ 的平穩分布為 $(0.75, 0.25)$——一般而言，若離開狀態 1 的轉換機率為 $a$、離開狀態 2 的為 $b$，平穩分布為 $(b, a)/(a + b)$——第二特徵值為 $1 - a - b = 0.6$。從狀態 1 出發，位於狀態 1 的機率為 $0.75 + 0.25\times0.6^t$：$1, 0.9, 0.84, 0.804, \dots$

若 $\pi_iP_{ij} = \pi_jP_{ji}$，鏈就相對於 $\pi$ 滿足**細緻平衡**；對 $i$ 求和即可證明此時 $\pi$ 是平穩的。馬可夫鏈蒙地卡羅方法就是為所要的 $\pi$ 構造具有這個性質的鏈。

## H.2 馬可夫決策過程與貝爾曼方程式

**馬可夫決策過程**加上了動作與獎勵：在狀態 $s$ 時代理採取動作 $a$，得到獎勵 $r(s, a)$，並移到 $s'\sim P(\cdot\mid s, a)$。**策略** $\pi(a\mid s)$ 選擇動作。以折扣 $\gamma\in[0, 1)$，從時間 $t$ 起的**回報**為 $G_t = \sum_{k\ge0}\gamma^kr_{t+k}$，狀態的**價值**為 $V^\pi(s) = \mathbb{E}_\pi[G_t\mid s_t = s]$（[Sutton & Barto 2018](/zh-tw/references/#sutton-2018)；[Puterman 1994](/zh-tw/references/#puterman-1994)）。

在固定策略下，狀態構成一條馬可夫鏈，其轉移矩陣為 $P_\pi(s, s') = \sum_a\pi(a\mid s)P(s'\mid s, a)$，期望獎勵為 $r_\pi(s) = \sum_a\pi(a\mid s)r(s, a)$。把回報拆成第一個獎勵與其餘部分，就得到**貝爾曼期望方程式**
$$
V^\pi = r_\pi + \gamma P_\pi V^\pi\quad\Longleftrightarrow\quad V^\pi = (I - \gamma P_\pi)^{-1}r_\pi .
$$
逆矩陣存在，因為 $\rho(\gamma P_\pi)\le\gamma < 1$（隨機矩陣的譜半徑為 1），其 Neumann 級數 $\sum_t\gamma^tP_\pi^t$ 就是期望的折扣造訪次數。

**例題 H.2。** 兩個狀態，$P_\pi = \begin{pmatrix}0.9 & 0.1\\0.5 & 0.5\end{pmatrix}$，狀態 1 的獎勵為 0、狀態 2 為 1，$\gamma = 0.9$。解 $\begin{pmatrix}0.19 & -0.09\\-0.45 & 0.55\end{pmatrix}V = \begin{pmatrix}0\\1\end{pmatrix}$ 得 $V = (45/32, 95/32) = (1.406, 2.969)$。狀態 2 的價值是狀態 1 的兩倍以上，雖然每次造訪只付 1，因為從狀態 1 很少到達它。

**動作價值** $Q^\pi(s, a) = r(s, a) + \gamma\sum_{s'}P(s'\mid s, a)V^\pi(s')$ 滿足 $V^\pi(s) = \sum_a\pi(a\mid s)Q^\pi(s, a)$，而**優勢**為 $A^\pi = Q^\pi - V^\pi$。**貝爾曼最佳性方程式**
$$
V^\star(s) = \max_a\Big[r(s, a) + \gamma\sum_{s'}P(s'\mid s, a)V^\star(s')\Big]
$$
刻畫最佳價值；任何相對於 $V^\star$ 為貪婪的策略都是最佳的。

## H.3 壓縮映射與價值迭代

把最佳性方程式的右邊寫成作用在價值函數上的算子 $\mathcal T$。

**定理 H.1。** $\mathcal T$ 在上確界範數下是 $\gamma$-壓縮：$\lVert\mathcal TU - \mathcal TV\rVert_\infty\le\gamma\lVert U - V\rVert_\infty$。

*證明。* 對每個 $s$，$|\max_af_a - \max_ag_a|\le\max_a|f_a - g_a|$。套用到 $f_a = r + \gamma\sum_{s'}PU$ 與 $g_a = r + \gamma\sum_{s'}PV$，差至多為 $\gamma\max_a\sum_{s'}P(s'\mid s, a)|U(s') - V(s')|\le\gamma\lVert U - V\rVert_\infty$。$\blacksquare$

由 Banach 不動點定理，$\mathcal T$ 有唯一的不動點 $V^\star$，而**價值迭代** $V_{k+1} = \mathcal TV_k$ 以 $\lVert V_k - V^\star\rVert_\infty\le\gamma^k\lVert V_0 - V^\star\rVert_\infty$ 收斂到它。同樣的論證，把 $\max_a$ 換成 $\pi$ 下的平均，就證明了迭代期望方程式會收斂到 $V^\pi$。

**例題 H.3。** 對例題 H.2，從 $V_0 = 0$ 出發，界 $0.9^k\times2.969 < 10^{-3}$ 需要 $k\ge76$；實際誤差在 $k = 71$ 時第一次降到 $10^{-3}$ 以下。折扣是使問題良定的原因：當 $\gamma\to1$，壓縮變弱，迭代次數像 $1/(1 - \gamma)$ 那樣增長。

**策略迭代**交替進行 $V^\pi$ 的精確評估（一次線性求解）與貪婪改進；**策略改進定理**保證每個新策略在每個狀態上都至少一樣好，而對有限 MDP，這個過程在有限步內終止於最佳策略。

## H.4 策略梯度定理

令 $\pi_\theta$ 對 $\theta$ 可微，$J(\theta) = \mathbb{E}_{\pi_\theta}[G_0]$，回合從固定的初始分布開始。

**定理 H.2（回合形式）。** 對有限長度的回合與任何不依賴 $a_t$ 的基準 $b(s_t)$，
$$
\nabla_\theta J = \mathbb{E}\Big[\sum_t\gamma^t\,\nabla_\theta\ln\pi_\theta(a_t\mid s_t)\,\big(G_t - b(s_t)\big)\Big].
$$
證明即單元 09 定理 9.1：對軌跡分布套用分數函數恆等式，再移除早於 $t$ 的獎勵（它們不依賴 $a_t$），第 $t$ 步那一項剩下 $\sum_{k\ge t}\gamma^kr_k = \gamma^tG_t$。

**折扣佔用分布。** 定義 $d^\pi_\gamma(s) = (1 - \gamma)\sum_t\gamma^t\Pr(s_t = s)$，一個狀態上的機率分布，較早的造訪權重較大。

**定理 H.3（佔用形式）。** 對 $\gamma < 1$，
$$
\nabla_\theta J = \frac{1}{1 - \gamma}\,\mathbb{E}_{s\sim d^\pi_\gamma,\ a\sim\pi_\theta(\cdot\mid s)}\big[Q^\pi(s, a)\,\nabla_\theta\ln\pi_\theta(a\mid s)\big].
$$
*證明概要。* 對 $V^\pi(s) = \sum_a\pi(a\mid s)Q^\pi(s, a)$ 微分並用 $Q^\pi = r + \gamma PV^\pi$：
$\nabla V^\pi(s) = \sum_a\nabla\pi(a\mid s)Q^\pi(s, a) + \gamma\sum_a\pi(a\mid s)\sum_{s'}P(s'\mid s, a)\nabla V^\pi(s')$。展開遞迴，第一項在所有未來狀態上累積，每個以 $\gamma^t\Pr(s_t = s)$ 加權；把權重正規化就得到 $d^\pi_\gamma$。$\blacksquare$

兩種形式一致：定理 H.2 中的 $\gamma^t$ 正是定理 H.3 的佔用加權，而 $\mathbb{E}[\gamma^tG_t\mid s_t, a_t] = \gamma^tQ^\pi(s_t, a_t)$。許多實作**省略 $\gamma^t$**，並從未折扣的造訪分布抽樣狀態；結果不是 $J$ 的梯度，而是另一個加權方式不同之目標的梯度（$\gamma = 1$ 時兩者一致）。**例題 H.4**（單元 09，§9.3.2）：$t = 0$ 只有一個動作；$t = 1$ 是 logit 為 $\theta = 0$ 的伯努利動作，只有為 1 時才得到獎勵；$\gamma = 0.5$。真實導數為 $0.5\,\sigma'(0) = 0.125$；不含 $\gamma^t$ 的式子給出 $0.25$。

## H.5 基準：偏差與變異數

**不偏性。** 只要 $\mathbb{E}[b\,\nabla\ln\pi_\theta(a_t\mid s_t)] = 0$，減去 $b$ 就不改變梯度。若 $b$ 是 $s_t$ 的函數（以及任何在抽出 $a_t$ 之前就已確定、或與它獨立抽出的東西），這就成立，因為 $\sum_a\pi(a\mid s)\nabla\ln\pi(a\mid s) = \nabla\sum_a\pi(a\mid s) = 0$。若 $b$ 依賴 $a_t$——包括透過樣本自己的獎勵——它就不成立。

**批次與群組平均。** 以 $B$ 個獨立樣本，並對每個樣本都用基準 $\bar r = \frac1B\sum_jr_j$，$\bar r$ 中樣本自身獎勵的那一份與 $a_i$ 相關，而對獨立同分布樣本，
$$
\mathbb{E}\Big[\frac1B\sum_i(r_i - \bar r)\nabla\ln\pi(a_i)\Big] = \Big(1 - \frac1B\Big)\nabla J .
$$
**留一法**基準 $b_i = \frac1{B-1}\sum_{j\ne i}r_j$ 獨立於 $a_i$ 且不偏，它與批次平均只差優勢上的一個因子 $B/(B - 1)$。除以群組標準差的群組相對方法，會引入另一個依資料而定的縮放。主幹專案使用留一法基準（[勘誤](/zh-tw/start/errata/)，K03）。

**變異數。** 在單一座標的常數基準中，使變異數最小的選擇是以分數加權的平均 $b^\star = \mathbb{E}[r\,s^2]/\mathbb{E}[s^2]$，$s = \partial\ln\pi$；對狀態相依基準與整個梯度，$b^\star(s) = \mathbb{E}_a[Q^\pi(s, a)\lVert\nabla\ln\pi(a\mid s)\rVert^2]/\mathbb{E}_a[\lVert\nabla\ln\pi(a\mid s)\rVert^2]$（[Greensmith et al. 2004](/zh-tw/references/#greensmith-2004)）。價值 $V^\pi(s)$ 是常見的實用選擇，而不是最佳選擇。**例題 H.5：**對 logit 參數化、$p = 0.8$、動作 1 獎勵為 1 的伯努利策略，$V = 0.8$ 而 $b^\star = 0.2$，後者使單樣本估計對任一動作都恰好是 $0.16$（單元 09，§9.5.1）。

## H.6 熵與 KL 正則化的決策問題

在每一步的獎勵上加上 $\tau H(\pi(\cdot\mid s))$——或減去 $\beta\,\mathrm{KL}(\pi(\cdot\mid s)\,\Vert\,\pi_{\text{ref}}(\cdot\mid s))$——就得到正則化的 MDP。單步時，吉布斯變分原理（附錄 F.6）解出它：$\pi^\star\propto\pi_{\text{ref}}\,e^{r/\beta}$，值為 $\beta\ln Z$（單元 09，命題 9.1）。多步時，同樣的原理在每個狀態上適用，只是把 $r$ 換成動作價值，得到**軟貝爾曼方程式**
$$
V^\star(s) = \beta\ln\sum_a\pi_{\text{ref}}(a\mid s)\exp\Big(\frac{r(s, a) + \gamma\,\mathbb{E}_{s'}V^\star(s')}{\beta}\Big),
$$
其算子同樣是 $\gamma$-壓縮，而最佳策略為 $\pi^\star(a\mid s)\propto\pi_{\text{ref}}(a\mid s)\exp(Q^\star(s, a)/\beta)$。精確解保有 $\pi_{\text{ref}}$ 的完整支撐；實務上多樣性的喪失來自小的 $\beta$、受限的策略類別與不完整的最佳化（單元 09，§9.5.2；附錄 D.7）。

## 習題

**H.1** 求 $P = \begin{pmatrix}0.5 & 0.5 & 0\\0.25 & 0.5 & 0.25\\0 & 0.5 & 0.5\end{pmatrix}$ 的平穩分布，並檢查細緻平衡。

**H.2** 證明 $(I - \gamma P_\pi)^{-1}$ 的元素非負，且每一列的總和為 $1/(1 - \gamma)$。

**H.3** 對例題 H.2，分別以 $\gamma = 0.5$ 與 $\gamma = 0.99$ 計算 $V^\pi$。哪個狀態的價值改變較多？為什麼？

**H.4** 證明期望算子 $V\mapsto r_\pi + \gamma P_\pi V$ 在上確界範數下是 $\gamma$-壓縮。

**H.5** 明確計算 $d^\pi_\gamma$ 與 $Q^\pi$，在例題 H.4 上驗證定理 H.3。

**H.6** ★ 證明 H.6 的軟貝爾曼算子是 $\gamma$-壓縮。*（提示：log-sum-exp 在上確界範數下是 1-Lipschitz 的。）*

**解答。** **H.1** $\pi = (0.25, 0.5, 0.25)$；細緻平衡成立（每對相鄰狀態都有 $0.25\times0.5 = 0.5\times0.25$）。**H.3** $\gamma = 0.5$：$V = (0.125, 1.375)$；$\gamma = 0.99$：$V = (16.39, 18.05)$——兩者都隨有效時程 $1/(1-\gamma)$ 增長，而狀態 1 的價值從狀態 2 的 9% 升到 91%，因為它最終造訪狀態 2 的價值，如今幾乎與立即造訪一樣多。

## 延伸閱讀

[Sutton & Barto 2018](/zh-tw/references/#sutton-2018) 第 3–4 章與第 13 章；[Puterman 1994](/zh-tw/references/#puterman-1994) 為 MDP 的完整理論；[Levin & Peres 2017](/zh-tw/references/#levin-2017) 講馬可夫鏈與混合；[Greensmith et al. 2004](/zh-tw/references/#greensmith-2004) 講策略梯度的變異數降低。
