---
title: "9.3 數學建構"
description: 馬可夫決策過程、帶基準與優勢的策略梯度定理、價值函數與貝爾曼方程式、KL 正則化獎勵最大化的封閉形式最佳解、由偏好得到的獎勵模型、PPO 的截斷目標函數、DPO 的推導，以及可驗證獎勵與群組基準。
sidebar: { label: "9.3 數學建構", order: 3 }
---

## 9.3.1 馬可夫決策過程

一個**馬可夫決策過程（Markov decision process，MDP）**包含狀態 $s$、動作 $a$、轉移分布 $P(s'\mid s, a)$、獎勵 $r(s, a)$ 與折扣 $\gamma\in[0, 1]$（[Sutton & Barto 2018](/zh-tw/references/#sutton-2018)）。策略 $\pi_\theta(a\mid s)$ 產生軌跡；從時間 $t$ 起的**回報（return）**是 $G_t = \sum_{k\ge0}\gamma^kr_{t+k}$，目標函數為 $J(\theta) = \mathbb{E}_{\tau\sim\pi_\theta}[G_0]$。

文字生成是一個結構特別簡單的 MDP：狀態是提示加上目前已生成的詞元，動作是下一個詞元，轉移以確定的方式把它接上去，而獎勵通常只在最後出現一次。主幹專案的算術任務是一步的 MDP——一個*情境式吃角子老虎機（contextual bandit）*。

## 9.3.2 策略梯度定理

**定理 9.1（策略梯度，回合式折扣目標）。** 令 $J(\theta) = \mathbb{E}_{\tau\sim\pi_\theta}[G_0]$，其中對有限長度的回合 $G_t = \sum_{k\ge0}\gamma^kr_{t+k}$。對任何不依賴 $a_t$ 的基準 $b(s_t)$，
$$
\nabla_\theta J(\theta) = \mathbb{E}_{\tau\sim\pi_\theta}\Big[\sum_{t}\gamma^t\,\nabla_\theta\ln\pi_\theta(a_t\mid s_t)\,\big(G_t - b(s_t)\big)\Big].
$$

*證明。* 軌跡機率為 $p_\theta(\tau) = p(s_0)\prod_t\pi_\theta(a_t\mid s_t)P(s_{t+1}\mid s_t, a_t)$；只有策略因子依賴 $\theta$，所以 $\nabla\ln p_\theta(\tau) = \sum_t\nabla\ln\pi_\theta(a_t\mid s_t)$。於是 $\nabla J = \nabla\int p_\theta(\tau)G_0(\tau) = \mathbb{E}[G_0\sum_t\nabla\ln\pi_\theta(a_t\mid s_t)]$。兩個改進。第一，時間 $t$ 之前得到的獎勵在抽出 $a_t$ 之前就已確定，因此它與 $\nabla\ln\pi(a_t\mid s_t)$ 的乘積期望為零；$G_0 = \sum_k\gamma^kr_k$ 在第 $t$ 步那一項中剩下的是 $\sum_{k\ge t}\gamma^kr_k = \gamma^tG_t$（因果性）。第二， $\mathbb{E}_{a_t}[b(s_t)\nabla\ln\pi_\theta(a_t\mid s_t)\mid s_t] = b(s_t)\nabla\sum_{a}\pi_\theta(a\mid s_t) = 0$，因此任何與動作無關的基準都可以減去（其權重 $\gamma^t$ 是常數）。$\blacksquare$

**因子 $\gamma^t$。** 它很容易被漏掉，而許多實作是刻意省略它：$\sum_t\nabla\ln\pi(a_t\mid s_t)(G_t - b)$ 是*另一個*目標函數的梯度——它把每個造訪到的狀態都當成回合的起點來加權。$\gamma = 1$ 時兩者一致。*例子：*一個兩步的回合，$t = 0$ 只有一個動作；$t = 1$ 時是 logit 為 $\theta = 0$ 的伯努利動作，若為 1 則得到獎勵 1。以 $\gamma = 0.5$，$J = 0.5\,\sigma(\theta)$，$\partial J/\partial\theta = 0.125$。定理給出 0.125；省略 $\gamma^t$ 的式子給出 0.25。主幹專案的單步任務只有 $t = 0$，此時 $\gamma^0 = 1$，因此無法揭露這個差異；`test_stage09_discounted_policy_gradient_carries_gamma_t` 檢查了兩步的情況。

對有限動作集上 logits 為 $z$ 的 softmax 策略，$\nabla_z\ln\pi(a) = e_a - \pi$（單元 03，§3.3.5）。因此其梯度即為 REINFORCE 估計的單樣本損失是 $\ell = -(G - b)\ln\pi(a)$，且 $\partial\ell/\partial z = (G - b)(\pi - e_a)$——朝抽到之動作的交叉熵，以其優勢縮放。`spine/stage09_post_training.py` 正好實作了這一行，並以有限差分穿過整個 Transformer 檢查它。

## 9.3.3 價值函數與貝爾曼方程式

在 $\pi$ 下一個狀態的**價值（value）**是 $V^\pi(s) = \mathbb{E}_\pi[G_t\mid s_t = s]$，**動作價值（action value）**是 $Q^\pi(s, a) = \mathbb{E}_\pi[G_t\mid s_t = s, a_t = a]$。它們滿足**貝爾曼方程式（Bellman equation）**（[Bellman 1957](/zh-tw/references/#bellman-1957)）：

$$
V^\pi(s) = \sum_a\pi(a\mid s)\Big[r(s, a) + \gamma\sum_{s'}P(s'\mid s, a)V^\pi(s')\Big],
$$

對固定策略是一個線性方程組；把平均換成 $\max_a$，就是以動態規劃求解的最佳性方程式。$V^\pi(s_t)$ 是最常用的狀態相依基準，但一般而言*不是*使變異數最小的那一個：§9.5.1 推導出最佳解，是以分數平方加權的動作價值平均，可能與 $V^\pi$ 相去甚遠——對 $p = 0.8$、動作 1 得到獎勵 1 的伯努利策略，$V = 0.8$，最佳基準卻是 0.2，並使估計的變異數為零（[Greensmith et al. 2004](/zh-tw/references/#greensmith-2004)）。**優勢** $A^\pi(s, a) = Q^\pi(s, a) - V^\pi(s)$ 衡量一個動作比策略的平均好多少。行動者–評論者（actor–critic）方法以第二個網路學習 $V$，並以 $r_t + \gamma V(s_{t+1}) - V(s_t)$ 或其平滑版本（[Schulman et al. 2016](/zh-tw/references/#schulman-2016)）作為優勢——以一點偏差換取比原始回報低得多的變異數。

## 9.3.4 KL 正則化目標函數及其最佳解

固定一個狀態，把 $\pi(\cdot\mid s)$ 寫成 $\pi$。考慮

$$
J_\beta(\pi) = \sum_a\pi(a)r(a) - \beta\sum_a\pi(a)\ln\frac{\pi(a)}{\pi_{\text{ref}}(a)} .
$$

**命題 9.1。** 在所有分布中，$J_\beta$ 的最大化者為 $\pi^\star(a) = \pi_{\text{ref}}(a)\,e^{r(a)/\beta}/Z$，其中 $Z = \sum_a\pi_{\text{ref}}(a)e^{r(a)/\beta}$，最大值為 $\beta\ln Z$。

*證明。* 改寫為 $J_\beta(\pi) = -\beta\sum_a\pi(a)\ln\frac{\pi(a)}{\pi_{\text{ref}}(a)e^{r(a)/\beta}} = -\beta\,\mathrm{KL}(\pi\,\Vert\,\pi^\star) + \beta\ln Z$。由吉布斯不等式（單元 01，定理 1.1），KL 項非負，且恰在 $\pi = \pi^\star$ 時為零。$\blacksquare$

傾斜後的分布保留參考分布在獎勵相同的動作之間的相對偏好，而每多 $\Delta r$ 單位的獎勵，就把該動作的勝算乘上 $e^{\Delta r/\beta}$。主幹專案的測試檢查對 $J_\beta$ 做梯度上升會收斂到 $\pi^\star$，精確到小數點後六位。把這個關係反過來，就得到 DPO 需要的那一步：

$$
r(a) = \beta\ln\frac{\pi^\star(a)}{\pi_{\text{ref}}(a)} + \beta\ln Z .
$$

## 9.3.5 獎勵模型與 PPO

**獎勵模型。** 收集附帶成對回應的提示，以及人類（或 AI）對哪一個較好的標註。以最大化 Bradley–Terry 對數概似 $\sum\ln\sigma\big(r_\phi(y_w) - r_\phi(y_l)\big)$ 擬合 $r_\phi$——對獎勵差做邏輯斯迴歸（[Christiano et al. 2017](/zh-tw/references/#christiano-2017)；[Stiennon et al. 2020](/zh-tw/references/#stiennon-2020)）。只有差值可被辨識：每個獎勵都加上一個常數不會改變任何事。

**PPO。** 接著以 KL 懲罰針對 $r_\phi$ 最佳化策略。因為每批樣本會被重複用於好幾個梯度步，正在更新的策略與產生樣本的策略不同；近端策略最佳化（Proximal Policy Optimization，PPO）以重要性比值 $\rho = \pi_\theta(a\mid s)/\pi_{\text{old}}(a\mid s)$ 修正，並在目標函數 $\mathbb{E}[\min(\rho A, \mathrm{clip}(\rho, 1-\epsilon, 1+\epsilon)A)]$ 中把它**截斷（clip）**在 $[1-\epsilon, 1+\epsilon]$（[Schulman et al. 2017](/zh-tw/references/#schulman-2017)）。截斷移除的是把被抽到之動作的比值推到區間之外的*誘因*；它不是硬性限制。未被抽到之動作的比值，以及相鄰策略之間的 KL 散度，都不受它約束，大幅更新仍可能發生——這就是為什麼實作會監控 KL 的估計值，並在它變大時提早結束該輪。這條三階段流程——監督式微調、獎勵模型、帶 KL 的 PPO——產生了第一批被廣泛使用、能遵循指令的助理（[Ouyang et al. 2022](/zh-tw/references/#ouyang-2022)）。

## 9.3.6 直接偏好最佳化

把 §9.3.4 反解出的最佳解代入 Bradley–Terry 模型。對同一個提示的兩個回應而言，$\beta\ln Z$ 項相等而相消：

$$
P(y_w\succ y_l) = \sigma\Big(\beta\ln\frac{\pi(y_w)}{\pi_{\text{ref}}(y_w)} - \beta\ln\frac{\pi(y_l)}{\pi_{\text{ref}}(y_l)}\Big).
$$

在此模型下最大化觀測到之偏好的概似，就得到 **DPO 損失**（[Rafailov et al. 2023](/zh-tw/references/#rafailov-2023)）：

$$
\mathcal L_{\text{DPO}} = -\mathbb{E}\Big[\ln\sigma\big(u\big)\Big],\qquad u = \beta\Big[\big(\ln\pi(y_w) - \ln\pi_{\text{ref}}(y_w)\big) - \big(\ln\pi(y_l) - \ln\pi_{\text{ref}}(y_l)\big)\Big].
$$

沒有獎勵模型、訓練時不需抽樣、沒有 PPO：它是成對資料上的分類損失。它對對數機率的梯度是 $-\beta\sigma(-u)\big(\nabla\ln\pi(y_w) - \nabla\ln\pi(y_l)\big)$——提高偏好的回應、降低不偏好的回應，權重是目前模型把兩者排錯的程度。對單一詞元的答案，兩個對數機率梯度中的 softmax 項相消，在 logits 上只剩 $-\beta\sigma(-u)(e_{y_w} - e_{y_l})$；主幹專案以有限差分檢查它。

## 9.3.7 可驗證獎勵與群組基準

當正確性能由程式檢查——算術、單元測試、證明檢查器——獎勵就不需要人，也不需要學習出來的模型。對同一個提示抽樣 $G$ 個答案，以它們的平均獎勵作為基準，得到優勢 $A_i = r_i - \bar r$（可再除以該群組的標準差），不需要價值網路（[Shao et al. 2024](/zh-tw/references/#shao-2024)）。因為 $\bar r$ 包含 $r_i$ 本身，這個基準並不獨立於 $a_i$：期望上它把梯度縮放了 $(G-1)/G$。**留一法（leave-one-out）**基準 $b_i = \frac{1}{G-1}\sum_{j\ne i}r_j$ 獨立於 $a_i$，使估計量保持不偏；它與群組平均只差優勢上的一個因子 $G/(G-1)$。除以群組標準差則引入另一個依資料而定的縮放。以這種方式在數學與程式上訓練大型模型，並讓它在作答前寫出長串推理，帶來了強大的推理表現（[DeepSeek-AI 2025](/zh-tw/references/#deepseek-2025)）**[E]**。主幹專案的階段 09 是這個配方的單詞元版本，以批次上的留一法作為基準；它用有錯檢查器的那次執行，則是警示版本（§9.7.1）。因為一個批次中的 64 個提示各不相同，這個基準是常數，而不是狀態的函數——不偏，但降低變異數的效果不如逐提示的基準。（v2.1 之前的版本使用單純的批次平均，使獎勵梯度向下偏差 1/64；`test_stage09_batch_mean_baseline_is_biased_and_leave_one_out_is_not` 檢查兩者。）
