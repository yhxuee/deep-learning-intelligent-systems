---
title: "1.3 數學建構"
description: 母體風險與經驗風險、交叉熵 = 熵 + KL 與 Gibbs 不等式、以拉格朗日乘數推導最大概似二元語法、把加 α 平滑看成 Dirichlet 後驗平均、Hoeffding 不等式、挑選的代價，以及保留資料損失的標準誤。
sidebar: { label: "1.3 數學建構", order: 3 }
---

本節把 §1.2 轉化為數學。每個小節陳述一個結果、完整證明，並說明它對二元語法的意義。

## 1.3.1 設定：分布、預測器與風險

資料是從未知的**資料分布** $p_{\text{data}}$ 抽出的配對 $(x, y)$：對語言模型而言，$x$ 是上下文（對二元語法來說就是前一個字元），$y$ 是下一個字元。**預測器** $q$ 把每個 $x$ 對應到 $y$ 的 $V$ 種可能值上的一個分布 $q(\cdot\mid x)$。$q$ 在單一樣本上的**損失**是 $\ell(q; x, y) = -\ln q(y\mid x)$。

我們在意的是在*新*樣本上的期望損失，即**母體風險（population risk）**：

$$
L(q) = \mathbb{E}_{(x,y)\sim p_{\text{data}}}\big[-\ln q(y\mid x)\big].
$$

我們無法計算它，因為 $p_{\text{data}}$ 未知。我們能計算的是樣本 $\mathcal{D} = \{(x_i, y_i)\}_{i=1}^n$ 上的**經驗風險（empirical risk）**：

$$
\hat L(q) = \frac{1}{n}\sum_{i=1}^n -\ln q(y_i\mid x_i).
$$

若樣本是從 $p_{\text{data}}$ 抽出的 i.i.d. 樣本，而且 $q$ 是在*沒看過* $\mathcal{D}$ 的情況下選定的，則 $\hat L(q)$ 是 $L(q)$ 的不偏估計：由期望值的線性，$\mathbb{E}[\hat L(q)] = L(q)$。這一行就是保留資料評估之所以有效的原因，也是訓練損失無效的原因：當 $q$ 是*利用* $\mathcal{D}$ 選出時，這個期望值論證就不再適用（§1.3.6）。

**經驗風險最小化（empirical risk minimization, ERM）**從一個模型族 $\mathcal{Q}$ 中選出讓訓練集上 $\hat L$ 最小的 $q$。學習理論問的是：ERM 選出的 $q$ 什麼時候也有小的 $L$？

## 1.3.2 交叉熵、熵與 KL 散度

固定一個上下文 $x$ 並在記號中省略它。令 $p$ 為 $y$ 的真實分布，$q$ 為模型的分布。$q$ 在 $p$ 下的期望對數損失是**交叉熵（cross-entropy）**

$$
H(p, q) = -\sum_{y} p(y)\ln q(y).
$$

加上再減去 $\sum_y p(y)\ln p(y)$，得到

$$
H(p, q) = \underbrace{-\sum_y p(y)\ln p(y)}_{H(p)\ \text{（熵）}} \;+\; \underbrace{\sum_y p(y)\ln\frac{p(y)}{q(y)}}_{\mathrm{KL}(p\,\Vert\, q)\ \text{（KL 散度）}} .
$$

**定理 1.1（Gibbs 不等式）。** $\mathrm{KL}(p\,\Vert\,q)\ge 0$，等號成立若且唯若 $q = p$。

*證明。* $\ln$ 是凹函數，所以由 Jensen 不等式，對任何正隨機變數 $Z$ 有 $\mathbb{E}_p[\ln Z] \le \ln \mathbb{E}_p[Z]$。取 $Z = q(y)/p(y)$，範圍限於 $p(y) > 0$ 的結果：

$$
-\mathrm{KL}(p\,\Vert\,q) = \sum_{y:\,p(y)>0} p(y)\ln\frac{q(y)}{p(y)} \le \ln\sum_{y:\,p(y)>0} p(y)\frac{q(y)}{p(y)} = \ln\sum_{y:\,p(y)>0} q(y) \le \ln 1 = 0 .
$$

Jensen 不等式取等號需要 $q(y)/p(y)$ 在 $p>0$ 處為常數，最後一步取等號則需要 $q$ 在 $p = 0$ 處沒有機率質量；兩者合起來就是 $q = p$。$\blacksquare$

**系理 1.1。** 期望對數損失 $H(p,q)$ 恰在 $q = p$ 時最小，最小值為熵 $H(p)$。每個模型的母體風險都可分解為

$$
L(q) = \underbrace{\mathbb{E}_x\big[H(p(\cdot\mid x))\big]}_{\text{不可約減}} + \underbrace{\mathbb{E}_x\big[\mathrm{KL}\big(p(\cdot\mid x)\,\Vert\, q(\cdot\mid x)\big)\big]}_{\text{可約減}} .
$$

這有兩個貫穿全書的後果。第一，沒有任何模型的保留資料損失能低於資料的條件熵：存在一個下限，而且通常不是零（單元 06 把它稱為縮放定律中的*不可約減損失* $E$）。第二，超出下限的**額外**損失是一個 KL 散度——量測模型的信念離真相多遠。對數損失之所以是 §1.2.3 所說的恰當評分，正是因為系理 1.1。

## 1.3.3 最大概似二元語法

二元語法模型族是所有滿足 $q(b\mid a)\ge0$、且對每個 $a$ 都有 $\sum_b q(b\mid a) = 1$ 的表 $q(b\mid a)$。序列 $x_1,\dots,x_n$ 上的訓練對數損失為

$$
n\hat L(q) = -\sum_{t=2}^{n}\ln q(x_t\mid x_{t-1}) = -\sum_{a}\sum_{b} c(a\to b)\ln q(b\mid a),
$$

這裡把相同的配對歸併在一起。總和拆成彼此獨立的問題，每一列 $a$ 一個。

**命題 1.1。** 對每個 $c(a)>0$ 的 $a$，在 $\sum_b q(b\mid a) = 1$ 的限制下，$-\sum_b c(a\to b)\ln q(b\mid a)$ 的最小值發生在 $q(b\mid a) = c(a\to b)/c(a)$。

*以拉格朗日乘數證明。* 記 $c_b = c(a\to b)$、$q_b = q(b\mid a)$，最小化 $F = -\sum_b c_b \ln q_b + \lambda(\sum_b q_b - 1)$。令 $\partial F/\partial q_b = -c_b/q_b + \lambda = 0$ 得 $q_b = c_b/\lambda$；由限制條件得 $\lambda = \sum_b c_b = c(a)$。目標函數對 $q$ 是凸的（$-\ln$ 的總和），所以這個駐點就是全域最小值。$\blacksquare$

*不用微積分的第二種證明。* 令 $\hat p_b = c_b/c(a)$。則 $-\sum_b c_b\ln q_b = c(a)\,H(\hat p, q) = c(a)\,[H(\hat p) + \mathrm{KL}(\hat p\,\Vert\,q)]$，由 Gibbs 不等式，它在 $q = \hat p$ 時最小。**最大概似就是最小化從經驗分布到模型的 KL 散度。** 這個觀點可推廣到本書的每個模型：以交叉熵訓練神經網路，就是在 KL 意義下把 $q_\theta$ 擬合到經驗分布。

由此得到 §1.2.8 與單元 03 主幹專案測試所用的主張：**任何二元語法模型——計數的、平滑的或神經的——在一份訓練文本上的訓練損失，都不可能低於該文本的最大概似表。** 主幹專案階段 03 驗證其以梯度訓練的二元語法從上方收斂到這個值，而且從不低於它。

## 1.3.4 把平滑看成貝氏推論

把一列 $\theta = (\theta_1,\dots,\theta_V)$、$\theta_b = q(b\mid a)$ 視為一個未知的隨機向量，並給它參數全為 $\alpha$ 的 **Dirichlet 先驗**：

$$
\pi(\theta) \propto \prod_b \theta_b^{\alpha-1}, \qquad \theta_b\ge 0,\ \textstyle\sum_b\theta_b = 1 .
$$

觀察到的計數的概似為 $\prod_b \theta_b^{c_b}$。相乘之後，後驗仍是 Dirichlet 分布，參數為 $c_b + \alpha$：

$$
\pi(\theta\mid c) \propto \prod_b \theta_b^{c_b + \alpha - 1}.
$$

參數為 $\beta_b$ 的 Dirichlet 分布的平均是 $\beta_b/\sum_{b'}\beta_{b'}$，所以**後驗平均**為

$$
\mathbb{E}[\theta_b \mid c] = \frac{c_b + \alpha}{c(a) + \alpha V},
$$

恰好就是加 $\alpha$ 平滑。而且在對數損失下，後驗平均就是貝氏最佳預測：把我們對 $\theta$ 的不確定性平均掉之後，*下一個*字元是 $b$ 的機率就是 $\mathbb{E}[\theta_b\mid c]$。所以平滑不是權宜之計；它是一個在看到資料前相信每一列都來自對稱 Dirichlet 分布的學習者所做的預測。小的 $\alpha$ 表示相信每列都很集中（多數配對從不出現）；大的 $\alpha$ 表示相信每列都接近均勻。$\alpha = 1$、$V = 2$ 時就是拉普拉斯的繼起法則：$n$ 次試驗中成功 $s$ 次後，預測 $(s+1)/(n+2)$。

先驗的選擇是一種**歸納偏差（inductive bias）**——一個不在資料中的假設。每個學習者都有歸納偏差；平滑讓這一個變得明確且可調整。單元 02 會引入一個強得多的歸納偏差：相似的字元應該有相似的列。

## 1.3.5 量測到的損失可能離真值多遠？

對一個*固定的*預測器 $q$，保留資料損失 $\hat L(q)$ 是 $n$ 個平均為 $L(q)$ 的 i.i.d. 項的平均。它可能有多接近真值？

**定理 1.2（Hoeffding 不等式，[Hoeffding 1963](/zh-tw/references/#hoeffding-1963)）。** 若 $Z_1,\dots,Z_n$ 彼此獨立且 $Z_i\in[a,b]$，則對每個 $\varepsilon > 0$

$$
\Pr\Big(\Big|\tfrac{1}{n}\textstyle\sum_i Z_i - \mathbb{E}\big[\tfrac1n\sum_i Z_i\big]\Big| \ge \varepsilon\Big) \le 2\exp\!\Big(-\frac{2n\varepsilon^2}{(b-a)^2}\Big).
$$

（證明見 §1.5.1 ★。）令右側等於 $\delta$ 並解出：以至少 $1-\delta$ 的機率，

$$
|\hat L(q) - L(q)| \le (b-a)\sqrt{\frac{\ln(2/\delta)}{2n}} .
$$

對數損失一般來說沒有上界——機率接近零時損失極大——但平滑後的模型有：使用加 $\alpha$ 平滑時，每個機率至少是 $\alpha/(c(a) + \alpha V)$，所以損失至多是 $\ln((c(a)+\alpha V)/\alpha)$。更重要的是，這個界顯示了**速率**：保留資料估計的誤差以 $1/\sqrt n$ 的速度縮小。保留資料變成四倍，不確定性減半。詳解例題 1.6 代入了數字。

## 1.3.6 在許多模型中挑選需要新的資料

Hoeffding 不等式適用於在看到資料*之前*就固定的預測器。假設改為在同一份資料上評估 $K$ 個預測器 $q_1,\dots,q_K$ 並保留最好的那個。每一個都有 $\delta/K$ 的小機率出現大偏差；由**聯集界（union bound）**（$\Pr(\cup_k A_k)\le\sum_k\Pr(A_k)$），以至少 $1-\delta$ 的機率，*全部* $K$ 個估計都在其真值的

$$
\varepsilon_K = (b-a)\sqrt{\frac{\ln K + \ln(2/\delta)}{2n}}
$$

範圍內。在 $K$ 個候選中挑選的代價只隨 $\sqrt{\ln K}$ 成長——但它確實是代價。如果你嘗試一千種設定並回報最好的驗證分數，這個分數最多會樂觀 $\varepsilon_K$，只有**只使用一次的測試集**能給出不偏的數字。把同樣的論證套用在訓練集上，「候選」就是所有可能的參數值，這正是 §1.5.1 均勻收斂理論的起點。

這個界也說明了為什麼*適應性（adaptive）*重複使用比聯集界所暗示的更糟：如果第 500 個候選的選擇取決於前 499 個的分數，候選就不再是事先固定的，一個有心的分析者用遠少於 $e^{n}$ 次的嘗試就能過擬合驗證集（[Dwork et al. 2015](/zh-tw/references/#dwork-2015)）。這就是為什麼基準測試的分數會隨多年的重複使用而上漂（§1.7.3；[Recht et al. 2019](/zh-tw/references/#recht-2019)）。

## 1.3.7 保留資料損失的標準誤

Hoeffding 給的是最壞情況的界。實務上我們回報的是**標準誤（standard error）**。若每個樣本的損失 $Z_i$ 彼此獨立、變異數為 $\sigma^2$，則平均值的變異數為 $\sigma^2/n$，所以

$$
\mathrm{se}(\hat L) = \frac{\sigma}{\sqrt n},
$$

以樣本標準差代替 $\sigma$ 來估計。近似的 95% 區間為 $\hat L \pm 1.96\,\mathrm{se}$（由中央極限定理）。

「獨立」這個詞在這裡很關鍵。一個故事中每個字元的損失並不獨立：在主幹語料上，二元語法下相鄰字元損失的相關係數是 $-0.32$。誠實的抽樣單位是**故事**，因為被獨立抽出的是故事，而不是字元。以每個故事的平均損失計算標準誤（詳解例題 1.7）是最簡單的正確做法；單元 10 會發展自助法（bootstrap），它能處理任何統計量。

**要多少保留樣本才能分辨兩個模型？** 假設模型 B 平均每個樣本比模型 A 好 $d$ nats，而兩者每個樣本損失*差*的標準差為 $\sigma_d$。顯著水準 5%、檢定力 80% 的雙尾檢定大約需要

$$
n \approx \Big(\frac{(1.96 + 0.84)\,\sigma_d}{d}\Big)^2
$$

個獨立樣本。注意是 $\sigma_d$，而不是任一模型損失的標準差：在*相同*樣本上評估兩個模型（配對）能抵銷每個樣本共有的難度，讓 $\sigma_d$ 小得多（單元 10，§10.3）。

## 1.3.8 單位：nats、bits 與困惑度

文獻中會看到同一個數字的三種等價報告方式：

| 量 | 公式 | 主幹二元語法（保留資料） |
|---|---|---|
| 以 nats／字元計的損失 | $\hat L$ | 2.124 |
| bits／字元 | $\hat L/\ln 2$ | 3.065 |
| 困惑度（perplexity） | $e^{\hat L}$ | 8.37 |

困惑度有一個直觀的讀法：困惑度 8.37 的模型，平均而言就像在 8.37 個等可能的字元中均勻挑選一樣不確定。均勻模型的困惑度恰好是 $V$。因為困惑度是指數，損失的小差異會變成看起來很大的困惑度差異；本書回報 nats。
