---
title: "3.5 一般理論：梯度下降法為什麼收斂、收斂多快"
description: 泰勒展開與 Hessian 矩陣；平滑性與下降引理；收斂到駐點；在二次函數上的精確分析與條件數；強凸性；隨機梯度與雜訊底限；動量與預條件化。
sidebar: { label: "3.5 一般理論", order: 5 }
---

§3.2 分析了一條拋物線。本節證明關於梯度下降法一般能說什麼——同樣重要的是，不能說什麼。所有證明只用到泰勒定理、柯西–施瓦茨不等式與線性代數。這些內容的標準參考書為 [Nocedal & Wright 2006](/zh-tw/references/#nocedal-2006)、[Boyd & Vandenberghe 2004](/zh-tw/references/#boyd-2004)、[Nesterov 2018](/zh-tw/references/#nesterov-2018)，隨機情況則見 [Bottou、Curtis & Nocedal 2018](/zh-tw/references/#bottou-2018)。

## 3.5.1 二階的圖像：Hessian 矩陣與局部曲率

在 §3.2 中，我們利用梯度構建了損失函數的一階局部線性近似。然而線性模型完全無法感知「地勢的彎曲」。為了理解優化算法的收斂極限，必須進入二階泰勒展開的世界。

對二次連續可微的損失函數 $L(\theta)$，泰勒定理在局部將損失展開為：
$$
L(\theta + \delta) = L(\theta) + \nabla L(\theta)^{\top}\delta + \tfrac12\,\delta^{\top}H(\theta)\,\delta + o(\|\delta\|^2),
\qquad H_{ij} = \frac{\partial^2L}{\partial\theta_i\,\partial\theta_j}.
$$

**Hessian 矩陣（黑塞矩陣）的幾何解構：**
$H$ 是一個 $n \times n$ 的實對稱矩陣。根據線性代數的譜定理（Spectral Theorem），$H$ 必然存在一組由正交單位特徵向量構成的基底 $q_1,\dots,q_n$，以及對應的實數特徵值 $\lambda_1,\dots,\lambda_n$：
$H = Q\Lambda Q^{\top} = \sum_{i=1}^n \lambda_i q_i q_i^\top$。

若將位移向量投影到特徵基底下 $\delta = \sum_i \alpha_i q_i$，二次項精確解耦為無交互作用的獨立平方和：
$\frac{1}{2} \delta^\top H \delta = \frac{1}{2} \sum_{i=1}^n \lambda_i \alpha_i^2$。

這意味著：**在高維參數空間的任意局部，損失曲面沿著特徵向量 $q_i$ 方向的行為，完全等價於一條曲率為 $\lambda_i$ 的一維拋物線！**
- **正特徵值（$\lambda_i > 0$）：** 沿該方向曲面向上彎曲，如山谷碗底（凸曲率）；
- **負特徵值（$\lambda_i < 0$）：** 沿該方向曲面向下彎曲，如山脊穹頂（凹曲率）；
- **零特徵值（$\lambda_i = 0$）：** 平坦方向（退化山谷）。
- **鞍點（Saddle Point）：** 同時存在正特徵值與負特徵值。在幾百萬至數千億維度的高維非凸深度學習損失曲面中，駐點（$\nabla L = 0$）幾乎從來不是局部極小值，而是絕大多數特徵方向均為負或近零的**高階鞍點**（[Dauphin et al. 2014](/zh-tw/references/#dauphin-2014)）。

## 3.5.2 平滑性與下降引理

如果損失曲面的曲率無界劇烈震盪，任何固定步長的局部梯度更新都可能引發災難性數值爆炸。為了給出優化的數學保證，我們需要限制曲率的上限——這就是**平滑性（Smoothness）**。

**定義。** 若 $L$ 的梯度是 $\beta$-Lipschitz 連續的，即對所有 $x, y$ 滿足 $\|\nabla L(x) - \nabla L(y)\| \le \beta\|x - y\|$，則稱 $L$ 為 **$\beta$-平滑（$\beta$-smooth）**。
在二次可微條件下，這等價於 Hessian 矩陣的最大特徵值處處受限：$\lambda_{\max}(H(x)) \le \beta$。也就是說，曲面在任何方向上向上的彎曲程度都不會超過一條開口係數為 $\beta/2$ 的二次拋物線。

**引理 3.1（下降引理 / Descent Lemma）。** 若 $L$ 為 $\beta$-平滑，則對所有 $x, y$
$$
L(y) \le L(x) + \nabla L(x)^{\top}(y - x) + \frac\beta2\|y - x\|^2 .
$$

*證明。* 考慮連接 $x$ 與 $y$ 的線段路徑 $\gamma(s) = x + s(y-x)$，其中 $s\in[0,1]$。由微積分基本定理：
$$
L(y) - L(x) - \nabla L(x)^{\top}(y-x) = \int_0^1\bigl(\nabla L(x + s(y-x)) - \nabla L(x)\bigr)^{\top}(y-x)\,ds .
$$
由柯西–施瓦茨不等式與梯度的 $\beta$-Lipschitz 平滑性，被積函數上界為：
$\|\nabla L(x + s(y-x)) - \nabla L(x)\| \cdot \|y-x\| \le \beta s \|y-x\|^2$。
將 $\beta s \|y-x\|^2$ 沿 $s \in [0, 1]$ 積分，得 $\frac{\beta}{2}\|y-x\|^2$。整理即得下降引理。$\blacksquare$

**系理（梯度下降單步的下降保證）。** 在下降引理中令下一步更新為 $y = x - \eta\nabla L(x)$：
$$
L\bigl(x - \eta\nabla L(x)\bigr) \le L(x) - \eta\Bigl(1 - \frac{\beta\eta}{2}\Bigr)\|\nabla L(x)\|^2 .
$$

這項基本不等式徹底確立了梯度下降的安全工作區間：
- 只要學習率滿足 $0 < \eta < \frac{2}{\beta}$，係數 $\eta(1 - \frac{\beta\eta}{2}) > 0$，因此只要梯度非零，**損失保證在每一步嚴格單調下降**！
- 當 $\eta = \frac{1}{\beta}$ 時，係數取得最大值 $\frac{1}{2\beta}$，單步理論保證下降量最大：$L(x_{k+1}) \le L(x_k) - \frac{1}{2\beta}\|\nabla L(x_k)\|^2$。

:::note[主張檢核]
- **主張。** 每一個 $0 < \eta < 2/\beta$ 的梯度下降步都會降低損失。
- **假設。** $L$ 在整條路徑上都是 $\beta$-平滑的，梯度是精確的（全批次），而且 $\beta$ 已知。
- **檢驗。** 在最大特徵值已知的二次函數上，分別以略低於與略高於 $2/\lambda_{\max}$ 的步長執行（實驗 3.1）。
- **限制。** ReLU 網路並非處處平滑（§3.7.1），小批次梯度會帶來雜訊（§3.5.6），而曲率在訓練中會改變；所以這個界限是選擇學習率的指引，而不是直接決定學習率（§3.7.8）。
:::


## 3.5.3 沒有凸性時能保證什麼

神經網路的損失不是凸的。那還能證明什麼？

**定理 3.1（收斂到駐點）。** 設 $L$ 為 $\beta$-平滑且下有界於 $L^*$。以 $\eta = 1/\beta$ 做梯度下降，滿足

$$
\min_{0\le k<K}\|\nabla L(\theta_k)\|^2 \;\le\; \frac{2\beta\,\bigl(L(\theta_0) - L^*\bigr)}{K}.
$$

*證明。* 由系理，$\|\nabla L(\theta_k)\|^2 \le 2\beta\bigl(L(\theta_k) - L(\theta_{k+1})\bigr)$。對 $k = 0,\dots,K-1$ 加總，右邊伸縮相消為 $2\beta\bigl(L(\theta_0) - L(\theta_K)\bigr) \le 2\beta\bigl(L(\theta_0) - L^*\bigr)$。$K$ 個數的最小值至多是它們的平均。$\blacksquare$

**它說了什麼。** $K$ 步之後，某個迭代值的梯度範數至多為 $O(1/\sqrt K)$，常數與參數數量無關。**它沒說什麼。** 梯度小代表近似的*駐點（stationary point）*——可能是最小值、鞍點，也可能是平坦的高原。這裡完全沒說最小值是全域的，也沒說它會泛化。對非凸損失，這些都是真正的開放問題，以實證與特殊情況的理論來研究（§3.7.2）。

## 3.5.4 精確分析二次函數：條件數與「狹長峽谷」困境

在 Hessian 正定的極小值 $\theta^*$ 附近，任何光滑損失函數都可以被二階泰勒展開極佳地近似。因此，研究梯度下降法在純二次模型
$$
L(\theta) = \tfrac12(\theta - \theta^*)^{\top}H(\theta - \theta^*),\qquad 0<\lambda_{\min}\le\lambda_i\le\lambda_{\max},
$$
上的動力學，就精確揭示了它在一般光滑非凸曲面上的局部收斂本質。

### 誤差動力學的完全解耦
由於梯度為 $\nabla L(\theta) = H(\theta - \theta^*)$，定義第 $k$ 步的參數誤差為 $e_k = \theta_k - \theta^*$。梯度下降迭代給出：
$e_{k+1} = e_k - \eta H e_k = (I - \eta H)e_k$。

將誤差向量在 Hessian 的正交特徵向量基底展開：$e_k = \sum_i c_{k,i}q_i$。由於 $H q_i = \lambda_i q_i$，算子作用在特徵向量上化為純量相乘：
$$
c_{k+1,i} = (1-\eta\lambda_i)\,c_{k,i}\qquad\Longrightarrow\qquad c_{k,i} = (1-\eta\lambda_i)^k\,c_{0,i}.
$$

**驚人的正交解耦結論：** 高維二次函數上的多變數優化，被完美拆解成了 $n$ 個互不干擾、各自獨立沿著特徵軸演化的一維拋物線！

### 幾何困境：病態條件與狹長峽谷
每個特徵方向都要求 $|1 - \eta\lambda_i| < 1$ 才能保證收斂。由此導出優化的兩大衝突法則：
1. **穩定性鐵律：** 為了防止在最陡峭方向上爆炸，全局學習率必須受制於**最大特徵值**：$\boxed{\eta < \frac{2}{\lambda_{\max}}}$。
2. **進展瓶頸：** 最平緩特徵方向的收斂速度取決於 $(1 - \eta\lambda_{\min})$。當我們為了最陡方向的安全而被迫選取極小的 $\eta$ 時，平緩方向的誤差衰減比率極其接近 1，幾乎停滯不前！

定義**條件數（Condition Number）**為最大與最小曲率之比：
$$
\kappa = \frac{\lambda_{\max}}{\lambda_{\min}} \ge 1.
$$

**狹長峽谷（Narrow Ravine）的幾何圖景：**
當 $\kappa \gg 1$ 時（例如 $\kappa = 10^4$），等高線是極度拉長的狹長橢圓。
- 負梯度方向垂直於等高線，因此在峽谷的絕大部分區域，**負梯度幾乎直指對面陡峭的崖壁，而非指向遠處沿著峽谷底部的真實極小值點**！
- 傳統梯度下降法被迫在兩側峭壁之間劇烈「鋸齒狀振盪」（Zig-zagging），同時以極其緩慢的速度沿著平坦的峽谷底部蠕動。

### 最優步長與收斂率
最差情況的收斂速率由最慢和最震盪的方向共同決定：$r(\eta) = \max\bigl(|1-\eta\lambda_{\min}|,\ |1-\eta\lambda_{\max}|\bigr)$。
當兩端誤差衰減幅度平衡時取得最優步長：$1 - \eta\lambda_{\min} = \eta\lambda_{\max} - 1$，由此解得：
$$
\eta^* = \frac{2}{\lambda_{\min}+\lambda_{\max}},\qquad r^* = \frac{\lambda_{\max}-\lambda_{\min}}{\lambda_{\max}+\lambda_{\min}} = \frac{\kappa-1}{\kappa+1},\qquad \kappa = \frac{\lambda_{\max}}{\lambda_{\min}} .
$$

若要將誤差縮小至原始的 $\varepsilon$ 倍，所需迭代步數滿足 $(r^*)^k \le \varepsilon$，即 $k = \ln(1/\varepsilon)/(-\ln r^*)$。當條件數 $\kappa$ 較大時，泰勒近似 $-\ln\frac{\kappa-1}{\kappa+1} \approx \frac{2}{\kappa}$，得到：
$$
k \approx \frac\kappa2\,\ln\frac1\varepsilon .
$$

**核心結論：梯度下降法的迭代步數隨條件數 $\kappa$ 線性增長！**
若損失曲面中不同特徵方向的曲率相差 $10^4$ 倍，無論工程師如何精細調節學習率，基礎梯度下降法都需要成千上萬倍的漫長步數。

這正是為什麼後續的發明如此不可或缺：
- **動量法（Polyak Momentum，§3.5.7）：** 利用物理慣性抵消橫跨崖壁的高頻振盪，將迭代步數依賴從 $O(\kappa)$ 大幅降低到 $O(\sqrt{\kappa})$；
- **自適應學習率（Adam/AdamW，§3.3.8）：** 針對不同坐標軸的歷史梯度平方進行對角預條件化，使各軸步長動態適應局部曲率；
- **初始化與殘差連接（§3.7）：** 從網絡架構源頭壓制特徵值發散，保持條件數 $\kappa \approx 1$。

## 3.5.5 強凸性與線性收斂

**定義。** 若對所有 $x, y$ 有 $L(y)\ge L(x) + \nabla L(x)^{\top}(y-x) + \frac\mu2\|y-x\|^2$，則稱 $L$ 為 **$\mu$-強凸（strongly convex）**（$\mu > 0$）——一個二次下界，是引理 3.1 的鏡像。對二次可微的 $L$，這表示 Hessian 的所有特徵值至少為 $\mu$。

**Polyak–Łojasiewicz 不等式。** 把定義兩邊對 $y$ 取最小值。左邊的最小值為 $L^*$；右邊是 $y$ 的二次函數，在 $y = x - \nabla L(x)/\mu$ 取最小值 $L(x) - \frac{1}{2\mu}\|\nabla L(x)\|^2$。因此

$$
\tfrac12\|\nabla L(x)\|^2 \;\ge\; \mu\bigl(L(x) - L^*\bigr):
$$

*損失值離最佳值越遠，梯度就越大。*

**定理 3.2（線性收斂）。** 若 $L$ 為 $\beta$-平滑且 $\mu$-強凸，以 $\eta = 1/\beta$ 做梯度下降滿足

$$
L(\theta_{k+1}) - L^* \;\le\; \Bigl(1 - \frac{\mu}{\beta}\Bigr)\bigl(L(\theta_k) - L^*\bigr).
$$

*證明。* 引理 3.1 的系理給出 $L(\theta_{k+1}) \le L(\theta_k) - \frac1{2\beta}\|\nabla L(\theta_k)\|^2$，PL 不等式則給出 $\|\nabla L\|^2 \ge 2\mu(L(\theta_k) - L^*)$。兩邊減去 $L^*$。$\blacksquare$

比值 $\beta/\mu$ 又是條件數；這個定理是 §3.5.4 在一般函數上的對應。證明只用到 PL 不等式，所以同樣的線性速率對任何滿足它的平滑函數都成立——包括某些非凸函數。

## 3.5.6 隨機梯度：進展加上雜訊

把 $\nabla L$ 換成不偏估計 $g$，滿足 $\mathbb E[g] = \nabla L(\theta)$ 且 $\mathbb E\|g - \nabla L(\theta)\|^2 \le \sigma^2$（對小批次，$\sigma^2 = \mathrm{tr}\,\Sigma/B$，§3.3.7）。

**引理 3.2（一步的期望進展）。** 對 $\beta$-平滑的 $L$ 與一步 $\theta' = \theta - \eta g$，

$$
\mathbb E\bigl[L(\theta')\bigr] \;\le\; L(\theta) - \eta\Bigl(1 - \frac{\beta\eta}{2}\Bigr)\|\nabla L(\theta)\|^2 + \frac{\beta\eta^2}{2}\,\sigma^2 .
$$

*證明。* 以 $y - x = -\eta g$ 套用引理 3.1，得 $L(\theta') \le L(\theta) - \eta\nabla L^{\top}g + \frac{\beta\eta^2}{2}\|g\|^2$。取期望值：由不偏性 $\mathbb E[\nabla L^{\top}g] = \|\nabla L\|^2$，且 $\mathbb E\|g\|^2 = \|\nabla L\|^2 + \mathbb E\|g - \nabla L\|^2 \le \|\nabla L\|^2 + \sigma^2$（交叉項的平均為零）。整理各項即得。$\blacksquare$

第一項是 §3.5.2 的確定性進展；第二項是**雜訊的代價**，與 $\eta^2\sigma^2$ 成正比。對 $K$ 步加總、取 $\eta\le1/\beta$（使 $1 - \beta\eta/2 \ge \tfrac12$），並像定理 3.1 一樣伸縮相消：

$$
\frac1K\sum_{k<K}\mathbb E\|\nabla L(\theta_k)\|^2 \;\le\; \frac{2\bigl(L(\theta_0) - L^*\bigr)}{\eta K} + \beta\eta\,\sigma^2 .
$$

解讀這兩項：

- **步長固定時，SGD 有雜訊底限（noise floor）** $\beta\eta\sigma^2$：它不會收斂到駐點，而是在一個區域內徘徊，區域大小隨步長與梯度雜訊增長。
- **縮小步長能降低底限，但會減慢進展。** 滿足 $\eta_k\to0$、$\sum_k\eta_k = \infty$、$\sum_k\eta_k^2<\infty$ 的排程（例如 $\eta_k\propto1/k$）能讓兩項都趨於零——這是 [Robbins & Monro（1951）](/zh-tw/references/#robbins-1951) 的經典條件。
- **批次大小與步長互相取捨。** 因為 $\sigma^2\propto1/B$，底限與 $\eta/B$ 成正比。固定 $\eta/B$ 就固定了底限，這是「學習率隨批次大小縮放」的理論動機（§3.8.4）。

## 3.5.7 ★ 二次函數上的動量

把 §3.3.8 的重球更新套用到 §3.5.4 的二次函數。在每個特徵方向上，係數滿足一個二項遞迴：

$$
c_{k+1} = (1 + \beta - \eta\lambda)\,c_k - \beta\,c_{k-1},
$$

其解是 $r^k$ 的組合，$r$ 為 $r^2 - (1+\beta-\eta\lambda)r + \beta = 0$ 的根。若兩根為複數，它們的乘積為 $\beta$ 且模長相等，所以 $|r| = \sqrt\beta$——**與 $\lambda$ 無關**。兩根為複數或重根的條件是 $(1+\beta-\eta\lambda)^2\le4\beta$，即

$$
(1-\sqrt\beta)^2 \;\le\; \eta\lambda \;\le\; (1+\sqrt\beta)^2 .
$$

選擇 $\eta$ 與 $\beta$，讓整個區間 $[\lambda_{\min}, \lambda_{\max}]$ 恰好落入其中：$\eta\lambda_{\min} = (1-\sqrt\beta)^2$、$\eta\lambda_{\max} = (1+\sqrt\beta)^2$。兩式相除得 $\kappa = \bigl(\tfrac{1+\sqrt\beta}{1-\sqrt\beta}\bigr)^2$，所以

$$
\sqrt\beta = \frac{\sqrt\kappa-1}{\sqrt\kappa+1},\qquad \eta = \frac{4}{(\sqrt{\lambda_{\max}}+\sqrt{\lambda_{\min}})^2},\qquad \text{速率 } \frac{\sqrt\kappa-1}{\sqrt\kappa+1}.
$$

迭代次數現在隨 $\sqrt\kappa$ 而非 $\kappa$ 增長——當 $\kappa = 10^4$，少了一百倍。詳解例題 3.5 對 $\kappa = 10$ 檢查了這些數字。Nesterov 加速梯度法對一般平滑強凸函數也達到同樣的 $\sqrt\kappa$ 依賴，而對這類函數的一階方法而言，這個依賴是最佳的（[Nesterov 2018](/zh-tw/references/#nesterov-2018)）。

## 3.5.8 ★ 預條件化

**預條件化（preconditioning）** 的一步 $\theta \leftarrow \theta - \eta P^{-1}\nabla L$，就是在由 $P$ 定義的幾何下做最陡下降（§3.2.5）。在二次函數上，誤差迭代變成 $e_{k+1} = (I - \eta P^{-1}H)e_k$，所以重要的是 $P^{-1}H$ 的條件數：

- $P = H$（牛頓法）使 $P^{-1}H = I$，一步收斂——代價是建立並求逆一個 $n\times n$ 矩陣，$n$ 很大時不可能。
- 等於 $H$ 對角部分的對角 $P$，能去除與座標軸對齊的病態條件，但無法去除旋轉過的病態條件。

Adam 的分母 $\sqrt{\hat v}$ 是一個由梯度大小（而非曲率）建立的對角 $P$。兩者在某些情況下相關，在其他情況下無關，所以「Adam 是對角二階方法」只是一種啟發式詮釋，不是定理。以可負擔的成本建立有結構、非對角預條件器的最佳化器，在 §3.8.6 討論。

## 3.5.9 ★ 反向模式成本的正式敘述

設一個程式以一連串基本運算計算 $L$，每個運算的局部偏導數都能以至多該運算本身常數倍的成本計算。則梯度可以用至多「計算 $L$ 之成本的小常數倍」（與 $n$ 無關）的運算數算出，另加與儲存的中間值數量成正比的記憶體。[Griewank & Walther（2008）](/zh-tw/references/#griewank-2008)在數種成本模型下給出精確敘述與常數。記憶體項可以用重新計算來換：只儲存選定的檢查點，在反向傳遞時重算各段——記憶體可降到運算數的對數級，計算量增加對數倍（他們的「revolve」排程）；對 $\ell$ 層網路，也可以約多一次前向傳遞換得 $O(\sqrt{\ell})$ 的記憶體（[Chen et al. 2016](/zh-tw/references/#chen-2016)）。
