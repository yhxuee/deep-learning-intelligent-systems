---
title: "3.5 一般理論：梯度下降法為什麼收斂、收斂多快"
description: 泰勒展開與 Hessian 矩陣；平滑性與下降引理；收斂到駐點；在二次函數上的精確分析與條件數；強凸性；隨機梯度與雜訊底限；動量與預條件化。
sidebar: { label: "3.5 一般理論", order: 5 }
---

§3.2 分析了一條拋物線。本節證明關於梯度下降法一般能說什麼——同樣重要的是，不能說什麼。所有證明只用到泰勒定理、柯西–施瓦茨不等式與線性代數。這些內容的標準參考書為 [Nocedal & Wright 2006](/zh-tw/references/#nocedal-2006)、[Boyd & Vandenberghe 2004](/zh-tw/references/#boyd-2004)、[Nesterov 2018](/zh-tw/references/#nesterov-2018)，隨機情況則見 [Bottou、Curtis & Nocedal 2018](/zh-tw/references/#bottou-2018)。

## 3.5.1 二階的圖像：Hessian 矩陣

對二次可微的 $L$，泰勒定理在 §3.2.4 的局部線性模型上加入二次項：

$$
L(\theta + \delta) = L(\theta) + \nabla L(\theta)^{\top}\delta + \tfrac12\,\delta^{\top}H(\theta)\,\delta + o(\|\delta\|^2),
\qquad H_{ij} = \frac{\partial^2L}{\partial\theta_i\,\partial\theta_j}.
$$

**Hessian 矩陣（黑塞矩陣）** $H$ 是對稱的，由譜定理，它有實數特徵值 $\lambda_1,\dots,\lambda_n$ 與一組正交單位特徵向量基底 $q_1,\dots,q_n$：$H = Q\Lambda Q^{\top}$。沿方向 $q_i$，損失像拋物線 $\tfrac12\lambda_i t^2$ 一樣彎曲。正特徵值是碗狀方向，負特徵值是山丘狀方向（兩者同時出現就是鞍點），而特徵值的*分散程度*——不只是它們的大小——將決定梯度下降法的速度。

## 3.5.2 平滑性與下降引理

**定義。** 若 $L$ 的梯度是 $\beta$-Lipschitz，即對所有 $x, y$ 有 $\|\nabla L(x) - \nabla L(y)\| \le \beta\|x - y\|$，則稱 $L$ 為 **$\beta$-平滑（$\beta$-smooth）**。對二次可微的 $L$，這等價於 Hessian 的所有特徵值處處落在 $[-\beta, \beta]$。平滑性限制了斜率能變化多快，也就限制了線性模型能錯多少。

**引理 3.1（下降引理）。** 若 $L$ 為 $\beta$-平滑，則對所有 $x, y$

$$
L(y) \le L(x) + \nabla L(x)^{\top}(y - x) + \frac\beta2\|y - x\|^2 .
$$

*證明。* 沿線段 $x + s(y-x)$，$s\in[0,1]$，由微積分基本定理：

$$
L(y) - L(x) - \nabla L(x)^{\top}(y-x) = \int_0^1\bigl(\nabla L(x + s(y-x)) - \nabla L(x)\bigr)^{\top}(y-x)\,ds .
$$

由柯西–施瓦茨不等式與平滑性，被積函數至多為 $\beta s\|y-x\|\cdot\|y-x\|$。把 $\beta s$ 從 0 積到 1 得 $\beta/2$。$\blacksquare$

**系理（梯度下降法的一步）。** 令 $y = x - \eta\nabla L(x)$：

$$
L\bigl(x - \eta\nabla L(x)\bigr) \le L(x) - \eta\Bigl(1 - \frac{\beta\eta}{2}\Bigr)\|\nabla L(x)\|^2 .
$$

每一個 $0 < \eta < 2/\beta$ 的步都會降低損失（除非梯度為零）。保證的下降量在 $\eta = 1/\beta$ 時最大，等於 $\|\nabla L\|^2/(2\beta)$。這是拋物線結果 $\eta < 2/a$ 的多變數、最壞情況版本。

## 3.5.3 沒有凸性時能保證什麼

神經網路的損失不是凸的。那還能證明什麼？

**定理 3.1（收斂到駐點）。** 設 $L$ 為 $\beta$-平滑且下有界於 $L^*$。以 $\eta = 1/\beta$ 做梯度下降，滿足

$$
\min_{0\le k<K}\|\nabla L(\theta_k)\|^2 \;\le\; \frac{2\beta\,\bigl(L(\theta_0) - L^*\bigr)}{K}.
$$

*證明。* 由系理，$\|\nabla L(\theta_k)\|^2 \le 2\beta\bigl(L(\theta_k) - L(\theta_{k+1})\bigr)$。對 $k = 0,\dots,K-1$ 加總，右邊伸縮相消為 $2\beta\bigl(L(\theta_0) - L(\theta_K)\bigr) \le 2\beta\bigl(L(\theta_0) - L^*\bigr)$。$K$ 個數的最小值至多是它們的平均。$\blacksquare$

**它說了什麼。** $K$ 步之後，某個迭代值的梯度範數至多為 $O(1/\sqrt K)$，常數與參數數量無關。**它沒說什麼。** 梯度小代表近似的*駐點（stationary point）*——可能是最小值、鞍點，也可能是平坦的高原。這裡完全沒說最小值是全域的，也沒說它會泛化。對非凸損失，這些都是真正的開放問題，以實證與特殊情況的理論來研究（§3.7.2）。

## 3.5.4 精確分析二次函數：條件數

在 Hessian 正定的最小值 $\theta^*$ 附近，損失近似於二次函數。所以梯度下降法在

$$
L(\theta) = \tfrac12(\theta - \theta^*)^{\top}H(\theta - \theta^*),\qquad 0<\lambda_{\min}\le\lambda_i\le\lambda_{\max},
$$

上的行為，描述了它在任何平滑損失上的*局部*行為。在這裡一切都可以精確計算。

梯度為 $H(\theta - \theta^*)$，所以誤差 $e_k = \theta_k - \theta^*$ 滿足 $e_{k+1} = (I - \eta H)e_k$。在特徵基底下展開誤差 $e_k = \sum_ic_{k,i}q_i$。因為 $Hq_i = \lambda_iq_i$，

$$
c_{k+1,i} = (1-\eta\lambda_i)\,c_{k,i}\qquad\Longrightarrow\qquad c_{k,i} = (1-\eta\lambda_i)^k\,c_{0,i}.
$$

每個特徵方向都是 §3.2.3 拋物線的獨立複本。因此：

1. **穩定性：** 從任何起點收斂的充要條件是對所有 $i$ 有 $|1 - \eta\lambda_i| < 1$，即 $\boxed{\eta < 2/\lambda_{\max}}$。
2. **速率：** 誤差像 $r(\eta)^k$ 一樣縮小，其中 $r(\eta) = \max\bigl(|1-\eta\lambda_{\min}|,\ |1-\eta\lambda_{\max}|\bigr)$。
3. **最佳固定步長：** 當兩項相等時 $r(\eta)$ 最小，即 $1 - \eta\lambda_{\min} = \eta\lambda_{\max} - 1$：

$$
\eta^* = \frac{2}{\lambda_{\min}+\lambda_{\max}},\qquad r^* = \frac{\lambda_{\max}-\lambda_{\min}}{\lambda_{\max}+\lambda_{\min}} = \frac{\kappa-1}{\kappa+1},\qquad \kappa = \frac{\lambda_{\max}}{\lambda_{\min}} .
$$

**條件數（condition number）** $\kappa$ 單獨決定了可達到的最佳速率。要把誤差縮小 $\varepsilon$ 倍需要 $k = \ln(1/\varepsilon)/(-\ln r^*)$ 步；由於 $\kappa$ 很大時 $-\ln\frac{\kappa-1}{\kappa+1}\approx\frac2\kappa$，

$$
k \approx \frac\kappa2\,\ln\frac1\varepsilon .
$$

**迭代次數隨條件數線性增長。** 曲率範圍相差 $10^4$ 倍的損失，不論步長怎麼選，都比條件良好的損失多需要數千倍的步數。因此，讓 $\kappa$ 變小——透過初始化、正規化、預條件化或架構——是實務最佳化的主要目標之一（§3.8）。

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
