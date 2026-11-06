---
title: "3.5 一般理論：梯度下降法為什麼收斂、收斂多快"
description: 泰勒展開與 Hessian 矩陣；平滑性與下降引理；近似駐點；二次函數與條件數的精確分析；強凸性；隨機梯度與雜訊；動量、預條件化與反向模式成本。
sidebar: { label: "3.5 一般理論", order: 5 }
---

§3.2 追蹤了梯度下降法在一條拋物線上的行為。現在問題變得更難：其中哪些現象能延伸到一般的多變數損失？

本節的論證分三層。第一，平滑性把一個局部梯度步變成不等式。第二，把這個不等式沿時間加總，得到收斂界限。第三，加入正定二次函數或強凸性等額外結構，把較弱的保證升級為幾何速率。閱讀時要留意量詞：「前幾步中最好的一個迭代點」、「對隨機執行取平均」與「每個迭代點」是三種不同主張。

證明會用到泰勒定理、柯西–施瓦茨不等式、期望值與線性代數。標準參考書包括 [Nocedal & Wright（2006）](/zh-tw/references/#nocedal-2006)、[Boyd & Vandenberghe（2004）](/zh-tw/references/#boyd-2004)、[Nesterov（2018）](/zh-tw/references/#nesterov-2018)；隨機最佳化則見 [Bottou、Curtis & Nocedal（2018）](/zh-tw/references/#bottou-2018)。

## 3.5.1 二階圖像：Hessian 矩陣

梯度記錄局部斜率。再對梯度微分一次，得到 **Hessian 矩陣（黑塞矩陣）**：

$$
H_{ij}(\theta)=\frac{\partial^2L}{\partial\theta_i\,\partial\theta_j}.
$$

對二次可微的損失，泰勒定理把 §3.2.4 的局部線性模型延伸為

$$
L(\theta+\delta)
=L(\theta)+\nabla L(\theta)^\top\delta
+\tfrac12\delta^\top H(\theta)\delta
+o(\|\delta\|^2).
$$

依序讀這四項：目前的函數值、一階變化、二次曲率，以及當 $\delta\to0$ 時相對於 $\|\delta\|^2$ 可忽略的餘項。這是**局部**描述；它不表示同一個 Hessian 能代表離 $\theta$ 很遠的損失地形。

當混合偏導數連續時，$H$ 是對稱矩陣。譜定理因此給出實數特徵值 $\lambda_i$ 與正交單位特徵基底 $q_i$：

$$
H=Q\Lambda Q^\top,\qquad
\delta=\sum_i a_iq_i,\qquad
\delta^\top H\delta=\sum_i\lambda_i a_i^2.
$$

最後一個等式最有用。在特徵座標中，二次項沒有交叉項；沿 $q_i$ 方向的彎曲就像 $\tfrac12\lambda_i t^2$。正的 $\lambda_i$ 代表向上彎，負值代表向下彎。如果梯度也等於零，同時出現正、負特徵值就能判定為鞍點；若梯度不為零，Hessian 不定只描述曲率，不能單獨分類該點。

稍後會用到兩種不同的頻譜摘要。最大絕對曲率限制安全步長；對正定二次函數，最大與最小曲率的比值則控制調好步長後的最佳最壞情況速率。

## 3.5.2 平滑性與下降引理

**定義。** 若在一個凸區域內，$L$ 的梯度是 $\beta$-Lipschitz：

$$
\|\nabla L(x)-\nabla L(y)\|\le\beta\|x-y\|,
$$

就稱 $L$ 在該區域為 **$\beta$-平滑（$\beta$-smooth）**。平滑不代表凸；它只表示斜率不能任意快速改變。若 $L$ 二次連續可微，這個條件等價於整個區域內都有 $\|H(x)\|_2\le\beta$，也等價於 Hessian 的每個特徵值都落在 $[-\beta,\beta]$。

**引理 3.1（下降引理）。** 若 $L$ 在包含 $x$ 到 $y$ 線段的凸區域內為 $\beta$-平滑，則

$$
L(y)\le L(x)+\nabla L(x)^\top(y-x)+\frac{\beta}{2}\|y-x\|^2.
$$

右邊是線性預測，再加上一份二次餘裕，容許梯度在移動途中發生變化。

*證明。* 令 $d=y-x$，沿直線 $\gamma(s)=x+sd$ 移動，其中 $0\le s\le1$。微積分基本定理給出

$$
L(y)-L(x)=\int_0^1\nabla L(x+sd)^\top d\,ds.
$$

減去固定的線性預測 $\nabla L(x)^\top d$：

$$
L(y)-L(x)-\nabla L(x)^\top d
=\int_0^1\bigl(\nabla L(x+sd)-\nabla L(x)\bigr)^\top d\,ds.
$$

柯西–施瓦茨不等式把被積函數限制在兩個範數的乘積內；平滑性再把第一個範數限制為 $\beta s\|d\|$，所以

$$
\bigl(\nabla L(x+sd)-\nabla L(x)\bigr)^\top d
\le\beta s\|d\|^2.
$$

把 $s$ 從 0 積到 1，就得到因子 $1/2$。$\blacksquare$

**系理（一次精確梯度更新）。** 代入 $y=x-\eta\nabla L(x)$：

$$
L\bigl(x-\eta\nabla L(x)\bigr)
\le L(x)-\eta\left(1-\frac{\beta\eta}{2}\right)\|\nabla L(x)\|^2.
$$

因此，只要 $\nabla L(x)\ne0$，每個滿足 $0<\eta<2/\beta$ 的步長都保證降低損失。這個界限所保證的下降量在 $\eta=1/\beta$ 時最大，為 $\|\nabla L(x)\|^2/(2\beta)$。這是拋物線條件 $\eta<2/a$ 的多變數、最壞情況版本。

:::note[主張檢核]
- **主張。** 以 $0<\eta<2/\beta$ 做一次精確梯度更新會降低損失。
- **假設。** 更新經過的整條線段都位於 $L$ 為 $\beta$-平滑的區域；梯度精確；所用的 $\beta$ 確實是有效上界。
- **檢驗。** 對 $\lambda_{\max}$ 已知的二次函數，比較略低於與略高於 $2/\lambda_{\max}$ 的步長（實驗 3.1）。
- **限制。** ReLU 網路並非處處可微（§3.7.1），小批次會把精確梯度換成隨機估計（§3.5.6），曲率也會在訓練中改變。這個界限是指引，不是自動產生的學習率設定（§3.7.8）。
:::

## 3.5.3 沒有凸性時能保證什麼

神經網路的損失通常是非凸的。單靠平滑性無法保證全域最小值，卻能保證梯度下降法會找到至少一個梯度很小的迭代點。

**定理 3.1（近似駐點界限）。** 設 $L$ 為 $\beta$-平滑，且下有界於某個數 $L^*$。取 $\eta=1/\beta$，則

$$
\min_{0\le k<K}\|\nabla L(\theta_k)\|^2
\le\frac{2\beta\bigl(L(\theta_0)-L^*\bigr)}{K}.
$$

*證明。* 在系理中取 $\eta=1/\beta$，得到

$$
\|\nabla L(\theta_k)\|^2
\le2\beta\bigl(L(\theta_k)-L(\theta_{k+1})\bigr).
$$

把不等式從 $k=0$ 加到 $K-1$。每個中間損失都以正號出現一次、負號出現一次：

$$
\sum_{k=0}^{K-1}\|\nabla L(\theta_k)\|^2
\le2\beta\bigl(L(\theta_0)-L(\theta_K)\bigr)
\le2\beta\bigl(L(\theta_0)-L^*\bigr).
$$

兩邊除以 $K$。$K$ 個非負數中的最小值不可能大於它們的平均。$\blacksquare$

前 $K$ 個迭代點中**最好的一個**，其梯度平方為 $O(1/K)$，所以梯度範數為 $O(1/\sqrt K)$。公式沒有明寫參數數量，但常數 $\beta$ 與 $L(\theta_0)-L^*$ 本身仍可能隨模型規模改變。

這個定理刻意比「迭代點會收斂」更弱。它只指出至少有一個近似駐點；沒有說最後一個迭代點就是它、整個序列有極限，也沒有說小梯度點必為最小值。駐點可能是局部最小值、鞍點或平坦高原，而上述任何敘述都不保證泛化。

## 3.5.4 精確分析二次函數：條件數

若要得到精確速率，可先專注於正定二次函數：

$$
L(\theta)
=\tfrac12(\theta-\theta^*)^\top H(\theta-\theta^*),
\qquad
0<\lambda_{\min}\le\lambda_i\le\lambda_{\max}.
$$

這個模型對二次函數完全精確；在 Hessian 正定的平滑最小值附近，它是第一個局部近似。如果 Hessian 沿路變化很大，就不能期待它描述一整段很長的軌跡。

令 $e_k=\theta_k-\theta^*$。由於 $\nabla L(\theta_k)=He_k$，

$$
e_{k+1}=(I-\eta H)e_k.
$$

在 Hessian 的特徵基底中展開 $e_k=\sum_i c_{k,i}q_i$。每個方向都獨立演化：

$$
c_{k+1,i}=(1-\eta\lambda_i)c_{k,i},
\qquad
c_{k,i}=(1-\eta\lambda_i)^kc_{0,i}.
$$

矩陣問題現在變成每個特徵值各自對應一份一維拋物線。可直接得到三項結果。

1. **從每個起點都穩定。** 所有分量都收斂到零的充要條件為

   $$
   0<\eta<\frac{2}{\lambda_{\max}}.
   $$

2. **最壞情況的參數收縮。** 在歐幾里得範數下，

   $$
   \|e_k\|\le r(\eta)^k\|e_0\|,
   \qquad
   r(\eta)=\max_i|1-\eta\lambda_i|.
   $$

   若只知道區間 $[\lambda_{\min},\lambda_{\max}]$，最壞情況會出現在端點。

3. **在這個極小極大意義下的最佳固定步長。** 令兩個端點的絕對值平衡：

   $$
   1-\eta\lambda_{\min}
   =\eta\lambda_{\max}-1.
   $$

   得到

   $$
   \eta^*=\frac{2}{\lambda_{\min}+\lambda_{\max}},
   \qquad
   r^*=\frac{\lambda_{\max}-\lambda_{\min}}
   {\lambda_{\max}+\lambda_{\min}}
   =\frac{\kappa-1}{\kappa+1},
   \qquad
   \kappa=\frac{\lambda_{\max}}{\lambda_{\min}}.
   $$

**條件數（condition number）** $\kappa$ 控制最佳的最壞情況固定步長速率。把 $H$ 整體乘上一個常數，會改變合適的學習率，卻不改變 $\kappa$ 或調好後的收縮因子。要把最壞情況參數誤差縮小 $\varepsilon$ 倍，需要

$$
k=\frac{\ln(1/\varepsilon)}{-\ln r^*}.
$$

當 $\kappa$ 很大時，

$$
-\ln\frac{\kappa-1}{\kappa+1}\approx\frac{2}{\kappa},
\qquad
k\approx\frac{\kappa}{2}\ln\frac1\varepsilon.
$$

因此，最壞情況迭代次數隨 $\kappa$ 線性增長。這**不**表示每個初始化都需要這麼久；如果 $e_0$ 在緩慢特徵方向上沒有分量，實際速率可能更快。它真正表示的是：對跨越該頻譜區間的所有起始誤差，沒有任何固定步長能改善這項保證。初始化、正規化、預條件化與架構都會以不同方式嘗試改善有效條件數。

## 3.5.5 強凸性與線性收斂

二次函數很特殊，但它的幾何速率可以延伸到更廣的函數類別。

**定義。** 若 $\mu>0$，且定義域中的每個 $x,y$ 都滿足

$$
L(y)\ge L(x)+\nabla L(x)^\top(y-x)+\frac{\mu}{2}\|y-x\|^2,
$$

就稱 $L$ 為 **$\mu$-強凸（strongly convex）**。這是一個二次下界，是下降引理中二次上界的鏡像。對二次可微的 $L$，它表示處處都有 $H(x)\succeq\mu I$。

假設最小值點存在，並把最小函數值寫成 $L^*$。強凸性可推出 **Polyak–Łojasiewicz（PL）不等式**。把二次下界對 $y$ 最小化；它在 $y=x-\nabla L(x)/\mu$ 取得最小值 $L(x)-\|\nabla L(x)\|^2/(2\mu)$。真正的最小值不可能低於這個全域下界，因此

$$
\frac12\|\nabla L(x)\|^2
\ge\mu\bigl(L(x)-L^*\bigr).
$$

用白話說：有了這項結構，函數值遠高於最佳值就會迫使梯度很大。這正是前一個非凸定理缺少的連結。

**定理 3.2（函數值的線性收斂）。** 若 $L$ 為 $\beta$-平滑且 $\mu$-強凸，以 $\eta=1/\beta$ 做梯度下降，則

$$
L(\theta_{k+1})-L^*
\le\left(1-\frac{\mu}{\beta}\right)
\bigl(L(\theta_k)-L^*\bigr).
$$

*證明。* 下降引理給出

$$
L(\theta_{k+1})
\le L(\theta_k)-\frac{1}{2\beta}\|\nabla L(\theta_k)\|^2.
$$

以 PL 不等式把梯度平方替換成至少 $2\mu(L(\theta_k)-L^*)$，再從兩邊減去 $L^*$。$\blacksquare$

反覆套用這個界限，得到

$$
L(\theta_k)-L^*
\le\left(1-\frac{\mu}{\beta}\right)^k
\bigl(L(\theta_0)-L^*\bigr).
$$

這裡的「線性收斂」是指幾何衰減，不是圖上畫出一條直線。比值 $\beta/\mu$ 又是條件數。注意，證明直接用到的是平滑性加 PL 不等式，而不是凸性；因此，某些滿足 PL 的平滑非凸函數也有相同的函數值速率，強凸性只是取得 PL 的一個充分條件。

## 3.5.6 隨機梯度：進展加上雜訊

現在把精確梯度換成隨機估計 $g_k$。在給定目前迭代點 $\theta_k$ 的條件下，假設

$$
\mathbb E_k[g_k]=\nabla L(\theta_k),
\qquad
\mathbb E_k\|g_k-\nabla L(\theta_k)\|^2\le\sigma^2.
$$

下標提醒我們：取期望時固定 $\theta_k$，只對新一次取樣的隨機性平均。若批次大小為 $B$、樣本獨立且放回取樣，單筆樣本梯度的共變異數為 $\Sigma(\theta_k)$，則變異數項為 $\operatorname{tr}\Sigma(\theta_k)/B$。不放回取樣還有有限母體修正；樣本彼此相關時，也未必遵守簡單的 $1/B$ 定律。

**引理 3.2（一步的期望進展）。** 對 $\beta$-平滑損失與更新 $\theta_{k+1}=\theta_k-\eta g_k$，

$$
\mathbb E_k[L(\theta_{k+1})]
\le L(\theta_k)
-\eta\left(1-\frac{\beta\eta}{2}\right)
\|\nabla L(\theta_k)\|^2
+\frac{\beta\eta^2}{2}\sigma^2.
$$

*證明。* 在下降引理中代入 $y-x=-\eta g_k$：

$$
L(\theta_{k+1})
\le L(\theta_k)-\eta\nabla L(\theta_k)^\top g_k
+\frac{\beta\eta^2}{2}\|g_k\|^2.
$$

不偏性給出 $\mathbb E_k[\nabla L^\top g_k]=\|\nabla L\|^2$。展開 $g_k=\nabla L+(g_k-\nabla L)$，可得

$$
\mathbb E_k\|g_k\|^2
=\|\nabla L\|^2
+\mathbb E_k\|g_k-\nabla L\|^2
\le\|\nabla L\|^2+\sigma^2,
$$

因為交叉項的條件期望為零。代回並整理各項即得。$\blacksquare$

若使用固定的 $\eta\le1/\beta$，取完整期望、對 $K$ 步加總、讓損失像望遠鏡一樣逐項相消，再套用下界 $L^*$：

$$
\frac1K\sum_{k=0}^{K-1}
\mathbb E\|\nabla L(\theta_k)\|^2
\le
\frac{2\bigl(L(\theta_0)-L^*\bigr)}{\eta K}
+\beta\eta\sigma^2.
$$

第一項會隨步數增加而縮小，第二項不會。因此，這個上界在平均梯度平方中留下 $O(\eta\sigma^2)$ 的漸近**雜訊項**。當非零雜訊持續存在時，固定步長 SGD 通常會在解附近波動，而非精確停在解上。

但不要把這句話反過來當成普遍不可能定理。不等式是上界，沒有證明每次執行一定碰到這個底限；若梯度雜訊在最佳值附近消失，固定步長方法仍可能精確收斂。

對遞減步長 $\eta_k\le1/\beta$，同一個證明給出加權界限

$$
\frac{\sum_{k<K}\eta_k\,
\mathbb E\|\nabla L(\theta_k)\|^2}
{\sum_{k<K}\eta_k}
\le
\frac{
2\bigl(L(\theta_0)-L^*\bigr)
+\beta\sigma^2\sum_{k<K}\eta_k^2}
{\sum_{k<K}\eta_k}.
$$

經典 Robbins–Monro 條件為

$$
\eta_k\to0,\qquad
\sum_k\eta_k=\infty,\qquad
\sum_k\eta_k^2<\infty.
$$

在上述一致變異數界限下，這些條件會讓右邊趨近零；$\eta_k\propto1/k$ 是標準例子。若要對迭代點本身提出更強的收斂主張，還需要額外假設。

最後，若獨立批次近似 $\sigma^2\propto1/B$ 成立，固定步長的雜訊項就隨 $\eta/B$ 縮放。這是讓 $\eta$ 隨 $B$ 增加的理論動機，卻不保證線性縮放永遠穩定：曲率仍會限制 $\eta$，而且批次夠大後，變異數模型不再是完整故事（§3.8.4）。

## 3.5.7 ★ 二次函數上的動量

把 §3.3.8 的重球更新套用到 §3.5.4 的正定二次函數。沿曲率為 $\lambda$ 的特徵方向，誤差係數滿足

$$
c_{k+1}=(1+\beta-\eta\lambda)c_k-\beta c_{k-1}.
$$

嘗試解 $c_k=r^k$，得到特徵方程式

$$
r^2-(1+\beta-\eta\lambda)r+\beta=0.
$$

當兩根互為共軛複數時，其乘積為 $\beta$，所以每個根的模長都是 $\sqrt\beta$。在重根邊界，頻譜半徑相同，但重根可能帶來 $k r^k$ 之類的多項式前因子。兩根為複數或重根的條件是

$$
(1+\beta-\eta\lambda)^2\le4\beta,
$$

等價於

$$
(1-\sqrt\beta)^2
\le\eta\lambda
\le(1+\sqrt\beta)^2.
$$

選擇 $\eta$ 與 $\beta$，讓整個曲率區間恰好塞進兩個端點之間：

$$
\eta\lambda_{\min}=(1-\sqrt\beta)^2,
\qquad
\eta\lambda_{\max}=(1+\sqrt\beta)^2.
$$

兩式相除並求解，得到

$$
\sqrt\beta
=\frac{\sqrt\kappa-1}{\sqrt\kappa+1},
\qquad
\eta
=\frac{4}
{(\sqrt{\lambda_{\max}}+\sqrt{\lambda_{\min}})^2}.
$$

因此，最壞情況的漸近根因子為

$$
\rho_{\mathrm{HB}}
=\frac{\sqrt\kappa-1}{\sqrt\kappa+1}.
$$

它的迭代數依賴為 $O(\sqrt\kappa\log(1/\varepsilon))$，優於梯度下降法的 $O(\kappa\log(1/\varepsilon))$。當 $\kappa=10^4$，這在增長階上約改善一百倍；它並不承諾每個有限步執行都恰好早一百倍結束。初期暫態、重根前因子與參數失調都會改變實測步數；詳解例題 3.5 在 $\kappa=10$ 時展示了這一點。

這段重球推導只針對二次函數；把同一組固定參數搬到一般平滑強凸目標，不會自動繼承全域加速保證。Nesterov 加速梯度法採用不同更新，能對這個更廣的函數類別達到相同的 $\sqrt\kappa$ 依賴。在標準一階預言機模型下，這個依賴對該函數類別是最佳的（[Nesterov 2018](/zh-tw/references/#nesterov-2018)）。

## 3.5.8 ★ 預條件化

令 $P$ 為對稱正定矩陣。**預條件化（preconditioning）** 更新為

$$
\theta\leftarrow\theta-\eta P^{-1}\nabla L(\theta).
$$

若以 $P$-範數 $\|d\|_P^2=d^\top Pd$ 衡量距離，這就是最陡下降（§3.2.5）。對二次函數，

$$
e_{k+1}=(I-\eta P^{-1}H)e_k.
$$

$P^{-1}H$ 外觀看來不一定對稱，但它與對稱矩陣 $P^{-1/2}HP^{-1/2}$ 相似，兩者有相同的正特徵值。因此，真正重要的是這些**預條件化後**特徵值的分散程度。

- 若 $P=H$ 且 $\eta=1$，則 $P^{-1}H=I$，精確二次函數一步就能解完。對大型模型而言，建立、儲存並求解一個稠密的 $n\times n$ Hessian 通常不可行。
- 若 $H$ 為對角矩陣且 $P=\operatorname{diag}(H)$，預條件化後的 Hessian 恰好是 $I$。如果困難方向相對座標軸旋轉過，一個對角 $P$ 通常無法消除它們。

Adam 的分母 $\sqrt{\hat v}+\epsilon$，作用像由過去梯度平方建立、隨時間變化的對角重新縮放，而不是 Hessian 本身。在某些問題中，梯度尺度與曲率相關；在另一些問題中則不相關。因此，「Adam 是對角二階方法」只是啟發式類比，不是定理。帶結構的非對角預條件器會在 §3.8.6 討論。

## 3.5.9 ★ 反向模式成本的正式敘述

「一次反向傳遞得到所有參數梯度」是在談**成本比值**，不是說絕對成本與模型大小無關。

假設一個程式用一連串基本運算計算純量 $L$，而每個運算的局部偏導數，都能以相對於該運算至多常數倍的成本算出。反向模式自動微分便能用「前向計算成本的小常數倍」算出完整梯度。這個倍數不會只因輸入座標更多而增長；但模型變大時，前向程式本身當然也可能更昂貴。[Griewank & Walther（2008）](/zh-tw/references/#griewank-2008) 在多種成本模型下給出精確敘述。

基本反向傳遞會保留局部導數規則所需的前向中間值，所以記憶體可能隨實際執行的計算長度增長。**檢查點法（checkpointing）**以重新計算換取記憶體：只存選定狀態，反向傳遞需要時重跑省略的前向區段，使用後再丟棄。不同排程位於不同的時間–空間取捨點。在相應模型下，*revolve* 之類的二項式排程能以對數級記憶體換取對數級重新計算因子；對 $\ell$ 層鏈，較簡單的分段方式可用 $O(\sqrt\ell)$ 個儲存活化值，代價約為多一次前向傳遞（[Chen et al. 2016](/zh-tw/references/#chen-2016)）。

所以，反向模式移除了逐座標微分額外付出的「參數數量」因子；它沒有讓計算或記憶體變成免費。
