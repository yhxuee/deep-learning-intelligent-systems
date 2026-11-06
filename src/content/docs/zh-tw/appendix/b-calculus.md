---
title: "B · 多變數與矩陣微積分"
description: 梯度、雅可比矩陣與多變數連鎖律；泰勒定理與黑塞矩陣；矩陣微分與反向傳播所用的恆等式；softmax、log-sum-exp 與交叉熵的導數；積分的變數變換；積分號下求導；有限差分及其誤差。
sidebar: { label: "B · 微積分", order: 2 }
---

單元 03 從頭建構反向模式微分；本附錄把它所依賴的微積分集中陳述，並補上使「穿過線性代數的反向傳播」變得機械化的矩陣恆等式。例題中的數字由 `labs/test_appendices.py` 檢查。

### 初學者圖像：導數是一項局部預測

導數首先不是搬動符號的規則，而是輸入有小變化時，輸出如何改變的最佳線性預測。若 $f(3)=9$、$f'(3)=6$，則

$$
f(3+h)\approx 9+6h
$$

對多個輸入，梯度預測純量變化 $\Delta f\approx\nabla f^{\top}\Delta x$；雅可比矩陣預測向量變化 $\Delta y\approx J\Delta x$。反向傳播反覆回答同一題：**已知輸出端的敏感度，每個輸入應分到多少敏感度？**

形狀是第一個錯誤偵測器。若 $x\in\mathbb R^n$、$y=f(x)\in\mathbb R^m$，則 $J_f$ 是 $m\times n$。在相乘雅可比矩陣以前，先把形狀寫在每個因子旁。這個習慣能在做任何代數以前抓到大量推導與程式錯誤。

| 小節 | 用於 |
|---|---|
| B.1–B.2 梯度、雅可比矩陣、連鎖律 | 03（反向傳播）、04、05（BPTT） |
| B.3 泰勒定理、黑塞矩陣 | 03（步長、下降引理）、06（精度） |
| B.4 矩陣微分 | 02（嵌入）、03、04（注意力的反向傳播） |
| B.5 Softmax 與交叉熵 | 01、03、04、07、09 |
| B.6 積分與變數變換 | 08（流）、09（分數函數梯度） |
| B.7 有限差分 | 03（梯度檢查）、每一個主幹專案測試 |

## B.1 導數與梯度

對 $f:\mathbb{R}^n\to\mathbb{R}$，**梯度（gradient）** $\nabla f(x)\in\mathbb{R}^n$ 是偏導數構成的向量。當 $f$ 在 $x$ 可微時，它是使線性模型在一階上精確的那個向量：
$$
f(x + h) = f(x) + \nabla f(x)^{\top}h + o(\lVert h\rVert).
$$
沿單位向量 $u$ 的**方向導數**是 $\nabla f(x)^{\top}u$；由柯西–施瓦茨（附錄 A，定理 A.1），它在 $u = \nabla f/\lVert\nabla f\rVert$ 時最大，這就是 $-\nabla f$ 為最陡下降方向的原因。

**例題 B.1。** $f(x, y) = x^2y + \sin y$ 的 $\nabla f = (2xy,\ x^2 + \cos y)$。在 $(1, 0)$：$\nabla f = (0, 2)$，所以一階近似下，$x$ 變動時 $f$ 不變，$y$ 變動時 $f$ 以速率 2 增加。

所有偏導數存在並不保證可微；偏導數連續才保證。本書用到的每個函數，除了孤立的折點——ReLU 在 0、最大池化中的最大值——之外都可微；在折點上，框架回傳**次微分**中的一個元素，這個選擇幾乎處處無害（單元 03，§3.7.1）。

## B.2 雅可比矩陣與連鎖律

對 $f:\mathbb{R}^n\to\mathbb{R}^m$，**雅可比矩陣（Jacobian）** $J_f(x)\in\mathbb{R}^{m\times n}$ 的元素為 $\partial f_i/\partial x_j$，且 $f(x + h) = f(x) + J_f(x)h + o(\lVert h\rVert)$。

**定理 B.1（連鎖律）。** 若 $g$ 在 $x$ 可微、$f$ 在 $g(x)$ 可微，則 $J_{f\circ g}(x) = J_f(g(x))\,J_g(x)$。

*證明概要。* $f(g(x + h)) = f(g(x) + J_gh + o(h)) = f(g(x)) + J_f(J_gh + o(h)) + o(J_gh + o(h)) = f(g(x)) + J_fJ_gh + o(\lVert h\rVert)$。$\blacksquare$

對以純量損失結尾的合成 $L = f_k\circ\dots\circ f_1$，梯度是乘積 $\nabla L^{\top} = J_{f_k}J_{f_{k-1}}\cdots J_{f_1}$。矩陣乘法滿足結合律，所以這個乘積可以用任何順序計算，而順序決定成本：

- **前向模式**從右邊開始乘，先算 $J_{f_1}$：每一趟計算一個雅可比–向量乘積 $Jv$，所以對 $n$ 個輸入的完整梯度需要 $n$ 趟。
- **反向模式**從左邊、從純量損失開始乘：每一趟計算一個向量–雅可比乘積 $u^{\top}J$，所以完整梯度只需一次反向傳遞，成本是前向傳遞的小常數倍（單元 03，§3.3.4 與 §3.5.9）。

中間量 $v$ 的**伴隨值（adjoint）**為 $\bar v = \partial L/\partial v$。反向模式就是規則 $\bar v_p = \sum_{c\,\in\,\text{children}(p)}\bar v_c\,\partial v_c/\partial v_p$：伴隨值在從 $p$ 到損失的每一條路徑上*累加*（單元 03，§3.3.3）。

## B.3 泰勒定理與黑塞矩陣

對二次連續可微的 $f:\mathbb{R}^n\to\mathbb{R}$，**黑塞矩陣（Hessian）** $\nabla^2 f(x)$ 是二階偏導數構成的對稱矩陣。

**定理 B.2（帶餘項的泰勒定理）。** 對每個 $x, h$，存在 $x$ 到 $x + h$ 線段上的 $\xi$，使
$$
f(x + h) = f(x) + \nabla f(x)^{\top}h + \tfrac12h^{\top}\nabla^2 f(\xi)\,h .
$$

*證明。* 對 $\phi(s) = f(x + sh)$ 套用一維帶拉格朗日餘項的泰勒定理，其導數為 $\phi'(s) = \nabla f(x + sh)^{\top}h$ 與 $\phi''(s) = h^{\top}\nabla^2f(x + sh)h$。$\blacksquare$

**推論：二次上界。** 若黑塞矩陣的特徵值從不超過 $L$——等價地，梯度是 $L$-Lipschitz 的——則
$$
f(y)\le f(x) + \nabla f(x)^{\top}(y - x) + \tfrac L2\lVert y - x\rVert^2 .
$$
對 $y$ 最小化右式，得到步驟 $y = x - \nabla f(x)/L$ 與保證的下降量 $\lVert\nabla f(x)\rVert^2/(2L)$：這就是單元 03（§3.5.2）與附錄 F 中每個收斂證明背後的**下降引理**。

**二階條件。** 在臨界點（$\nabla f = 0$），正定的黑塞矩陣表示嚴格局部極小，負定表示局部極大，不定則表示**鞍點**。在高維中，隨機函數的臨界點裡鞍點遠多於極小點，而帶雜訊的梯度方法會逃離它們（單元 03，§3.7.2）。

## B.4 矩陣微分

對矩陣求梯度，最可靠的方式是用**微分**：把 $df$ 寫成 $dX$ 的線性函數，再從 $df = \mathrm{tr}(G^{\top}dX) = \sum_{ij}G_{ij}\,dX_{ij}$ 讀出梯度 $G$。規則與純量微積分相同，只是要保持不可交換因子的順序（[Magnus & Neudecker 2019](/zh-tw/references/#magnus-2019)）：
$$
d(XY) = (dX)Y + X(dY),\qquad d(X^{-1}) = -X^{-1}(dX)X^{-1},\qquad d\ln\det X = \mathrm{tr}(X^{-1}dX).
$$

| 函數 | 微分 | 梯度 |
|---|---|---|
| $a^{\top}x$ | $a^{\top}dx$ | $a$ |
| $x^{\top}Ax$ | $x^{\top}(A + A^{\top})dx$ | $(A + A^{\top})x$ |
| $\lVert Ax - b\rVert^2$ | $2(Ax - b)^{\top}A\,dx$ | $2A^{\top}(Ax - b)$ |
| $\mathrm{tr}(AX)$ | $\mathrm{tr}(A\,dX)$ | $A^{\top}$ |
| $\ln\det X$（$X\succ0$） | $\mathrm{tr}(X^{-1}dX)$ | $X^{-\top}$ |

**穿過矩陣乘積的反向傳播。** 若 $Y = XW$，上游梯度為 $\bar Y = \partial L/\partial Y$，則 $dL = \mathrm{tr}(\bar Y^{\top}dY) = \mathrm{tr}(\bar Y^{\top}(dX)W) + \mathrm{tr}(\bar Y^{\top}X\,dW)$，循環置換跡即得
$$
\bar X = \bar YW^{\top},\qquad \bar W = X^{\top}\bar Y .
$$
主幹專案手寫反向傳播中的每一個線性層，就是這兩行。**形狀檢查**能抓到大部分錯誤：$\bar X$ 必須與 $X$ 同形狀，而用手邊的矩陣湊出這個形狀，通常只有一種乘法方式。

**例題 B.2（查找的梯度）。** 嵌入查找是 $H = UE$，$U$ 的每一列是獨熱向量，所以 $\bar E = U^{\top}\bar H$：$\bar E$ 的每一列，是查找到它的那些上游列的*總和*（單元 02，命題 2.1）。

## B.5 Softmax、log-sum-exp 與交叉熵

$z\in\mathbb{R}^K$ 的 **softmax** 是 $p_k = e^{z_k}/\sum_je^{z_j}$；**log-sum-exp** 是 $\mathrm{LSE}(z) = \ln\sum_je^{z_j}$，所以 $\ln p_k = z_k - \mathrm{LSE}(z)$。

**命題 B.3。** (i) $\nabla\mathrm{LSE}(z) = p$。(ii) softmax 的雅可比矩陣為 $J = \mathrm{diag}(p) - pp^{\top}$。(iii) 對交叉熵損失 $\ell = -\ln p_y$，$\nabla_z\ell = p - e_y$。(iv) $\nabla^2\mathrm{LSE} = \mathrm{diag}(p) - pp^{\top}\succeq0$，所以 LSE 是凸的。

*證明。* (i) $\partial\mathrm{LSE}/\partial z_k = e^{z_k}/\sum_je^{z_j} = p_k$。(iii) $\ell = -z_y + \mathrm{LSE}(z)$，所以 $\nabla\ell = -e_y + p$。(ii) 與 (iv) 把 (i) 再微分一次：$\partial p_k/\partial z_j = p_k(\mathbb{1}[j = k] - p_j)$。這個矩陣是一個共變異數——$Y\sim p$ 時獨熱向量 $e_Y$ 的共變異數——因此是半正定的。$\blacksquare$

**例題 B.3。** $z = (1, 0, 0)$ 給出 $p = (0.5761, 0.2119, 0.2119)$，且
$$
J = \begin{pmatrix}0.2442 & -0.1221 & -0.1221\\-0.1221 & 0.1670 & -0.0449\\-0.1221 & -0.0449 & 0.1670\end{pmatrix}.
$$
每一列總和為零，因為對所有 logit 加上同一個常數不會改變任何事。目標為 $y = 2$ 時，交叉熵梯度為 $p - e_2 = (0.5761, -0.7881, 0.2119)$：提高目標的 logit，並依機率比例降低其他 logit。

**數值穩定性。** 直接計算 $e^{z_k}$，在 32 位元運算中 $z_k\gtrsim 89$ 就會溢位。改用 $\mathrm{LSE}(z) = m + \ln\sum_je^{z_j - m}$，其中 $m = \max_jz_j$；平移後的指數至多為 1。對 $z = (1000, 999)$，這給出 $1000 + \ln(1 + e^{-1}) = 1000.3133$，而天真的公式會回傳無窮大。交叉熵應以 $\mathrm{LSE}(z) - z_y$ 計算，而不是對一個算出來的機率取 $-\ln$——那個機率可能下溢成 0。

**書中用到的其他導數。** 邏輯斯函數 $\sigma'(u) = \sigma(u)(1 - \sigma(u))$；$\tanh'(u) = 1 - \tanh^2u$；層正規化與 GELU 請見主幹專案的 `model.py`，其導數以有限差分檢查。

## B.6 積分、變數變換與積分號下求導

**變數變換。** 若 $g:\mathbb{R}^n\to\mathbb{R}^n$ 是可微的雙射且其反函數可微，則
$$
\int_{g(\Omega)}f(y)\,dy = \int_\Omega f(g(x))\,\big|\det J_g(x)\big|\,dx .
$$
套用到機率密度，就是正規化流背後的公式 $p_Y(y) = p_X(g^{-1}(y))\,|\det J_{g^{-1}}(y)|$（單元 08，§8.3.2；附錄 C）。

**高斯積分。** $\int_{-\infty}^{\infty}e^{-x^2/2}dx = \sqrt{2\pi}$。*證明：*把它平方，得 $\iint e^{-(x^2 + y^2)/2}dx\,dy$，換成極座標，其雅可比行列式為 $r$：$\int_0^{2\pi}\int_0^\infty e^{-r^2/2}r\,dr\,d\theta = 2\pi$。$\blacksquare$

**積分號下求導。** 若 $p_\theta(x)$ 及其對 $\theta$ 的導數連續，且被一個可積函數所控制，則 $\nabla_\theta\int f(x)p_\theta(x)\,dx = \int f(x)\nabla_\theta p_\theta(x)\,dx$。兩個推論支撐了單元 08 與 09：

- **分數函數恆等式。** $\nabla_\theta\mathbb{E}_{p_\theta}[f] = \mathbb{E}_{p_\theta}[f\,\nabla_\theta\ln p_\theta]$，用到 $\nabla p = p\nabla\ln p$。它不需要 $f$ 的導數——這是 REINFORCE 的基礎（單元 09）。
- **重參數化（reparameterization）。** 若 $x = g_\theta(\varepsilon)$，$\varepsilon$ 來自固定分布，則 $\nabla_\theta\mathbb{E}[f(x)] = \mathbb{E}_\varepsilon[\nabla_\theta f(g_\theta(\varepsilon))]$。它需要 $f$ 可微，但通常變異數低得多——這是變分自編碼器所用的梯度（單元 08，§8.3.3）。

## B.7 有限差分

要檢查導數，就把它與差商比較。**前向差分** $(f(x + h) - f(x))/h$ 的截斷誤差為 $\tfrac h2f''(x) + O(h^2)$；**中央差分** $(f(x + h) - f(x - h))/(2h)$ 的誤差為 $\tfrac{h^2}{6}f'''(x) + O(h^4)$，因為偶數階項相消。

浮點捨入會加入約 $\epsilon_{\text{mach}}|f|/h$ 的誤差，隨 $h$ 變小而增大。平衡兩者，中央差分的最佳 $h\approx\epsilon_{\text{mach}}^{1/3}$——64 位元運算中約為 $6\times10^{-6}$。

**例題 B.4。** 對 $f = \exp$ 在 $0$（真實導數為 1），64 位元運算中中央差分的誤差為：

| $h$ | $10^{-1}$ | $10^{-2}$ | $10^{-3}$ | $10^{-5}$ | $10^{-8}$ | $10^{-11}$ |
|---|---|---|---|---|---|---|
| 誤差 | $1.67\times10^{-3}$ | $1.67\times10^{-5}$ | $1.67\times10^{-7}$ | $1.2\times10^{-11}$ | $6\times10^{-9}$ | $8\times10^{-8}$ |

前三個精確地遵循 $h^2/6$；在 $h\approx10^{-5}$ 以下，捨入誤差接手，誤差又增大。這就是為什麼主幹專案的梯度檢查使用 64 位元運算與 $h = 10^{-6}$，並比較*相對*誤差（單元 03，§3.7.5）。

## 習題

**B.1** 計算 $f(x) = x/\lVert x\rVert$ 的雅可比矩陣，並證明 $J_fx = 0$。加以詮釋。

**B.2** 對 $Y = XW + \mathbf 1b^{\top}$ 推導 $\partial L/\partial X$ 與 $\partial L/\partial W$，並求 $\partial L/\partial b$。

**B.3** 證明對任何機率向量 $q$，$\nabla_z\big(-\sum_kq_k\ln p_k\big) = p - q$（軟目標）。

**B.4** 以微分證明 $\nabla_X\,\mathrm{tr}(X^{\top}AX) = (A + A^{\top})X$。

**B.5** 對 $f(x) = x^3$ 在 $x = 1$、$h = 0.1$，計算前向與中央差分的誤差，並與預測的 $\tfrac h2f''$ 與 $\tfrac{h^2}6f'''$ 比較。

**B.6** ★ 證明線性 softmax 模型 $z = Wx$ 的交叉熵損失對 $\mathrm{vec}(W)$ 的黑塞矩陣為 $(\mathrm{diag}(p) - pp^{\top})\otimes xx^{\top}$，並推論損失在 $W$ 上是凸的。

**解答。** **B.1** $J = (I - uu^{\top})/\lVert x\rVert$，其中 $u = x/\lVert x\rVert$：正規化忽略沿 $x$ 方向的改變。**B.2** $\bar X = \bar YW^{\top}$、$\bar W = X^{\top}\bar Y$、$\bar b = \bar Y^{\top}\mathbf 1$。**B.5** 前向：$(1.331 - 1)/0.1 - 3 = 0.31$（預測 $0.3$）；中央：$(1.331 - 0.729)/0.2 - 3 = 0.01$（預測 $0.01$，對三次函數精確）。

## 延伸閱讀

[Magnus & Neudecker 2019](/zh-tw/references/#magnus-2019) 講矩陣微分；[Griewank & Walther 2008](/zh-tw/references/#griewank-2008) 講自動微分；[Deisenroth et al. 2020](/zh-tw/references/#deisenroth-2020) 第 5 章講機器學習中的向量微積分；[Higham 2002](/zh-tw/references/#higham-2002) 講浮點誤差。
