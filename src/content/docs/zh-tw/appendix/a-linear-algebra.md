---
title: "A · 線性代數"
description: 向量、內積與範數；線性映射、秩與投影；最小平方法；特徵值與譜定理；正定矩陣；奇異值分解；矩陣範數與條件數；矩陣的冪、譜半徑與暫態增長。
sidebar: { label: "A · 線性代數", order: 1 }
---

本書的每個模型都由矩陣乘積構成，而它的理論大多是譜理論：步長上限是特徵值（單元 03），記憶的衰減是譜半徑（單元 05），低秩結構是奇異值分解（單元 02 與 04）。本附錄收錄各單元所用的事實，證明簡短且有啟發性時附上證明。例題中的數字由 `labs/test_appendices.py` 檢查。

| 小節 | 用於 |
|---|---|
| A.1 內積與範數 | 02（相似度、JL 引理）、03（最陡下降）、04（注意力分數） |
| A.2 秩與投影 | 02（瓶頸）、04（$QK^{\top}$ 的秩）、05（回憶的秩上限） |
| A.3 最小平方法 | 01、02（線性讀出）、06（擬合規模定律） |
| A.4–A.5 特徵值、正定性 | 03（黑塞矩陣、條件數）、05（穩定性）、10（共變異數） |
| A.6 SVD | 02（Eckart–Young）、04（低秩注意力）、11（電路的秩） |
| A.7 範數與條件數 | 03（病態條件）、06（數值精度） |
| A.8 冪與譜半徑 | 03（深層乘積）、05（遞迴、暫態增長） |

## A.1 向量、內積與範數

向量是行向量，$x\in\mathbb{R}^n$。**內積（inner product）**為 $\langle x, y\rangle = x^{\top}y = \sum_ix_iy_i$，**歐幾里得範數**為 $\lVert x\rVert = \sqrt{x^{\top}x}$。其他常見範數：$\lVert x\rVert_1 = \sum_i|x_i|$ 與 $\lVert x\rVert_\infty = \max_i|x_i|$。

**定理 A.1（柯西–施瓦茨，Cauchy–Schwarz）。** $|x^{\top}y|\le\lVert x\rVert\,\lVert y\rVert$，等號成立若且唯若 $x$ 與 $y$ 平行。

*證明。* 若 $y = 0$ 則無需證明。否則，對每個實數 $t$，$0\le\lVert x - ty\rVert^2 = \lVert x\rVert^2 - 2t\,x^{\top}y + t^2\lVert y\rVert^2$。一個永不為負的 $t$ 的二次式，判別式不為正：$4(x^{\top}y)^2 - 4\lVert x\rVert^2\lVert y\rVert^2\le0$。等號恰在某個 $t$ 使 $x - ty = 0$ 時成立。$\blacksquare$

書中用到的推論：**三角不等式** $\lVert x + y\rVert\le\lVert x\rVert + \lVert y\rVert$（兩邊平方，對交叉項用柯西–施瓦茨）；用來比較嵌入的**餘弦相似度** $\cos\angle(x, y) = x^{\top}y/(\lVert x\rVert\lVert y\rVert)\in[-1, 1]$（單元 02）；以及在所有單位向量 $u$ 中，方向導數 $\nabla f^{\top}u$ 在 $u = \nabla f/\lVert\nabla f\rVert$ 時最大——梯度是最陡上升方向（單元 03，§3.2.5）。

**正交歸一基。** 若 $q_i^{\top}q_j = \mathbb{1}[i = j]$，向量 $q_1,\dots,q_k$ 就是**正交歸一的（orthonormal）**。把它們排成 $Q\in\mathbb{R}^{n\times k}$ 的各行，就是 $Q^{\top}Q = I_k$。當 $k = n$ 時，$Q$ 是**正交矩陣**：$Q^{-1} = Q^{\top}$，且對每個 $x$ 都有 $\lVert Qx\rVert = \lVert x\rVert$——正交映射是旋轉與鏡射，保持長度與角度。

## A.2 線性映射、秩與投影

矩陣 $A\in\mathbb{R}^{m\times n}$ 是線性映射 $\mathbb{R}^n\to\mathbb{R}^m$。它的**行空間**（值域）$\mathcal{R}(A) = \{Ax\}$ 與**零空間** $\mathcal{N}(A) = \{x: Ax = 0\}$ 都是子空間，**秩（rank）** $r = \dim\mathcal R(A)$ 等於線性獨立之行的數目，也等於線性獨立之列的數目。

**定理 A.2（秩–零度定理）。** $\dim\mathcal R(A) + \dim\mathcal N(A) = n$。

*證明。* 取 $\mathcal N(A)$ 的一組基 $v_1,\dots,v_k$，並擴充成 $\mathbb{R}^n$ 的一組基 $v_1,\dots,v_n$。向量 $Av_{k+1},\dots,Av_n$ 張成 $\mathcal R(A)$（因為 $i\le k$ 時 $Av_i = 0$），而且它們線性獨立：若 $\sum_{i>k}c_iAv_i = 0$，則 $\sum_{i>k}c_iv_i\in\mathcal N(A)$，所以它是 $v_1,\dots,v_k$ 的線性組合，而整組基的獨立性迫使每個 $c_i = 0$。因此 $\dim\mathcal R(A) = n - k$。$\blacksquare$

**乘積的秩。** $\mathrm{rank}(AB)\le\min(\mathrm{rank}\,A, \mathrm{rank}\,B)$：$AB$ 的各行落在 $\mathcal R(A)$ 中，且 $\mathcal N(B)\subseteq\mathcal N(AB)$。這就是為什麼無論序列多長，注意力頭的 $QK^{\top} = XW_QW_K^{\top}X^{\top}$ 的秩至多為 $d_k$（單元 04），也是為什麼具有 $N$ 維狀態的遞迴，從過去傳到未來的獨立數字至多 $N$ 個（單元 05，§5.5.4）。

**正交投影。** 令 $U\in\mathbb{R}^{n\times k}$ 的各行為子空間 $\mathcal S$ 的一組正交歸一基。矩陣 $P = UU^{\top}$ 是到 $\mathcal S$ 的**正交投影**：$P^2 = P$、$P^{\top} = P$，而且對每個 $y$，殘差 $y - Py$ 與 $\mathcal S$ 正交。

**定理 A.3（投影就是最近點）。** 對每個 $s\in\mathcal S$，$\lVert y - s\rVert^2 = \lVert y - Py\rVert^2 + \lVert Py - s\rVert^2$；因此 $Py$ 是 $\mathcal S$ 中離 $y$ 最近的唯一一點。

*證明。* $y - s = (y - Py) + (Py - s)$；第一項與 $\mathcal S$ 正交，第二項在 $\mathcal S$ 中，所以交叉項為零（畢氏定理）。$\blacksquare$

**例題 A.1。** 把 $y = (1, 2, 2)$ 投影到 $a = (1, 1, 0)$ 張成的直線上。以 $u = a/\lVert a\rVert$，$Py = a\,(a^{\top}y)/(a^{\top}a) = (1, 1, 0)\cdot 3/2 = (1.5, 1.5, 0)$。殘差 $(-0.5, 0.5, 2)$ 與 $a$ 正交，長度為 $\sqrt{4.5} = 2.121$：直線上沒有任何一點更近。

## A.3 最小平方法

給定行彼此獨立的 $X\in\mathbb{R}^{n\times p}$ 與 $y\in\mathbb{R}^n$，**最小平方（least squares）**問題是 $\min_\beta\lVert y - X\beta\rVert^2$。由定理 A.3，最小化者使 $X\beta$ 成為 $y$ 在 $\mathcal R(X)$ 上的投影，所以殘差與 $X$ 的每一行正交：

$$
X^{\top}(y - X\hat\beta) = 0 \quad\Longleftrightarrow\quad X^{\top}X\hat\beta = X^{\top}y \qquad\text{（正規方程式）}
$$

當各行獨立時，$X^{\top}X$ 可逆（A.5），$\hat\beta = (X^{\top}X)^{-1}X^{\top}y$。數值計算上，應透過 QR 分解或 SVD 求解，而不是組出 $X^{\top}X$——它的條件數是 $X$ 的平方（A.7）。

**例題 A.2。** 以直線 $y\approx\beta_0 + \beta_1x$ 擬合 $(0, 1), (1, 2), (2, 2)$。正規方程式 $\begin{pmatrix}3 & 3\\3 & 5\end{pmatrix}\beta = \begin{pmatrix}5\\6\end{pmatrix}$ 給出 $\hat\beta = (7/6, 1/2) = (1.1667, 0.5)$。擬合值為 $(1.1667, 1.6667, 2.1667)$，殘差 $(-0.1667, 0.3333, -0.1667)$ 總和為零——殘差與常數行正交。

單元 06 的規模定律擬合是非線性最小平方；對純冪律取對數 $\ln L = \ln A - \alpha\ln N$ 之後，就正好變成這個問題。

## A.4 特徵值與譜定理

若非零的 $v$ 滿足 $Av = \lambda v$，它就是**特徵向量**，$\lambda$ 是**特徵值**。特徵值是特徵多項式 $\det(\lambda I - A)$ 的根，所以 $n\times n$ 矩陣計重數有 $n$ 個特徵值，可能是複數。兩個恆等式反覆用到：$\mathrm{tr}\,A = \sum_i\lambda_i$ 與 $\det A = \prod_i\lambda_i$。

若 $A$ 有 $n$ 個獨立的特徵向量，排成 $S$ 的各行，則 $A = S\Lambda S^{-1}$，其中 $\Lambda = \mathrm{diag}(\lambda_i)$，任何冪都很容易計算：$A^t = S\Lambda^tS^{-1}$。並非每個矩陣都可對角化（$\begin{pmatrix}0&1\\0&0\end{pmatrix}$ 就不行），而即使可對角化，$S$ 也可能條件很差——這是暫態增長的來源（A.8）。

**定理 A.4（對稱矩陣的譜定理）。** 若 $A = A^{\top}\in\mathbb{R}^{n\times n}$，則它的特徵值全為實數，並存在正交矩陣 $Q$ 使 $A = Q\Lambda Q^{\top}$。

*證明概要。* 若 $Av = \lambda v$ 且 $v$ 為複向量，則 $\bar v^{\top}Av = \lambda\lVert v\rVert^2$；取共軛轉置並利用對稱性，同一個量等於 $\bar\lambda\lVert v\rVert^2$；所以 $\lambda$ 是實數。對應相異特徵值的特徵向量互相正交：$\lambda_1v_2^{\top}v_1 = v_2^{\top}Av_1 = (Av_2)^{\top}v_1 = \lambda_2v_2^{\top}v_1$。完整陳述：取一個單位特徵向量 $q_1$；$A$ 把 $q_1$ 的正交補空間映到自身（若 $x\perp q_1$，則 $q_1^{\top}Ax = (Aq_1)^{\top}x = \lambda_1q_1^{\top}x = 0$），再對維度歸納即可。$\blacksquare$

**瑞利商（Rayleigh quotient）。** 對特徵值為 $\lambda_{\min}\le\dots\le\lambda_{\max}$ 的對稱矩陣 $A$，寫 $x = Qc$，
$$
\frac{x^{\top}Ax}{x^{\top}x} = \frac{\sum_i\lambda_ic_i^2}{\sum_ic_i^2}\in[\lambda_{\min}, \lambda_{\max}],
$$
極值在對應的特徵向量處達到。二次損失沿某個方向的曲率，就是其黑塞矩陣的瑞利商；最大曲率 $\lambda_{\max}$ 決定了梯度下降法的步長上限 $\eta < 2/\lambda_{\max}$（單元 03，§3.2.3 與 §3.5.4；附錄 F）。

**例題 A.3。** $A = \begin{pmatrix}2 & 1\\1 & 2\end{pmatrix}$ 的特徵值為 $1$ 與 $3$，特徵向量為 $(1, -1)/\sqrt2$ 與 $(1, 1)/\sqrt2$；跡 $4 = 1 + 3$，行列式 $3 = 1\cdot3$。二次形式 $x^{\top}Ax$ 沿 $(1, 1)$ 的彎曲速度是沿 $(1, -1)$ 的三倍：條件數為 3。

## A.5 正定矩陣

若對所有 $x$ 都有 $x^{\top}Ax\ge0$，對稱矩陣 $A$ 就是**半正定的**（$A\succeq0$）；若對 $x\ne0$ 嚴格成立，就是**正定的**（$A\succ0$）。由譜定理，這分別等於所有特徵值 $\ge0$、$>0$。

它們出現在：
- **Gram 矩陣與共變異數矩陣。** $X^{\top}X\succeq0$，因為 $x^{\top}X^{\top}Xx = \lVert Xx\rVert^2$；它恰在 $X$ 的各行獨立時為正定。每個共變異數矩陣 $\Sigma = \mathbb{E}[(z - \mu)(z - \mu)^{\top}]$ 都是半正定的，因為 $x^{\top}\Sigma x = \mathrm{Var}(x^{\top}z)\ge0$。
- **黑塞矩陣。** 二次可微函數為凸，若且唯若它的黑塞矩陣處處半正定（附錄 F）；在臨界點，正定的黑塞矩陣保證嚴格局部極小。
- **分解。** $A\succ0$ 有唯一的 **Cholesky 分解** $A = LL^{\top}$，$L$ 為下三角且對角線為正；也有唯一的正定平方根 $A^{1/2} = Q\Lambda^{1/2}Q^{\top}$。從 $N(\mu, \Sigma)$ 抽樣就是 $\mu + Lz$，其中 $z\sim N(0, I)$（附錄 C）。

## A.6 奇異值分解

**定理 A.5（SVD）。** 每個 $A\in\mathbb{R}^{m\times n}$ 都可寫成 $A = U\Sigma V^{\top} = \sum_{i=1}^{r}\sigma_iu_iv_i^{\top}$，其中 $U\in\mathbb{R}^{m\times m}$ 與 $V\in\mathbb{R}^{n\times n}$ 為正交矩陣，$\Sigma$ 為對角矩陣，$\sigma_1\ge\sigma_2\ge\dots\ge\sigma_r > 0$，且 $r = \mathrm{rank}\,A$。

*證明概要。* $A^{\top}A$ 是對稱半正定的，所以 $A^{\top}A = V\,\mathrm{diag}(\sigma_i^2)\,V^{\top}$，$\sigma_i\ge0$。對 $\sigma_i > 0$，令 $u_i = Av_i/\sigma_i$。它們正交歸一：$u_i^{\top}u_j = v_i^{\top}A^{\top}Av_j/(\sigma_i\sigma_j) = \mathbb{1}[i = j]$。把它們擴充成 $\mathbb{R}^m$ 的正交歸一基。於是 $AV = U\Sigma$，即 $A = U\Sigma V^{\top}$。$\blacksquare$

從幾何上讀，每個線性映射都是一次旋轉（$V^{\top}$）、一次沿座標軸的伸縮（$\Sigma$），再一次旋轉（$U$）。奇異值就是伸縮倍數：$\sigma_1 = \max_{\lVert x\rVert = 1}\lVert Ax\rVert$。

**低秩近似。** 截取最大的 $k$ 項，$A_k = \sum_{i\le k}\sigma_iu_iv_i^{\top}$，在譜範數與 Frobenius 範數下都是最佳的秩 $k$ 近似，誤差分別為 $\sigma_{k+1}$ 與 $\sqrt{\sum_{i>k}\sigma_i^2}$（Eckart–Young 定理，證明於單元 02，§2.5.5）。

**偽逆。** $A^{+} = V\Sigma^{+}U^{\top}$，把非零奇異值取倒數。$A^{+}y$ 是 $Ax\approx y$ 的最小範數最小平方解——在欠定線性問題上，從零出發的梯度下降法正是收斂到它，這是隱式偏誤最簡單的例子（單元 01，§1.5.4）。

**例題 A.4。** $A = \begin{pmatrix}3 & 0\\4 & 5\end{pmatrix}$。$A^{\top}A = \begin{pmatrix}25 & 20\\20 & 25\end{pmatrix}$ 的特徵值為 45 與 5，所以 $\sigma_1 = \sqrt{45} = 6.708$、$\sigma_2 = \sqrt5 = 2.236$，右奇異向量為 $(1, 1)/\sqrt2$ 與 $(1, -1)/\sqrt2$。最佳秩 1 近似在兩種範數下的誤差都是 $\sigma_2 = 2.236$；條件數為 $\sigma_1/\sigma_2 = 3$。注意 $A$ 本身的特徵值是 3 與 5——對非對稱矩陣，特徵值與奇異值不同。

## A.7 矩陣範數與條件數

**譜範數** $\lVert A\rVert = \sigma_1$ 是 $A$ 伸長向量的最大倍數；**Frobenius 範數** $\lVert A\rVert_F = \sqrt{\sum_{ij}A_{ij}^2} = \sqrt{\sum_i\sigma_i^2}$ 是各元素的歐幾里得範數。兩者都滿足次乘性：$\lVert AB\rVert\le\lVert A\rVert\,\lVert B\rVert$。這個不等式是所有關於深層乘積之論證的骨幹——$L$ 個層雅可比矩陣乘積的範數可能隨 $L$ 呈幾何增長或縮小（單元 03，§3.7.4）。

可逆矩陣 $A$ 的**條件數（condition number）**為 $\kappa(A) = \sigma_1/\sigma_n$。它量測求解 $Ax = b$ 時相對誤差被放大多少：若 $A(x + \delta x) = b + \delta b$，則
$$
\frac{\lVert\delta x\rVert}{\lVert x\rVert}\le\kappa(A)\,\frac{\lVert\delta b\rVert}{\lVert b\rVert},
$$
因為 $\lVert\delta x\rVert\le\lVert A^{-1}\rVert\lVert\delta b\rVert = \lVert\delta b\rVert/\sigma_n$ 且 $\lVert b\rVert\le\sigma_1\lVert x\rVert$。在單位捨入誤差為 $u$ 的浮點運算中（32 位元約 $6\times10^{-8}$，bfloat16 約 $5\times10^{-4}$），預期會損失約 $\log_{10}\kappa$ 位有效數字（[Higham 2002](/zh-tw/references/#higham-2002)）。對最佳化而言，同一個數有第二個意義：對黑塞矩陣為 $H$ 的二次函數，$\kappa(H)$ 決定梯度下降法的收斂速率（附錄 F）。

## A.8 矩陣的冪與譜半徑

**譜半徑**為 $\rho(A) = \max_i|\lambda_i|$。遞迴 $h_t = Ah_{t-1}$ 給出 $h_t = A^th_0$，所以它的長期行為由 $A^t$ 決定。

**定理 A.6。** $t\to\infty$ 時 $A^t\to0$，若且唯若 $\rho(A) < 1$。此外，對每個 $\epsilon > 0$ 都有常數 $c_\epsilon$ 使 $\lVert A^t\rVert\le c_\epsilon(\rho(A) + \epsilon)^t$，且 $\lim_t\lVert A^t\rVert^{1/t} = \rho(A)$（Gelfand 公式）。

*證明概要。* 若某個特徵值 $|\lambda|\ge1$、特徵向量為 $v$，則 $\lVert A^tv\rVert = |\lambda|^t\lVert v\rVert\not\to0$。反之，把 $A$ 寫成 Jordan 形式 $A = SJS^{-1}$。特徵值為 $\lambda$、大小為 $m$ 的 Jordan 塊，其 $t$ 次冪的元素為 $\binom{t}{j}\lambda^{t-j}$，$j < m$，每個都被 $t$ 的多項式乘 $|\lambda|^t$ 所界定。多項式乘 $|\lambda|^t$ 至多為 $c_\epsilon(|\lambda| + \epsilon)^t$，在 $|\lambda| < 1$ 時趨於零。$\blacksquare$

常數 $c_\epsilon$ 在實務上很重要。當 $A$ 是**正規的（normal）**（$AA^{\top} = A^{\top}A$，例如對稱矩陣）時，$\lVert A^t\rVert = \rho(A)^t$ 精確成立，衰減是單調的。若不是，$\lVert A^t\rVert$ 可能在最終衰減之前先增長許多步——**暫態增長（transient growth）**（單元 05，§5.3.5；[Trefethen & Embree 2005](/zh-tw/references/#trefethen-2005)）。

**例題 A.5。** $A = \begin{pmatrix}0.5 & 2\\0 & 0.5\end{pmatrix}$ 的 $\rho(A) = 0.5$，然而
$$
\lVert A^t\rVert = 1,\ 2.118,\ 2.031,\ 1.510,\ 1.004,\ 0.627,\ 0.376,\ \dots\qquad(t = 0, 1, 2, \dots).
$$
前四步這個映射都在放大，直到 $t = 5$ 才開始收縮。原因是非對角的耦合：$A^t = \begin{pmatrix}0.5^t & 4t\,0.5^{t}\\0 & 0.5^t\end{pmatrix}$，其右上元素先上升，幾何因子才勝出。因此，譜半徑小於一的學習遞迴，仍可能在中等時程內放大擾動與梯度。

**線性系統的穩定性**——內部穩定與輸入–輸出穩定，以及為什麼轉移函數的極點不一定是全部的特徵值——在附錄 G 中展開。

## 習題

**A.1** 證明 $\mathrm{rank}(A^{\top}A) = \mathrm{rank}(A)$。*（提示：證明 $\mathcal N(A^{\top}A) = \mathcal N(A)$。）*

**A.2** 證明正交矩陣的特徵值模為 1，且 $\lVert Q\rVert = 1$。

**A.3** 對 $A = \begin{pmatrix}4 & 0\\0 & 1\end{pmatrix}$，計算 $\kappa(A)$，以及 $b$ 的相對誤差為 $10^{-3}$ 時 $x$ 的最大相對誤差。

**A.4** 證明對秩為 $r$ 的矩陣，$\lVert A\rVert\le\lVert A\rVert_F\le\sqrt{r}\,\lVert A\rVert$。

**A.5** 對 $A = \begin{pmatrix}0.9 & 0\\0 & 0.5\end{pmatrix}$ 計算 $\lVert A^t\rVert$，並說明為什麼沒有暫態增長。

**A.6** ★ 令 $A$ 為特徵值 $\lambda_1\ge\dots\ge\lambda_n$ 的對稱矩陣。證明 Courant–Fischer 刻畫 $\lambda_k = \max_{\dim\mathcal S = k}\min_{x\in\mathcal S, x\ne0}x^{\top}Ax/x^{\top}x$。

**解答。** **A.3** $\kappa = 4$；相對誤差至多 $4\times10^{-3}$。**A.5** $\lVert A^t\rVert = 0.9^t$：這個矩陣是對角的，因此是正規的。

## 延伸閱讀

[Strang 2016](/zh-tw/references/#strang-2016) 提供幾何直覺；[Axler 2024](/zh-tw/references/#axler-2024) 以證明為主；[Horn & Johnson 2012](/zh-tw/references/#horn-2012) 與 [Golub & Van Loan 2013](/zh-tw/references/#golub-2013) 供查閱；[Trefethen & Embree 2005](/zh-tw/references/#trefethen-2005) 討論非正規矩陣；[Deisenroth et al. 2020](/zh-tw/references/#deisenroth-2020) 在機器學習的脈絡中講述這些內容。
