---
title: "2.4 Worked examples"
description: Eight worked examples — hand-computing XOR, one-hot readout as a table, parameter scaling for long contexts, embedding width, geometry of learned representations, count versus neural 4-grams, lookup gradients, and Cover counting for XOR.
sidebar: { label: "2.4 Worked examples", order: 4 }
---

The previous sections established representations and geometric manifolds; this page translates those abstractions into concrete numerical calculations and engineering diagnostics. These eight worked examples address a single foundational proposition: **how does deep learning, by reconstructing representations in continuous space, shatter the classical barriers of separability, parameter scaling, and transfer generalization?**

- **Examples 2.1–2.2:** Prove the algebraic impossibility of XOR under 2D hyperplanes, show how non-linear feature expansion resolves it, and demonstrate that one-hot representations are algebraically equivalent to unshared lookup tables.
- **Examples 2.3–2.4:** Contrast exponential discrete count scaling against linear neural parameter growth, sweeping embedding dimension $d$ to trace geometric capacity.
- **Examples 2.5–2.6:** Inspect the spontaneous emergence of semantic manifolds (such as digit clustering) under self-supervision, quantifying sample efficiency trade-offs between neural sharing and discrete counting.
- **Examples 2.7–2.8:** Trace the outer-product mechanics of the scatter-add gradient, applying Cover's counting theorem to compute the exact combinatorial limits of hyperplane classification.

All numbers on this page are recomputed and verified in continuous integration by `labs/test_worked_examples_u01_u02.py`; trained model results (Examples 2.4–2.6) are produced by `spine/stage02_embeddings.py`.

## Worked Example 2.1 — XOR: first impossible, then trivial

**Question.** Why is a single-layer perceptron algebraically incapable of solving the XOR problem in two dimensions, and how does adding a single non-linear feature untangle the geometry?

**Setting.** The four vertices of the XOR problem in $\mathbb{R}^2$: $(0,0) \to 0$, $(0,1) \to 1$, $(1,0) \to 1$, $(1,1) \to 0$.

**Step 1: Algebraic proof of linear inseparability in 2D.**
Suppose there exists an affine hyperplane $f(x) = w_1 x_1 + w_2 x_2 + b$ that correctly classifies all four points. This requires simultaneously satisfying four linear inequalities:
1. Input $(0, 0)$, label $0$: $w_1(0) + w_2(0) + b \le 0 \implies b \le 0$;
2. Input $(0, 1)$, label $1$: $w_1(0) + w_2(1) + b > 0 \implies w_2 + b > 0$;
3. Input $(1, 0)$, label $1$: $w_1(1) + w_2(0) + b > 0 \implies w_1 + b > 0$;
4. Input $(1, 1)$, label $0$: $w_1(1) + w_2(1) + b \le 0 \implies w_1 + w_2 + b \le 0$.

Add inequalities (2) and (3):
$(w_1 + b) + (w_2 + b) > 0 \implies w_1 + w_2 + 2b > 0 \implies w_1 + w_2 + b > -b$.
However, from inequality (1), we have $b \le 0 \implies -b \ge 0$.
Therefore: $w_1 + w_2 + b > 0$.
This directly contradicts inequality (4), which requires $w_1 + w_2 + b \le 0$.
**No straight line in the plane can separate the XOR pattern.**

**Step 2: Lifting into feature space.**
Construct a non-linear interaction feature: $\phi(x) = (x_1, x_2, x_1 x_2) \in \mathbb{R}^3$.
Select weight vector $w = (1, 1, -2)$ and bias $b = -0.5$. Evaluate the linear score $w^\top \phi(x) + b$:

| Input $x$ | Representation $\phi(x)$ | Score $w^{\top}\phi + b$ | Prediction | Target Label |
|---|---|---|---|---|
| **(0, 0)** | (0, 0, 0) | $1(0) + 1(0) - 2(0) - 0.5 = \mathbf{-0.5}$ | 0 | 0 |
| **(0, 1)** | (0, 1, 0) | $1(0) + 1(1) - 2(0) - 0.5 = \mathbf{+0.5}$ | 1 | 1 |
| **(1, 0)** | (1, 0, 0) | $1(1) + 1(0) - 2(0) - 0.5 = \mathbf{+0.5}$ | 1 | 1 |
| **(1, 1)** | (1, 1, 1) | $1(1) + 1(1) - 2(1) - 0.5 = \mathbf{-0.5}$ | 0 | 0 |

**Interpretation.**
In $\mathbb{R}^3$, every data point achieves a functional margin of absolute magnitude 0.5. The geometric margin is $0.5 / \lVert w \rVert = 0.5 / \sqrt{1^2 + 1^2 + (-2)^2} = 0.5 / \sqrt{6}$.
A single hidden ReLU neuron computes this exact interaction: for binary inputs, $\mathrm{ReLU}(x_1 + x_2 - 1)$ outputs 0 at $(0,0), (0,1), (1,0)$, and activates to $1+1-1=1$ at $(1,1)$—reproducing the non-linear interaction $x_1 x_2$. The final read-out remains strictly linear, but operates on untangled geometric representations.

## Worked Example 2.2 — One-hot readout is a table

**Question.** When a linear read-out operates directly on discrete one-hot vectors, why is it mathematically identical to an unshared lookup table?

**Setting.** The three-symbol alphabet $\{\texttt a, \texttt b, \texttt \_\}$ from Worked Example 1.2 with smoothed bigram transitions.

Let the linear classifier weights equal the log-probability matrix: $W[a, b] = \ln q(b\mid a)$.
For input `b`, the one-hot vector is $u_b = (0, 1, 0)^\top$.
Computing the linear logit vector:
$z = W^{\top} u_b = W[\texttt b, \cdot] = (\ln\tfrac26, \ln\tfrac16, \ln\tfrac36)^\top \approx (-1.0986, -1.7918, -0.6931)^\top$.
Passing $z$ through the softmax function:
$q = \mathrm{softmax}(z) = (\tfrac26, \tfrac16, \tfrac36)$,
which **identically reproduces row `b` of the bigram table**!

**Interpretation.**
Multiplying $W \in \mathbb{R}^{V\times V}$ by a one-hot vector activates only the row corresponding to that symbol, completely zeroing out all other rows.
**Under a one-hot representation with an unconstrained linear read-out, no parameters are shared across distinct symbols.** Learning transitions for `a` provides zero statistical transfer to `b`. Discrete orthogonal representations inherently prohibit generalization across semantically related tokens.

## Worked Example 2.3 — Parameter scaling with long context

**Question.** As context length $k$ expands, what is the contrast in parameter scaling between discrete count tables and neural language models?

**Setting.** Vocabulary size $V = 50$. Predict the next character given the past $k$ characters. Compare a discrete count model against a neural $n$-gram with embedding dimension $d = 16$ and hidden width $H = 128$:

| Context Length $k$ | Discrete Table Capacity $V^k$ | Contexts Observed in Spine Corpus (385k chars) | Neural $n$-gram Parameters |
|---|---|---|---|
| **1** | 50 | 50 | 9,426 |
| **2** | 2,500 | 401 | 11,474 |
| **3** | 125,000 | 2,078 | 13,522 |
| **4** | 6,250,000 | 4,502 | 15,570 |
| **6** | 15,625,000,000 | 12,903 | 19,666 |

The neural parameter formula sums five components:
$\mathrm{Params} = Vd + kdH + H + HV + V = (50\times 16) + k(16\times 128) + 128 + (128\times 50) + 50 = \mathbf{7\,378 + 2\,048k}$.

**Interpretation.**
- **Discrete count tables explode exponentially ($V^k$):** At context $k = 6$, the discrete table requires over 15.6 billion rows. Even with sparse storage, 385,000 characters observe only 12,903 patterns (less than one in a million). Discrete models hit an insurmountable sparsity wall.
- **Neural models scale linearly ($O(k)$):** Each added context step adds merely $dH = 16 \times 128 = 2\,048$ parameters. Because all contexts share representations in continuous space, neural networks break the curse of dimensionality.

## Worked Example 2.4 — How wide must embeddings be?

**Question.** How does embedding dimension $d$ act as a direct capacity knob in factorized bilinear models?

**Setting.** Train the factorized bigram $z_{ab} = e_a^\top M e_b + c_b$ on the spine corpus using Adam for 1,500 steps (batch size 4,096), varying bottleneck dimension $d$:

| Embedding Width $d$ | 2 | 4 | 8 | 16 | 32 | 50 | Counting Bigram |
|---|---|---|---|---|---|---|---|
| **Trainable Parameters** | 154 | 266 | 514 | 1,106 | 2,674 | 5,050 | 2,500 logits (2,450 free parameters) |
| **Held-Out NLL (nats/char)** | 2.65 | 2.37 | 2.21 | **2.15** | 2.15 | 2.15 | 2.12 |

**Interpretation.**
1. **Bottleneck regime ($d < 16$):** At $d = 2$, squeezing 50 symbols into 2 dimensions incurs severe information loss (loss 2.65 nats, 0.53 nats worse than counting); increasing to $d = 8$ quickly recovers performance to 2.21 nats.
2. **Saturation plateau ($d \ge 16$):** At $d = 16$, with only 1,106 parameters (less than half the counting table), the model reaches 2.15 nats, within 0.03 nats of unconstrained counting. Expanding to $d = 32$ or $d = 50$ yields no further gain. The effective low-rank structure of this transition matrix is fully captured at $d \approx 16$.

## Worked Example 2.5 — What the learned representation captures

**Question.** What semantic geometry emerges spontaneously in embedding space under self-supervised next-character prediction?

**Setting.** Take the trained $d = 16$ embedding matrix from Worked Example 2.4. Query nearest neighbors using cosine similarity $\cos(e_i, e_j) = \frac{e_i^\top e_j}{\lVert e_i\rVert \lVert e_j\rVert}$:

| Query Character | Top-4 Nearest Neighbors (Cosine Similarity) | Emergent Geometric Manifold |
|---|---|---|
| `3` | `4` (0.98), `2` (0.98), `1` (0.97), `5` (0.97) | Digits cluster tightly into a dedicated numerical subspace |
| `T` | `Z` (0.86), `N` (0.85), `S` (0.83), `L` (0.76) | Capital initial letters cluster together |
| `a` | `u` (0.45), `e` (0.39), `A` (0.23) | Vowels exhibit mutual affinity |
| `.` | `,` (0.59) | Punctuation marks group together |

**Interpretation.**
No human supervisor tagged characters as "digits" or "vowels."
Because input and output embeddings are tied, characters that predict similar successors and are preceded by similar antecedents are pulled together by gradient descent. **Geometric proximity emerges spontaneously as an automated proxy for functional linguistic role.**

## Worked Example 2.6 — When is sharing worthwhile?

**Question.** Across what data regimes does neural feature sharing dominate discrete counting tables, and where do discrete models catch up?

**Setting.** Context length $k = 4$. Split 1,400 stories into 1,260 train, 70 validation, and 70 test stories. Compare tuned discrete 4-grams against a fixed neural 4-gram ($d=16, H=128$) across training prefix sizes:

| Training Characters $n$ | 5,000 | 20,000 | 80,000 | 385,274 |
|---|---|---|---|---|
| **Selected Smoothing α** | 0.003 | 0.003 | 0.01 | 0.01 |
| **Counting 4-gram Test NLL** | 1.07 | 0.67 | 0.56 | **0.51** |
| **Neural 4-gram Test NLL** | **0.94** | **0.62** | **0.55** | 0.53 |
| **Neural Advantage** | **-0.13 nats** | **-0.05 nats** | **-0.01 nats** | +0.02 nats |

**Interpretation.**
- **Data-scarce regime ($n = 5\,000$):** Discrete 4-gram counts are sparse, whereas neural parameter sharing across positions yields a massive lead of **0.94 vs 1.07** (0.13 nats lower loss).
- **Asymptotic regime ($n = 385\,274$):** As text approaches 400,000 characters, discrete cells accumulate sufficient empirical support, slightly overtaking the neural network (0.51 vs 0.53 nats) because the neural network is constrained by its 16-dimensional bottleneck.
Parameter sharing is a strategy that exchanges **inductive bias for sample efficiency**: transformative when data is scarce, while unbiased tables approach Bayes optimality with infinite data.

## Worked Example 2.7 — Gradient of a lookup

**Question.** When a batch contains repeated references to the same token, how does backpropagation accumulate gradients via matrix outer products, and what happens if an engineer implements plain assignment?

**Setting.** Vocabulary $V = 3$, embedding dimension $d = 2$. Batch index vector $x = (0, 2, 0)$. Upstream gradient matrix $G \in \mathbb{R}^{3\times 2}$:
$G = \begin{pmatrix}1&2\\3&4\\5&6\end{pmatrix}$ (row 1 corresponds to $E[0]$, row 2 to $E[2]$, row 3 to $E[0]$).

**Step 1: Scatter-add accumulation (Proposition 2.1).**
Symbol `0` was referenced at positions 1 and 3; its gradient row must accumulate both contributions:

$$
\frac{\partial L}{\partial E} = \begin{pmatrix}1+5 & 2+6\\ 0 & 0\\ 3 & 4\end{pmatrix} = \begin{pmatrix}6&8\\0&0\\3&4\end{pmatrix} = U^{\top}G,\quad U = \begin{pmatrix}1&0&0\\0&0&1\\1&0&0\end{pmatrix}.
$$

The tensor algebra $U^\top G$ matches the physical accumulation exactly.

**Step 2: The classic assignment bug.**
If implemented with plain assignment `dE[x] = G`, row 3's $(5, 6)$ simply overwrites row 1's $(1, 2)$, resulting in $dE[0] = (5, 6)$! Half of the training signal is silently erased. This bug is insidious because the model will still slowly decrease loss, but convergence is crippled. Only gradient checking (Unit 03) catches this failure reliably.

## Worked Example 2.8 — What a hyperplane can separate: Cover counting

**Question.** In two dimensions, how many binary labeling configurations can an affine hyperplane separate on 4 points in general position, and how does this explain XOR?

**Setting.** Four points in general position in $\mathbb{R}^2$ with affine classifier $\mathrm{sign}(w^\top x + b)$.

**Applying Cover's counting theorem (Theorem 2.3, §2.5.2):**
For $N$ points in general position in $\mathbb{R}^p$, the number of linearly separable binary dichotomies is:
$C(N, p+1) = 2\sum_{k=0}^{p}\binom{N-1}{k}$.

For $N = 4$ points in $p = 2$ dimensions (affine dimension $p+1 = 3$):
$C(4, 3) = 2\Big(\binom{3}{0} + \binom{3}{1} + \binom{3}{2}\Big) = 2(1 + 3 + 3) = \mathbf{14}$.

**Interpretation.**
Four binary labels admit $2^4 = 16$ possible labeling assignments.
Cover's theorem proves that a 2D line can realize at most 14 of them. **The two impossible labelings are precisely XOR and its complement!**
When lifted to $\mathbb{R}^3$ via $\phi(x) = (x_1, x_2, x_1 x_2)$, four points become affinely independent in 3D, and the Cover count reaches $C(4, 4) = 2\sum_{k=0}^3 \binom{3}{k} = 16$. All 16 labelings become linearly separable. Lifting dimensions unlocks geometric classification capacity.
