---
title: "2.3 Mathematical construction"
description: Linear separability, the embedding lookup as a matrix product and its scatter-add gradient, the rank-d factorized bigram with its gradients, the neural n-gram, best low-rank approximation, and invariance and equivariance defined precisely.
sidebar: { label: "2.3 Construction", order: 3 }
---

This section translates the geometric intuitions of §2.2 into the rigorous tensor algebra of modern deep learning. When reading each matrix equation, always follow a four-step mental protocol:

1. **Identify the entity:** State the real-world or semantic object represented by each symbol;
2. **Shape check:** Strictly verify the tensor dimensions of every factor;
3. **Identify parameter reuse:** Detect which underlying weights participate simultaneously in multiple forward or backward paths;
4. **Trace the computational graph:** Follow the exact routes along which gradients propagate and aggregate.

The recurring dimensions throughout this book are: vocabulary size $V$, embedding dimension $d$, batch size $N$, context length $k$, and hidden layer width $H$. A shape mismatch is not a trivial typographic flaw; in deep learning systems, it almost invariably flags an erroneous transpose or a gradient routed to the wrong computational node.

## 2.3.1 Linear read-outs and separability

Let an input object $x$ be mapped to a continuous vector $\phi(x) \in \mathbb{R}^d$ in feature space. A binary linear classifier defined on this representation predicts labels according to the algebraic sign of $w^{\top}\phi(x) + b$, parameterized by hyperplane normal $w \in \mathbb{R}^d$ and bias $b \in \mathbb{R}$.

A labeled dataset is **linearly separable in $\phi$** if there exists at least one parameter pair $(w, b)$ that correctly classifies every single example.
Notice a crucial insight: **separability is never an intrinsic property of raw data in isolation; it is a relational property between data and a chosen geometric representation $\phi$.** The same set of inputs (such as the XOR problem) may be hopelessly inseparable in raw input coordinates, yet become effortlessly linearly separable when projected into a transformed representation space $\phi$.

When probabilistic predictions are required, binary classification employs a **logistic (sigmoid)** read-out, while multi-class tasks (such as predicting the next token across vocabulary $V$) employ a **softmax** read-out:

$$
q(y\mid x) = \frac{\exp(w_y^{\top}\phi(x) + b_y)}{\sum_{y'}\exp(w_{y'}^{\top}\phi(x) + b_{y'})}.
$$

### Why is softmax called a "linear read-out" despite its exponentials?
Beginners are often puzzled: the softmax operation involves non-linear exponential activations and a normalizer; why is it classified as a linear model?
Examine the ratio of predicted probabilities between any two candidate classes $y$ and $y'$. Their common normalizing denominator cancels out identically:

$$
\ln\frac{q(y\mid x)}{q(y'\mid x)}
=(w_y-w_{y'})^{\top}\phi(x)+(b_y-b_{y'}).
$$

Taking the natural logarithm reveals that **the pairwise log-odds (logit difference) between any two classes is an exact affine linear function of the representation vector $\phi(x)$!**
This gives a precise geometric formulation to the statement "the task is simple in this representation": a superior representation $\phi(x)$ untangles the data manifold so that the log-odds hyperplanes required by the task align linearly with the coordinates of the representation space.

## 2.3.2 The embedding lookup is a matrix product

Let $E \in \mathbb{R}^{V\times d}$ denote the embedding table, whose $a$-th row stores the continuous $d$-dimensional semantic coordinate vector of vocabulary symbol $a$.
To avoid notational ambiguity, let $\mathbf{u}_a \in \{0, 1\}^V$ denote the standard one-hot basis vector for symbol $a$, and let $e_a = E[a] \in \mathbb{R}^d$ denote its dense embedding vector. The lookup operation can be written strictly as a matrix product:

$$
E^{\top}\mathbf u_a=e_a \qquad(\text{row }a\text{ of }E,\text{ written as a column}).
$$

For a batch of $N$ input indices $x_1, \dots, x_N$, stack their one-hot vectors into row matrix $U \in \{0, 1\}^{N\times V}$. The batched lookup across all inputs is algebraically identical to a standard matrix multiplication:

$$
H=UE\in\mathbb R^{N\times d}.
$$

Shape check: $(N\times V) \times (V\times d) = N\times d$. In high-performance GPU kernels, the system performs direct memory pointer indexing (`E[x]`), never allocating the sparse matrix $U$. However, formulating the lookup as a matrix product makes backpropagation mathematically transparent.

**Proposition 2.1 (gradient of an embedding lookup).** If scalar loss $L$ depends on embedding matrix $E$ only through batch representation $H = UE$, and the upstream gradient arriving at $H$ is $G = \partial L/\partial H \in \mathbb{R}^{N\times d}$, then the gradient with respect to embedding table $E$ is:

$$
\frac{\partial L}{\partial E} = U^{\top}G,\qquad\text{i.e.}\qquad \frac{\partial L}{\partial E[a]} = \sum_{i:\,x_i = a} G[i].
$$

Before following the derivation, grasp its physical meaning: whenever symbol $a$ is referenced at multiple positions within a batch, each occurrence looks up the identical parameter row $E[a]$. By the multivariate chain rule, **all gradient contributions flowing back from every position referencing that symbol must be routed back and accumulated into that single row!**

*Proof.* By definition, $H_{ik} = \sum_{v=1}^V U_{iv} E_{vk}$, so $\partial H_{ik}/\partial E_{vk} = U_{iv}$.
Applying the chain rule across all downstream dependencies:
$\frac{\partial L}{\partial E_{vk}} = \sum_{i=1}^N \frac{\partial L}{\partial H_{ik}} \frac{\partial H_{ik}}{\partial E_{vk}} = \sum_{i=1}^N G_{ik} U_{iv} = \sum_{i=1}^N (U^{\top})_{vi} G_{ik} = (U^{\top}G)_{vk}$.
Looking at row $a$: since $U_{ia} = 1$ if and only if $x_i = a$ (and 0 otherwise), row $a$ of $U^{\top}G$ sums exactly the rows of $G$ at indices where symbol $a$ occurred. $\blacksquare$

### Engineering pitfall: scatter-add and concurrent race conditions
In deep learning frameworks, this accumulation is performed via **scatter-add**: `np.add.at(dE, x, G)` or `dE.index_add_(0, x, G)`.
A classic implementation error is using plain indexing assignment: `dE[x] = G`. If a symbol appears multiple times in a single batch, plain assignment causes subsequent occurrences to overwrite earlier gradients, leading to severe gradient loss. On parallel GPU architectures, concurrent writes to the same memory row $E[a]$ require hardware-level atomic addition (`atomicAdd`) to eliminate race conditions.

## 2.3.3 The factorized bigram and its gradients

We now reformulate Unit 01's bigram model using continuous vector representations. The transition score for antecedent $a$ followed by successor $b$ is computed as an unnormalized logit $z_{ab}$:

$$
z_{ab} = e_a^{\top}Me_b + c_b,\qquad q(b\mid a) = \mathrm{softmax}_b(z_{a\cdot}),
$$

This bilinear formulation encapsulates profound geometric design principles:
1. **Weight tying:** The same embedding table $E \in \mathbb{R}^{V\times d}$ is reused on both sides—$e_a = E[a]$ represents the observed input entity, while $e_b = E[b]$ represents the candidate prediction;
2. **The transition operator $M \in \mathbb{R}^{d\times d}$:** If $M$ were omitted ($M = I$), the score would collapse to a pure inner product $e_a^\top e_b + c_b$. Maximizing transition probabilities would force frequently adjacent tokens (such as `q` and `u`) to have identical embeddings, destroying semantic distinctions. **Matrix $M$ acts as a geometric rotation and shearing operator**: it transforms current token vector $e_a$ into the expected feature vector for the successor, $M^\top e_a$, which is then matched against candidate vectors $e_b$;
3. **Next-token bias $c_b$:** Captures the marginal unigram frequency of token $b$.

For a batch of inputs with looked-up representations $H = E[x] \in \mathbb{R}^{N\times d}$, the logit matrix across all $V$ candidate tokens is:

$$
Z=HME^{\top}+\mathbf1c^{\top}\in\mathbb R^{N\times V}.
$$

Shape check: $(N\times d) \times (d\times d) \times (d\times V) + (N\times 1)(1\times V) = N\times V$.
The interaction term $HME^{\top}$ has algebraic rank at most $d$. We have replaced a discrete $V^2$ lookup table with a compact continuous geometric system containing only $Vd + d^2 + V$ parameters!

Let $\Delta = \partial L/\partial Z = (Q - Y)/N \in \mathbb{R}^{N\times V}$ denote the logit error matrix from cross-entropy (derived in Unit 03, §3.3.5). Differentiating $Z = HME^{\top} + \mathbf{1}c^{\top}$ with respect to each parameter yields:

$$
\frac{\partial L}{\partial c} = \Delta^{\top}\mathbf 1,\qquad
\frac{\partial L}{\partial M} = H^{\top}\Delta E,\qquad
\frac{\partial L}{\partial E}\Big|_{\text{output}} = \Delta^{\top}HM,\qquad
\frac{\partial L}{\partial H} = \Delta EM^{\top}.
$$

### Dual-route gradient aggregation
Because embedding table $E$ is tied between input and output, it receives backpropagated gradients along two distinct computational paths:
- **Output route:** Every candidate vector in $E^\top$ participates in scoring, contributing $\Delta^\top HM$;
- **Input route:** Observed tokens form $H$; the gradient arriving at $H$ is $\Delta EM^\top$, which Proposition 2.1 scatters back via $U^\top$.

The two routes aggregate additively:

$$
\frac{\partial L}{\partial E} = \Delta^{\top}HM + U^{\top}(\Delta EM^{\top}).
$$

Each equation follows the universal matrix calculus rule: "for $Y = AXB$, $\partial L/\partial X = A^\top (\partial L/\partial Y) B^\top$". Shape verification confirms: $H^\top \Delta E$ has dimension $(d\times N) \times (N\times V) \times (V\times d) = d\times d$, matching $M$ exactly.

## 2.3.4 The neural n-gram

To process an extended context of $k$ preceding tokens $(x_{t-k}, \dots, x_{t-1})$, the classic Bengio architecture ([Bengio et al. 2003](/references/#bengio-2003)) concatenates their looked-up embeddings into a vector $u = [e_{x_{t-k}}; \dots; e_{x_{t-1}}] \in \mathbb{R}^{kd}$, synthesizing features through a multi-layer perceptron (MLP):

$$
a = W_1^{\top}u + b_1,\quad h = \tanh(a),\quad z = W_2^{\top}h + b_2,\quad q = \mathrm{softmax}(z),
$$

where $W_1 \in \mathbb{R}^{kd\times H}$ is the projection weight and $W_2 \in \mathbb{R}^{H\times V}$ is the output classifier.
The total parameter count is $Vd + kdH + H + HV + V$.
For typical dimensions $V = 50, d = 16, k = 4, H = 128$:
Total parameters = $800 + 8\,192 + 128 + 6\,400 + 50 = \mathbf{15\,570}$.
In contrast, a discrete count table for the same context length would require $V^k = 50^4 = \mathbf{6\,250\,000}$ rows and over 312 million cells! The neural network compresses parameter volume by a factor of 20,000, overcoming the combinatorial cliff.

For a single example with logit error $\delta_z = q - \mathbf{u}_y \in \mathbb{R}^V$, the backward pass traverses the forward graph in reverse order:

$$
\frac{\partial L}{\partial W_2} = h\,\delta_z^{\top},\quad \frac{\partial L}{\partial b_2} = \delta_z,\quad
\delta_a = (W_2\delta_z)\odot(1 - h\odot h),\quad
\frac{\partial L}{\partial W_1} = u\,\delta_a^{\top},\quad
\frac{\partial L}{\partial u} = W_1\delta_a .
$$

where $\odot$ denotes elementwise Hadamard multiplication, and $1 - h\odot h$ is the exact derivative of the hyperbolic tangent $\tanh'(a) = 1 - \tanh^2(a)$.
Finally, slice $\partial L/\partial u \in \mathbb{R}^{kd}$ into $k$ segments of length $d$. The $j$-th segment belongs to the token looked up at position $j$, which is scatter-added back into that symbol's embedding row.

## 2.3.5 Best low-rank approximation

When can a low-rank matrix approximation accurately compress a massive discrete table? Singular value decomposition (SVD) provides the exact mathematical diagnostic.

Any real matrix $A \in \mathbb{R}^{m\times n}$ of rank $r$ can be decomposed into an orthogonal sum of rank-one matrices:

$$
A=\sum_{i=1}^{r}\sigma_i u_i v_i^{\top},
$$

where left singular vectors $\{u_i\}$ and right singular vectors $\{v_i\}$ form orthonormal bases, with singular values ordered monotonically: $\sigma_1 \ge \sigma_2 \ge \dots \ge \sigma_r > 0$.

**Theorem 2.1 (Eckart–Young, [Eckart & Young 1936](/references/#eckart-1936)).** For any positive integer $d < r$, the truncated SVD matrix $A_d = \sum_{i=1}^{d}\sigma_iu_iv_i^{\top}$ is the unique minimizer of the Frobenius distance $\lVert A - B\rVert_F$ over all matrices $B$ of rank at most $d$, with minimal error:

$$\min_{\mathrm{rank}(B)\le d} \lVert A - B\rVert_F = \lVert A - A_d \rVert_F = \sqrt{\sum_{i=d+1}^{r}\sigma_i^2}.$$

(Proof in §2.5.5 ★.)
If the singular values of a co-occurrence matrix decay rapidly, Theorem 2.1 guarantees that low-rank truncation discards negligible information. Classical latent semantic analysis directly computed this truncation ([Deerwester et al. 1990](/references/#deerwester-1990)), while modern Word2Vec algorithms implicitly factorize shifted pointwise mutual information matrices at optimality ([Levy & Goldberg 2014](/references/#levy-2014)).

## 2.3.6 Invariance and equivariance

The second fundamental mechanism by which representations empower deep learning is **actively filtering out task-irrelevant variations**. Let $g$ denote a group transformation acting on input space (e.g., spatial translation, rotation, or permutation):

- **Invariance:** A representation $\phi$ is **invariant** to $g$ if $\phi(g\cdot x) = \phi(x)$. Transforming the input leaves the representation unchanged.
- **Equivariance:** A mapping $f$ is **equivariant** to $g$ if $f(g\cdot x) = g\cdot f(x)$. Transforming the input transforms the output by the exact corresponding transformation.

The choice between invariance and equivariance depends on task semantics:
- For image classification, shifting an object across the frame should not alter the category label; thus the classification head demands **translation invariance**;
- For object detection or semantic segmentation, moving the object requires the bounding box or segmentation mask to move synchronously; thus the feature backbone demands **translation equivariance** (Theorem 4.2);
- In Unit 04, we show that standard self-attention without positional encodings is strictly **permutation equivariant** (Proposition 4.2).

Remember the foundational trade-off of geometric deep learning (§2.7.1): **a representation that discards task-critical information cannot be rescued by any downstream learner.** Invariance is a deliberate architectural bet: making a digit recognizer invariant to small translations boosts generalization, but making it invariant to 180-degree rotations destroys the ability to distinguish `6` from `9`!
