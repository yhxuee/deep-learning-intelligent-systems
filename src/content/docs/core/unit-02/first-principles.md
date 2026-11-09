---
title: "2.2 First principles"
description: Features and linear read-outs, XOR, one-hot codes as count tables, embeddings as learned shared coordinates, the distributional hypothesis, tied input and output embeddings, and longer contexts without exponential tables.
sidebar: { label: "2.2 First principles", order: 2 }
---

The core mission of deep learning is representation learning. Objects in the physical and symbolic world—characters, spoken syllables, pixel arrays—possess no native linear algebraic structure. The words "cat" and "dog" are discrete symbols; they cannot be added, scaled, or differentiated. The first step of machine learning is to map these discrete entities into a continuous geometric vector space where distances, angles, and algebraic transformations carry operational predictive meaning.

## 2.2.1 A representation is a choice of coordinates

A machine never receives abstract concepts in the wild; it operates exclusively on arrays of real numbers. Formally, a **representation** is a mapping:

$$
\phi:x\longmapsto\phi(x)\in\mathbb R^d
$$

that transforms an arbitrary raw input $x$ into a $d$-dimensional feature vector in Euclidean space $\mathbb R^d$. Here $x$ may represent a discrete text token, a visual patch, or an audio waveform; $\phi(x)$ constitutes the actual numerical coordinate vector read by subsequent computation; and $d$ denotes the representation dimension. This mapping may be handcrafted via domain engineering or learned end-to-end entirely from data via differentiable parameters.

To isolate what a representation uniquely contributes, we hold the downstream decision rule fixed. We begin with the simplest and most transparent decision mechanism: a **linear read-out**. For binary classification, a linear read-out computes an inner product followed by a scalar bias shift:

$$
s(x)=w^{\top}\phi(x)+b
$$

and predicts class 1 whenever the score crosses zero ($s(x) > 0$). In multi-class classification, the readout assigns an independent weight vector to each class, producing a vector of logits that can be mapped to probabilities via the Softmax operator. Geometrically, a linear read-out is constrained to draw flat hyperplanes—straight lines in $\mathbb R^2$, planes in $\mathbb R^3$, and affine hyperplanes in $\mathbb R^d$.

This yields the foundational axiom of deep learning:

> Under a linear read-out, learning is mathematically straightforward whenever the predictive target is linearly separable in the chosen representation space.

This insight explicitly focuses on the **coupled pair** of representation and read-out. An isolated one-hot encoding paired with an unrestricted linear table behaves identically to a memorized lookup table; but when the same input is mapped into an expressive continuous latent space, a simple linear hyperplane achieves robust generalization. The essence of deep learning is not to invent increasingly tortuous decision surfaces, but to discover coordinates $\phi$ in which complex real-world relationships become linearly untangled.

## 2.2.2 XOR: impossible, then easy

To see concretely how coordinates govern learnability, consider the classical Exclusive-OR (XOR) problem. Take four inputs with two binary features, where the target label is 1 if and only if exactly one coordinate equals 1:

| $x_1$ | $x_2$ | Target Label $y$ |
|---|---|---|
| 0 | 0 | 0 |
| 0 | 1 | 1 |
| 1 | 0 | 1 |
| 1 | 1 | 0 |

Using raw coordinates $\phi(x)=(x_1,x_2)$, suppose a linear classifier predicts 1 when $w_1x_1+w_2x_2+b>0$ and 0 otherwise. Reading the four truth table rows establishes four simultaneous linear inequalities:

$$
b\le0,\qquad w_2+b>0,\qquad w_1+b>0,\qquad w_1+w_2+b\le0.
$$

Summing the two strict middle inequalities yields $w_1+w_2+2b>0$, which rearranges to $w_1+w_2+b > -b$. Because the first condition requires $b \le 0 \implies -b \ge 0$, we obtain $w_1+w_2+b > 0$. This directly contradicts the fourth requirement $w_1+w_2+b \le 0$!

This algebraic contradiction proves that **no line in $\mathbb R^2$ can separate XOR**. [Minsky & Papert (1969)](/references/#minsky-1969) proved this fundamental limitation of single-layer perceptrons, triggering the first historical AI winter.

Now lift the feature representation into three dimensions by adding a nonlinear interaction coordinate: $\phi(x)=(x_1, x_2, x_1x_2)$. Assign weights $w=(1, 1, -2)$ with bias $b=-0.5$. Evaluating the linear readout across the four points gives:
- For $(0,0)$: $s = 1(0) + 1(0) - 2(0) - 0.5 = -0.5 \le 0$ (class 0)
- For $(0,1)$: $s = 1(0) + 1(1) - 2(0) - 0.5 = +0.5 > 0$ (class 1)
- For $(1,0)$: $s = 1(1) + 1(0) - 2(0) - 0.5 = +0.5 > 0$ (class 1)
- For $(1,1)$: $s = 1(1) + 1(1) - 2(1) - 0.5 = -0.5 \le 0$ (class 0)

The exact same linear hyperplane now separates XOR without error (Worked Example 2.1)! The new coordinate $x_1x_2$ tracks whether both inputs are simultaneously active, geometrically lifting $(1,1)$ along the third axis so that a flat plane can slice cleanly between classes.

Two generalized lessons follow:
1. **A good representation makes the target simple—ideally linear—in its coordinates.**
2. **Hidden layers in deep networks automate this feature lifting.** In this toy example, we handcrafted the interaction $x_1x_2$. Deep learning replaces manual feature engineering with optimization: gradient descent searches a parameterized space of nonlinear compositions to discover features that linearize downstream prediction (§2.5.1).

## 2.2.3 One-hot codes: the representation that shares nothing

Suppose our vocabulary contains $V$ discrete tokens. The most primitive encoding assigns a private orthogonal basis dimension to each symbol. Token $a$ becomes a **one-hot** vector $u_a\in\mathbb R^V$, with a 1 at index $a$ and 0 everywhere else. For a tiny alphabet of three symbols:

$$
u_1=(1,0,0),\qquad u_2=(0,1,0),\qquad u_3=(0,0,1).
$$

In Euclidean geometry, any two distinct one-hot vectors are completely orthogonal ($u_a^\top u_b = 0$) with Euclidean distance $\|u_a - u_b\|_2 = \sqrt{2}$. This geometry enforces an extreme inductive assumption: **every symbol is equally and totally unrelated to every other symbol**. The encoding cannot represent that "3" and "4" are digits, or that "cat" and "dog" are domestic animals.

Now pair this representation with an unconstrained linear readout matrix $W\in\mathbb R^{V\times V}$. Given input token $a$, the output logit vector is:

$$
z=W^{\top}u_a=W[a],
$$

where $W[a]$ denotes column $a$ of $W^\top$ (the $a$-th row of $W$). Multiplying by a one-hot vector is **computationally identical to table lookup**. When followed by a Softmax layer and trained under maximum likelihood, this reproduces the count-based bigram table of Unit 01. Updating parameters for token $a$ produces zero gradient signal for token $b$.

The fundamental conclusion is that **one-hot encodings preclude all statistical sharing across symbols.** In a one-hot representation, every symbol resides in an isolated orthogonal compartment.

## 2.2.4 Embeddings: learned shared coordinates

The solution to one-hot isolation is to compress the sparse $V$-dimensional space into a compact, continuous vector $e_a\in\mathbb R^d$, where the embedding dimension is dramatically smaller than the vocabulary size ($d\ll V$). Storing these vectors as rows in an **embedding table** $E\in\mathbb R^{V\times d}$, retrieving symbol $a$ extracts $E[a]=e_a$. These $d$ coordinates are continuous, differentiable parameters learned through gradient descent.

How do low-dimensional embeddings enable generalization?
Suppose token embeddings feed into a smooth parameterized predictor $f(e)$. If training places $e_{\texttt3}$ and $e_{\texttt4}$ in close Euclidean proximity, and $f$ possesses bounded Lipschitz curvature, their model predictions will naturally align. Crucially: **a gradient update that improves $f$ on sentences containing "3" simultaneously improves predictions on sentences containing "4"!** This phenomenon is **statistical sharing**.

This represents the foundational insight of modern neural language modeling formulated by [Bengio et al. (2003)](/references/#bengio-2003): "associate with each word in the vocabulary a distributed word feature vector... and learn simultaneously the word feature vectors and the parameters of that probability function." The word "distributed" denotes that concepts are represented across $d$ shared continuous coordinates rather than a single isolated index.

Furthermore, low-dimensional embeddings introduce a parameter bottleneck. A full bigram table stores $V^2$ independent logits (effective degrees of freedom $V(V-1)$). In contrast, factorized bilinear scoring:

$$
z_{ab}=e_a^{\top}Me_b+c_b
$$

requires only $Vd+d^2+V$ trainable parameters (§2.3.3). When $V=50$ and $d=16$, parameter count drops from 2,500 to 1,106. On the textbook spine corpus, this factorized model matches the held-out loss of the full count table to within 0.03 nats (Worked Example 2.4). The dimensional bottleneck functions as an inductive bias, preventing the network from memorizing idiosyncratic noise.

## 2.2.5 The distributional hypothesis

Why should words appearing in similar contexts share similar geometric representations?
Linguist Zellig Harris argued that semantic differences correlate with differences in distributional environments ([Harris 1954](/references/#harris-1954)); J. R. Firth famously summarized: "You shall know a word by the company it keeps." For machine learning, we rely on a practical predictive formulation:

> If two symbols are mutually interchangeable across all task-relevant contexts, the optimal predictor must treat them identically. Pooling evidence across interchangeable symbols reduces statistical estimation variance without sacrificing task-relevant information.

Importantly, geometric proximity reflects **similarity relative to the training objective**, rather than universal metaphysical truth. For example, antonyms like "hot" and "cold" frequently appear in identical syntactic frames ("the soup is too ___ to drink"), causing language model objectives to place them close in cosine similarity.

## 2.2.6 Tied embeddings: using one table for input and output

In autoregressive language modeling, every token plays a dual role:
1. As an **input context token**: informing the model of past history;
2. As a **target output token**: scored by the model's logits for next-token prediction.

With separate matrices, the input embedding for "3" shares no parameters with the readout weights that score "3". **Weight tying** ([Press & Wolf 2017](/references/#press-2017)) reuses the identical matrix $E$ for both roles, scoring candidate output $b$ via the dot product with $e_b$. This cuts parameters in half and imposes the structural inductive bias that the geometric manifold of input comprehension should align with that of output generation.

## 2.2.7 Longer contexts without exponential tables

When expanding context length, classical count tables suffer catastrophic combinatorial explosion. An $n$-gram table conditioning on $k$ prior tokens requires $V^k$ rows. For $V=50$ and $k=4$, this demands 6.25 million rows; for $k=10$, it exceeds $9.7 \times 10^{16}$ configurations, outstripping available memory.

A **neural n-gram** bypasses this exponential barrier by retrieving the $d$-dimensional embeddings of the $k$ context tokens, **concatenating** them into a single vector of length $kd$, and processing them through a multilayer perceptron:

$$
h = \tanh\big(W_1[e_{x_{t-k}};\ \dots;\ e_{x_{t-1}}] + b_1\big),\qquad z = W_2h + b_2 .
$$

Reading from inside out: the brackets concatenate $k$ vectors into $\mathbb R^{kd}$; projection $W_1$ mixes coordinates across temporal positions; $\tanh$ injects nonlinearity to form representation $h$; and $W_2$ produces logits across vocabulary $V$.
For fixed $d$, hidden width $H$, and vocabulary $V$, parameter count scales **linearly with context length** ($O(kdH)$) rather than exponentially ($O(V^k)$)!

Contexts such as `ad 3 ` and `ad 4 ` map to adjacent locations in $\mathbb R^{kd}$. The continuous smoothness of the neural network transfers statistical strength between similar histories, allowing neural models to generalize across contexts (Worked Example 2.6).

## 2.2.8 Summary of representation paradigms

| Representation Architecture | Sharing Mechanism | Parameters for $k$-token Context | Spine Benchmark Performance |
|---|---|---|---|
| One-hot + Unconstrained Table | Zero sharing across tokens and contexts | Up to $V^{k+1}$ logits | Bigram NLL 2.12 nats (Unit 01) |
| Continuous Embeddings + Factorized Bigram | Manifold sharing across similar tokens | $Vd + d^2 + V$ | NLL 2.15 nats at $d = 16$ (1,106 parameters) |
| Continuous Embeddings + Neural $n$-gram | Shared representations across tokens & contexts | $Vd + kdH + H + HV + V$ | NLL 0.94 vs Count-table 1.07 ($k=4$, 5k tokens) |

The unifying thread of this unit is established: **the geometry of the representation space and the functional class operating upon it collectively dictate which statistical evidence is shared, and which distinctions are preserved.**
