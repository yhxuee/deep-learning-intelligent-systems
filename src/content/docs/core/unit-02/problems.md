---
title: "2.10 Problems and further study"
description: Twenty graded problems on separability, embeddings and their gradients, low-rank structure, random projections and symmetry, with selected answers and an annotated reading list.
sidebar: { label: "2.10 Problems", order: 10 }
---

Difficulty: ★ routine · ★★ combines ideas · ★★★ proof or open-ended. **[AI-off]** problems are for independent practice and assessment.

## Computation

**2.1 ★ [AI-off]** Find a different feature map $\phi:\{0,1\}^2\to\mathbb{R}^2$ (two coordinates only) under which XOR is linearly separable, and give the separating line.

**2.2 ★ [AI-off]** Redo Worked Example 2.7 with $x = (1, 1, 1)$ and $G$ as given. What does the buggy assignment `dE[x] = G` produce?

**2.3 ★** Count the parameters of a neural 8-gram with $V = 256$ (bytes), $d = 32$, $H = 512$. How many rows could a count 8-gram need?

**2.4 ★** Compute $C(N, p)$ from Theorem 2.3 for $N = 10$ points in $\mathbb{R}^3$ (homogeneous). What fraction of all labellings is separable?

**2.5 ★★** Using Theorem 2.4, what dimension guarantees that 10 000 embeddings keep all pairwise squared distances within 10%? Within 1%?

**2.6 ★★** A one-hot model and a $d$-dimensional embedding model with a linear read-out are trained on the same bigram data. Show that for $d\ge V$ the embedding model can represent every bigram table, and for $d = 1$ it can represent only tables whose log-odds matrix has rank at most 2 (after accounting for the bias).

## Derivation

**2.7 ★ [AI-off]** Prove Proposition 2.1 (the gradient of a lookup) and explain in one sentence why a duplicated index must accumulate.

**2.8 ★★ [AI-off]** For $Y = AXB$ and scalar $L(Y)$, show $\partial L/\partial X = A^{\top}(\partial L/\partial Y)B^{\top}$. Use it to derive all four gradients of §2.3.3.

**2.9 ★★** Derive the gradients of the *untied* factorized bigram $z_{ab} = e_a^{\top}Mf_b + c_b$. Show that the tied gradient of §2.3.3 equals the sum of the untied gradients for $E$ and $F$ evaluated at $F = E$.

**2.10 ★★** Derive the backward pass of the neural n-gram (§2.3.4) for a batch, including the case where the same symbol occurs twice in one context.

**2.11 ★★** Show that the loss of the factorized bigram is unchanged under $E\mapsto EQ$, $M\mapsto Q^{-1}MQ^{-\top}$ for any invertible $Q$. What does this imply about the uniqueness of learned embeddings, and about comparing two runs' embeddings coordinate by coordinate?

## Proof

**2.12 ★★ [AI-off]** Prove that no affine classifier separates XOR (Worked Example 2.1) and that a two-layer ReLU network with two hidden units can.

**2.13 ★★** Prove Proposition 2.2 and show that the sum of embeddings of a sequence is invariant to permutations. Construct two different sentences with the same bag-of-words representation and different correct outputs for a task of your choice.

**2.14 ★★★** Prove that every linear map $\mathbb{R}^T\to\mathbb{R}^T$ that commutes with all cyclic shifts is a circular convolution. (Compare with Unit 04, Theorem 4.2.)

**2.15 ★★★** Complete the proof of Cover's recursion $C(N+1, p) = C(N, p) + C(N, p-1)$ and solve it.

**2.16 ★★★** Prove the Johnson–Lindenstrauss lemma with the constants of Theorem 2.4, including the Chernoff bound for $\chi^2_d$.

## Implementation and design

**2.17 ★ [AI-off]** Implement the factorized bigram's forward and backward pass from memory and pass the spine's gradient check.

**2.18 ★★** Train the factorized bigram with $d = 2$ and plot all 50 characters in the plane. Which groups form? Which characters are placed badly, and why can two dimensions not separate them?

**2.19 ★★** Replace the concatenation in the neural n-gram by the *sum* of the context embeddings. Predict, then measure, the effect on held-out loss. Relate it to §2.7.1.

**2.20 ★★★** Design an experiment that tests whether the spine model's embedding geometry depends more on the data or on the random seed: what would you compare, with what similarity measure, and what result would favour each answer? (A preview of §2.8.5 and Unit 12.)

## Selected answers

- **2.1** $\phi(x) = (x_1 + x_2,\ x_1x_2)$ maps the points to $(0,0)$, $(1,0)$, $(1,0)$, $(2,1)$; the rule $\phi_1 - 2\phi_2 > 0.5$ separates them (scores $0, 1, 1, 0$).
- **2.2** Correct gradient: row 1 receives all three rows, $(9, 12)$; rows 0 and 2 are zero. Buggy: row 1 equals $(5, 6)$.
- **2.3** $256\cdot32 + 8\cdot32\cdot512 + 512 + 512\cdot256 + 256 = 8\,192 + 131\,072 + 512 + 131\,072 + 256 = 271\,104$ parameters; up to $256^8\approx1.8\times10^{19}$ rows.
- **2.4** $C(10, 3) = 2(1 + 9 + 36) = 92$ of $2^{10} = 1\,024$ labellings: 9.0%.
- **2.5** $d\ge 8\ln(10^4)/\varepsilon^2$: $\varepsilon = 0.1$ gives $d\ge 7\,369$; $\varepsilon = 0.01$ gives $d\ge 736\,828$. (The bound is loose; in practice far fewer dimensions suffice.)

## Further reading

**Representation learning.** [Bengio, Courville & Vincent 2013](/references/#bengio-2013) — the review that framed the field; read §§1–4. [Bengio et al. 2003](/references/#bengio-2003) — the neural language model, still clear and short.

**Embeddings.** [Mikolov et al. 2013](/references/#mikolov-2013), [Pennington et al. 2014](/references/#pennington-2014), and [Levy & Goldberg 2014](/references/#levy-2014) — the last explains what the first two compute.

**Geometry and symmetry.** [Bronstein et al. 2021](/references/#bronstein-2021) — invariance and equivariance as the organizing principle of architectures; read the first two chapters before Unit 04. [Dasgupta & Gupta 2003](/references/#dasgupta-2003) — a four-page proof of Johnson–Lindenstrauss.

**Inside learned representations.** [Elhage et al. 2022](/references/#elhage-2022) and [Park et al. 2024](/references/#park-2024) — superposition and linear structure, the starting point for Unit 11.
