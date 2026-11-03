---
title: "2.10 Problems and further study"
description: Twenty graded problems on separability, embeddings and their gradients, low-rank structure, random projections and symmetry, with selected answers and an annotated reading list.
sidebar: { label: "2.10 Problems", order: 10 }
---

Difficulty: ★ routine · ★★ combines ideas · ★★★ proof or open-ended. **[AI-off]** problems are for independent practice and assessment.

Do not treat the stars as a speed test. For a derivation, first write every matrix shape; for a proof, state the claim and assumptions before manipulating symbols; for an experiment, define the split and success criterion before running code. The selected answers show a checkable endpoint, not every intermediate step.

Unless a problem says otherwise, a bigram logit table has rows indexed by the current symbol $a$ and columns by the next symbol $b$.

## Computation

**2.1 ★ [AI-off] Feature design.** Find a feature map $\phi:\{0,1\}^2\to\mathbb{R}^2$, different from Worked Example 2.1, under which XOR is linearly separable. List the four transformed points and give an explicit affine separating line. Verify all four inequalities; a drawing alone is not enough.

**2.2 ★ [AI-off] Repeated lookup.** Redo Worked Example 2.7 with $x=(1,1,1)$ and the same upstream-gradient matrix $G$. Compute the correct scatter-add result row by row, then execute the buggy assignment `dE[x] = G` mentally. Which contribution survives, and why is ordinary indexed assignment not a derivative here?

**2.3 ★ Parameter counting.** For a neural 8-gram with $V=256$ byte symbols, embedding width $d=32$, and hidden width $H=512$, count each of $E,W_1,b_1,W_2,b_2$ separately. Then compute the maximum number $V^8$ of context rows in a count 8-gram. Compare stored scalar parameters, not the number of possible complete sequences.

**2.4 ★ Cover counting.** For $N=10$ general-position points in $\mathbb{R}^3$, use the *homogeneous* formula in Theorem 2.3 to compute $C(N,p)$. Divide by $2^{10}$ to obtain the fraction of all binary labelings that are separable. Do not append the affine bias coordinate in this problem.

**2.5 ★★ A sufficient JL dimension.** Insert $n=10{,}000$ and, separately, $\varepsilon=0.1$ and $0.01$ into the *displayed sufficient condition* of Theorem 2.4. Because the theorem uses a strict inequality, return the smallest integer strictly above the bound. Explain why these numbers are sufficient upper bounds, not necessary dimensions.

**2.6 ★★ Capacity of a linear embedding model.** Let the logit table be $Z=EW+\mathbf1c^\top$, where $E\in\mathbb R^{V\times d}$ and $W\in\mathbb R^{d\times V}$. First construct $E,W,c$ that represent an arbitrary $V\times V$ logit table when $d\ge V$. Then set $d=1$, choose a reference output $b_0$, and form

$$
L_{ab}=\log\frac{p(b\mid a)}{p(b_0\mid a)}=Z_{ab}-Z_{ab_0}.
$$

Show that $L$ is the sum of at most two rank-one matrices and therefore has rank at most 2. State why softmax row offsets do not affect this conclusion.

## Derivation

**2.7 ★ [AI-off] Lookup gradient.** Prove Proposition 2.1 component by component: write $H_{tj}=E_{x_tj}$ and differentiate with respect to $E_{ij}$. Express the result with an indicator, then explain in one sentence why a duplicated index must accumulate.

**2.8 ★★ [AI-off] Matrix chain rule.** Let $A\in\mathbb R^{m\times r}$, $X\in\mathbb R^{r\times s}$, and $B\in\mathbb R^{s\times n}$, so $Y=AXB$. Using differentials or indices, show

$$
\frac{\partial L}{\partial X}=A^\top\frac{\partial L}{\partial Y}B^\top.
$$

Then derive the four parameter gradients in §2.3.3. Annotate every final gradient with its shape and account for both routes into the tied table $E$.

**2.9 ★★ Untied then tied.** For the untied score $z_{ab}=e_a^\top M f_b+c_b$, derive gradients with respect to $E,F,M,c$ for a batch. Only after that, set $F=E$. Use the multivariable chain rule to show that the tied gradient is the sum of the two untied table gradients, rather than either gradient alone.

**2.10 ★★ Neural n-gram backward pass.** Starting from batch logits, propagate an upstream gradient through $W_2$, `tanh`, $W_1$, concatenation, and the embedding lookup of §2.3.4. Give the shape at every step. Finish with scatter-add into $E$, explicitly showing what happens when the same symbol appears twice in one context or across two batch items.

**2.11 ★★ Change of embedding basis.** For invertible $Q\in\mathbb R^{d\times d}$, substitute $E'=EQ$ and $M'=Q^{-1}MQ^{-\top}$ into $E'M'E'^\top$. Show that every logit, and hence the loss, is unchanged. Explain separately what this says about (a) uniqueness of parameters and (b) coordinate-by-coordinate comparison across training runs. Does it say that every two trained models are functionally identical?

## Proof

**2.12 ★★ [AI-off] XOR capacity.** Prove that no affine classifier separates the four XOR points by writing the four required score inequalities and deriving a contradiction. Then construct a two-hidden-unit ReLU network explicitly. One useful route is to let both hidden units depend only on $s=x_1+x_2$ and build a piecewise-linear function that is 0, 1, 0 at $s=0,1,2$.

**2.13 ★★ Invariance and its cost.** Prove Proposition 2.2 by re-indexing the group sum. Then prove directly that the sum of a sequence's embeddings is invariant to any permutation. Finally, specify a prediction task and construct two sentences with the same multiset of words but different correct outputs. Your example must show which task-relevant distinction the invariant representation discarded.

**2.14 ★★★ Shift equivariance.** Let $S$ be the one-step cyclic-shift matrix and let $A$ represent a linear map. Prove that $AS=SA$ implies every column of $A$ is a cyclic shift of its first column; conclude that $A$ is circulant and the map is circular convolution. Then show the converse. Compare the result with Unit 04, Theorem 4.2.

**2.15 ★★★ Cover's recursion.** For points in general position, justify the correspondence between dichotomies whose separating hyperplane can pass through the new point and dichotomies after projection to dimension $p-1$. This should yield

$$
C(N+1,p)=C(N,p)+C(N,p-1).
$$

Using the boundary cases $C(1,p)=2$ and $C(N,1)=2$, solve the recursion with Pascal's identity and recover Theorem 2.3.

**2.16 ★★★ Johnson–Lindenstrauss from concentration.** For fixed unit $u$, show that $d\lVert Ru\rVert^2\sim\chi_d^2$. Derive the upper and lower Chernoff tails, combine them into the bound displayed in Theorem 2.4, and apply it to all nonzero pairwise differences with a union bound. Track the factor of 2 and the number of unordered pairs; your final strict condition on $d$ must match the theorem.

## Implementation and design

**2.17 ★ [AI-off] Forward/backward implementation.** Without copying the reference, implement the factorized bigram's forward loss and analytical backward pass. First test shapes and the $M=0,c=0$ loss; then compare several randomly chosen gradient entries with centered finite differences; finally run the spine gradient check. Passing only the last check is not a substitute for explaining the two contributions to `dE`.

**2.18 ★★ Two-dimensional bottleneck.** Train the factorized bigram with $d=2$ for at least three seeds and plot all 50 characters directly in the learned plane. Compare nearest neighbors and held-out NLL with the $d=16$ model. Which relationships are stable across seeds, and which distinctions visible at $d=16$ are lost or distorted at $d=2$? Do not infer impossibility merely from overlapping labels in one plot.

**2.19 ★★ Removing order.** Replace concatenation in the neural n-gram with the *sum* of the context embeddings. Before training, prove which permutations of a fixed context now receive the same representation and predict the direction of the held-out NLL change. Train both variants with matched budgets and at least three seeds. Relate the result to §2.7.1; if the summed model ever wins, explain why that does not invalidate the information-loss argument.

**2.20 ★★★ Data or seed?** Design a two-factor experiment that varies both training-corpus sample and random initialization. Specify repeated runs, a shared evaluation set of symbols or contexts, and a representation-similarity measure that is not fooled by a rotation or change of basis (for example, aligned distances or CKA). State the statistic that compares within-data/across-seed variation with across-data variation, include uncertainty, and preregister what result would favor each conclusion. This previews §2.8.5 and Unit 12.

## Selected answers

- **2.1** $\phi(x) = (x_1 + x_2,\ x_1x_2)$ maps the points to $(0,0)$, $(1,0)$, $(1,0)$, $(2,1)$; the rule $\phi_1 - 2\phi_2 > 0.5$ separates them (scores $0, 1, 1, 0$).
- **2.2** Correct gradient: row 1 receives all three rows, $(9, 12)$; rows 0 and 2 are zero. Buggy: row 1 equals $(5, 6)$.
- **2.3** $256\cdot32 + 8\cdot32\cdot512 + 512 + 512\cdot256 + 256 = 8\,192 + 131\,072 + 512 + 131\,072 + 256 = 271\,104$ parameters; up to $256^8\approx1.8\times10^{19}$ rows.
- **2.4** $C(10, 3) = 2(1 + 9 + 36) = 92$ of $2^{10} = 1\,024$ labellings: 9.0%.
- **2.5** Theorem 2.4 requires $d>4\ln(10^4)/(\varepsilon^2/2-\varepsilon^3/3)$. The smallest integers satisfying it are $7\,895$ for $\varepsilon=0.1$ and $741\,773$ for $\varepsilon=0.01$. These are conservative sufficient dimensions; the theorem does not claim smaller dimensions fail.
- **2.6** With $d=V$, choose $E=I$, $W=Z$, and $c=0$. For $d=1$, write $E=u$ and $W=v^\top$; relative to $b_0$, $L=u(v-v_{b_0}\mathbf1)^\top+\mathbf1(c-c_{b_0}\mathbf1)^\top$, a sum of two rank-one matrices.
- **2.11** Direct substitution gives $(EQ)(Q^{-1}MQ^{-\top})(EQ)^\top=EME^\top$. Coordinates are therefore not identifiable without alignment, but this symmetry does not say independently trained runs must compute the same logits.

## Further reading

**Representation learning.** Start with §§1–2 of [Bengio, Courville & Vincent (2013)](/references/#bengio-2013) for the motivation, then use §§3–4 to connect distributed representations to depth. Read [Bengio et al. (2003)](/references/#bengio-2003) next: identify the lookup, concatenation, hidden layer, and output table in the notation of §2.3.4.

**Embeddings.** [Mikolov et al. (2013)](/references/#mikolov-2013) and [Pennington et al. (2014)](/references/#pennington-2014) introduce two influential objectives. Read [Levy & Goldberg (2014)](/references/#levy-2014) afterward to see how a predictive objective relates to factorizing a transformed co-occurrence matrix. Compare objectives and assumptions, not only nearest-neighbor examples.

**Geometry and symmetry.** Read the opening sections of [Bronstein et al. (2021)](/references/#bronstein-2021) for the distinction between invariance and equivariance before Unit 04. [Dasgupta & Gupta (2003)](/references/#dasgupta-2003) is a compact Johnson–Lindenstrauss proof; while reading, match its tail bound, pair count, and constants to Theorem 2.4 rather than memorizing only the $O(\varepsilon^{-2}\log n)$ rate.

**Inside learned representations.** [Elhage et al. (2022)](/references/#elhage-2022) studies when toy models choose superposition; separate those controlled results from claims about large networks. [Park et al. (2024)](/references/#park-2024) develops versions of the linear representation hypothesis. Record which claims are definitions, derivations, empirical observations, and open hypotheses; Unit 11 returns to that distinction.
