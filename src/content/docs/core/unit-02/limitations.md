---
title: "2.7 Limitations and counterexamples"
description: Representations that discard needed information, static embeddings and context-dependent meaning, learned bias, fragile geometry, superposition, and the missing closed form that motivates Unit 03.
sidebar: { label: "2.7 Limitations", order: 7 }
---

Labels: **[D]** derivation, **[E]** experimental evidence, **[O]** engineering observation, **[H]** open hypothesis.

## 2.7.1 A representation can throw away what the task needs

No learner can recover information its input representation has discarded **[D]**: if $\phi(x) = \phi(x')$ but the correct outputs differ, every function of $\phi$ errs on one of them. *Constructed example:* represent a sentence by the sum of its word embeddings (a bag of words, invariant to order by Proposition 2.2). "The dog bit the man" and "the man bit the dog" have the same representation, so no read-out can say who bit whom. *Spine example:* the factorized bigram represents the context by one character; "Now Ben has " and "Ben had 3 " both end in a space, so it predicts the same distribution after both and cannot know the answer to the arithmetic. Invariance is a bet that the discarded information is irrelevant; when the bet is wrong, the error is permanent.

## 2.7.2 One vector per symbol cannot capture context-dependent meaning

A static embedding gives "bank" one vector whether the sentence is about rivers or money **[D]**. Character embeddings face the same problem in miniature: `.` ends sentences and appears in numbers in other corpora. Any representation that is a function of the symbol alone must average over its uses. The fix is a representation that depends on the surrounding context — **contextual** representations, computed by the attention and recurrence of Units 04–05 **[E]**.

## 2.7.3 Embeddings learn the regularities of their data, including the unwanted ones

Embeddings trained on human text encode associations present in that text, including social stereotypes: [Bolukbasi et al. (2016)](/references/#bolukbasi-2016) found occupational analogies such as "man : computer programmer :: woman : homemaker" in word2vec trained on news **[E]**. The learning rule is doing exactly what §2.2.5 says it should — making interchangeable-in-the-data things similar. Whether a regularity of the data is acceptable in a deployed model is a decision about the task, not about the representation; Unit 10 treats it as an evaluation question.

## 2.7.4 Similarity in embedding space is not similarity in the world

Cosine similarity measures interchangeability *for the training objective* **[D]**. `3` and `4` are nearly identical in Worked Example 2.5 because the corpus uses them in the same slots — but they are different numbers, and an arithmetic task needs them apart. Learned spaces are also often **anisotropic**: most vectors point into a narrow cone, so raw cosines are uniformly high and nearest-neighbour lists are dominated by frequency **[O]**. Analogy arithmetic works for some relations and fails for many others, and reported successes depended on evaluation choices such as excluding the query words from the candidates **[E]**.

## 2.7.5 More features than dimensions

The Johnson–Lindenstrauss corollary (§2.5.3) says many nearly orthogonal directions fit in a small space. Networks appear to use this: in toy models, they store more sparse features than dimensions, each feature smeared across many coordinates ([Elhage et al. 2022](/references/#elhage-2022)) **[E]**. The consequence for interpretation is that individual coordinates of a representation need not mean anything; directions might **[H]**. Unit 11 develops the tools for finding them.

## 2.7.6 The limitation that motivates Unit 03: no closed form

The count table of Unit 01 had a closed-form fit: count and normalize (Proposition 1.1). The factorized bigram and the neural n-gram do not **[D]**. Their losses are non-convex in the parameters — swapping two hidden units, or rotating $E$ by an orthogonal matrix while counter-rotating $M$, leaves the loss unchanged, so the minimizers come in families, and the loss surface between them need not be bowl-shaped. Every number in Worked Examples 2.4–2.6 was obtained by *searching*: repeatedly computing the gradient of the loss and taking a small step. How to compute that gradient for any model, at a cost comparable to evaluating the model, and when such a search is guaranteed to make progress, is the question of [Unit 03](/core/unit-03/).
