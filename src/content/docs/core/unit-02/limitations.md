---
title: "2.7 Limitations and counterexamples"
description: Representations that discard needed information, static embeddings and context-dependent meaning, learned bias, fragile geometry, superposition, and the missing closed form that motivates Unit 03.
sidebar: { label: "2.7 Limitations", order: 7 }
---

Labels: **[D]** derivation, **[E]** experimental evidence, **[O]** engineering observation, **[H]** open hypothesis.

Representation learning is powerful precisely because it keeps some distinctions and suppresses others. That compression creates failure modes. The useful question is therefore not “is this representation good?” in isolation, but “which distinctions does it preserve, for which task and data?”

## 2.7.1 A representation can throw away what the task needs

Suppose a downstream predictor has the form $g(\phi(x))$. If $\phi(x)=\phi(x')$, then it must produce the same output for both inputs. Therefore, if their correct outputs differ, **no choice of $g$ can be correct on both** **[D]**. Training longer or enlarging the read-out cannot restore a distinction that never reaches it.

*Constructed example.* Sum the word embeddings in a sentence. By Proposition 2.2, this bag-of-words representation is invariant to word order. “The dog bit the man” and “the man bit the dog” then become identical, so no read-out can recover who bit whom.

*Spine example.* The factorized bigram represents its context with only the final character. “Now Ben has ” and “Ben had 3 ” both end in a space, so the model must predict the same next-character distribution after both. The arithmetic answer is present earlier in the text, but absent from the model's representation. Every invariance is thus a bet that the discarded variation is irrelevant; a wrong bet imposes an irreducible error for that architecture.

## 2.7.2 One vector per symbol cannot capture context-dependent meaning

A **static embedding** is a lookup: the token identity selects one vector. It therefore gives “bank” the same input vector in “river bank” and “bank account” **[D]**. Character embeddings face the same issue at a smaller scale: `.` can end a sentence or separate digits. Training may place a static vector at a useful compromise among these uses, but the lookup itself cannot choose a different meaning from context.

Do not confuse that input lookup with the representation produced by the whole network. A **contextual representation** combines the token with its surroundings, so two occurrences of “bank” can start from the same embedding and end with different hidden states. Attention and recurrence construct such states in Units 04–05; models such as BERT provide large-scale evidence that this approach is useful ([Devlin et al. 2019](/references/#devlin-2019)) **[E]**. Context dependence mitigates the ambiguity; it does not guarantee that the correct sense will be recovered.

## 2.7.3 Embeddings learn the regularities of their data, including the unwanted ones

An embedding objective cannot automatically distinguish “useful pattern” from “historical stereotype.” It rewards whatever statistical regularities help predict its training data. [Bolukbasi et al. (2016)](/references/#bolukbasi-2016), for example, reported occupational analogies such as “man : computer programmer :: woman : homemaker” in word2vec vectors trained on news **[E]**.

This does **not** mean that every association in an embedding will appear unchanged in every output: later layers, prompts, fine-tuning data, and decision rules also matter. Nor does changing a few vectors prove that a deployed system is fair. The practical lesson is to evaluate the complete system on the groups and decisions that matter, trace failures back through data, objective, representation, and read-out, and test any mitigation again. Unit 10 develops that evaluation view.

## 2.7.4 Similarity in embedding space is not similarity in the world

Cosine similarity answers a geometric question: how small is the angle between two nonzero vectors? Calling that angle “semantic similarity” is an interpretation supplied by the training objective and data, not a mathematical consequence **[D]**. In Worked Example 2.5, `3` and `4` are close because the corpus places them in similar character contexts. For arithmetic, however, their numerical difference is essential.

The measurement itself can also mislead. Many learned spaces are **anisotropic**: vectors concentrate in a narrow region, making unrelated pairs have high raw cosine; frequency can then dominate nearest-neighbour lists **[O]**. Reported word-vector analogies succeed for some relations and fail for others ([Mikolov et al. 2013](/references/#mikolov-2013); [Bolukbasi et al. 2016](/references/#bolukbasi-2016)) **[E]**. Their measured accuracy also depends on evaluation choices such as normalization, candidate vocabulary and whether query words are excluded. Always validate a similarity measure against the downstream relation of interest rather than reading meaning directly from a plot.

## 2.7.5 More features than dimensions

The Johnson–Lindenstrauss discussion (§2.5.3) establishes a geometric possibility: a modest-dimensional space can contain many directions with small pairwise interference. It does not prove that trained networks use those directions as features. Controlled toy models nevertheless show a related phenomenon: when features are sparse, a model can encode more features than it has dimensions by distributing each feature across coordinates ([Elhage et al. 2022](/references/#elhage-2022)) **[E]**.

This is **superposition**. Its interpretability cost is immediate: inspecting coordinate 17 may reveal a mixture rather than a human concept. Useful variables may instead correspond to directions, possibly drawn from an overcomplete dictionary **[H]**. Even then, finding a direction that correlates with a concept does not by itself show that the model causally uses it. Unit 11 separates these claims.

## 2.7.6 The limitation that motivates Unit 03: no closed form

The count table of Unit 01 has a direct maximum-likelihood solution: count transitions and normalize each row (Proposition 1.1). The factorized bigram and neural n-gram instead define the answer implicitly as parameters that minimize a loss; in general, there is no count-and-normalize formula for those parameters **[D]**.

Two issues are easy to conflate:

1. **Non-identifiability.** Different parameter settings can represent the same function. Hidden units can be permuted. In compatible factorizations, an embedding basis can be changed while the next linear map is changed inversely. Thus “the learned parameters” need not be unique.
2. **Non-convex optimization.** The neural loss is not generally bowl-shaped in all parameters jointly. A zero gradient may be a minimum, a maximum, or a saddle; reaching a low training loss need not mean reaching the global minimum.

These statements are related but neither one alone proves the other. What the worked examples actually did was numerical search: evaluate a loss, compute its gradient, update the parameters, and repeat. [Unit 03](/core/unit-03/) explains how reverse-mode differentiation computes those gradients at a cost comparable to a small multiple of the forward pass, why sufficiently small steps decrease a smooth loss locally, and why that local guarantee is not a guarantee of finding the best solution.
