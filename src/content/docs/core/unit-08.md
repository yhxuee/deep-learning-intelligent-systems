---
title: "Unit 08 · How can models generate new data?"
description: "Latent variables, VAEs, autoregressive factorization, diffusion, score-based learning and flow matching."
sidebar: { label: "08 · Generation", order: 8 }
unit:
  number: 8
  part: 3
  question: "How can a learned distribution be sampled or transformed?"
  anchor: "Flow matching on a two-dimensional toy distribution, compared with autoregressive sampling."
  spine: "Side branch — compare greedy, temperature and nucleus sampling from the spine model; a 2-D flow-matching model as the second exemplar."
  checkpoint: "Compare two generative mechanisms in terms of objectives, sampling procedure and approximation error."
  status: outline
  atlas: []
---

:::caution[Planning outline — not yet textbook text]
This page records the unit's question, anchor exemplar, outcomes and activities. It is a syllabus, not a chapter: it states topics without teaching them. It will be rewritten into the ten-part structure of the [writing standard](/project/writing-standard/), as Units 03–05 have been (see the [roadmap](/project/roadmap/)). Contributions welcome.
:::

## The problem

Two models assign the same likelihood to a test set. One produces visibly better samples. How is that possible?

:::note[Predict before reading on]
Write your answer and the assumptions it depends on before continuing.
:::

## Core content

- Three ways to represent a distribution: factorize it (autoregressive), introduce latents (VAE [Kingma & Welling 2014](/references/#kingma-2014)), or learn a transport from noise.
- Diffusion and score-based models [Ho et al. 2020](/references/#ho-2020); [Song et al. 2021](/references/#song-2021): learning to reverse a noising process.
- **Flow matching** [Lipman et al. 2023](/references/#lipman-2023): regress a velocity field along simple probability paths — the anchor because its objective fits on one page and is easy to verify on toy data.
- Sampling as a design choice: temperature, nucleus sampling, number of integration steps; the quality–diversity–compute trade-off.
- Genealogy: Boltzmann machines and GANs as historical branches; score matching as the bridge to modern diffusion.

## Learning outcomes by ownership tier

| Tier | Outcomes |
|---|---|
| <span class="tier tier-internalize">Internalize</span> | autoregressive factorization; the ELBO; the flow-matching objective |
| <span class="tier tier-verify">Understand & verify</span> | diffusion noise schedules; guidance; evaluation metrics for samples |
| <span class="tier tier-access">Access on demand</span> | specific image/video model architectures and samplers |

## Learning activities

**Construction (AI-off).** Implement conditional flow matching for a 2-D mixture of Gaussians in under 80 lines; verify the learned velocity field transports samples to the target (moment and two-sample tests).

**Challenge (Critic mode).** An AI states that "diffusion models are strictly better than autoregressive models." List the dimensions on which that claim could be true or false.

**Experiment (Collaborator mode).** Vary the number of integration steps; predict how sample quality and compute trade off; measure with a two-sample statistic.

**Spine increment.** Side branch — compare greedy, temperature and nucleus sampling from the spine model; a 2-D flow-matching model as the second exemplar.

**Transfer & defense (AI-off).** Given a new data type (e.g. molecular graphs), choose a generative mechanism and justify it in terms of symmetry, likelihood and sampling cost.

## Key sources

[kingma-2014](/references/#kingma-2014) · [ho-2020](/references/#ho-2020) · [song-2021](/references/#song-2021) · [lipman-2023](/references/#lipman-2023)
