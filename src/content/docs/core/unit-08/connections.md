---
title: "8.6 Connections"
description: Generation in statistical physics, Monte Carlo methods, numerical analysis, optimal transport and fluid mechanics, and coding theory — and within the book.
sidebar: { label: "8.6 Connections", order: 6 }
---

## 8.6.1 Statistical physics: temperature and diffusion

The softmax with temperature is the Boltzmann distribution: probabilities proportional to $e^{-E/kT}$ with energy $E = -z$. Lowering the temperature concentrates probability in low-energy states — greedy decoding is the zero-temperature ground state; raising it approaches uniform disorder. Diffusion models take their name and their first formulation from non-equilibrium thermodynamics: a forward process that adds noise until structure is destroyed, and a learned reverse process that restores it ([Sohl-Dickstein et al. 2015](/references/#sohl-dickstein-2015)). Langevin dynamics, used to sample with a learned score, is the physicist's model of a particle in a potential with thermal noise.

## 8.6.2 Monte Carlo methods

Sampling from a distribution known only up to a constant is the central problem of Markov chain Monte Carlo. Langevin and Hamiltonian Monte Carlo use the score $\nabla\ln p$ — exactly what score-based generative models learn. The difference is where the difficulty lies: MCMC is given the density and must mix; a generative model is given samples and must learn the density or its score. Speculative decoding (§8.8.4) uses a rejection-sampling argument from the same toolbox to make a fast approximate sampler exact.

## 8.6.3 Numerical analysis: ODE and SDE solvers

Generation with diffusion or flow matching is numerical integration, and everything known about solvers applies: order of accuracy, stiffness, adaptive step sizes, stability regions. Unit 05 met the same ideas when discretizing continuous-time memory (exact zero-order hold versus Euler); here the trade-off is samples per second against sample fidelity (§8.5.3). Neural ODEs ([Chen et al. 2018](/references/#chen-2018)) introduced training continuous-time models through ODE solvers; flow matching avoids simulating the ODE during training altogether.

## 8.6.4 Optimal transport and fluid mechanics

The continuity equation $\partial_tp + \nabla\cdot(pv) = 0$ of §8.3.5 is the conservation of mass from fluid dynamics; a generative flow is a fluid of probability. The Benamou–Brenier formulation of optimal transport finds the velocity field of least kinetic energy carrying one density to another — straight-line flow, as in §8.5.1 ([Villani 2009](/references/#villani-2009)). Rectified flow and minibatch optimal-transport couplings are approximations of this classical solution.

## 8.6.5 Coding theory

An autoregressive model plus arithmetic coding is a compressor (Unit 01, §1.6.1); running the decoder on random bits instead of a compressed message produces exactly a sample from the model. Sampling and decompression are the same operation, which is one way to see why ancestral sampling is exact.

## 8.6.6 Inside the book

| This unit | Returns in |
|---|---|
| ancestral sampling is exact | Unit 09 (policy gradients sample from the model to estimate expectations) |
| temperature | Unit 04 (attention's $\beta$), Unit 09 (exploration) |
| ODE discretization error | Unit 05 (Euler vs. exact discretization) |
| likelihood vs. sample quality | Unit 10 (choosing metrics) |
| speculative decoding's exact acceptance rule | Unit 06 (memory-bound decoding) |
