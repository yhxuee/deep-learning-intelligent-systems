---
title: "9.9 Experiments"
description: Seven prediction-first experiments that connect policy-gradient algebra to observable behaviour: estimator variance, KL tilting, spine post-training, regularization, offline versus online feedback, reward hacking, and checker audits.
sidebar: { label: "9.9 Experiments", order: 9 }
---

The point of these experiments is not to obtain the highest reward. It is to learn which measurement explains each change. For every run, keep a small lab table:

| field | record before running |
|---|---|
| question | the single claim being tested |
| prediction | direction and, when possible, a number |
| intervention | the one factor deliberately changed |
| controls | seeds, prompts, reference policy, sample budget |
| measurements | proxy reward **and** an independent task metric |
| decision | what result would change your mind |

Code: [`spine/stage09_post_training.py`](https://github.com/yhxuee/deep-learning-intelligent-systems/tree/main/spine) and verification tests in `spine/test_spine.py`. Commands below assume the repository root and the supplied virtual environment. If the Stage 07 checkpoint is absent, run the Stage 07 script first.

## Experiment 9.1 — See a policy gradient one sample at a time

**Question.** Why is REINFORCE unbiased yet noisy, and what exactly does a baseline change?

Use a three-action bandit with

$$
\pi=(1/3,1/3,1/3),\qquad r=(1,0,0).
$$

For logit $z_1$, one sample gives

$$
g_1(a)=(r(a)-b)\left(\mathbf 1[a=1]-\pi_1\right).
$$

**Predict first.** Compute the three possible values of $g_1$ for $b=0$ and $b=1/3$. From those values, predict the mean and variance before simulation. The known answers in Worked Examples 9.1–9.2 are means $0.2222$ in both cases and variances $0.0988$ versus $0.0247$.

**Procedure.** Draw 100, 1,000 and 100,000 actions. At each sample size, report the sample mean, variance and error from the exact gradient. Repeat for ten seeds.

**Interpretation.** The baseline changes variance, not the expected gradient. The mean-reward baseline helps in this example, but do not turn that into a universal ordering: an arbitrary baseline can increase variance. Proposition 9.2 gives the component-specific variance-minimizing constant.

**Failure check.** If the mean shifts systematically when a constant baseline is added, the implementation probably differentiated through the sampled baseline or made it action-dependent.

## Experiment 9.2 — Watch KL regularization tilt a distribution

**Question.** Does numerical optimization recover the closed-form optimum, and what role does $\beta$ play?

Use

$$
\pi_{\mathrm{ref}}=(0.2,0.5,0.3),\qquad r=(1,0,0),
$$

so that

$$
\pi^\star(a)=\frac{\pi_{\mathrm{ref}}(a)e^{r(a)/\beta}}{\sum_b\pi_{\mathrm{ref}}(b)e^{r(b)/\beta}}.
$$

**Predict first.** Calculate $\pi^\star$ for $\beta\in\{0.1,0.5,2\}$. Which value stays closest to the reference? Which concentrates most on action 1?

**Procedure.** First run the closed-form test:

```bash
LAB_IMPL=reference .venv/bin/python -m pytest spine/test_spine.py -k tilted -q
```

Then optimize logits by exact gradient ascent from three different initial distributions. Plot $J_\beta$, $D_{\mathrm{KL}}(\pi\|\pi_{\mathrm{ref}})$ and $D_{\mathrm{KL}}(\pi\|\pi^\star)$. Finally replace the exact gradient with REINFORCE batches of 16 and 256.

**Interpretation.** Exact optimization should approach the same interior optimum from all three starts when step sizes are stable. Stochastic runs fluctuate around it; a larger batch reduces sampling noise, not objective misspecification. A small $\beta$ makes reward differences dominate, while a large $\beta$ preserves more of the reference distribution.

**Failure check.** Verify that the implemented KL direction is $D_{\mathrm{KL}}(\pi\|\pi_{\mathrm{ref}})$. Reversing it changes both the gradient and boundary behaviour.

## Experiment 9.3 — Post-train the spine model and read every meter

**Question.** Does post-training increase true arithmetic correctness, or merely the number printed as “reward”?

**Predict first.** Draw four curves against training step: sampled training reward, held-out greedy accuracy, held-out mean probability of the correct digit, and KL from the frozen reference. Predict the sign of each change, then assign a plausible range rather than demanding monotonicity.

**Procedure.** From the `spine` directory, run:

```bash
cd spine
../.venv/bin/python stage09_post_training.py --steps 150
```

The checked-in deterministic setup gives a useful orientation, not a universal benchmark: the pretrained model has held-out greedy accuracy about $0.32$; the exact-reward REINFORCE and DPO runs reach about $0.82$ and $0.78$, respectively. Repeat the run with seeds 0–4 and report mean, standard deviation and every individual result.

**Interpretation.** Greedy accuracy and mean $p(\text{correct})$ answer different questions. The first asks which token wins; the second asks how much probability mass correctness receives. KL measures change from the reference, not correctness. A larger KL can accompany improvement, damage or both.

**Failure check.** Do not evaluate on the prompts sampled for updates. Freeze held-out prompts before training, and use the same prompts for every method and seed.

## Experiment 9.4 — Turn the KL “leash” without calling it safety

**Question.** How does the KL coefficient trade task improvement against distributional change?

**Predict first.** For $\beta\in\{0,0.01,0.05,0.5,5\}$, rank the likely update size. Do **not** predict that the best held-out accuracy must occur at the smallest $\beta$; optimization noise and overfitting can break that ordering.

**Procedure.** Keep the initialization, prompts, seeds, batch size, step count and learning rate fixed. For each $\beta$, record:

- arithmetic greedy accuracy and $p(\text{correct})$;
- KL on arithmetic prompts;
- next-character NLL on a frozen non-arithmetic story set;
- entropy and the frequency of each output digit.

Plot accuracy against KL, not only against $\beta$. Repeat for at least five seeds.

**Interpretation.** $\beta$ is a Lagrange-like trade-off parameter, not a safety dial. Two runs with the same mean KL can damage different behaviours. If story NLL degrades while arithmetic improves, the experiment has revealed interference that arithmetic reward alone could not see.

**Failure check.** A KL value is meaningful only with its direction, averaging convention, token scope and logarithm base. Record all four.

## Experiment 9.5 — Offline pairs versus current-policy samples

**Question.** When does a fixed preference dataset become stale?

**Predict first.** Construct “easy” pairs in which the rejected answer is far from the correct digit, and “near-boundary” pairs in which it differs by one. Which set should keep providing useful gradients after the model has learned to avoid obviously wrong digits?

**Procedure.** Compare three budgets:

1. DPO on one fixed set of easy pairs;
2. DPO on one fixed set of near-boundary pairs;
3. REINFORCE using fresh current-policy samples and the exact checker.

Match the number of prompt evaluations, and separately report wall-clock time and checker calls. A “step” is not a fair compute unit across algorithms. Track held-out accuracy, KL and the fraction of training pairs on which the policy already strongly prefers the winner.

**Interpretation.** Fixed pairs make the experiment reproducible; fresh samples follow the current policy's mistakes. DPO's policy update does not require on-policy sampling, but its pair dataset still had to come from somewhere. REINFORCE does not solve tasks without a checker, and DPO does not solve them without a source of trustworthy comparisons.

**Failure check.** If the three methods see different prompts or different numbers of feedback queries, label the comparison as system-level, not optimizer-only.

## Experiment 9.6 — Find the reward hack before the policy does

**Question.** Can you predict the optimizer's “clever” solution by inspecting the checker alone?

Audit three candidate rewards for target $t=a+b$:

```python
reward_a = int(answer >= t)
reward_b = int(answer in "0123456789")
reward_c = int(abs(answer - t) <= 1)
```

**Predict first.** Enumerate all ten digit actions for every possible target $t\in\{2,\ldots,9\}$. Mark which actions maximize each reward.

**Procedure.** Train the same policy with each checker. At every checkpoint record both the candidate reward and exact-match correctness. Also record the output histogram by target.

**Interpretation.** Checker (a) rewards large digits and can collapse toward 9. Checker (b) only distinguishes digits from non-digits; among digits it supplies no correctness signal. Checker (c) defines a three-value acceptance band except at boundaries, so exact correctness is only one of several equally rewarded behaviours. The pretrained distribution and optimizer decide which accepted behaviour wins.

**Failure check.** Test the checker before training with a truth table, boundary cases, equivalent valid forms and adversarial outputs. The policy should not be the first serious user of the evaluator.

## Experiment 9.7 — Build a checker audit set

**Question.** Can a small, deliberately structured audit predict a failure that average validation reward hides?

**Predict first.** Write at least five failure families before generating examples: correct answer, off by one, too large, too small, non-digit, malformed text, and—if the answer format is extended—an equivalent form such as `05` versus `5`.

**Procedure.** Create a table with columns `prompt`, `candidate`, `intended_valid`, `checker_reward`, `failure_family`. Do not sample it from the training policy. Require:

- every family to contain multiple targets;
- both ordinary and boundary targets;
- zero false positives on safety-critical invalid cases;
- a reported confusion matrix, not only accuracy.

Run this audit against the exact and three buggy checkers from Experiment 9.6. Then decide, *before policy training*, which checkers are admissible.

**Interpretation.** Random validation estimates average behaviour under a sampling distribution. A structured audit asks whether known semantic requirements are represented at all. Both are needed: the audit finds named holes, while random evaluation estimates their prevalence.

**Failure check.** Passing a finite audit does not prove a checker complete. Record the uncovered input space and carry independent correctness evaluation into training. That discipline is the bridge to Unit 10.

## Minimum experiment report

A credible report includes the prediction, exact command or code revision, checkpoint identity, seeds, prompt split, sample or feedback budget, all proxy and independent metrics, raw per-seed values, and at least one failed or surprising run. “Reward increased” is not a conclusion until you state which reward, who computed it, and what independent measurement moved with it.
