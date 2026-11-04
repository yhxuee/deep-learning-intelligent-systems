---
title: "6.4 Worked examples"
description: Eight examples — every FLOP of the spine model, the 6N rule for GPT-3, a 7B training budget, arithmetic intensity, decoding latency, a pre-registered scaling fit that missed, the flat compute-optimal optimum, and energy per FLOP — with what the Landauer limit does and does not say.
sidebar: { label: "6.4 Worked examples", order: 4 }
---

Exact numbers are recomputed by `labs/test_worked_examples_u06_u08.py`; the FLOP identity is asserted by `spine/test_model.py`; the scaling fit comes from `spine/stage06_scaling.py`.

Use the same five-line worksheet for every example:

1. **Given:** copy the quantities with units.
2. **Find:** name one requested quantity.
3. **Model:** write the formula and its assumptions.
4. **Compute:** substitute before using a calculator.
5. **Interpret:** state what the result does *and does not* establish.

Several examples deliberately use optimistic lower bounds. A lower bound on time is not a promise that an implementation will attain it.

## Worked Example 6.1 — Every FLOP of the spine model

**Given.** The configuration later used in Stage 07: $V=50$, $d=64$, $L=3$, $h=4$, $T=128$, batch $B=16$. **Find:** counted training FLOPs per token, per step and over 4,000 steps.

**Try first.** Compute $BT$. Then estimate whether $6N_{\mathrm{mm}}$ or $12LTd$ is larger; this predicts whether weight products or attention products dominate.

**Model and computation.** Proposition 6.1 counts the declared matrix products:

$$
N_{\mathrm{mm}}=12Ld^2+Vd
=12(3)(64^2)+50(64)
=147{,}456+3{,}200=150{,}656.
$$

Per token,

$$
6N_{\mathrm{mm}}+12LTd
=903{,}936+294{,}912
=1{,}198{,}848\ \text{FLOPs/token}.
$$

One step contains $BT=16\cdot128=2{,}048$ tokens, so

$$
F_{\mathrm{step}}=2{,}048(1{,}198{,}848)
=2{,}455{,}240{,}704\ \text{FLOPs}.
$$

This is exactly the instrumented matrix-product counter. Over 4,000 steps the same convention gives $9.82\times10^{12}$ FLOPs.

**Interpretation.** Attention contributes $294{,}912/1{,}198{,}848=24.6\%$ of the total counted FLOPs. If the readout is excluded, the block-only shortcut gives $T/(T+6d)=128/(128+384)=25\%$. The model has 160,704 parameters in total; the 16-byte recipe estimates $2{,}571{,}264$ bytes, about 2.57 MB of parameter, gradient and optimizer state. It does not include activations or workspaces. The prediction prompt's answer is therefore about 2.5 billion **counted matrix-product FLOPs** per step, not 2.5 billion machine instructions.

## Worked Example 6.2 — The 6N rule for GPT-3

**Given.** 175 billion parameters and 300 billion training tokens. **Find:** the $6ND$ training estimate and the idealized A100 time at 40% useful-FLOP utilization.

**Model and computation.** Treat the dense model's active matrix weights as $N$:

$$
C\approx6ND
=6(1.75\times10^{11})(3\times10^{11})
=3.15\times10^{23}\ \text{FLOPs}.
$$

The reported figure is $3.14\times10^{23}$ ([Brown et al. 2020](/references/#brown-2020)). At $0.4(3.12\times10^{14})$ useful FLOP/s per A100-equivalent device,

$$
t=\frac{3.15\times10^{23}}{0.4(3.12\times10^{14})}
=2.52\times10^9\ \text{device-seconds}
\approx7.01\times10^5\ \text{device-hours}.
$$

**Interpretation.** Two scale variables and the factor 6 reproduce the published compute within about 0.3%; that is why the rule is useful. The device-hour result is an equivalence, not a proposed schedule: 1,000 such devices would ideally take about 701 hours before communication losses, failures and data-centre overheads. It also does not claim GPT-3 was trained on A100s.

## Worked Example 6.3 — What a 7B model needs to train

**Given.** $N=7\times10^9$, using the §6.2.4 mixed-precision Adam recipe. **Find:** model-state capacity before activations.

**Model and computation.** Write the byte ledger before multiplying:

| item | bytes/parameter |
|---|---:|
| 16-bit model weight | 2 |
| 16-bit gradient | 2 |
| 32-bit master weight | 4 |
| Adam first moment | 4 |
| Adam second moment | 4 |
| **total** | **16** |

Therefore $7\times10^9\times16=112\times10^9$ bytes = 112 GB (decimal), or about 104 GiB.

**Interpretation.** The model state alone exceeds one 80 GB device. Perfectly sharding all 112 GB across eight devices would average 14 GB per device, but that is only the partitioned state: activations, temporary buffers, allocator fragmentation and any replicated tensors still require space. The example motivates sharding (§6.5.5); it does not prove that every 7B training configuration fits on eight devices.

## Worked Example 6.4 — Arithmetic intensity

**Given.** The illustrative A100 roofline: 312 TFLOP/s and 2.039 TB/s, hence a balance point of 153 FLOP/byte. **Find:** which roof bounds two idealized product shapes.

**Large square product, $n=4{,}096$, 16-bit.** The optimistic intensity is $I=n/3=1{,}365$ FLOP/byte. Because $1{,}365>153$, the roofline bound is the compute roof: a well-tiled implementation has enough reuse to *be capable of* becoming compute-bound.

**Matrix--vector product, 16-bit.** $I\approx1$ FLOP/byte, so the bandwidth roof is $\beta I\approx2.039$ TFLOP/s, about $1/153=0.65\%$ of the 312-TFLOP/s arithmetic peak.

**Interpretation.** Shape changes the *ceiling*: the square product may be compute-bound, while the matrix--vector product cannot approach peak arithmetic rate if each weight comes from high-bandwidth memory only once. Nothing here guarantees that the square kernel reaches 99% of peak; launch overhead, tiling and problem size still matter. Batching creates reuse and moves decoding from the second case toward the first.

## Worked Example 6.5 — Why decoding is slow

**Given.** A dense 7-billion-parameter model, 16-bit weights, batch 1, short context, one illustrative A100. **Find:** compute-only and weight-bandwidth lower bounds per generated token.

**Model and computation.** Ignore cache traffic and all overheads to obtain optimistic bounds:

$$
t_{\mathrm{compute}}\ge
\frac{1.4\times10^{10}}{3.12\times10^{14}}
=0.045\ \text{ms/token},
$$

$$
t_{\mathrm{weights}}\ge
\frac{1.4\times10^{10}\ \text{bytes}}{2.039\times10^{12}\ \text{bytes/s}}
=6.87\ \text{ms/token}.
$$

The roofline lower bound is the larger value, 6.87 ms/token, corresponding to an upper bound of about 146 tokens/s.

**Interpretation.** The compute-only time is about 150 times smaller than the weight-streaming time. The 22,000-token/s estimate used the wrong roof. Actual generation will be slower because the calculation omitted KV-cache reads, non-matrix kernels and imperfect bandwidth. A batch of 64 can reuse weights and may raise throughput substantially while the workload remains memory-bound; a 64-fold gain and unchanged step latency are optimistic limits, not guarantees. Waiting to assemble that batch may also worsen request latency.

## Worked Example 6.6 — A pre-registered scaling fit that missed

**Given.** A 2-layer spine model at widths 8–32, each trained for 2,000 steps of $16\times64$ characters: $2{,}048{,}000$ training tokens per run. **Protocol:** fit $L=E+AN^{-\alpha}$ to those five completed runs, save the width-48 prediction, and only then train width 48. **Find:** the prediction error and what can legitimately be learned from it.

| width $d$ | 8 | 12 | 16 | 24 | 32 | **48** |
|---|---|---|---|---|---|---|
| non-embedding parameters $N$ | 1 696 | 3 696 | 6 464 | 14 304 | 25 216 | 56 256 |
| held-out NLL | 2.04 | 1.74 | 1.04 | 0.50 | 0.39 | **0.35** |

**Registered fit.** Fitting all five points gives $E=0.00$ and $\alpha=0.67$, with a saved width-48 prediction of **0.23 nats/token**. The observed value is **0.35**, a residual of $+0.12$ nats/token. Because this demonstration uses one deterministic training configuration rather than repeated independent runs, the residual is evidence of predictive failure for this run; it is not by itself an estimate of a noise distribution.

**Post-hoc diagnosis.** The five losses show strong curvature on the proposed scale, especially between widths 12 and 16. This is consistent with a change of regime—perhaps representational capacity, optimization, or both—rather than one power law over all widths. Fitting only the three larger completed widths gives $E=0.33$ and a width-48 value of 0.34, within 0.01 of the observation. That second fit is a useful description made *after* seeing the failure; it is not an independently validated prediction.

**Interpretation.** The honest result is not “the law was right after all.” It is: the registered model missed; inspecting the residual suggested a narrower regime; that new choice must be registered and tested on another unseen width, such as 64. Selecting points after seeing width 48 and then counting width 48 as confirmation would be the analytic flexibility Unit 12 warns about. Finally, the fitted $E=0.33$ is not automatically the corpus entropy: with finite data, a fixed optimizer and a 2-layer architecture, it is merely an asymptote parameter in this local fit.

## Worked Example 6.7 — The flat compute-optimal optimum

**Given.** Hoffmann et al.'s approach-3 fit: $E=1.69$, $A=406.4$, $B=410.7$, $\alpha=0.34$, $\beta=0.28$; training budget $C=5.76\times10^{23}$ FLOPs. **Find:** the optimum predicted by this fit and the penalty for using the 70B/1.4T allocation instead.

**Computation.** Substitution into Proposition 6.2 gives

$$
N^\star=3.22\times10^{10},\qquad
D^\star=2.98\times10^{12},
$$

or about 93 tokens per parameter, with predicted loss 1.9307 nats/token. Evaluating the *same fitted surface* at 70 billion parameters and 1.4 trillion tokens gives 1.9366 nats/token. The predicted difference is only 0.0059.

**Interpretation.** Under this particular fit, increasing model size by a factor of about 2.2 while reducing tokens costs only 0.006 nats/token. The valley is shallow: the exact location of its minimum can move substantially when fitted constants change, while nearby allocations have almost indistinguishable predicted losses. This is why alternative estimation methods and a later reanalysis ([Besiroglu et al. 2024](/references/#besiroglu-2024)) can disagree about the optimal size without implying a comparably large performance penalty. The calculation compares predictions from one fitted surface; it is not a theorem that every twofold allocation error is cheap.

## Worked Example 6.8 — Energy per FLOP, and what Landauer's limit does not say

**Given.** An illustrative A100 drawing 400 W while delivering useful model FLOPs at 40% of 312 TFLOP/s. **Find:** accelerator joules per useful FLOP and the corresponding GPT-3 accelerator-energy estimate.

**Computation.**

$$
e=\frac{400\ \mathrm{J/s}}{0.4(3.12\times10^{14})\ \mathrm{FLOP/s}}
=3.2\times10^{-12}\ \mathrm{J/FLOP}.
$$

Multiplying by GPT-3's reported $3.14\times10^{23}$ FLOPs gives $1.0\times10^{12}$ J. Since $1\ \mathrm{MWh}=3.6\times10^9$ J, this is about 280 MWh of accelerator energy under the illustrative assumptions. [Patterson et al. (2021)](/references/#patterson-2021) estimate 1,287 MWh for the actual run on older V100 hardware and with whole-data-centre effects. At 300 K, Landauer's limit is $kT\ln2=2.87\times10^{-21}$ J **per bit irreversibly erased**.

**Interpretation.** The factor-of-four gap is not one clean “overhead factor”: the hardware generation, achieved utilization, power model and system boundary all differ. Landauer's number cannot simply be divided into joules per FLOP, because a floating-point multiply--add is not identical to erasing one bit; a circuit-level account of irreversible erasures would be required. The useful conclusion is narrower: present energy per useful model FLOP depends strongly on hardware, utilization and data movement. Arithmetic intensity is one lever, not a thermodynamic conversion formula.

**Next:** [§6.5 General theory](/core/unit-06/theory/) turns the compute ledger into a constrained optimization problem and asks how much trust to place in the fitted loss surface.
