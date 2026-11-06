---
title: "6.10 Problems and further study"
description: Twenty graded problems on FLOPs, memory, rooflines, scaling laws, compute allocation and distributed training, with a learning path, hints, checks, selected answers and annotated readings.
sidebar: { label: "6.10 Problems", order: 10 }
---

Difficulty: ★ routine application · ★★ combines ideas · ★★★ proof or open-ended design. Problems marked **[AI-off]** are intended for unaided practice and assessment; for the others, make a first attempt before using the [Mentor](/mentor/).

If this is your first pass, do **6.1, 6.3, 6.7 and 6.11** first. Then attempt **6.2, 6.5, 6.9, 6.13 and 6.17**. Enter the open-ended problems only after you can keep four ledgers separate:

1. **work:** FLOPs;
2. **capacity:** stored bytes;
3. **traffic:** bytes moved per operation or token;
4. **time:** seconds after work and traffic meet hardware ceilings.

For every numerical answer, write one line of assumptions and one dimensional check. A bare number is not complete when changing “GB” to “GiB” or “peak” to “sustained” changes the conclusion.

## Computation

**6.1 ★ [AI-off]** A dense model has $N=10^9$ matrix-multiplied parameters and trains on $D=2\times10^{10}$ tokens. Ignore attention-specific products. (a) Compute $C=6ND$. (b) For this exercise only, assume an A100 peak of 312 TFLOP/s and 40% useful-FLOP utilization. Convert the time to GPU-days. (c) Name two reasons elapsed time on one real device can be longer. *Check: FLOPs divided by FLOP/s must leave seconds.*

**6.2 ★ [AI-off]** A dense decoder has $d=4{,}096$, $L=32$, $V=32{,}000$ and tied input/output embeddings. Use $N_{\text{mm}}=12Ld^2+Vd$. Compute $N_{\text{mm}}$ and

$$f_{\text{attn}}=\frac{12LTd}{6N_{\text{mm}}+12LTd}$$

at $T=2{,}048$ and $T=32{,}768$. Before calculating, predict which term grows with $T$. *Check: the fraction must lie in $[0,1]$.*

**6.3 ★** Under the illustrative mixed-precision Adam ledger of 16 bytes/parameter, compute training-state memory for 1B, 7B and 70B parameters. Using decimal GB and 80 GB/device, give the mathematical minimum number of devices under perfect even partitioning. Then list memory omitted by that minimum. *Hint: use a ceiling, not rounding to nearest.*

**6.4 ★** Multiply a $B\times4{,}096$ matrix by a $4{,}096\times4{,}096$ matrix in 16-bit arithmetic. Under the optimistic traffic model “read each input once, write the output once,” derive $I(B)$ and evaluate it for $B=1,16,256$. Under the illustrative A100 roofline $312$ TFLOP/s and $2.039$ TB/s, which shapes have enough intensity to be compute-bound? Explain why “enough intensity” is not “guaranteed peak performance.”

**6.5 ★★** A 13B-parameter model has 16-bit weights, $L=40$, $d=5{,}120$, context $T=2{,}048$, and ordinary multi-head attention (no grouped/multi-query attention). (a) Derive KV-cache bytes/sequence. (b) Using **80 billion bytes**, and ignoring runtime buffers, fragmentation and all non-weight storage, find the largest integer batch that fits beside the 26 GB of weights. (c) Recompute with 80 GiB and explain why neither result is a safe deployment batch. *Hint: $M_{KV}=2LTdb$ for batch one.*

**6.6 ★★** Use the §6.5.1 approach-3 constants $E=1.69$, $A=406.4$, $B=410.7$, $\alpha=0.34$, $\beta=0.28$. For $C=10^{21}$ and $10^{24}$ FLOPs, compute $N^\star$, $D^\star$, $D^\star/N^\star$ and predicted loss. *Checkpoint: increasing $C$ by $10^3$ should multiply $N^\star$ by $10^{3\beta/(\alpha+\beta)}$, not by $10^3$.* State why these outputs are conditional predictions rather than hardware recommendations.

## Derivation

**6.7 ★ [AI-off]** For $Y=XW$ with $X\in\mathbb R^{m\times k}$ and $W\in\mathbb R^{k\times n}$, write the two backward products and their shapes. Under the $2mnk$ convention, show that computing both costs twice the forward product. Which input-gradient product can be omitted if the input and every upstream parameter are fixed? *Do not omit a gradient merely because $X$ is data; an earlier trainable layer may have produced it.*

**6.8 ★★ [AI-off]** Prove Proposition 6.1. Make a table with one row for QKV, output projection, the two MLP projections, unembedding, $QK^\top$ and $AV$. Give forward and backward cost for each, then sum. Explicitly state why embedding lookup and elementwise operations are outside this exact matrix-product count.

**6.9 ★★ [AI-off]** Starting from $L=E+AN^{-\alpha}+BD^{-\beta}$ and $6ND=C$, prove Proposition 6.2 without quoting it. Next use the stationary balance condition to show

$$L^\star-E\propto C^{-\alpha\beta/(\alpha+\beta)}.$$

*Hint: after substitution, both excess-loss terms have the same power of $C$ at $N^\star$.* Check the exponent for $\alpha=0.34$, $\beta=0.28$.

**6.10 ★★** An ideal ring all-reduce handles a $G$-byte gradient on $p$ devices. In reduce-scatter, each device sends $(p-1)$ chunks of size $G/p$; all-gather repeats the pattern. Derive bytes **sent per device**, bytes received per device, and the sum over all devices. Separate the bandwidth term from the $2(p-1)$ message-startup latencies. *Check the cases $p=1$ and $p\to\infty$.*

**6.11 ★★** A decoder block has an MLP expansion factor 4. For one forward pass: (a) compare only position-mixing attention products $QK^\top$ and $AV$ with the two MLP products; (b) repeat after including QKV and output projections in “attention.” Derive the threshold on $T/d$ in both definitions. *Check: the answers differ; this is why the phrase “attention cost” needs a declared boundary.*

## Proof and analysis

**6.12 ★★** Divide an $L$-block network into $s$ consecutive segments. Store each segment boundary; during backward, recompute and temporarily retain one segment's internal activations. Show that retained activation memory is $O(s+L/s)$ block states and the additional arithmetic is at most one forward pass. Minimize the memory expression over $s$ to obtain $O(\sqrt L)$. State what this abstraction omits about unequal layers and non-activation memory.

**6.13 ★★** Set $u=\ln N$ and prove that

$$f(e^u)=Ae^{-\alpha u}+B(C/6)^{-\beta}e^{\beta u}$$

is strictly convex when $A,B,\alpha,\beta>0$. Conclude that Proposition 6.2's stationary point is the unique global minimum over $N>0$. Why does this not prove that the empirical loss surface is correct?

**6.14 ★★** Expand $f(e^u)$ to second order around $u^\star=\ln N^\star$. For the approach-3 constants and $C=5.76\times10^{23}$, approximate the excess loss when $N=2N^\star$ and compare with direct substitution and Worked Example 6.7. *Hint: $\Delta u=\ln2$ and the linear term vanishes.* Explain what the small error says about the **fitted valley**, not all training runs.

**6.15 ★★★** Generalize the heuristic of §6.5.4. If cell width is $\ell\propto N^{-1/d_{\text{int}}}$ and a degree-$k$ local polynomial has pointwise error $O(\ell^{k+1})$, derive the squared-error exponent. Treat piecewise constant as $k=0$ and the text's piecewise linear case as $k=1$. What intrinsic dimension would an exponent near 0.1 imply in each case? Identify at least three assumptions that prevent this inversion from being a measurement of language's “true dimension.”

**6.16 ★★★** Amdahl holds total work fixed. Derive Gustafson's scaled-speedup law when serial time remains $s$ of the one-device run but parallel work grows with $p$. Then analyze data-parallel training with fixed per-device batch: distinguish throughput scaling for a growing global batch from time-to-train at a fixed optimization target. Which law is informative for each question, and what communication term is missing from both?

## Implementation and design

**6.17 ★ [AI-off]** Write a function that accepts $(V,d,L,T,B)$, bytes/dtype, MLP expansion and number of KV heads, then returns a **breakdown** of training matrix-product FLOPs/token, illustrative training-state bytes, and KV-cache bytes/sequence. Test the default dense case against `model.FLOPS` and `kv_cache_bytes`. Add tests for $T=1$, grouped-query attention and a disabled MLP. A total without component checks is not sufficient.

**6.18 ★★** Add uncertainty to the §6.5.2 fit. Because five points are very few, compare two procedures: (a) resample seeds within each width after collecting at least three seeds; (b) leave one width out. Report the width-48 predictive interval and how often the nonlinear fit hits a boundary such as $E=0$. Explain why naïvely resampling only five widths changes the design distribution as well as the noise.

**6.19 ★★** At fixed width, measure held-out NLL over a preregistered grid of training-token budgets and at least three seeds; fit $L(D)$. Combine this with Experiment 6.4's width sweep to fit $L(N,D)$. Hold evaluation data and tokenizer fixed, plot residuals, reserve at least one $(N,D)$ pair for validation, and compare separable versus interaction-augmented fits. State which conclusion the held-out pair supports and which it cannot.

**6.20 ★★★** “Serve 10,000 users below 100 ms/token” is under-specified. First list the missing workload quantities: concurrent active requests, arrival rate, prompt-length distribution, output length, latency percentile, quality target and hardware budget. Then declare one numerical scenario and design model size, precision, tensor/data parallel layout, batch policy and KV-cache strategy. Provide capacity, bandwidth, compute and queueing ledgers. Finally stress-test at double the arrival rate and double the context. A valid answer may conclude that the service-level objective is infeasible on the proposed hardware.

## Selected answers and checks

- **6.1** $C=6(10^9)(2\times10^{10})=1.2\times10^{20}$ FLOPs. The assumed useful rate is $0.4(312\times10^{12})=1.248\times10^{14}$ FLOP/s, giving $9.62\times10^5$ s, or **11.1 GPU-days**. This ignores attention, non-matmul work, communication, input stalls and failures.
- **6.2** $N_{\text{mm}}=6{,}573{,}522{,}944$. The attention fractions are about **7.55%** and **56.65%**. The formula includes unembedding but treats lookup separately, exactly as Proposition 6.1 declares.
- **6.3** 16, 112 and 1,120 decimal GB; the capacity-only minima are 1, 2 and 14 devices. Activations, temporary workspaces, fragmentation and replicated or imperfectly sharded state make real requirements larger.
- **6.4** $I(B)=B/(1+2B/4096)$ FLOP/byte: about 1.00, 15.88 and 227.56. The illustrative balance point is $312/2.039\approx153$ FLOP/byte, so only $B=256$ lies above it under this optimistic model.
- **6.5** The FP16 cache is $2(40)(2048)(5120)(2)=1{,}677{,}721{,}600$ bytes/sequence. Beside 26 billion weight bytes, the idealized limits are batch 32 in 80 billion bytes and batch 35 in 80 GiB. Neither leaves operational headroom.
- **6.6** For $10^{21}$ FLOPs: $N^\star\approx1.82$B, $D^\star\approx91.4$B, $D/N\approx50.1$, $L\approx2.329$. For $10^{24}$: $N^\star\approx41.3$B, $D^\star\approx4.04$T, $D/N\approx97.7$, $L\approx1.911$ nats/token.
- **6.9** Both excess terms scale as $C^{-\alpha\beta/(\alpha+\beta)}$; with the given exponents the power is about **$-0.154$**.
- **6.10** Each device sends and receives $2G(p-1)/p$ bytes. Summed sends across the ring are $2G(p-1)$; counting both send and receive endpoints doubles that network-endpoint total.
- **6.11** Position mixing alone costs $4T^2d$ versus MLP's $16Td^2$, so it exceeds the MLP at $T>4d$. Including attention projections adds $8Td^2$, changing the threshold to $T>2d$.
- **6.14** The quadratic approximation gives about **0.00551 nats/token**; direct substitution gives about **0.00545**, close to the 0.0059 comparison in Worked Example 6.7, which uses a nearby reported allocation rather than exactly $2N^\star$.
- **6.15** The exponent is $2(k+1)/d_{\text{int}}$. An observed 0.1 would imply $d_{\text{int}}=20$ for piecewise constant and 40 for piecewise linear—only inside the heuristic.

## Further study

**Start with accounting.** [Kaplan et al. 2020](/references/#kaplan-2020), §2, makes the parameter and compute conventions explicit. Read it beside [Korthikanti et al. 2023](/references/#korthikanti-2023), which shows why production activation memory needs a more detailed ledger than one bytes/parameter slogan.

**Treat scaling as measurement.** Read [Hoffmann et al. 2022](/references/#hoffmann-2022) together with the reanalysis by [Besiroglu et al. 2024](/references/#besiroglu-2024). The useful lesson is not which ratio to memorize, but how fitting choices move an optimum along a shallow loss valley. [Muennighoff et al. 2023](/references/#muennighoff-2023) asks what changes when unique data are limited.

**Connect algorithms to machines.** [Williams et al. 2009](/references/#williams-2009) introduces the roofline; [Pope et al. 2023](/references/#pope-2023) applies related ledgers to inference. [Rajbhandari et al. 2020](/references/#rajbhandari-2020) and [Narayanan et al. 2021](/references/#narayanan-2021) show how sharding and pipeline scheduling exchange memory for communication and idle time.

**Keep the system boundary visible.** [Patterson et al. 2021](/references/#patterson-2021) shows why energy and emissions depend on accelerator efficiency, utilization, data-centre overhead and electricity supply—not FLOPs alone.
