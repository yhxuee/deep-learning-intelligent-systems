---
title: "5.2 First principles: state, decay and time constants"
description: What a state is, the exponential moving average derived and unrolled, half-life, the noise–lag trade-off, and the continuous-time system behind the decay factor.
sidebar: { label: "5.2 First principles", order: 2 }
---

This section builds the simplest useful memory from nothing but arithmetic, then asks what it costs. Every later model in the unit is a generalization of what we construct here.

## 5.2.1 What is a state?

A device that processes a stream $x_1, x_2, \dots$ one element at a time keeps some internal quantity $s_t$ and updates it with a fixed rule:

$$
s_t = F(s_{t-1},\, x_t).
$$

We call $s_t$ a **state** if everything the device will ever output in the future depends on the past inputs **only through** $s_t$. The state is the device's entire memory: two different histories that lead to the same state are, from that moment on, indistinguishable.

The size of the state is the size of the memory. Some questions about a stream can be answered with a tiny state; others cannot. Consider four:

| Question about $x_1,\dots,x_t$ | Minimal state | Update rule |
|---|---|---|
| largest reading so far | 1 number $m_t$ | $m_t = \max(m_{t-1}, x_t)$ |
| mean reading so far | 2 numbers $(S_t, t)$ | $S_t = S_{t-1} + x_t$; mean $= S_t/t$ |
| mean and variance so far | 3 numbers | Welford's update ([Welford 1962](/references/#welford-1962)) |
| the reading exactly 1000 s ago | 1000 numbers | a shift register: drop the oldest, append $x_t$ |

The first three summaries are *compressions*: a fixed number of values answers the question exactly, no matter how long the stream. The fourth is not compressible at all; §5.5.2 proves that no state with fewer distinguishable values than there are possible windows — in bits, fewer than the window's own information content — can answer it for arbitrary inputs. (With infinitely precise real numbers the count is unbounded; the statement is about states of finite precision, which is what every real device has.)

**The task defines what memory is needed.** "Remember the past" is not a well-posed goal; "remember what is needed to answer *this* question about the past" is. This sentence will return at the end of the unit, when we compare recurrent models with attention.

## 5.2.2 A first attempt: the running mean

For the temperature sensor, the obvious summary is the running mean, which needs only two numbers. Written as an update,

$$
m_t = m_{t-1} + \frac{1}{t}\,(x_t - m_{t-1}).
$$

Read it as: *move the old estimate a fraction $1/t$ of the way toward the new reading*. That is a good rule for a temperature that never changes — each new reading gets exactly as much weight as every earlier one. But the room's temperature *does* change. Suppose the sensor has read 20 °C for 1000 seconds and the room then genuinely warms to 23 °C. Ten seconds later,

$$
m_{1010} = \frac{1000\cdot 20 + 10\cdot 23}{1010} \approx 20.03 .
$$

The mean has barely moved. Because the step size $1/t$ shrinks forever, the running mean becomes ever more confident and ever less responsive. For a world that changes, *equal weighting of all history is the wrong prior*. Recent readings should count more than old ones.

## 5.2.3 The exponential moving average

The fix is to stop shrinking the step. Replace $1/t$ by a fixed fraction $1-\alpha$:

$$
\boxed{\,h_t = \alpha\, h_{t-1} + (1-\alpha)\, x_t, \qquad 0 < \alpha < 1.\,}
$$

This is the **exponential moving average (EMA)**. Each update blends the previous summary (weight $\alpha$) with the new reading (weight $1-\alpha$). Because the two weights are positive and sum to one, $h_t$ always lies between $h_{t-1}$ and $x_t$: it is a *convex combination*, a compromise between old belief and new evidence.

Apply it to the sensor readings from §5.1 with $\alpha = 0.8$ and $h_0 = 20$:

| $t$ | $x_t$ | $h_t = 0.8\,h_{t-1} + 0.2\,x_t$ |
|---|---|---|
| 1 | 20.0 | $0.8(20.000) + 0.2(20.0) = 20.000$ |
| 2 | 20.4 | $0.8(20.000) + 0.2(20.4) = 20.080$ |
| 3 | 19.8 | $0.8(20.080) + 0.2(19.8) = 20.024$ |
| 4 | **25.0** | $0.8(20.024) + 0.2(25.0) = 21.019$ |
| 5 | 20.1 | $0.8(21.019) + 0.2(20.1) = 20.835$ |
| 6 | 20.0 | $0.8(20.835) + 0.2(20.0) = 20.668$ |

The 5 °C glitch moved $h$ by only about 1 °C — exactly $1-\alpha = 20\%$ of the jump — and its effect then fades. With a fan threshold of 24 °C, the glitch no longer triggers the fan. The device stores **one number**.

We have a memory. Now we need to understand it.

## 5.2.4 Unrolling: what does the state actually contain?

The update rule hides the answer to question 1 from §5.1 ("why does this system retain past information?"). To expose it, substitute the rule into itself. Since $h_{t-1} = \alpha h_{t-2} + (1-\alpha)x_{t-1}$,

$$
h_t = \alpha\bigl[\alpha h_{t-2} + (1-\alpha)x_{t-1}\bigr] + (1-\alpha)x_t
    = \alpha^2 h_{t-2} + \alpha(1-\alpha)\,x_{t-1} + (1-\alpha)\,x_t .
$$

Substituting once more for $h_{t-2}$ gives $\alpha^3 h_{t-3} + \alpha^2(1-\alpha)x_{t-2} + \alpha(1-\alpha)x_{t-1} + (1-\alpha)x_t$. The pattern is clear. Continuing all the way back to $h_0$:

$$
\boxed{\,h_t = \alpha^t h_0 + (1-\alpha)\sum_{k=1}^{t} \alpha^{\,t-k}\, x_k\,}
$$

**Proof by induction.** For $t = 1$ the formula reads $h_1 = \alpha h_0 + (1-\alpha)x_1$, which is the update rule. Assume it holds for $t-1$. Then

$$
h_t = \alpha h_{t-1} + (1-\alpha)x_t
    = \alpha\Bigl[\alpha^{t-1}h_0 + (1-\alpha)\sum_{k=1}^{t-1}\alpha^{t-1-k}x_k\Bigr] + (1-\alpha)x_t
    = \alpha^{t}h_0 + (1-\alpha)\sum_{k=1}^{t-1}\alpha^{t-k}x_k + (1-\alpha)\alpha^0 x_t ,
$$

and the last term is the $k = t$ term of the sum. $\blacksquare$

The closed form exposes a structural fact. **The device never stores a past reading, yet its state is a linear combination of every reading it has ever seen**, with weight

$$
w_k = (1-\alpha)\,\alpha^{\,t-k}
$$

on the reading from $t-k$ steps ago. The weights shrink geometrically with age: that is the "exponential" in the name.

Do the weights form an average? Sum them with the geometric series $\sum_{j=0}^{n-1}\alpha^j = (1-\alpha^n)/(1-\alpha)$:

$$
\sum_{k=1}^{t} w_k = (1-\alpha)\sum_{j=0}^{t-1}\alpha^{j} = 1 - \alpha^t .
$$

Together with the weight $\alpha^t$ on the initial state $h_0$, the weights sum to exactly 1. So $h_t$ is a **weighted average** of the initial guess and all inputs — which is why, in the table above, $h_t$ always stayed in the range of the readings.

## 5.2.5 How long does the memory last?

Compare the influence of a reading that is $n$ steps old with that of the newest reading: the ratio of their weights is $\alpha^n$. With $\alpha = 0.9$, after 20 steps

$$
0.9^{20} \approx 0.122 ,
$$

so only about 12% of the original influence remains. No reading is ever forgotten completely, but its influence decays geometrically.

A convenient single number is the **half-life**: the age $n$ at which the influence has dropped to one half. Solve $\alpha^{n} = \tfrac12$ by taking logarithms:

$$
t_{1/2} = \frac{\ln(1/2)}{\ln \alpha} = \frac{\ln 2}{-\ln\alpha}.
$$

For $\alpha = 0.9$, $t_{1/2} = 0.6931/0.1054 \approx 6.58$ steps.

When $\alpha$ is close to 1, write $\alpha = 1-\varepsilon$ and use $\ln(1-\varepsilon) \approx -\varepsilon$ for small $\varepsilon$:

$$
t_{1/2} \approx \frac{\ln 2}{1-\alpha} \approx \frac{0.693}{1-\alpha}.
$$

For $\alpha = 0.99$ this gives 69.3, against the exact 68.97. **Memory length scales like $1/(1-\alpha)$**: halving the distance of $\alpha$ from 1 doubles the memory.

A second useful number is the **mean age** of the information in $h_t$ (for a long stream, ignoring $h_0$): $\sum_{j\ge 0} j\,(1-\alpha)\alpha^{j} = \alpha/(1-\alpha)$, which is 9 steps for $\alpha = 0.9$. (Problem 5.3 asks you to derive this by differentiating the geometric series.)

## 5.2.6 The price of memory: noise against lag

We can now make the trade-off in §5.1 precise. Model the sensor as a true value plus independent noise, $x_t = \mu + \varepsilon_t$, where the $\varepsilon_t$ are independent with mean 0 and variance $\sigma^2$.

**Noise reduction.** Long after start-up the $\alpha^t h_0$ term is negligible, and $h_t - \mu \approx (1-\alpha)\sum_{j\ge0}\alpha^j\varepsilon_{t-j}$. Variances of independent terms add, and $\mathrm{Var}(c\,\varepsilon) = c^2\sigma^2$, so

$$
\mathrm{Var}(h_t) = (1-\alpha)^2 \sum_{j\ge 0}\alpha^{2j}\,\sigma^2 = \frac{(1-\alpha)^2}{1-\alpha^2}\,\sigma^2 = \frac{1-\alpha}{1+\alpha}\,\sigma^2 ,
$$

using $1-\alpha^2 = (1-\alpha)(1+\alpha)$. Larger $\alpha$ means a quieter estimate.

**Lag.** Now suppose there is no noise, the state has settled at $h_0 = 20$, and the true temperature jumps to 23 at $t = 1$ and stays there. The closed form gives

$$
h_t = 20\,\alpha^t + 23\,(1-\alpha)\sum_{k=1}^{t}\alpha^{t-k} = 20\,\alpha^t + 23\,(1-\alpha^t) = 23 - 3\,\alpha^t .
$$

The remaining error after $t$ steps is the fraction $\alpha^t$ of the jump. The state is within 5% of the new value once $\alpha^t \le 0.05$, i.e. after $t_{95} = \ln 0.05/\ln\alpha$ steps.

| $\alpha$ | half-life (steps) | 95% response time (steps) | noise variance factor $\tfrac{1-\alpha}{1+\alpha}$ | noise std factor |
|---|---|---|---|---|
| 0.5 | 1.00 | 4.3 | 0.333 | 0.577 |
| 0.9 | 6.58 | 28.4 | 0.0526 | 0.229 |
| 0.99 | 68.97 | 298.1 | 0.00503 | 0.0709 |

![Noisy temperature readings with a genuine step from 20 to 23 °C at t = 40 and a spike at t = 70, filtered by three EMAs. α = 0.5 follows the step within a few seconds but stays noisy and reacts to the spike; α = 0.98 is smooth but takes tens of seconds to reach the new level.](/figures/u05-ema-noisy-step.svg)

*Figure 5.1 — The noise–lag trade-off. One decay factor cannot give both a fast response to genuine change and strong suppression of noise. Generated by `figures/make_figures.py`.*

<div class="dlis-widget not-content" data-widget="ema-filter">Interactive figure — requires JavaScript. The static figure and tables above contain the same information.</div>

**No single $\alpha$ answers question 3 from §5.1.** A rule that forgets noise quickly also forgets real changes quickly. To do better, the update would have to *know* which inputs matter — to use a large weight for a genuine change and a small one for a glitch. That requires the decay itself to depend on the input. We will build exactly that in §5.3.8 (gates) and §5.3.10 (selective state-space models), and we will see a completely different solution — store everything, then choose at read time — in §5.3.11 (attention).

## 5.2.7 Where does α come from? Memory as a physical system

So far $\alpha$ was a number we picked. There is a deeper way to see it. A real thermometer does not jump to the room's temperature; it relaxes toward it, faster when the gap is larger. Newton's law of cooling says the rate of change of the reading is proportional to the gap:

$$
\tau\,\frac{dh}{dt} = x(t) - h(t),
$$

where $x(t)$ is the true temperature, $h(t)$ the thermometer's reading, and $\tau > 0$ the **time constant** (seconds). This is the simplest continuous-time memory: $h$ "remembers" past values of $x$ because it cannot change instantly.

**Solving it exactly over one sampling interval.** Our device sees $x$ only at sample times $t_0, t_0+\Delta t, \dots$. Assume the input is held constant at its sampled value $x$ during each interval — the **zero-order hold** (ZOH) assumption. Then on $[t_0, t_0+\Delta t]$, define the gap $g(t) = h(t) - x$. Since $x$ is constant, $dg/dt = dh/dt = -g/\tau$, whose solution is $g(t) = g(t_0)\,e^{-(t-t_0)/\tau}$. Therefore

$$
h(t_0+\Delta t) = x + \bigl(h(t_0) - x\bigr)e^{-\Delta t/\tau}
              = e^{-\Delta t/\tau}\, h(t_0) + \bigl(1 - e^{-\Delta t/\tau}\bigr)\, x .
$$

This is exactly the EMA, with

$$
\boxed{\,\alpha = e^{-\Delta t/\tau}.\,}
$$

The decay factor is not arbitrary: it is the sampled decay of a physical time constant. Several facts now follow at once:

- $0 < \alpha < 1$ automatically, for every $\tau > 0$ and $\Delta t > 0$.
- The half-life in **seconds** is $t_{1/2}\,\Delta t = \tau\ln 2$, independent of the sampling rate. Sample twice as fast and $\alpha$ moves closer to 1 by exactly the amount needed to keep the physical memory the same.
- For $\tau = 10$ s sampled every second, $\alpha = e^{-0.1} \approx 0.905$; for $\tau = 1$ s, $\alpha = e^{-1} \approx 0.368$.

The same construction — write a continuous-time system, then sample it exactly — is how modern **state-space models** (§5.3.9) obtain their recurrences, with a learned time step $\Delta$.

## 5.2.8 A warning: a careless discretization can make memory explode

There is a tempting shortcut. Approximate the derivative by a forward difference, $dh/dt \approx (h_t - h_{t-1})/\Delta t$ — the **forward Euler** method — evaluated at the old state:

$$
\tau\,\frac{h_t - h_{t-1}}{\Delta t} = x_t - h_{t-1}
\quad\Longrightarrow\quad
h_t = \Bigl(1 - \frac{\Delta t}{\tau}\Bigr)h_{t-1} + \frac{\Delta t}{\tau}\,x_t .
$$

This is again an EMA-shaped update, now with $\alpha_{\text{Euler}} = 1 - \Delta t/\tau$. Compare it with the exact factor:

| $\Delta t/\tau$ | exact $\alpha = e^{-\Delta t/\tau}$ | Euler $\alpha = 1 - \Delta t/\tau$ | Euler behaviour |
|---|---|---|---|
| 0.1 | 0.905 | 0.900 | close to exact |
| 1.0 | 0.368 | 0.000 | memory erased every step |
| 2.5 | 0.082 | −1.500 | **oscillates with growing amplitude** |

With $\alpha = -1.5$ the recurrence multiplies the state by $-1.5$ every step: a perturbation of size 1 becomes $-1.5, 2.25, -3.375, \dots$ The *physical* system is perfectly stable, but the *discrete* memory explodes. The recurrence $h_t = \alpha h_{t-1} + \dots$ is stable exactly when $|\alpha| < 1$, which for Euler means

$$
\Bigl|1 - \frac{\Delta t}{\tau}\Bigr| < 1 \iff 0 < \Delta t < 2\tau .
$$

![Decay factor α as a function of Δt/τ for the exact (zero-order hold) discretization, which stays between 0 and 1, and the forward Euler discretization, which leaves the stable band |α| < 1 when Δt exceeds 2τ.](/figures/u05-zoh-vs-euler.svg)

*Figure 5.2 — Exact discretization is stable for every step size; forward Euler only for $\Delta t < 2\tau$.*

<div class="dlis-widget not-content" data-widget="discretization">Interactive figure — requires JavaScript. The static figure and tables above contain the same information.</div>

:::tip[You have seen this inequality before]
In Unit 03, gradient descent on $L(w) = \tfrac{\lambda}{2}w^2$ updates $w \leftarrow (1-\eta\lambda)\,w$ and is stable exactly when $\eta < 2/\lambda$. That update *is* forward Euler applied to the continuous "gradient flow" $dw/dt = -\lambda w$, with time step $\eta$. The learning-rate limit of Unit 03 and the sampling limit here are the same theorem: forward Euler on a decaying mode with rate $\lambda$ is stable iff the step is below $2/\lambda$. §5.6 returns to this connection.
:::

## 5.2.9 What we have established

- A **state** is the summary through which all future behaviour depends on the past. What can be compressed into a small state depends on the question being asked.
- The **EMA** $h_t = \alpha h_{t-1} + (1-\alpha)x_t$ is a weighted average of the entire history with geometrically decaying weights $(1-\alpha)\alpha^{t-k}$.
- Its memory has half-life $\ln 2/(-\ln\alpha) \approx 0.693/(1-\alpha)$ steps.
- A fixed $\alpha$ trades noise suppression, $\tfrac{1-\alpha}{1+\alpha}$, against lag, $\alpha^t$; doing better requires a decay that depends on the input.
- $\alpha = e^{-\Delta t/\tau}$ is the exact sampled decay of a first-order system with time constant $\tau$; forward Euler gives $1 - \Delta t/\tau$ and is unstable for $\Delta t > 2\tau$.

**Next:** one decay rate is one timescale. Real signals carry information on many timescales at once — the last word, the current sentence, the topic of the chapter. In [§5.3](/core/unit-05/construction/) we give the memory a vector state and a matrix of decay rates, and discover that its eigenvalues are its memory modes.
