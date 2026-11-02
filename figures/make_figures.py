"""Regenerate every figure in public/figures/ from code, so each figure is reproducible.

    pip install numpy matplotlib
    python figures/make_figures.py

Figures are static SVGs on a light surface. Palette: validated categorical slots 1-3
(blue #2a78d6, orange #eb6834, aqua #1baf7a); at most three series per figure;
every multi-series figure has a legend; numbers that matter are stated in the text.
"""
import math
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

OUT = pathlib.Path(__file__).resolve().parent.parent / "public" / "figures"
OUT.mkdir(parents=True, exist_ok=True)
C1, C2, C3 = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False, "lines.linewidth": 2,
    "font.size": 11, "legend.frameon": False, "svg.fonttype": "none",
    "font.family": ["DejaVu Sans"],
})


def save(fig, name):
    fig.tight_layout()
    fig.savefig(OUT / f"{name}.svg", format="svg")
    plt.close(fig)
    print("wrote", name)


# ------------------------------------------------------------------ Unit 03
def u03_gd_step_sizes():
    a, w0, k = 4.0, 1.0, np.arange(0, 9)
    fig, ax = plt.subplots(figsize=(7, 3.6))
    for eta, c in [(0.1, C1), (0.4, C2), (0.6, C3)]:
        ax.plot(k, w0 * (1 - eta * a) ** k, "-o", color=c, ms=5, label=f"η = {eta}  (factor {1 - eta * a:+.1f})")
    ax.axhline(0, color=INK2, lw=1)
    ax.set_ylim(-4, 4)
    ax.set_xlabel("iteration k")
    ax.set_ylabel("$w_k$")
    ax.set_title("Gradient descent on L(w) = 2w²  (a = 4, stable iff η < 0.5)", fontsize=11, loc="left")
    ax.legend(loc="upper left")
    save(fig, "u03-gd-step-sizes")


def u03_quadratic_paths():
    lam = np.array([1.0, 10.0])
    def run(eta, beta, steps=40):
        x, prev, path = np.array([1.0, 1.0]), np.array([1.0, 1.0]), [np.array([1.0, 1.0])]
        for _ in range(steps):
            new = x - eta * lam * x + beta * (x - prev)
            prev, x = x, new
            path.append(x)
        return np.array(path)
    gd = run(2 / 11, 0.0)
    r = (math.sqrt(10) - 1) / (math.sqrt(10) + 1)
    hb = run(4 / (math.sqrt(10) + 1) ** 2, r ** 2)
    X, Y = np.meshgrid(np.linspace(-1.2, 1.2, 300), np.linspace(-1.2, 1.2, 300))
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    ax.contour(X, Y, 0.5 * (X ** 2 + 10 * Y ** 2), levels=[0.01, 0.05, 0.2, 0.5, 1, 2, 4], colors=GRID, linewidths=1)
    ax.plot(gd[:, 0], gd[:, 1], "-o", color=C1, ms=3.5, lw=1.5, label="gradient descent, η = 2/11")
    ax.plot(hb[:, 0], hb[:, 1], "-o", color=C2, ms=3.5, lw=1.5, label="heavy-ball momentum (tuned)")
    ax.plot([0], [0], marker="*", color=INK, ms=12)
    ax.set_aspect("equal")
    ax.set_xlabel("$\\theta_1$  (curvature 1)")
    ax.set_ylabel("$\\theta_2$  (curvature 10)")
    ax.set_title("κ = 10: GD zig-zags across the narrow valley", fontsize=11, loc="left")
    ax.legend(loc="lower left")
    save(fig, "u03-quadratic-paths")


def u03_finite_difference_error():
    hs = np.logspace(-13, -1, 200)
    x0 = 1.0
    c = np.abs((np.sin(x0 + hs) - np.sin(x0 - hs)) / (2 * hs) - np.cos(x0))
    f = np.abs((np.sin(x0 + hs) - np.sin(x0)) / hs - np.cos(x0))
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.loglog(hs, f, color=C2, label="forward difference  (error ∝ h, then ε/h)")
    ax.loglog(hs, c, color=C1, label="central difference  (error ∝ h², then ε/h)")
    ax.axvline((3 * 2.0 ** -52) ** (1 / 3), color=C1, lw=1, ls="--")
    ax.axvline(math.sqrt(2.0 ** -52), color=C2, lw=1, ls="--")
    ax.set_xlabel("step size h")
    ax.set_ylabel("|error| in d/dx sin(x) at x = 1")
    ax.set_title("Truncation error falls with h; rounding error rises (float64)", fontsize=11, loc="left")
    ax.legend(loc="upper right")
    save(fig, "u03-finite-difference-error")


def u03_vanishing_gradients():
    rng = np.random.default_rng(0)
    n, depth, trials = 256, 50, 8
    acts = {
        "sigmoid, N(0, 1/n) weights": (lambda z: 1 / (1 + np.exp(-z)), lambda z: (s := 1 / (1 + np.exp(-z))) * (1 - s), 1.0, C2),
        "tanh, Glorot N(0, 1/n)": (np.tanh, lambda z: 1 - np.tanh(z) ** 2, 1.0, C1),
        "ReLU, He N(0, 2/n)": (lambda z: np.maximum(z, 0), lambda z: (z > 0).astype(float), 2.0, C3),
    }
    fig, ax = plt.subplots(figsize=(7, 4))
    for label, (f, df, gain, col) in acts.items():
        norms = np.zeros(depth + 1)
        for _ in range(trials):
            Ws = [rng.normal(0, math.sqrt(gain / n), (n, n)) for _ in range(depth)]
            h, zs = rng.normal(size=n), []
            for W in Ws:
                z = W @ h
                zs.append(z)
                h = f(z)
            g = rng.normal(size=n)
            g /= np.linalg.norm(g)
            out = [1.0]
            for W, z in zip(reversed(Ws), reversed(zs)):
                g = W.T @ (g * df(z))
                out.append(np.linalg.norm(g))
            norms += np.log10(np.array(out))
        ax.plot(np.arange(depth + 1), norms / trials, color=col, label=label)
    ax.set_xlabel("layers back from the output")
    ax.set_ylabel("log₁₀ ‖gradient‖")
    ax.set_title("Backpropagated gradient norm through a 50-layer MLP (width 256)", fontsize=11, loc="left")
    ax.legend(loc="lower left")
    save(fig, "u03-vanishing-gradients")


# ------------------------------------------------------------------ Unit 04
def u04_soft_lookup():
    b = np.linspace(0, 10, 200)
    w = np.exp(b) / (np.exp(b) + 3)
    out = w * 7 + (1 - w) / 3 * 4
    fig, ax = plt.subplots(figsize=(7, 3.6))
    ax.plot(b, out, color=C1)
    ax.axhline(7, color=INK2, lw=1, ls="--")
    ax.text(10, 7.1, "hard lookup: 7", ha="right", va="bottom", color=INK2)
    for bb in [1, 2, 4, 8]:
        y = math.exp(bb) / (math.exp(bb) + 3) * 7 + (3 / (math.exp(bb) + 3)) / 3 * 4
        ax.plot([bb], [y], "o", color=C1, ms=6)
        ax.annotate(f"β={bb}: {y:.2f}", (bb, y), textcoords="offset points", xytext=(6, -14), color=INK)
    ax.set_xlabel("score scale β")
    ax.set_ylabel("retrieved value")
    ax.set_title("Soft lookup of key b in {a→3, b→7, c→1} (single series)", fontsize=11, loc="left")
    save(fig, "u04-soft-lookup")


def u04_attention_heatmap():
    A = np.array([[1, 0, 0], [0.330238, 0.669762, 0], [0.248255, 0.50349, 0.248255]])
    fig, ax = plt.subplots(figsize=(6, 4.4))
    ax.grid(False)
    ax.imshow(A, cmap="Blues", vmin=0, vmax=1)
    for i in range(3):
        for j in range(3):
            masked = j > i
            ax.text(j, i, "masked" if masked else f"{A[i, j]:.3f}", ha="center", va="center",
                    color=INK2 if masked else ("white" if A[i, j] > 0.6 else INK), fontsize=11)
    ax.set_xticks(range(3), ["key 1", "key 2", "key 3"])
    ax.set_yticks(range(3), ["query 1", "query 2", "query 3"])
    ax.set_title("Causal attention weights (Worked Example 4.2)", fontsize=11, loc="left")
    save(fig, "u04-attention-heatmap")


def u04_softmax_saturation():
    rng = np.random.default_rng(1)
    dks = [4, 8, 16, 32, 64, 128, 256, 512]
    T = 64
    res = {"unscaled  q·k": [], "scaled  q·k / √d_k": []}
    for dk in dks:
        q, k = rng.normal(size=(400, dk)), rng.normal(size=(T, dk))
        for name, s in [("unscaled  q·k", 1.0), ("scaled  q·k / √d_k", math.sqrt(dk))]:
            L = q @ k.T / s
            P = np.exp(L - L.max(1, keepdims=True))
            P /= P.sum(1, keepdims=True)
            res[name].append(P.max(1).mean())
    fig, ax = plt.subplots(figsize=(7, 3.8))
    ax.semilogx(dks, res["unscaled  q·k"], "-o", color=C2, base=2, ms=5, label="unscaled  q·k")
    ax.semilogx(dks, res["scaled  q·k / √d_k"], "-o", color=C1, base=2, ms=5, label="scaled  q·k / √d_k")
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("key dimension d_k")
    ax.set_ylabel("mean largest attention weight")
    ax.set_title("Without scaling, softmax over 64 keys collapses toward one-hot", fontsize=11, loc="left")
    ax.legend(loc="center right")
    save(fig, "u04-softmax-saturation")


def u04_flops_crossover():
    d = 768
    T = np.logspace(6, 15, 200, base=2)
    fig, ax = plt.subplots(figsize=(7, 3.8))
    ax.loglog(T, 24 * T * d * d, color=C1, base=2, label="projections + MLP  (24·T·d²)")
    ax.loglog(T, 4 * T * T * d, color=C2, base=2, label="attention scores and mixing  (4·T²·d)")
    ax.axvline(6 * d, color=INK2, lw=1, ls="--")
    ax.text(6 * d * 1.1, 1e9, "T = 6d = 4608", color=INK2)
    ax.set_xlabel("sequence length T  (d = 768)")
    ax.set_ylabel("forward FLOPs per block")
    ax.set_title("Where the compute goes in one Transformer block", fontsize=11, loc="left")
    ax.legend(loc="upper left")
    save(fig, "u04-flops-crossover")


# ------------------------------------------------------------------ Unit 05
def u05_ema_noisy_step():
    rng = np.random.default_rng(3)
    T = 120
    x = np.where(np.arange(T) < 40, 20.0, 23.0) + rng.normal(0, 0.8, T)
    x[70] += 6
    fig, ax = plt.subplots(figsize=(7.5, 4))
    ax.plot(x, color="#b9b8b2", lw=1, label="sensor reading $x_t$")
    for a, c in [(0.5, C1), (0.9, C2), (0.98, C3)]:
        h, hs = 20.0, []
        for xt in x:
            h = a * h + (1 - a) * xt
            hs.append(h)
        ax.plot(hs, color=c, label=f"EMA, α = {a}")
    ax.annotate("spike", (70, x[70]), textcoords="offset points", xytext=(8, 0), color=INK2)
    ax.set_xlabel("time step t (seconds)")
    ax.set_ylabel("temperature (°C)")
    ax.set_title("Larger α: smoother but slower to follow the true step at t = 40", fontsize=11, loc="left")
    ax.legend(loc="upper left", ncol=2)
    save(fig, "u05-ema-noisy-step")


def u05_impulse_responses():
    t = np.arange(0, 41)
    th = math.pi / 6
    A = 0.9 * np.array([[math.cos(th), -math.sin(th)], [math.sin(th), math.cos(th)]])
    h, rot = np.array([1.0, 0.0]), []
    for _ in t:
        rot.append(h[0])
        h = A @ h
    fig, ax = plt.subplots(figsize=(7, 3.8))
    ax.plot(t, 0.5 ** t, "-o", ms=3, color=C1, label="mode λ = 0.5 (fast)")
    ax.plot(t, 0.95 ** t, "-o", ms=3, color=C2, label="mode λ = 0.95 (slow)")
    ax.plot(t, rot, "-o", ms=3, color=C3, label="$\\lambda = 0.9\\,e^{\\pm i\\pi/6}$ (oscillating)")
    ax.axhline(0, color=INK2, lw=1)
    ax.set_xlabel("steps since the impulse")
    ax.set_ylabel("response")
    ax.set_title("Impulse responses: each eigenvalue is a memory mode", fontsize=11, loc="left")
    ax.legend(loc="upper right")
    save(fig, "u05-impulse-responses")


def u05_nonnormal():
    t = np.arange(0, 81)
    N = np.array([[0.9, 5.0], [0.0, 0.9]])
    norms_n = [np.linalg.norm(np.linalg.matrix_power(N, k), 2) for k in t]
    fig, ax = plt.subplots(figsize=(7, 3.8))
    ax.plot(t, norms_n, color=C2, label="non-normal  [[0.9, 5], [0, 0.9]]")
    ax.plot(t, 0.9 ** t, color=C1, label="normal  diag(0.9, 0.9)")
    ax.annotate("peak ≈ 19.4 at t = 9–10", (9.5, 19.4), textcoords="offset points", xytext=(12, -4), color=INK)
    ax.set_xlabel("t")
    ax.set_ylabel("‖Aᵗ‖₂")
    ax.set_title("Same eigenvalues (both 0.9), very different transients", fontsize=11, loc="left")
    ax.legend(loc="upper right")
    save(fig, "u05-nonnormal")


def u05_zoh_vs_euler():
    r = np.linspace(0, 3, 300)
    fig, ax = plt.subplots(figsize=(7, 3.8))
    ax.axhspan(-1, 1, color="#eef3fb", zorder=0)
    ax.text(2.95, 0.9, "stable band |α| < 1", ha="right", va="top", color=INK2)
    ax.plot(r, np.exp(-r), color=C1, label="exact (zero-order hold): $\\alpha = e^{-\\Delta t/\\tau}$")
    ax.plot(r, 1 - r, color=C2, label="forward Euler: $\\alpha = 1 - \\Delta t/\\tau$")
    ax.axvline(2, color=INK2, lw=1, ls="--")
    ax.text(2.03, -1.8, "Δt = 2τ", color=INK2)
    ax.set_ylim(-2.1, 1.2)
    ax.set_xlabel("Δt / τ")
    ax.set_ylabel("decay factor α")
    ax.set_title("Two discretizations of τ·dh/dt = x − h", fontsize=11, loc="left")
    ax.legend(loc="lower left")
    save(fig, "u05-zoh-vs-euler")


def u05_linear_attention_capacity():
    rng = np.random.default_rng(4)
    d = 16
    ns = [1, 2, 4, 8, 12, 16, 24, 32, 48, 64, 96, 128]
    emp = []
    for n in ns:
        errs = []
        for _ in range(40):
            K = rng.normal(size=(n, d))
            K /= np.linalg.norm(K, axis=1, keepdims=True)
            V = rng.normal(size=(n, 8))
            M = V.T @ K
            R = (M @ K.T).T
            errs.append(np.mean(np.linalg.norm(R - V, axis=1) / np.linalg.norm(V, axis=1)))
        emp.append(np.mean(errs))
    nn = np.linspace(1, 128, 200)
    fig, ax = plt.subplots(figsize=(7, 3.8))
    ax.plot(nn / d, np.sqrt((nn - 1) / d), color=C2, label="first-order estimate  $\\sqrt{(n-1)/d_k}$")
    ax.plot(np.array(ns) / d, emp, "o", color=C1, ms=6, label="measured, random unit keys")
    ax.axvline(1, color=INK2, lw=1, ls="--")
    ax.text(1.05, 2.4, "n = d_k", color=INK2)
    ax.set_xlabel("stored pairs per key dimension  n / d_k  (d_k = 16)")
    ax.set_ylabel("relative retrieval error")
    ax.set_title("Linear-attention memory: interference grows with load", fontsize=11, loc="left")
    ax.legend(loc="upper left")
    save(fig, "u05-linear-attention-capacity")


def u05_bptt_gradient_decay():
    rng = np.random.default_rng(5)
    n, T = 64, 60
    fig, ax = plt.subplots(figsize=(7, 3.8))
    for rho, c in [(0.5, C1), (1.0, C2), (1.5, C3)]:
        acc = np.zeros(T + 1)
        for _ in range(10):
            W = rng.normal(size=(n, n))
            W *= rho / np.max(np.abs(np.linalg.eigvals(W)))
            h, Js = np.zeros(n), []
            for _ in range(T):
                h = np.tanh(W @ h + rng.normal(0, 1, n))
                Js.append(np.diag(1 - h ** 2) @ W)
            P, out = np.eye(n), [0.0]
            for J in reversed(Js):
                P = P @ J
                out.append(np.log10(np.linalg.norm(P, 2) + 1e-300))
            acc += np.array(out)
        ax.plot(np.arange(T + 1), acc / 10, color=c, label=f"spectral radius of W = {rho}")
    ax.set_xlabel("time lag t − k")
    ax.set_ylabel("log₁₀ ‖∂h_t / ∂h_k‖₂")
    ax.set_title("tanh RNN (n = 64): gradients through time shrink even at ρ ≥ 1", fontsize=11, loc="left")
    ax.legend(loc="lower left")
    save(fig, "u05-bptt-gradient-decay")


if __name__ == "__main__":
    for fn in [u03_gd_step_sizes, u03_quadratic_paths, u03_finite_difference_error, u03_vanishing_gradients,
               u04_soft_lookup, u04_attention_heatmap, u04_softmax_saturation, u04_flops_crossover,
               u05_ema_noisy_step, u05_impulse_responses, u05_nonnormal, u05_zoh_vs_euler,
               u05_linear_attention_capacity, u05_bptt_gradient_decay]:
        fn()
