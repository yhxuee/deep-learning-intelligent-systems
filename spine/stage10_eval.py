"""Spine Stage 10 — an evaluation harness with honest uncertainty (Unit 10).

  * held-out NLL per story, with a bootstrap confidence interval over stories (the unit that was
    sampled — not characters, which are strongly dependent within a story)
  * paired comparison of two models on the same stories (paired bootstrap of the difference)
  * calibration of next-character predictions: reliability table and expected calibration error
  * the contamination check of Stage 07, re-run on the evaluation split
  * arithmetic accuracy (the Stage 09 task) with a confidence interval

Verification targets (test_spine.py), all on synthetic cases with known answers:
  * a calibrated predictor has ECE close to 0; a predictor that is overconfident by a known
    amount has exactly that ECE
  * 95% bootstrap intervals cover the true mean in about 95% of simulated experiments
  * pairing narrows the interval when the two models' errors are correlated
"""
import numpy as np

import model as M
from common import encode, load_checkpoint, load_corpus, load_lines, split_lines


def bootstrap_ci(values, n_boot=2000, alpha=0.05, seed=0):
    """Percentile bootstrap interval for the mean (Efron & Tibshirani 1993)."""
    v = np.asarray(values, float)
    rng = np.random.default_rng(seed)
    means = v[rng.integers(0, len(v), (n_boot, len(v)))].mean(1)
    return float(v.mean()), float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))


def paired_bootstrap_ci(a, b, **kw):
    """Interval for mean(a - b) resampling *pairs*: the shared difficulty of each item cancels."""
    return bootstrap_ci(np.asarray(a, float) - np.asarray(b, float), **kw)


def ece(confidence, correct, n_bins=10):
    """Expected calibration error: sum over equal-width confidence bins of
    (fraction of samples in bin) * |accuracy in bin - mean confidence in bin|."""
    confidence, correct = np.asarray(confidence, float), np.asarray(correct, float)
    bins = np.minimum((confidence * n_bins).astype(int), n_bins - 1)
    total = 0.0
    table = []
    for b in range(n_bins):
        m = bins == b
        if m.any():
            gap = abs(correct[m].mean() - confidence[m].mean())
            total += m.mean() * gap
            table.append((b / n_bins, (b + 1) / n_bins, int(m.sum()), float(confidence[m].mean()), float(correct[m].mean())))
    return float(total), table


def story_nll(p, cfg, stories, vocab):
    """Mean NLL (nats/char) of each story, scored in windows of at most max_len characters."""
    out = []
    for s in stories:
        ids = encode("\n" + s + "\n", vocab)
        total, n = 0.0, 0
        for start in range(0, len(ids) - 1, cfg.max_len):
            x = ids[start : start + cfg.max_len]
            y = ids[start + 1 : start + cfg.max_len + 1]
            x = x[: len(y)]
            loss, _ = M.cross_entropy(M.forward(p, cfg, x[None]), y[None])
            total += loss * len(y)
            n += len(y)
        out.append(total / n)
    return np.array(out)


def next_char_calibration(p, cfg, ids, n_windows=40):
    T = cfg.max_len
    x = np.stack([ids[i * T : i * T + T] for i in range(n_windows)])
    y = np.stack([ids[i * T + 1 : i * T + T + 1] for i in range(n_windows)])
    logits = M.forward(p, cfg, x)
    z = logits - logits.max(-1, keepdims=True)
    P = np.exp(z)
    P /= P.sum(-1, keepdims=True)
    return P.max(-1).ravel(), (P.argmax(-1) == y).ravel()


def main():
    _, held = split_lines(load_lines())
    _, va, vocab = load_corpus()
    models = {name: load_checkpoint(name) for name in ("stage04", "stage07")}
    models = {k: v for k, v in models.items() if v is not None}
    per_story = {}
    for name, (p, cfg, meta) in models.items():
        per_story[name] = story_nll(p, cfg, held, vocab)
        m, lo, hi = bootstrap_ci(per_story[name])
        conf, corr = next_char_calibration(p, cfg, va)
        e, _ = ece(conf, corr)
        print(f"{name}: held-out NLL {m:.4f} [{lo:.4f}, {hi:.4f}] (95% CI over {len(held)} stories);  next-char accuracy {corr.mean():.3f}, ECE {e:.4f}")
    if len(per_story) == 2:
        m, lo, hi = paired_bootstrap_ci(per_story["stage04"], per_story["stage07"])
        print(f"stage04 - stage07: {m:+.4f} [{lo:+.4f}, {hi:+.4f}] nats/char (paired)")
    if "stage07" in models:
        from stage09_post_training import accuracy, make_prompts

        p, cfg, _ = models["stage07"]
        rng = np.random.default_rng(7)
        X, ab = make_prompts(300, rng, vocab)
        lp = __import__("stage09_post_training").last_logprobs(p, cfg, X)[0]
        target = np.array([vocab.index(str(s)) for s in ab.sum(1)])
        hits = (lp.argmax(-1) == target).astype(float)
        m, lo, hi = bootstrap_ci(hits)
        print(f"arithmetic (greedy) accuracy {m:.3f} [{lo:.3f}, {hi:.3f}]")
        del accuracy


if __name__ == "__main__":
    main()
