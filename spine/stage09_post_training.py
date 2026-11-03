"""Spine Stage 09 — post-training with a verifiable reward (Unit 09).

Task.  The pretraining corpus contains sentences such as "Ben had 2 eggs. Then Ben found 3 more.
Now Ben has 5 eggs." — but 30% of the written answers are wrong (noisy text, see
data/make_tales.py).  A maximum-likelihood model therefore *imitates the noise*.  Post-training
asks the model to answer and rewards it only when a program confirms the answer: r = 1[answer = a+b].

Methods (all act on the single answer character):
  * REINFORCE with a leave-one-out baseline and a KL penalty to the pretrained (reference) model:
        A_i = r_i - mean_{j != i} r_j          (independent of a_i, so the estimator is unbiased)
        dlogits = (A_i / B) (pi - e_{a_i})  +  (beta / B) pi * (log pi - log pi_ref - KL)
    The plain batch mean, A_i = r_i - mean_j r_j, includes the sample's own reward and scales the
    expected reward gradient by (B - 1) / B (Unit 09, Section 9.3.2; audit finding K03).  The prompts in
    a batch differ, so neither baseline is state-conditioned; both are constants across actions.
  * DPO on (correct, incorrect) pairs (Rafailov et al. 2023):
        loss = -log sigmoid(beta [(log pi(y_w) - log ref(y_w)) - (log pi(y_l) - log ref(y_l))])
  * Reward-hacking audit: the same REINFORCE run with a buggy checker (answer >= a+b instead of ==).

Verification targets (test_spine.py):
  * the policy-gradient estimator is unbiased (Monte-Carlo mean = exact gradient on a bandit),
    with the leave-one-out baseline; the batch-mean baseline is biased by the factor (B - 1) / B
  * the discounted policy-gradient theorem carries the factor gamma^t (two-step MDP)
  * the KL-regularized optimum is pi* proportional to pi_ref * exp(r / beta)
  * the DPO and KL gradients pushed through the spine model match finite differences
"""
import argparse

import numpy as np

import model as M
from common import encode, load_checkpoint, load_corpus

NAMES = ["Mia", "Tom", "Lily", "Ben", "Sam", "Anna", "Max", "Zoe", "Leo", "Ivy"]
FRUIT = ["apples", "pears", "plums", "nuts", "eggs", "cakes"]
CTX = 96


def make_prompts(n, rng, vocab):
    """Prompts cropped to the last CTX characters, all the same length (absolute positions stay in
    the distribution the model was trained on: random windows of the corpus)."""
    X, ab = [], []
    for _ in range(n):
        name, friend = rng.choice(NAMES, 2, replace=False)
        a, b = int(rng.integers(1, 6)), int(rng.integers(1, 5))
        fruit = rng.choice(FRUIT)
        text = (f"{friend} had a red ball. One sunny day {name} went to the park with {friend}. "
                f"{name} had {a} {fruit}. Then {name} found {b} more. Now {name} has ")
        X.append(encode(text[-CTX:], vocab))
        ab.append((a, b))
    return np.stack(X), np.array(ab)


def last_logprobs(p, cfg, X):
    logits, cache = M.forward(p, cfg, X, keep=True)
    z = logits[:, -1]
    z = z - z.max(-1, keepdims=True)
    return z - np.log(np.exp(z).sum(-1, keepdims=True)), logits, cache


def grad_from_last(p, cfg, cache, logits, dlast):
    d = np.zeros_like(logits)
    d[:, -1] = dlast
    return M.backward(p, cfg, cache, d)


def advantages(rewards, baseline="loo"):
    """Reward minus a baseline that does not depend on the sample's own action.

    "loo":  b_i = mean of the other B - 1 rewards (unbiased);
    "mean": b_i = mean of all B rewards, including r_i (biased by the factor (B - 1) / B)."""
    rewards = np.asarray(rewards, float)
    B = len(rewards)
    if baseline == "mean":
        return rewards - rewards.mean()
    if baseline == "loo":
        return rewards - (rewards.sum() - rewards) / (B - 1)
    raise ValueError(baseline)


def reinforce_grad(p, cfg, X, actions, rewards, ref_logp, beta, baseline="loo"):
    """Gradient of  -(1/B) sum_i A_i log pi(a_i)  +  beta * mean KL(pi || pi_ref)  (to be minimized)."""
    logp, logits, cache = last_logprobs(p, cfg, X)
    pi = np.exp(logp)
    B, V = pi.shape
    A = advantages(rewards, baseline)
    d = pi * A[:, None]
    d[np.arange(B), actions] -= A
    kl = (pi * (logp - ref_logp)).sum(-1)
    d += beta * pi * (logp - ref_logp - kl[:, None])
    return grad_from_last(p, cfg, cache, logits, d / B), float(kl.mean())


def dpo_loss_grad(p, cfg, X, yw, yl, ref_logp, beta):
    logp, logits, cache = last_logprobs(p, cfg, X)
    B = len(X)
    i = np.arange(B)
    u = beta * ((logp[i, yw] - ref_logp[i, yw]) - (logp[i, yl] - ref_logp[i, yl]))
    loss = float(np.mean(np.logaddexp(0.0, -u)))            # -log sigmoid(u), computed stably
    s = 1.0 / (1.0 + np.exp(u))                               # sigmoid(-u)
    d = np.zeros_like(logp)
    d[i, yw] -= beta * s
    d[i, yl] += beta * s                                      # the pi terms of d log pi cancel
    return loss, grad_from_last(p, cfg, cache, logits, d / B)


def digit_ids(vocab):
    """Vocabulary index of each digit 0-9 (-1 if the digit never occurs in the corpus)."""
    return np.array([vocab.index(str(k)) if str(k) in vocab else -1 for k in range(10)])


def accuracy(p, cfg, X, ab, vocab, rng):
    logp = last_logprobs(p, cfg, X)[0]
    target = digit_ids(vocab)[ab.sum(1)]
    pi = np.exp(logp)
    return {
        "greedy": float((logp.argmax(-1) == target).mean()),
        "p_correct": float(pi[np.arange(len(X)), target].mean()),
    }


def post_train(p, cfg, vocab, method="reinforce", steps=150, B=64, beta=0.05, lr=1e-3, seed=0, reward="exact"):
    rng = np.random.default_rng(seed)
    ref = {k: v.copy() for k, v in p.items()}
    opt = M.AdamW(p, lr=lr, weight_decay=0.0)
    dig = digit_ids(vocab)
    log = []
    for step in range(steps):
        X, ab = make_prompts(B, rng, vocab)
        ref_logp = last_logprobs(ref, cfg, X)[0]
        if method == "reinforce":
            logp = last_logprobs(p, cfg, X)[0]
            pi = np.exp(logp)
            actions = np.array([rng.choice(len(row), p=row / row.sum()) for row in pi])
            ans = np.array([vocab[a] for a in actions])
            val = np.array([int(c) if c.isdigit() else -1 for c in ans])
            gold = (val == ab.sum(1)).astype(float)
            r = gold if reward == "exact" else (val >= ab.sum(1)).astype(float)
            g, kl = reinforce_grad(p, cfg, X, actions, r, ref_logp, beta)
            log.append((float(r.mean()), float(gold.mean()), kl))
        else:
            yw = dig[ab.sum(1)]
            wrong = 2 + (ab.sum(1) - 2 + rng.integers(1, 8, B)) % 8      # a different answer in 2..9
            yl = dig[wrong]
            loss, g = dpo_loss_grad(p, cfg, X, yw, yl, ref_logp, beta)
            log.append((loss,))
        opt.step(p, g)
    return np.array(log)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=150)
    args = ap.parse_args()
    ck = load_checkpoint("stage07")
    if ck is None:
        raise SystemExit("run stage07_pretrain.py first")
    base, cfg, _ = ck
    vocab = load_corpus()[2]
    rng = np.random.default_rng(123)
    Xe, abe = make_prompts(400, rng, vocab)
    print("pretrained:", accuracy(base, cfg, Xe, abe, vocab, rng))
    for method, reward in [("reinforce", "exact"), ("dpo", "exact"), ("reinforce", "buggy >=")]:
        p = {k: v.copy() for k, v in base.items()}
        log = post_train(p, cfg, vocab, method=method, steps=args.steps, reward="exact" if reward == "exact" else "buggy")
        acc = accuracy(p, cfg, Xe, abe, vocab, rng)
        extra = ""
        if method == "reinforce":
            extra = f"  last-20 train reward {log[-20:, 0].mean():.2f}, gold {log[-20:, 1].mean():.2f}, KL {log[-20:, 2].mean():.3f}"
        print(f"{method:9s} reward={reward:9s} -> held-out greedy acc {acc['greedy']:.2f}, mean p(correct) {acc['p_correct']:.2f}{extra}")


if __name__ == "__main__":
    main()
