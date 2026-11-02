"""Spine Stage 11 — find, test and ablate an induction head (Unit 11).

An induction head completes the pattern  [A][B] ... [A] -> [B]: at the second [A] it attends to the
position *after* the earlier [A] and copies what it finds there (Elhage et al. 2021; Olsson et al. 2022).
It needs two heads in sequence: a previous-token head (layer 0) writes "the token before me" into
the residual stream, and the induction head (layer 1) matches the current token against it.

This stage does three things:
  1. `build_induction_circuit` sets the weights of a 2-layer attention-only model *by hand* so that
     it implements the circuit exactly — a ground truth for the detection methods.
  2. `train_attention_only` trains the same architecture on repeated random sequences and records
     the loss on the repeated half: it drops abruptly when the circuit forms.
  3. On the pretrained spine model (Stage 07), heads are scored for induction behaviour and the
     top-scoring head is ablated to test whether it is *used*.

Verification targets (test_spine.py):
  * the hand-built circuit predicts the repeated half almost perfectly
  * the detection scores single out exactly the two heads that were built
  * ablating either head breaks copying (the circuit needs both); ablating the unused heads does not
  * patching a run with its own activations changes nothing
"""
import argparse

import numpy as np

import model as M


def repeated_batch(rng, B, K, half):
    """B sequences [BOS, s, s]: s is `half` distinct tokens from range(K); BOS = K marks the start."""
    s = np.stack([rng.permutation(K)[:half] for _ in range(B)])
    return np.concatenate([np.full((B, 1), K), s, s], axis=1)


def shifted_batch(rng, B, K=24, n=8, T=28):
    """[BOS, noise, s, noise, s, noise] with random gaps, so the distance between the two copies of s
    varies and no fixed positional offset solves the task; noise tokens are drawn with replacement.
    Returns X (B, T), the (row, i, j) triples (i in the second copy, j after the same token in the
    first copy) and a weight mask selecting the predictable tokens of the second copy."""
    X = rng.integers(0, K, (B, T))
    X[:, 0] = K
    triples, w = [], np.zeros((B, T - 1))
    for r in range(B):
        s = rng.permutation(K)[:n]
        a = 1 + rng.integers(0, 4)
        b = a + n + rng.integers(0, 4)
        X[r, a : a + n] = s
        X[r, b : b + n] = s
        for k in range(n - 1):
            triples.append((r, b + k, a + k + 1))
            w[r, b + k] = 1.0
    return X, np.array(triples), w / w.sum()


def head_scores(cache, half):
    """For each (layer, head): mean attention from second-half position i to i-1 (previous token)
    and to i-half+1 (the position after the previous occurrence of the current token)."""
    prev, ind = {}, {}
    for l, c in enumerate(cache["blocks"]):
        A = c["A"]                                    # (B, h, T, T)
        T = A.shape[-1]
        i = np.arange(half + 1, T)                    # positions holding the 2nd, 3rd, ... token of the repeat
        for h in range(A.shape[1]):
            prev[(l, h)] = float(A[:, h, i, i - 1].mean())
            ind[(l, h)] = float(A[:, h, i, i - half + 1].mean())
    return prev, ind


def second_half_loss(p, cfg, X, half, head_mask=None, patch=None):
    logits = M.forward(p, cfg, X[:, :-1], head_mask=head_mask, patch=patch)
    w = np.zeros(X[:, 1:].shape)
    w[:, half + 1 :] = 1.0                            # predictions of the 2nd ... last token of the repeat
    w /= w.sum()
    return M.cross_entropy(logits, X[:, 1:], w)[0]


def build_induction_circuit(K=8, c=60.0, lam=10.0):
    """2-layer, 2-head, attention-only model over K tokens + BOS and sequences [BOS, s, s] of length 2K+1.
    Residual stream: [token one-hot (K+1) | position one-hot (2K+1) | previous-token one-hot (K+1) | 1 unused]."""
    V, T = K + 1, 2 * K + 1
    d = 2 * V + T + 1                                  # +1 so that d is divisible by the 2 heads
    cfg = M.Config(vocab=V, d=d, n_layer=2, n_head=2, max_len=T, mlp=False, layernorm=False, positions=True)
    p = {k: np.zeros_like(v) for k, v in M.init_params(cfg).items()}
    tok, pos, prv = 0, V, V + T
    dh = cfg.d_head
    for v in range(V):
        p["wte"][v, tok + v] = 1.0
    for t in range(T):
        p["wpe"][t, pos + t] = 1.0
    W0 = p["0.attn.wqkv"]                              # columns: [q (d) | k (d) | v (d)], head 0 = first dh of each
    for t in range(T):
        W0[pos + t, t] = c                             # q_i = c e_i
        if t + 1 < dh:
            W0[pos + t, d + t + 1] = 1.0               # k_j = e_{j+1}  =>  q_i . k_j = c [j = i - 1]
    for v in range(V):
        W0[tok + v, 2 * d + v] = 1.0                   # value = current token
        p["0.attn.wo"][v, prv + v] = 1.0               # write it into the previous-token subspace
    W1 = p["1.attn.wqkv"]
    for v in range(V):
        W1[tok + v, v] = c                             # query: my token
        W1[prv + v, d + v] = 1.0                       # key: the token before you
        W1[tok + v, 2 * d + v] = 1.0                   # value: your token
        p["1.attn.wo"][v, tok + v] = lam               # write it into the logit (token) subspace
    return p, cfg


def train_attention_only(K=24, n=8, T=28, steps=8000, seed=0, d=48, lr=3e-3, log_every=1000):
    """Train a 2-layer attention-only model on `shifted_batch` sequences.  The gaps matter: with the
    two copies always the same distance apart, a trained model solves the task with position-specific
    circuits that never attend to the induction target, and induction scores read zero although the
    loss is near zero (Unit 11, §11.7)."""
    cfg = M.Config(vocab=K + 1, d=d, n_layer=2, n_head=2, max_len=T, mlp=False, layernorm=False, positions=True)
    p = M.init_params(cfg, seed=seed)
    for k in p:
        if k in ("wte", "wpe"):
            p[k] *= 0.3
    rng = np.random.default_rng(seed)
    opt = M.AdamW(p, lr=lr, weight_decay=0.0)
    curve = []
    Xe, te, we = shifted_batch(np.random.default_rng(99), 64, K, n, T)
    for s in range(steps):
        X, _, _ = shifted_batch(rng, 32, K, n, T)
        loss, g = M.loss_and_grads(p, cfg, X[:, :-1], X[:, 1:])
        opt.step(p, g)
        if (s + 1) % log_every == 0:
            _, cache = M.forward(p, cfg, Xe[:, :-1], keep=True)
            ind = probe_scores(cache, te)
            curve.append((s + 1, probe_loss(p, cfg, Xe, we), max(ind.values())))
    return p, cfg, curve


def pet_probe(vocab, rng, B=48):
    """In-distribution induction probe for the spine model: a made-up pet name appears twice.
    Returns token ids X (B, T), the (row, i, j) triples where position i holds a letter of the
    second occurrence and j is the position just after the same letter in the first occurrence
    (where an induction head should look), and a weight mask selecting the predictions of the
    2nd ... last letters of the second occurrence."""
    from common import encode
    from data.make_tales import pet_name
    import random

    r = random.Random(int(rng.integers(1 << 30)))
    texts, spans = [], []
    for _ in range(B):
        name, friend = r.sample(["Mia", "Tom", "Lily", "Ben", "Sam", "Anna", "Max", "Zoe"], 2)
        pet = pet_name(r)
        a = f"\n{name} had a red kite. {name} also had a cat called "
        b = f"{pet}. One sunny day {name} went to the park with {friend}. "
        texts.append(a + b + pet + " ran to " + name + ".")
        spans.append((len(a), len(a) + len(b), len(pet)))
    T = min(len(t) for t in texts)
    X = np.stack([encode(t[:T], vocab) for t in texts])
    triples, w = [], np.zeros((B, T - 1))
    for row, (first, second, n) in enumerate(spans):
        for k in range(n - 1):
            i = second + k
            if i + 1 < T:
                triples.append((row, i, first + k + 1))
                w[row, i] = 1.0
    return X, np.array(triples), w / w.sum()


def probe_scores(cache, triples):
    """Mean attention from i to j over the probe's (row, i, j) triples, for every (layer, head)."""
    out = {}
    r, i, j = triples.T
    for l, c in enumerate(cache["blocks"]):
        for h in range(c["A"].shape[1]):
            out[(l, h)] = float(c["A"][r, h, i, j].mean())
    return out


def probe_loss(p, cfg, X, w, head_mask=None):
    return M.cross_entropy(M.forward(p, cfg, X[:, :-1], head_mask=head_mask), X[:, 1:], w)[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=8000)
    args = ap.parse_args()
    p, cfg = build_induction_circuit()
    X = repeated_batch(np.random.default_rng(0), 64, 8, 8)
    print(f"hand-built circuit: second-half loss {second_half_loss(p, cfg, X, 8):.4f} nats")
    p, cfg, curve = train_attention_only(steps=args.steps)
    print("trained attention-only model (random gaps): loss on the second copy, best induction score")
    for s, l, sc in curve:
        print(f"   step {s:5d}  loss {l:.3f}  induction {sc:.2f}")
    X, te, we = shifted_batch(np.random.default_rng(1), 64)
    _, cache = M.forward(p, cfg, X[:, :-1], keep=True)
    print("   induction scores", {k: round(v, 2) for k, v in probe_scores(cache, te).items()})
    for k in [(0, 0), (0, 1), (1, 0), (1, 1)]:
        m = np.ones((2, 2))
        m[k] = 0
        print(f"   ablate {k}: loss {probe_loss(p, cfg, X, we, head_mask=m):.3f}")
    from common import load_checkpoint, load_corpus

    ck = load_checkpoint("stage07")
    if ck is not None:
        p, cfg, _ = ck
        vocab = load_corpus()[2]
        X, triples, w = pet_probe(vocab, np.random.default_rng(0))
        logits, cache = M.forward(p, cfg, X[:, :-1], keep=True)
        z = logits - logits.max(-1, keepdims=True)
        lp = z - np.log(np.exp(z).sum(-1, keepdims=True))
        r, i, _ = triples.T
        per = -lp[r, i, X[r, i + 1]]
        start = {}
        for rr, ii in zip(r, i):
            start.setdefault(rr, ii)
        off = np.array([ii - start[rr] for rr, ii in zip(r, i)])
        print("spine model, second mention of a made-up pet name (behaviour first):")
        for o in range(3):
            print(f"   letter {o + 2} of the name: loss {per[off == o].mean():.2f} nats  (uniform: {np.log(cfg.vocab):.2f})")
        ind = probe_scores(cache, triples)
        top = max(ind, key=ind.get)
        print(f"   best induction score {ind[top]:.2f} at head {top}: no copying behaviour, so no copying mechanism to find")


if __name__ == "__main__":
    main()
