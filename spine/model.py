"""The spine model: a small decoder-only Transformer in NumPy with hand-derived backward passes.

Every later spine stage uses this one file, so the model you build in Units 02-05 is the
model you size (06), pretrain (07), sample (08), post-train (09), evaluate (10), dissect (11)
and experiment on (12).

Why NumPy and hand-written gradients?  Unit 03 builds reverse-mode differentiation, Unit 04
derives the attention backward pass (§4.3.6), and this file is those derivations, executed.
`test_model.py` checks every gradient against central finite differences, so the reference
runs anywhere with no framework.  Learners may write their own stages with a framework's
autograd and compare against this reference.

Architecture (pre-LayerNorm, as in Unit 04 §4.3.9):
    x0 = E[tokens] + P[positions]
    for each block:  x = x + Attn(LN1(x));  x = x + MLP(LN2(x))
    logits = LN_f(x) E^T                       (tied embedding / unembedding, Unit 02)
Switches let Unit 11 build attention-only models (no MLP, no LayerNorm).
"""
from dataclasses import dataclass, asdict

import numpy as np

# ----------------------------------------------------------------------------- FLOP counter
FLOPS = {"count": 0}


def mm(a, b):
    """Matrix product that counts 2*m*n*k FLOPs (Unit 06 checks the 6N rule with this)."""
    out = a @ b
    k = a.shape[-1]
    FLOPS["count"] += 2 * out.size * k
    return out


@dataclass
class Config:
    vocab: int
    d: int = 64
    n_layer: int = 2
    n_head: int = 4
    max_len: int = 128
    mlp: bool = True
    layernorm: bool = True
    positions: bool = True

    @property
    def d_head(self):
        assert self.d % self.n_head == 0
        return self.d // self.n_head


# ----------------------------------------------------------------------------- parameters
def init_params(cfg: Config, seed=0, dtype=np.float64):
    """Variance-preserving initialization (Unit 03 §3.8): std 1/sqrt(fan_in); residual
    output projections scaled by 1/sqrt(2 * n_layer) so the residual stream does not grow with depth."""
    rng = np.random.default_rng(seed)
    d, L = cfg.d, cfg.n_layer
    n = lambda *shape, std: (rng.standard_normal(shape) * std).astype(dtype)
    p = {"wte": n(cfg.vocab, d, std=0.02 if cfg.layernorm else 1.0)}
    if cfg.positions:
        p["wpe"] = n(cfg.max_len, d, std=0.02 if cfg.layernorm else 1.0)
    res = 1.0 / np.sqrt(2 * L)
    for l in range(L):
        if cfg.layernorm:
            p[f"{l}.ln1.g"], p[f"{l}.ln1.b"] = np.ones(d, dtype), np.zeros(d, dtype)
        p[f"{l}.attn.wqkv"] = n(d, 3 * d, std=1 / np.sqrt(d))
        p[f"{l}.attn.wo"] = n(d, d, std=res / np.sqrt(d))
        if cfg.mlp:
            if cfg.layernorm:
                p[f"{l}.ln2.g"], p[f"{l}.ln2.b"] = np.ones(d, dtype), np.zeros(d, dtype)
            p[f"{l}.mlp.wfc"] = n(d, 4 * d, std=1 / np.sqrt(d))
            p[f"{l}.mlp.bfc"] = np.zeros(4 * d, dtype)
            p[f"{l}.mlp.wproj"] = n(4 * d, d, std=res / np.sqrt(4 * d))
            p[f"{l}.mlp.bproj"] = np.zeros(d, dtype)
    if cfg.layernorm:
        p["lnf.g"], p["lnf.b"] = np.ones(d, dtype), np.zeros(d, dtype)
    return p


def n_params(p, cfg: Config, embedding=True):
    total = sum(v.size for v in p.values())
    if not embedding:
        total -= p["wte"].size + (p["wpe"].size if "wpe" in p else 0)
    return total


def matmul_params(cfg: Config):
    """Parameters that enter a matrix product once per token: 12 L d^2 (+ V d for the tied read-out)."""
    per_layer = 4 * cfg.d ** 2 + (8 * cfg.d ** 2 if cfg.mlp else 0)
    return cfg.n_layer * per_layer + cfg.vocab * cfg.d


# ----------------------------------------------------------------------------- building blocks
def layernorm(x, g, b, eps=1e-5):
    mu = x.mean(-1, keepdims=True)
    var = x.var(-1, keepdims=True)
    rstd = 1.0 / np.sqrt(var + eps)
    xhat = (x - mu) * rstd
    return xhat * g + b, (xhat, rstd, g)


def layernorm_backward(dy, cache):
    xhat, rstd, g = cache
    dg = (dy * xhat).reshape(-1, xhat.shape[-1]).sum(0)
    db = dy.reshape(-1, xhat.shape[-1]).sum(0)
    dxhat = dy * g
    dx = rstd * (dxhat - dxhat.mean(-1, keepdims=True) - xhat * (dxhat * xhat).mean(-1, keepdims=True))
    return dx, dg, db


C_GELU = np.sqrt(2.0 / np.pi)


def gelu(u):
    t = np.tanh(C_GELU * u * (1.0 + 0.044715 * u * u))
    return 0.5 * u * (1.0 + t), t


def gelu_backward(du_out, u, t):
    return du_out * (0.5 * (1.0 + t) + 0.5 * u * (1.0 - t * t) * C_GELU * (1.0 + 3 * 0.044715 * u * u))


def softmax(s, axis=-1):
    s = s - s.max(axis=axis, keepdims=True)
    e = np.exp(s)
    return e / e.sum(axis=axis, keepdims=True)


def split_heads(x, h):
    B, T, d = x.shape
    return x.reshape(B, T, h, d // h).transpose(0, 2, 1, 3)


def merge_heads(x):
    B, h, T, dh = x.shape
    return x.transpose(0, 2, 1, 3).reshape(B, T, h * dh)


# ----------------------------------------------------------------------------- forward / backward
def forward(p, cfg: Config, idx, head_mask=None, patch=None, keep=False):
    """Logits for token ids idx (B, T).

    head_mask: optional (n_layer, n_head) array multiplying each head's output (0 = ablate).
    patch:     optional {(layer, head): array (B, T, d_head)} replacing a head's output
               (activation patching, Unit 11; inference only — patched heads get no gradient).
    keep:      also return the intermediate values needed by `backward` and by Unit 11's probes.
    """
    B, T = idx.shape
    assert T <= cfg.max_len
    h, dh = cfg.n_head, cfg.d_head
    x = p["wte"][idx]
    if cfg.positions:
        x = x + p["wpe"][:T]
    mask = np.triu(np.ones((T, T), dtype=bool), 1)
    caches = []
    for l in range(cfg.n_layer):
        c = {"x_in": x}
        if cfg.layernorm:
            a_in, c["ln1"] = layernorm(x, p[f"{l}.ln1.g"], p[f"{l}.ln1.b"])
        else:
            a_in = x
        c["a_in"] = a_in
        qkv = mm(a_in, p[f"{l}.attn.wqkv"])
        q, k, v = (split_heads(t, h) for t in np.split(qkv, 3, axis=-1))
        s = mm(q, k.transpose(0, 1, 3, 2)) / np.sqrt(dh)
        s = np.where(mask, -np.inf, s)
        A = softmax(s)
        o = mm(A, v)                                     # (B, h, T, dh) per-head outputs
        if patch:
            o = o.copy()
            for (pl, ph), val in patch.items():
                if pl == l:
                    o[:, ph] = val
        if head_mask is not None:
            o = o * head_mask[l][None, :, None, None]
        c.update(q=q, k=k, v=v, A=A, o=o)
        att = mm(merge_heads(o), p[f"{l}.attn.wo"])
        x = x + att
        if cfg.mlp:
            c["x_mid"] = x
            if cfg.layernorm:
                m_in, c["ln2"] = layernorm(x, p[f"{l}.ln2.g"], p[f"{l}.ln2.b"])
            else:
                m_in = x
            u = mm(m_in, p[f"{l}.mlp.wfc"]) + p[f"{l}.mlp.bfc"]
            gu, t = gelu(u)
            c.update(m_in=m_in, u=u, t=t, gu=gu)
            x = x + mm(gu, p[f"{l}.mlp.wproj"]) + p[f"{l}.mlp.bproj"]
        caches.append(c)
    if cfg.layernorm:
        xf, lnf = layernorm(x, p["lnf.g"], p["lnf.b"])
    else:
        xf, lnf = x, None
    logits = mm(xf, p["wte"].T)
    if keep:
        return logits, {"blocks": caches, "xf": xf, "lnf": lnf, "x_final": x, "idx": idx, "head_mask": head_mask}
    return logits


def cross_entropy(logits, targets, weights=None):
    """Loss = sum_i w_i * (-log p(target_i)); default w = 1/(number of positions) (mean NLL in nats).
    Returns (loss, dlogits).  Positions with weight 0 are ignored; negative weights are allowed
    (policy-gradient updates in Unit 09 use them)."""
    B, T, V = logits.shape
    if weights is None:
        weights = np.full((B, T), 1.0 / (B * T))
    z = logits - logits.max(-1, keepdims=True)
    logp = z - np.log(np.exp(z).sum(-1, keepdims=True))
    nll = -np.take_along_axis(logp, targets[..., None], -1)[..., 0]
    loss = float((weights * nll).sum())
    d = np.exp(logp)
    np.put_along_axis(d, targets[..., None], np.take_along_axis(d, targets[..., None], -1) - 1.0, -1)
    return loss, d * weights[..., None]


def backward(p, cfg: Config, cache, dlogits):
    """Gradients of a scalar loss with respect to every parameter, given dL/dlogits.
    Each line is the adjoint rule derived in Unit 03 (§3.3) or Unit 04 (§4.3.6)."""
    g = {k: np.zeros_like(v) for k, v in p.items()}
    h, dh = cfg.n_head, cfg.d_head
    xf = cache["xf"]
    g["wte"] += mm(dlogits.reshape(-1, dlogits.shape[-1]).T, xf.reshape(-1, xf.shape[-1]))
    dx = mm(dlogits, p["wte"])
    if cfg.layernorm:
        dx, g["lnf.g"], g["lnf.b"] = layernorm_backward(dx, cache["lnf"])
    for l in reversed(range(cfg.n_layer)):
        c = cache["blocks"][l]
        if cfg.mlp:
            # x_out = x_mid + gu @ wproj + bproj
            g[f"{l}.mlp.wproj"] += mm(c["gu"].reshape(-1, 4 * cfg.d).T, dx.reshape(-1, cfg.d))
            g[f"{l}.mlp.bproj"] += dx.reshape(-1, cfg.d).sum(0)
            dgu = mm(dx, p[f"{l}.mlp.wproj"].T)
            du = gelu_backward(dgu, c["u"], c["t"])
            g[f"{l}.mlp.wfc"] += mm(c["m_in"].reshape(-1, cfg.d).T, du.reshape(-1, 4 * cfg.d))
            g[f"{l}.mlp.bfc"] += du.reshape(-1, 4 * cfg.d).sum(0)
            dm_in = mm(du, p[f"{l}.mlp.wfc"].T)
            if cfg.layernorm:
                dm_in, g[f"{l}.ln2.g"], g[f"{l}.ln2.b"] = layernorm_backward(dm_in, c["ln2"])
            dx = dx + dm_in
        # x_mid = x_in + merge(o) @ wo
        om = merge_heads(c["o"])
        g[f"{l}.attn.wo"] += mm(om.reshape(-1, cfg.d).T, dx.reshape(-1, cfg.d))
        do = split_heads(mm(dx, p[f"{l}.attn.wo"].T), h)
        if cache["head_mask"] is not None:
            do = do * cache["head_mask"][l][None, :, None, None]
        A, q, k, v = c["A"], c["q"], c["k"], c["v"]
        dA = mm(do, v.transpose(0, 1, 3, 2))
        dv = mm(A.transpose(0, 1, 3, 2), do)
        ds = A * (dA - (dA * A).sum(-1, keepdims=True)) / np.sqrt(dh)   # softmax VJP, Unit 04 §4.3.6
        dq = mm(ds, k)
        dk = mm(ds.transpose(0, 1, 3, 2), q)
        dqkv = np.concatenate([merge_heads(dq), merge_heads(dk), merge_heads(dv)], axis=-1)
        g[f"{l}.attn.wqkv"] += mm(c["a_in"].reshape(-1, cfg.d).T, dqkv.reshape(-1, 3 * cfg.d))
        da_in = mm(dqkv, p[f"{l}.attn.wqkv"].T)
        if cfg.layernorm:
            da_in, g[f"{l}.ln1.g"], g[f"{l}.ln1.b"] = layernorm_backward(da_in, c["ln1"])
        dx = dx + da_in
    # x0 = wte[idx] + wpe[:T]   (lookup: scatter-add, Unit 02 §2.3)
    idx = cache["idx"]
    np.add.at(g["wte"], idx.reshape(-1), dx.reshape(-1, cfg.d))
    if cfg.positions:
        g["wpe"][: idx.shape[1]] += dx.sum(0)
    return g


def loss_and_grads(p, cfg, idx, targets, weights=None, head_mask=None):
    logits, cache = forward(p, cfg, idx, head_mask=head_mask, keep=True)
    loss, dlogits = cross_entropy(logits, targets, weights)
    return loss, backward(p, cfg, cache, dlogits)


# ----------------------------------------------------------------------------- optimizer
class AdamW:
    """Adam with decoupled weight decay (Atlas: opt-adamw). Decay applies to matrices only."""

    def __init__(self, params, lr=3e-3, betas=(0.9, 0.99), eps=1e-8, weight_decay=0.01):
        self.lr, self.b1, self.b2, self.eps, self.wd = lr, betas[0], betas[1], eps, weight_decay
        self.m = {k: np.zeros_like(v) for k, v in params.items()}
        self.v = {k: np.zeros_like(v) for k, v in params.items()}
        self.t = 0

    def step(self, params, grads, lr=None):
        lr = self.lr if lr is None else lr
        self.t += 1
        for k in params:
            gk = grads[k]
            self.m[k] = self.b1 * self.m[k] + (1 - self.b1) * gk
            self.v[k] = self.b2 * self.v[k] + (1 - self.b2) * gk * gk
            mhat = self.m[k] / (1 - self.b1 ** self.t)
            vhat = self.v[k] / (1 - self.b2 ** self.t)
            if params[k].ndim == 2 and k != "wpe":
                params[k] -= lr * self.wd * params[k]
            params[k] -= lr * mhat / (np.sqrt(vhat) + self.eps)


def batches(ids, T, B, rng):
    """Random contiguous windows: inputs ids[s:s+T], targets ids[s+1:s+T+1]."""
    starts = rng.integers(0, len(ids) - T - 1, size=B)
    x = np.stack([ids[s : s + T] for s in starts])
    y = np.stack([ids[s + 1 : s + T + 1] for s in starts])
    return x, y


def evaluate(p, cfg, ids, T, max_windows=64):
    """Mean held-out NLL (nats/token) over non-overlapping windows."""
    n = min(max_windows, (len(ids) - 1) // T)
    x = np.stack([ids[i * T : i * T + T] for i in range(n)])
    y = np.stack([ids[i * T + 1 : i * T + T + 1] for i in range(n)])
    loss, _ = cross_entropy(forward(p, cfg, x), y)
    return loss


def train(p, cfg, train_ids, steps, T=64, B=16, lr=3e-3, seed=0, warmup=50, val_ids=None, log_every=0, weight_decay=0.01):
    """AdamW with linear warm-up and cosine decay (Unit 03 §3.8). Returns the training-loss history."""
    rng = np.random.default_rng(seed)
    opt = AdamW(p, lr=lr, weight_decay=weight_decay)
    hist = []
    for s in range(steps):
        x, y = batches(train_ids, T, B, rng)
        loss, g = loss_and_grads(p, cfg, x, y)
        frac = s / max(1, steps - 1)
        lr_s = lr * min(1.0, (s + 1) / warmup) * (0.1 + 0.9 * 0.5 * (1 + np.cos(np.pi * frac)))
        opt.step(p, g, lr=lr_s)
        hist.append(loss)
        if log_every and (s + 1) % log_every == 0:
            msg = f"step {s + 1:5d}  train {np.mean(hist[-log_every:]):.4f}"
            if val_ids is not None:
                msg += f"  val {evaluate(p, cfg, val_ids, T):.4f}"
            print(msg, flush=True)
    return np.array(hist)


# ----------------------------------------------------------------------------- KV-cache decoding (Unit 05)
class KVCache:
    def __init__(self, cfg: Config, B):
        self.k = [np.zeros((B, cfg.n_head, 0, cfg.d_head)) for _ in range(cfg.n_layer)]
        self.v = [np.zeros((B, cfg.n_head, 0, cfg.d_head)) for _ in range(cfg.n_layer)]

    @property
    def length(self):
        return self.k[0].shape[2]

    def nbytes(self):
        return sum(a.nbytes for a in self.k + self.v)


def decode_step(p, cfg: Config, tok, cache: KVCache):
    """Logits for the next position given one new token per sequence (B,), reusing cached keys/values."""
    t = cache.length
    h, dh = cfg.n_head, cfg.d_head
    x = p["wte"][tok][:, None, :]
    if cfg.positions:
        x = x + p["wpe"][t]
    for l in range(cfg.n_layer):
        a_in = layernorm(x, p[f"{l}.ln1.g"], p[f"{l}.ln1.b"])[0] if cfg.layernorm else x
        q, k, v = (split_heads(z, h) for z in np.split(mm(a_in, p[f"{l}.attn.wqkv"]), 3, axis=-1))
        cache.k[l] = np.concatenate([cache.k[l], k], axis=2)
        cache.v[l] = np.concatenate([cache.v[l], v], axis=2)
        A = softmax(mm(q, cache.k[l].transpose(0, 1, 3, 2)) / np.sqrt(dh))   # no mask needed: only the past is cached
        x = x + mm(merge_heads(mm(A, cache.v[l])), p[f"{l}.attn.wo"])
        if cfg.mlp:
            m_in = layernorm(x, p[f"{l}.ln2.g"], p[f"{l}.ln2.b"])[0] if cfg.layernorm else x
            x = x + mm(gelu(mm(m_in, p[f"{l}.mlp.wfc"]) + p[f"{l}.mlp.bfc"])[0], p[f"{l}.mlp.wproj"]) + p[f"{l}.mlp.bproj"]
    xf = layernorm(x, p["lnf.g"], p["lnf.b"])[0] if cfg.layernorm else x
    return mm(xf, p["wte"].T)[:, 0, :]


def kv_cache_bytes(cfg: Config, B, t, bytes_per=8):
    """Analytic cache size: 2 (K and V) x layers x heads x d_head x positions x batch (Unit 04 §4.3.10)."""
    return 2 * cfg.n_layer * cfg.n_head * cfg.d_head * t * B * bytes_per


def generate(p, cfg: Config, prompt, n_new, sampler, rng, use_cache=True):
    """Extend each prompt (B, t0) by n_new tokens chosen by sampler(logits, rng) -> ids (Unit 08)."""
    seq = np.asarray(prompt)
    if use_cache:
        cache = KVCache(cfg, seq.shape[0])
        for i in range(seq.shape[1]):
            logits = decode_step(p, cfg, seq[:, i], cache)
        for _ in range(n_new):
            nxt = sampler(logits, rng)
            seq = np.concatenate([seq, nxt[:, None]], axis=1)
            if cache.length >= cfg.max_len:
                break
            logits = decode_step(p, cfg, nxt, cache)
    else:
        for _ in range(n_new):
            logits = forward(p, cfg, seq[:, -cfg.max_len :])[:, -1]
            nxt = sampler(logits, rng)
            seq = np.concatenate([seq, nxt[:, None]], axis=1)
    return seq


def config_dict(cfg: Config):
    return asdict(cfg)
