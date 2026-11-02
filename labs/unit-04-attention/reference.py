"""Reference causal multi-head self-attention in NumPy (Unit 04).

Conventions (the tests rely on them):
  x            (T, d)          one sequence, no batch dimension
  Wq,Wk,Wv,Wo  (d, d)          y = concat_h(head_h) @ Wo, with head h using columns h*dh:(h+1)*dh
  cache        dict(K=(t, d), V=(t, d))   full-width keys/values of the t tokens seen so far
"""
import numpy as np


def softmax(s, axis=-1):
    m = np.max(s, axis=axis, keepdims=True)
    e = np.exp(s - m)
    return e / np.sum(e, axis=axis, keepdims=True)


def attention(q, k, v, causal=True):
    """Single-head scaled dot-product attention.

    q (Tq, dk), k (Tk, dk), v (Tk, dv). With causal=True, query i sits at absolute
    position Tk - Tq + i and may attend to keys j <= that position.
    Returns (out (Tq, dv), weights (Tq, Tk)).
    """
    Tq, dk = q.shape
    Tk = k.shape[0]
    scores = q @ k.T / np.sqrt(dk)
    if causal:
        qpos = np.arange(Tq)[:, None] + (Tk - Tq)
        kpos = np.arange(Tk)[None, :]
        scores = np.where(kpos <= qpos, scores, -np.inf)
    w = softmax(scores, axis=-1)
    return w @ v, w


def _split_heads(m, n_heads):
    T, d = m.shape
    return m.reshape(T, n_heads, d // n_heads).transpose(1, 0, 2)   # (h, T, dh)


def mha(x, Wq, Wk, Wv, Wo, n_heads, causal=True):
    Q, K, V = _split_heads(x @ Wq, n_heads), _split_heads(x @ Wk, n_heads), _split_heads(x @ Wv, n_heads)
    heads = [attention(Q[h], K[h], V[h], causal)[0] for h in range(n_heads)]
    return np.concatenate(heads, axis=-1) @ Wo


def mha_decode(x_new, cache, Wq, Wk, Wv, Wo, n_heads):
    """One decoding step. x_new (d,). Returns (y (d,), new_cache)."""
    q, k, v = x_new @ Wq, x_new @ Wk, x_new @ Wv
    K = np.vstack([cache["K"], k[None]]) if cache else k[None]
    V = np.vstack([cache["V"], v[None]]) if cache else v[None]
    Qh, Kh, Vh = _split_heads(q[None], n_heads), _split_heads(K, n_heads), _split_heads(V, n_heads)
    heads = [attention(Qh[h], Kh[h], Vh[h], causal=True)[0] for h in range(n_heads)]
    y = (np.concatenate(heads, axis=-1) @ Wo)[0]
    return y, {"K": K, "V": V}


def param_count(d, n_layers, vocab, n_ctx, tied=True):
    """Exact parameter count of a GPT-2-style decoder (biases + LayerNorms included)."""
    per_block = 12 * d * d + 13 * d
    embeddings = vocab * d + n_ctx * d
    final_ln = 2 * d
    unembed = 0 if tied else vocab * d
    return n_layers * per_block + embeddings + final_ln + unembed
