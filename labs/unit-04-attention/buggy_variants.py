"""Five plausible attention implementations, each with ONE planted bug (Unit 04 bug hunt).

These are the kind of mistakes that survive a quick read of AI-generated code.
For each variant: predict which invariant in invariants.py fails, write your prediction
in bug_hunt_predictions.py, then run  python -m pytest test_bug_hunt.py -v -s
Do NOT "fix" them here — they are test fixtures.
"""
import types

import numpy as np

import reference as R


def _variant(name, **overrides):
    """A module-like object: the reference implementation with some functions replaced."""
    ns = {f: getattr(R, f) for f in ("softmax", "attention", "mha", "mha_decode", "param_count")}
    ns.update(overrides)
    m = types.SimpleNamespace(**ns)
    m.__name__ = name
    # Rebind mha/mha_decode to this variant's attention so the bug propagates.
    if "attention" in overrides:
        def mha(x, Wq, Wk, Wv, Wo, n_heads, causal=True):
            Q, K, V = R._split_heads(x @ Wq, n_heads), R._split_heads(x @ Wk, n_heads), R._split_heads(x @ Wv, n_heads)
            heads = [m.attention(Q[h], K[h], V[h], causal)[0] for h in range(n_heads)]
            return np.concatenate(heads, axis=-1) @ Wo

        def mha_decode(x_new, cache, Wq, Wk, Wv, Wo, n_heads):
            q, k, v = x_new @ Wq, x_new @ Wk, x_new @ Wv
            K = np.vstack([cache["K"], k[None]]) if cache else k[None]
            V = np.vstack([cache["V"], v[None]]) if cache else v[None]
            Qh, Kh, Vh = R._split_heads(q[None], n_heads), R._split_heads(K, n_heads), R._split_heads(V, n_heads)
            heads = [m.attention(Qh[h], Kh[h], Vh[h], causal=True)[0] for h in range(n_heads)]
            return (np.concatenate(heads, axis=-1) @ Wo)[0], {"K": K, "V": V}

        m.mha, m.mha_decode = mha, mha_decode
    return m


def _mask(Tq, Tk, offset=0):
    qpos = np.arange(Tq)[:, None] + (Tk - Tq)
    kpos = np.arange(Tk)[None, :]
    return kpos <= qpos + offset


# A ----------------------------------------------------------------------------------
def _attention_a(q, k, v, causal=True):
    scores = q @ k.T / np.sqrt(q.shape[1])
    if causal:
        scores = np.where(_mask(len(q), len(k)), scores, -np.inf)
    w = R.softmax(scores, axis=0)
    return w @ v, w


# B ----------------------------------------------------------------------------------
def _attention_b(q, k, v, causal=True):
    scores = q @ k.T
    if causal:
        scores = np.where(_mask(len(q), len(k)), scores, -np.inf)
    w = R.softmax(scores, axis=-1)
    return w @ v, w


# C ----------------------------------------------------------------------------------
def _attention_c(q, k, v, causal=True):
    w = R.softmax(q @ k.T / np.sqrt(q.shape[1]), axis=-1)
    if causal:
        w = w * _mask(len(q), len(k))
    return w @ v, w


# D ----------------------------------------------------------------------------------
def _attention_d(q, k, v, causal=True):
    scores = q @ k.T / np.sqrt(q.shape[1])
    if causal:
        scores = np.where(_mask(len(q), len(k), offset=1), scores, -np.inf)
    w = R.softmax(scores, axis=-1)
    return w @ v, w


# E ----------------------------------------------------------------------------------
def _split_heads_e(m, n_heads):
    T, d = m.shape
    return m.reshape(T, d // n_heads, n_heads).transpose(2, 0, 1)


def _mha_e(x, Wq, Wk, Wv, Wo, n_heads, causal=True):
    Q, K, V = _split_heads_e(x @ Wq, n_heads), _split_heads_e(x @ Wk, n_heads), _split_heads_e(x @ Wv, n_heads)
    heads = [R.attention(Q[h], K[h], V[h], causal)[0] for h in range(n_heads)]
    return np.concatenate(heads, axis=-1) @ Wo


VARIANTS = {
    "A": _variant("A", attention=_attention_a),
    "B": _variant("B", attention=_attention_b),
    "C": _variant("C", attention=_attention_c),
    "D": _variant("D", attention=_attention_d),
    "E": _variant("E", mha=_mha_e),
}
