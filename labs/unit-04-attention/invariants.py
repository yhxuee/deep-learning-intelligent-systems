"""Executable invariants for Atlas node mech-attention.

Each check takes an implementation module `impl` (with softmax, attention, mha,
mha_decode) and returns (passed: bool, detail: str). They are used both by
test_attention.py (to test an implementation) and test_bug_hunt.py (to test that
the invariants are strong enough to catch realistic bugs).
"""
import numpy as np


def _weights(d=32, n_heads=4, seed=0):
    rng = np.random.default_rng(seed)
    return [rng.normal(0, d ** -0.5, (d, d)) for _ in range(4)]


def rows_sum_to_one(impl):
    rng = np.random.default_rng(1)
    q, k, v = rng.normal(size=(7, 8)), rng.normal(size=(7, 8)), rng.normal(size=(7, 5))
    for causal in (True, False):
        _, w = impl.attention(q, k, v, causal=causal)
        if not np.all(np.isfinite(w)):
            return False, f"non-finite weights (causal={causal})"
        err = np.max(np.abs(w.sum(axis=-1) - 1.0))
        if err > 1e-9:
            return False, f"row sums deviate by {err:.2e} (causal={causal})"
        if np.any(w < 0):
            return False, "negative weights"
    return True, "every row is a probability vector"


def causality(impl):
    rng = np.random.default_rng(2)
    T, d, h = 10, 32, 4
    W = _weights(d, h)
    x = rng.normal(size=(T, d))
    y = impl.mha(x, *W, n_heads=h, causal=True)
    for t in range(T - 1):
        x2 = x.copy()
        x2[t + 1:] += rng.normal(size=(T - t - 1, d)) * 3.0
        y2 = impl.mha(x2, *W, n_heads=h, causal=True)
        if not np.allclose(y[: t + 1], y2[: t + 1], rtol=0, atol=1e-12):
            return False, f"output at positions <= {t} changed when only positions > {t} changed"
    return True, "future perturbations never affect past outputs"


def permutation_equivariance(impl):
    rng = np.random.default_rng(3)
    T, d, h = 9, 32, 4
    W = _weights(d, h)
    x = rng.normal(size=(T, d))
    P = np.eye(T)[rng.permutation(T)]
    lhs = impl.mha(P @ x, *W, n_heads=h, causal=False)
    rhs = P @ impl.mha(x, *W, n_heads=h, causal=False)
    err = np.max(np.abs(lhs - rhs))
    return (err < 1e-10, f"max |Attn(Px) - P Attn(x)| = {err:.2e}")


def kv_cache_equivalence(impl):
    rng = np.random.default_rng(4)
    T, d, h = 8, 32, 4
    W = _weights(d, h)
    x = rng.normal(size=(T, d))
    full = impl.mha(x, *W, n_heads=h, causal=True)
    cache = None
    for t in range(T):
        y, cache = impl.mha_decode(x[t], cache, *W, n_heads=h)
        err = np.max(np.abs(y - full[t]))
        if err > 1e-10:
            return False, f"step {t}: cached output differs from full pass by {err:.2e}"
    return True, "cached decoding reproduces the full forward pass"


def logit_scale(impl):
    """With unit-variance q, k and dk = 256, scaled logits have variance ~1, so weights stay spread out."""
    rng = np.random.default_rng(5)
    T, dk = 64, 256
    q, k, v = rng.normal(size=(T, dk)), rng.normal(size=(T, dk)), rng.normal(size=(T, 4))
    _, w = impl.attention(q, k, v, causal=False)
    mean_max = float(np.mean(np.max(w, axis=-1)))
    return (mean_max < 0.5, f"mean max weight = {mean_max:.3f} (≈0.1 when scaled by 1/sqrt(dk); ≈1 if unscaled)")


def matches_reference(impl, reference):
    rng = np.random.default_rng(6)
    T, d, h = 6, 32, 4
    W = _weights(d, h, seed=7)
    x = rng.normal(size=(T, d))
    err = np.max(np.abs(impl.mha(x, *W, n_heads=h) - reference.mha(x, *W, n_heads=h)))
    return (err < 1e-10, f"max deviation from reference = {err:.2e}")


CHECKS = {
    "rows_sum_to_one": rows_sum_to_one,
    "causality": causality,
    "permutation_equivariance": permutation_equivariance,
    "kv_cache_equivalence": kv_cache_equivalence,
    "logit_scale": logit_scale,
}
