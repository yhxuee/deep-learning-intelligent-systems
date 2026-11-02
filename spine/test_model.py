"""Verification targets for the shared spine model (model.py)."""
import pathlib
import sys

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import model as M  # noqa: E402

CONFIGS = {
    "full": M.Config(vocab=11, d=8, n_layer=2, n_head=2, max_len=6),
    "attn_only": M.Config(vocab=11, d=8, n_layer=2, n_head=2, max_len=6, mlp=False, layernorm=False, positions=False),
}


def _data(cfg, B=2, T=5, seed=1):
    rng = np.random.default_rng(seed)
    return rng.integers(0, cfg.vocab, (B, T)), rng.integers(0, cfg.vocab, (B, T))


@pytest.mark.parametrize("name", list(CONFIGS))
def test_every_gradient_matches_central_differences(name):
    cfg = CONFIGS[name]
    p = M.init_params(cfg, seed=3)
    for k in p:  # move away from the symmetric initialization so every path carries gradient
        p[k] = p[k] + np.random.default_rng(len(k)).normal(0, 0.3, p[k].shape)
    x, y = _data(cfg)
    hm = np.array([[1.0, 0.5], [0.0, 1.0]])
    for head_mask in (None, hm):
        _, g = M.loss_and_grads(p, cfg, x, y, head_mask=head_mask)
        rng = np.random.default_rng(0)
        for k in p:
            for _ in range(4):
                i = tuple(rng.integers(0, s) for s in p[k].shape)
                old = p[k][i]
                p[k][i] = old + 1e-5
                lp = M.cross_entropy(M.forward(p, cfg, x, head_mask=head_mask), y)[0]
                p[k][i] = old - 1e-5
                lm = M.cross_entropy(M.forward(p, cfg, x, head_mask=head_mask), y)[0]
                p[k][i] = old
                assert (lp - lm) / 2e-5 == pytest.approx(g[k][i], rel=1e-5, abs=1e-9), (k, i)


def test_weighted_loss_gradient_is_linear_in_the_weights():
    cfg = CONFIGS["full"]
    p = M.init_params(cfg, seed=0)
    x, y = _data(cfg)
    w1, w2 = np.random.default_rng(0).normal(size=(2, *x.shape))
    _, g1 = M.loss_and_grads(p, cfg, x, y, weights=w1)
    _, g2 = M.loss_and_grads(p, cfg, x, y, weights=w2)
    _, g12 = M.loss_and_grads(p, cfg, x, y, weights=w1 - 2 * w2)
    for k in p:
        assert np.allclose(g12[k], g1[k] - 2 * g2[k], atol=1e-12)


def test_causal_future_tokens_do_not_change_past_logits():
    cfg = CONFIGS["full"]
    p = M.init_params(cfg, seed=0)
    x, _ = _data(cfg)
    a = M.forward(p, cfg, x)
    x2 = x.copy()
    x2[:, 3:] = (x2[:, 3:] + 1) % cfg.vocab
    b = M.forward(p, cfg, x2)
    assert np.allclose(a[:, :3], b[:, :3], atol=1e-12)
    assert not np.allclose(a[:, 3:], b[:, 3:])


@pytest.mark.parametrize("name", list(CONFIGS))
def test_kv_cache_decoding_equals_full_recompute(name):
    cfg = CONFIGS[name]
    p = M.init_params(cfg, seed=2)
    x, _ = _data(cfg, B=3, T=6)
    full = M.forward(p, cfg, x)
    cache = M.KVCache(cfg, B=3)
    for t in range(6):
        step = M.decode_step(p, cfg, x[:, t], cache)
        assert np.allclose(step, full[:, t], atol=1e-10)
    assert cache.nbytes() == M.kv_cache_bytes(cfg, B=3, t=6)


def test_training_flops_are_6N_per_token_plus_attention():
    """Unit 06: forward + backward cost = 6 x (matmul parameters) per token + 12 L T d for attention."""
    cfg = M.Config(vocab=13, d=16, n_layer=3, n_head=4, max_len=10)
    p = M.init_params(cfg)
    B, T = 4, 10
    x = np.zeros((B, T), dtype=int)
    M.FLOPS["count"] = 0
    M.loss_and_grads(p, cfg, x, x)
    expected = B * T * (6 * M.matmul_params(cfg) + 12 * cfg.n_layer * T * cfg.d)
    assert M.FLOPS["count"] == expected


def test_initial_loss_is_close_to_ln_V():
    cfg = M.Config(vocab=50, d=32, n_layer=2, n_head=4, max_len=16)
    p = M.init_params(cfg, seed=0)
    rng = np.random.default_rng(0)
    x = rng.integers(0, 50, (8, 16))
    loss, _ = M.cross_entropy(M.forward(p, cfg, x), x)
    assert abs(loss - np.log(50)) < 0.05


def test_ablating_a_head_equals_patching_it_with_zeros():
    cfg = CONFIGS["full"]
    p = M.init_params(cfg, seed=4)
    x, _ = _data(cfg)
    hm = np.ones((2, 2))
    hm[1, 0] = 0
    a = M.forward(p, cfg, x, head_mask=hm)
    b = M.forward(p, cfg, x, patch={(1, 0): np.zeros((2, 5, cfg.d_head))})
    assert np.allclose(a, b, atol=1e-12)
