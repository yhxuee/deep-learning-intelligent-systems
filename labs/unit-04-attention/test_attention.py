"""Invariant tests for Atlas nodes mech-attention, mech-kv-cache, arch-transformer-block (Unit 04)."""
import pathlib
import sys

import pytest

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
from _loader import load, load_file  # noqa: E402

impl = load(HERE, "attention")
reference = load_file(HERE / "reference.py", "unit04_reference_for_tests")
invariants = load_file(HERE / "invariants.py", "unit04_invariants")


@pytest.mark.parametrize("name", list(invariants.CHECKS))
def test_invariant(name):
    ok, detail = invariants.CHECKS[name](impl)
    assert ok, detail


def test_matches_reference():
    ok, detail = invariants.matches_reference(impl, reference)
    assert ok, detail


def test_softmax_is_stable():
    import numpy as np

    p = impl.softmax(np.array([[1000.0, 0.0]]))
    assert np.allclose(p, [[1.0, 0.0]])


def test_first_position_attends_only_to_itself():
    import numpy as np

    rng = np.random.default_rng(0)
    q, k, v = rng.normal(size=(5, 8)), rng.normal(size=(5, 8)), rng.normal(size=(5, 3))
    out, w = impl.attention(q, k, v, causal=True)
    assert np.allclose(w[0], [1, 0, 0, 0, 0])
    assert np.allclose(out[0], v[0])


def test_gpt2_small_parameter_count():
    # Known value: GPT-2 small, tied embeddings, biases and LayerNorms included.
    assert impl.param_count(d=768, n_layers=12, vocab=50257, n_ctx=1024) == 124_439_808
    assert impl.param_count(d=768, n_layers=12, vocab=50257, n_ctx=1024) - (50257 + 1024) * 768 - 2 * 768 == 85_054_464
