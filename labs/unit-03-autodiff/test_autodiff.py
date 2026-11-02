"""Invariant tests for Atlas node mech-reverse-mode-autodiff (Unit 03)."""
import math
import pathlib
import random
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from _loader import load  # noqa: E402

engine = load(pathlib.Path(__file__).parent, "engine")
Value = engine.Value


# 1 — Known values from the worked example in Unit 03 §4 -------------------------
def test_worked_example_known_values():
    x1, x2 = Value(1.0), Value(2.0)
    w1, w2, b = Value(0.5), Value(-0.25), Value(0.1)
    t = 0.0
    z = w1 * x1 + w2 * x2 + b
    a = z.tanh()
    L = 0.5 * (a - t) ** 2
    L.backward()
    assert z.data == pytest.approx(0.1, abs=1e-12)
    assert a.data == pytest.approx(0.099668, abs=1e-6)
    assert L.data == pytest.approx(0.0049669, abs=1e-7)
    assert w1.grad == pytest.approx(0.098678, abs=1e-6)
    assert w2.grad == pytest.approx(0.197356, abs=1e-6)
    assert b.grad == pytest.approx(0.098678, abs=1e-6)


# 2 — Fan-out must accumulate ----------------------------------------------------
def test_fanout_mul_accumulates():
    a = Value(3.0)
    y = a * a
    y.backward()
    assert a.grad == pytest.approx(6.0)


def test_fanout_add_accumulates():
    a = Value(3.0)
    y = a + a
    y.backward()
    assert a.grad == pytest.approx(2.0)


def test_shared_subexpression():
    a = Value(-1.5)
    s = a * 2.0 + 1.0          # s is used twice below
    y = s * s + s.tanh()
    y.backward()
    sv = -2.0
    expected = (2 * sv + (1 - math.tanh(sv) ** 2)) * 2.0
    assert a.grad == pytest.approx(expected, rel=1e-12)


# 3 — Agreement with central finite differences on random expression graphs -------
def _apply(op, x, y):
    is_val = isinstance(x, Value)
    if op == "add":
        return x + y
    if op == "mul":
        return x * y
    if op == "sub":
        return x - y
    if op == "div":
        return x / y
    if op == "tanh":
        return x.tanh() if is_val else math.tanh(x)
    if op == "exp":
        return x.exp() if is_val else math.exp(x)
    if op == "logsq":  # log(1 + x^2) keeps the argument positive
        return (1 + x ** 2).log() if is_val else math.log(1 + x ** 2)
    if op == "pow3":
        return x ** 3
    raise ValueError(op)


def _random_program(rng, n_inputs=3, n_ops=12):
    prog = []
    for i in range(n_ops):
        op = rng.choice(["add", "mul", "sub", "div", "tanh", "exp", "logsq", "pow3"])
        hi = n_inputs + i
        prog.append((op, rng.randrange(hi), rng.randrange(hi)))
    return prog


def _run(prog, inputs):
    vals = list(inputs)
    for op, i, j in prog:
        x, y = vals[i], vals[j]
        if op == "div":            # keep denominators away from zero
            y = y * y + 1.0
        if op == "exp":            # keep magnitudes bounded
            x = x.tanh() if isinstance(x, Value) else math.tanh(x)
        if op == "pow3":
            x = x.tanh() if isinstance(x, Value) else math.tanh(x)
        vals.append(_apply(op, x, y))
    return vals[-1]


@pytest.mark.parametrize("seed", range(40))
def test_matches_central_differences(seed):
    rng = random.Random(seed)
    prog = _random_program(rng)
    x0 = [rng.uniform(-1.5, 1.5) for _ in range(3)]
    leaves = [Value(v) for v in x0]
    out = _run(prog, leaves)
    out.backward()
    h = 1e-6
    for k in range(3):
        xp, xm = list(x0), list(x0)
        xp[k] += h
        xm[k] -= h
        num = (_run(prog, xp) - _run(prog, xm)) / (2 * h)
        ana = leaves[k].grad
        denom = max(abs(num), abs(ana), 1e-8)
        assert abs(num - ana) / denom < 1e-6 or abs(num - ana) < 1e-9, (prog, k, num, ana)


# 4 — Linearity ----------------------------------------------------------------------
def test_linearity():
    def f(x):
        return x.tanh() * x

    def g(x):
        return (x * 3.0).exp()

    x0 = 0.3
    xa, xb, xc = Value(x0), Value(x0), Value(x0)
    f(xa).backward()
    g(xb).backward()
    (3.0 * f(xc) + 2.0 * g(xc)).backward()
    assert xc.grad == pytest.approx(3.0 * xa.grad + 2.0 * xb.grad, rel=1e-12)


# 5 — Accumulation is the contract: leaf gradients from separate graphs add up ---------
def test_leaf_grads_accumulate_across_graphs():
    a, b = Value(2.0), Value(-3.0)
    (a * b + a).backward()          # d/da = b + 1 = -2,  d/db = a = 2
    (a * a).backward()              # fresh graph, same leaf a:  d/da = 2a = 4
    assert a.grad == pytest.approx(-2.0 + 4.0)
    assert b.grad == pytest.approx(2.0)


# 6 — Operators with numbers on either side ---------------------------------------------
def test_reverse_operators():
    x = Value(4.0)
    ys = [2 + x, 2 * x, 2 - x, 2 / x, x / 2, x - 2, -x, x ** 0.5]
    expected = [1.0, 2.0, -1.0, -2 / 16, 0.5, 1.0, -1.0, 0.5 / 2.0]
    for y, e in zip(ys, expected):
        x.grad = 0.0
        y.backward()
        assert x.grad == pytest.approx(e), y


# 7 — Deep graphs must not overflow the call stack ---------------------------------------
def test_deep_graph_no_recursion_error():
    x = Value(1.0)
    y = x
    for _ in range(10_000):
        y = y + x
    y.backward()
    assert x.grad == pytest.approx(10_001.0)
