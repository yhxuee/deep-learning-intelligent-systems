"""YOUR scalar reverse-mode autodiff engine (Unit 03).  AI-OFF: write this yourself.

Implement a class `Value` wrapping one float, recording how it was computed, so that
calling `.backward()` on a scalar result fills `.grad` on every Value it depends on.

Required interface (the tests use exactly this):

    v = Value(2.0)          v.data -> 2.0, v.grad -> 0.0
    a + b, a * b, a - b, a / b, a ** k (k a Python number), -a
    2 + a, 2 * a, 2 - a, 2 / a          (mixed with plain numbers on either side)
    a.tanh(), a.exp(), a.log()
    out.backward()          sets out.grad = 1 and ACCUMULATES d(out)/d(v) into v.grad
                            for every v in the graph. Gradients from separate graphs that
                            share a leaf add up; zeroing them is the caller's job.

Hints you may use (they are in the Core text, not code):
  * adjoint recursion:  v̄_i = Σ_{j: i→j} v̄_j ∂v_j/∂v_i   — note the SUM
  * visit nodes in reverse topological order
  * graphs can be 10 000 nodes deep: avoid recursion in the topological sort

Run:  python -m pytest            (from this directory)
"""


class Value:
    def __init__(self, data, _children=(), _op=""):
        raise NotImplementedError("Implement Value — see the docstring at the top of engine.py")
