"""Reference scalar reverse-mode autodiff engine (Unit 03).

Read this only after your own engine.py passes the tests, or to settle a dispute
with the tests. ~90 lines; inspired by the design of micrograd (Karpathy).
"""
import math


class Value:
    def __init__(self, data, _children=(), _op=""):
        self.data = float(data)
        self.grad = 0.0
        self._backward = lambda: None
        self._prev = tuple(_children)
        self._op = _op

    # --- construction helpers -------------------------------------------------
    @staticmethod
    def _wrap(x):
        return x if isinstance(x, Value) else Value(x)

    # --- operations -------------------------------------------------------------
    def __add__(self, other):
        other = Value._wrap(other)
        out = Value(self.data + other.data, (self, other), "+")

        def _backward():
            self.grad += out.grad          # += : a value may feed several children
            other.grad += out.grad

        out._backward = _backward
        return out

    def __mul__(self, other):
        other = Value._wrap(other)
        out = Value(self.data * other.data, (self, other), "*")

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = _backward
        return out

    def __pow__(self, k):
        assert isinstance(k, (int, float)), "only constant exponents"
        out = Value(self.data ** k, (self,), f"**{k}")

        def _backward():
            self.grad += k * self.data ** (k - 1) * out.grad

        out._backward = _backward
        return out

    def tanh(self):
        t = math.tanh(self.data)
        out = Value(t, (self,), "tanh")

        def _backward():
            self.grad += (1.0 - t * t) * out.grad

        out._backward = _backward
        return out

    def exp(self):
        e = math.exp(self.data)
        out = Value(e, (self,), "exp")

        def _backward():
            self.grad += e * out.grad

        out._backward = _backward
        return out

    def log(self):
        out = Value(math.log(self.data), (self,), "log")

        def _backward():
            self.grad += out.grad / self.data

        out._backward = _backward
        return out

    def __neg__(self):
        return self * -1.0

    def __sub__(self, other):
        return self + (-Value._wrap(other))

    def __truediv__(self, other):
        return self * Value._wrap(other) ** -1

    def __radd__(self, other):
        return self + other

    def __rmul__(self, other):
        return self * other

    def __rsub__(self, other):
        return Value._wrap(other) - self

    def __rtruediv__(self, other):
        return Value._wrap(other) / self

    # --- reverse sweep --------------------------------------------------------
    def backward(self):
        # Iterative post-order DFS: a recursive version overflows the stack on deep graphs.
        topo, visited, stack = [], set(), [(self, False)]
        while stack:
            node, expanded = stack.pop()
            if expanded:
                topo.append(node)
            elif id(node) not in visited:
                visited.add(id(node))
                stack.append((node, True))
                stack.extend((child, False) for child in node._prev if id(child) not in visited)
        self.grad = 1.0
        for node in reversed(topo):
            node._backward()

    def __repr__(self):
        return f"Value(data={self.data:.6g}, grad={self.grad:.6g})"
