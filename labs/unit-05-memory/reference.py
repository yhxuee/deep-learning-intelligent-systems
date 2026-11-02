"""Reference implementations for Unit 05 (memory)."""
import numpy as np


def recurrence_loop(A, B, xs, h0):
    """h_t = A h_{t-1} + B x_t for t = 1..T.  xs (T, m), returns hs (T, N)."""
    h, out = np.array(h0, dtype=float), []
    for x in xs:
        h = A @ h + B @ x
        out.append(h)
    return np.array(out)


def recurrence_closed_form(A, B, xs, h0):
    """h_t = A^t h0 + sum_{k=1}^t A^{t-k} B x_k, evaluated directly (O(T^2), for verification)."""
    T = len(xs)
    powers = [np.eye(A.shape[0])]
    for _ in range(T):
        powers.append(A @ powers[-1])
    hs = []
    for t in range(1, T + 1):
        h = powers[t] @ h0
        for k in range(1, t + 1):
            h = h + powers[t - k] @ B @ xs[k - 1]
        hs.append(h)
    return np.array(hs)


def convolution_kernel(A, B, length):
    """K_j = A^j B for j = 0..length-1, so that with h0 = 0:  h_t = sum_{k<=t} K_{t-k} x_k."""
    K, P = [], np.eye(A.shape[0])
    for _ in range(length):
        K.append(P @ B)
        P = A @ P
    return np.array(K)


def influence_norms(A, max_lag):
    """Spectral norms ||A^j||_2 for j = 0..max_lag."""
    out, P = [], np.eye(A.shape[0])
    for _ in range(max_lag + 1):
        out.append(np.linalg.norm(P, 2))
        P = A @ P
    return np.array(out)


def half_life(a):
    """Steps until a scalar recurrence with coefficient 0 < a < 1 retains half its influence."""
    return np.log(0.5) / np.log(a)


class LinearAttentionMemory:
    """Linear attention as a recurrence with a matrix state: M <- M + v k^T,  read(q) = M q."""

    def __init__(self, d_k, d_v):
        self.M = np.zeros((d_v, d_k))

    def write(self, k, v):
        self.M = self.M + np.outer(v, k)

    def read(self, q):
        return self.M @ q
