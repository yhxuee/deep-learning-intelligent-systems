"""YOUR Unit 05 implementations.  AI-OFF: derive the closed form on paper first.

Implement, with the same signatures as described in the Core text:

  recurrence_loop(A, B, xs, h0) -> hs (T, N)          h_t = A h_{t-1} + B x_t
  recurrence_closed_form(A, B, xs, h0) -> hs (T, N)   h_t = A^t h0 + Σ_{k=1}^t A^{t-k} B x_k
  convolution_kernel(A, B, length) -> K               K_j = A^j B,  j = 0..length-1
  influence_norms(A, max_lag) -> array                ||A^j||_2 for j = 0..max_lag
  half_life(a) -> float                               scalar 0<a<1
  class LinearAttentionMemory(d_k, d_v)               .write(k, v): M += v k^T ; .read(q) -> M q

Run:  python -m pytest
"""


def recurrence_loop(A, B, xs, h0):
    raise NotImplementedError


def recurrence_closed_form(A, B, xs, h0):
    raise NotImplementedError


def convolution_kernel(A, B, length):
    raise NotImplementedError


def influence_norms(A, max_lag):
    raise NotImplementedError


def half_life(a):
    raise NotImplementedError


class LinearAttentionMemory:
    def __init__(self, d_k, d_v):
        raise NotImplementedError
