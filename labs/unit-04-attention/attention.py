"""YOUR causal multi-head self-attention in NumPy (Unit 04).  AI-OFF: write this from memory.

Implement the five functions below with exactly these signatures and conventions
(see reference.py's docstring for the conventions, but not its code).

  softmax(s, axis=-1)                                  numerically stable
  attention(q, k, v, causal=True) -> (out, weights)    q (Tq,dk), k (Tk,dk), v (Tk,dv);
                                                       causal: query i is at position Tk-Tq+i
  mha(x, Wq, Wk, Wv, Wo, n_heads, causal=True) -> y    x (T,d); head h uses columns h*dh:(h+1)*dh
  mha_decode(x_new, cache, Wq, Wk, Wv, Wo, n_heads)    one step with a KV cache -> (y, new_cache);
                                                       cache is None/{} at the first step
  param_count(d, n_layers, vocab, n_ctx, tied=True)    exact count for a GPT-2-style decoder

Budget: about 40 lines. Then run  python -m pytest  in this directory.
"""


def softmax(s, axis=-1):
    raise NotImplementedError


def attention(q, k, v, causal=True):
    raise NotImplementedError


def mha(x, Wq, Wk, Wv, Wo, n_heads, causal=True):
    raise NotImplementedError


def mha_decode(x_new, cache, Wq, Wk, Wv, Wo, n_heads):
    raise NotImplementedError


def param_count(d, n_layers, vocab, n_ctx, tied=True):
    raise NotImplementedError
