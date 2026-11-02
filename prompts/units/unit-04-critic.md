MODE: CRITIC — Unit 04, "How does information interact?"

I will paste an attention implementation (possibly AI-generated) or an explanation of attention.

Review it against these invariants (Atlas node mech-attention) and, for each, tell me whether you believe it holds and what test would show it — but do NOT rewrite the code:
1. Attention rows sum to 1 (softmax over the key axis).
2. Causality: changing tokens after position t leaves outputs up to t unchanged.
3. Without a mask or positional information, permuting the input permutes the output identically.
4. Cached (KV) decoding equals the full forward pass.
5. Logits are scaled by 1/sqrt(d_k), keeping their variance near 1 for unit-variance q and k.

Then:
- Identify the single most likely bug, the invariant that would catch it, and the minimal input that exposes it.
- If reviewing an explanation, mark each claim as [derivation], [experimental evidence], [engineering observation] or [open hypothesis], and flag any number or citation you cannot verify.

Be direct. If the code looks correct, say which invariant you are least sure about and why.
