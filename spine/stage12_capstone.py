"""Spine Stage 12 — a pre-registered ablation and its replication (Unit 12).

Question.  Does the spine model need learned positional embeddings?  A causal Transformer can
infer position from the mask alone (Haviv et al. 2022), so the answer is not obvious.

Protocol.
  1. Pre-register: hypothesis, prediction, primary metric, decision rule and seeds are written to
     a JSON file and its SHA-256 digest is printed *before* any run.  Editing the file afterwards
     is detected.
  2. Run: both conditions with the same seeds (paired), identical budgets.
  3. Analyse exactly as registered: paired bootstrap CI of the difference in held-out NLL.
  4. Replicate: repeat with fresh seeds; report whether the effect has the same sign and whether
     the replication's interval contains the original estimate.

Verification targets (test_spine.py):
  * identical seed -> identical result (the experiment is deterministic)
  * tampering with a pre-registration is detected
  * the registered decision rule is applied mechanically
"""
import argparse
import hashlib
import json
import pathlib

import numpy as np

import model as M
from common import CKPT_DIR, load_corpus
from stage10_eval import paired_bootstrap_ci


def _digest(record):
    return hashlib.sha256(json.dumps(record, sort_keys=True).encode()).hexdigest()


def preregister(path, record):
    path = pathlib.Path(path)
    path.write_text(json.dumps({"record": record, "sha256": _digest(record)}, indent=1, sort_keys=True))
    return _digest(record)


def verify_preregistration(path):
    z = json.loads(pathlib.Path(path).read_text())
    return z["sha256"] == _digest(z["record"]), z["record"]


def run_condition(positions, seed, tr, va, V, steps=400, d=32, T=64):
    cfg = M.Config(vocab=V, d=d, n_layer=2, n_head=4, max_len=T, positions=positions)
    p = M.init_params(cfg, seed=seed)
    M.train(p, cfg, tr, steps=steps, T=T, B=16, lr=3e-3, seed=seed)
    return M.evaluate(p, cfg, va, T)


def decide(diff_ci, threshold):
    """Registered rule: 'needed' if the whole 95% CI of NLL(no positions) - NLL(positions) is above
    `threshold`; 'not needed' if the whole CI is below it; otherwise 'inconclusive'."""
    _, lo, hi = diff_ci
    if lo > threshold:
        return "needed"
    if hi < threshold:
        return "not needed"
    return "inconclusive"


def experiment(seeds, tr, va, V, steps):
    with_pos = [run_condition(True, s, tr, va, V, steps) for s in seeds]
    without = [run_condition(False, s, tr, va, V, steps) for s in seeds]
    return with_pos, without


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=400)
    args = ap.parse_args()
    tr, va, vocab = load_corpus()
    V = len(vocab)
    CKPT_DIR.mkdir(exist_ok=True)
    reg = CKPT_DIR / "stage12_preregistration.json"
    record = {
        "hypothesis": "Learned positional embeddings are needed by the spine model.",
        "prediction": "Removing them raises held-out NLL by more than 0.02 nats/char.",
        "metric": "held-out NLL (nats/char), stage-04 architecture at d=32, 2 layers, 64-character context",
        "analysis": "paired bootstrap 95% CI over seeds of NLL(no positions) - NLL(positions); rule: decide()",
        "threshold": 0.02,
        "seeds": [0, 1, 2, 3, 4],
        "replication_seeds": [10, 11, 12, 13, 14],
        "steps": args.steps,
    }
    print("pre-registered, sha256", preregister(reg, record))
    ok, record = verify_preregistration(reg)
    assert ok
    a, b = experiment(record["seeds"], tr, va, V, record["steps"])
    ci = paired_bootstrap_ci(b, a)
    print(f"original:    with {np.mean(a):.4f}, without {np.mean(b):.4f}; difference {ci[0]:+.4f} [{ci[1]:+.4f}, {ci[2]:+.4f}] -> {decide(ci, record['threshold'])}")
    a2, b2 = experiment(record["replication_seeds"], tr, va, V, record["steps"])
    ci2 = paired_bootstrap_ci(b2, a2)
    print(f"replication: difference {ci2[0]:+.4f} [{ci2[1]:+.4f}, {ci2[2]:+.4f}] -> {decide(ci2, record['threshold'])};"
          f" same sign: {np.sign(ci[0]) == np.sign(ci2[0])}; original estimate inside replication CI: {ci2[1] <= ci[0] <= ci2[2]}")
    ok, _ = verify_preregistration(reg)
    print("pre-registration unchanged:", ok)


if __name__ == "__main__":
    main()
