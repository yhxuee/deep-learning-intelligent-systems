"""Meta-test: are the invariants strong enough to catch each planted bug?

Always runs against buggy_variants.py (not your implementation). Prints which
invariants each variant fails and, if you filled bug_hunt_predictions.py,
compares with your predictions.
"""
import os
import pathlib
import sys

import pytest

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
from _loader import load_file  # noqa: E402

reference = load_file(HERE / "reference.py", "reference")
sys.modules["reference"] = reference
invariants = load_file(HERE / "invariants.py", "unit04_invariants_bh")
buggy = load_file(HERE / "buggy_variants.py", "unit04_buggy")
predictions = load_file(HERE / "bug_hunt_predictions.py", "unit04_predictions").PREDICTIONS


def failing_checks(impl):
    failed = []
    for name, check in invariants.CHECKS.items():
        try:
            ok, _ = check(impl)
        except Exception:  # crashing counts as failing
            ok = False
        if not ok:
            failed.append(name)
    try:
        ok, _ = invariants.matches_reference(impl, reference)
    except Exception:
        ok = False
    if not ok:
        failed.append("matches_reference")
    return failed


def test_reference_passes_every_invariant():
    assert failing_checks(reference) == []


@pytest.mark.parametrize("variant", sorted(buggy.VARIANTS))
def test_each_planted_bug_is_caught(variant):
    failed = failing_checks(buggy.VARIANTS[variant])
    print(f"\nvariant {variant}: fails {failed}")
    assert failed, f"variant {variant} passes every invariant — the invariants are too weak"
    pred = predictions.get(variant)
    if pred and os.environ.get("CHECK_PREDICTIONS", "1") == "1":
        missed = set(failed) ^ set(pred.get("fails", []))
        print(f"  your prediction: {pred.get('fails')}  →  {'correct' if not missed else 'differs on ' + str(sorted(missed))}")
