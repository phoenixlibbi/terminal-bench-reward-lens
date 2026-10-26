"""Value checks for hardening verifiers against well-formatted wrong answers.

Format checks (file exists, columns, row count, keys) pass for any output with the right
shape, including all-zero or shuffled values. These helpers check the values themselves.
Each raises AssertionError with a readable message, so they drop into pytest tests.

Standard library only and Python 3.8+, so the file can be copied into a task's tests/
(`reward-lens convert --with-checks` does that) and imported in a container with
`from reward_lens_checks import ...`. Sequences can be lists, tuples or numpy arrays.

  assert_non_degenerate   catches constant outputs ("all zeros")
  assert_strict_intervals catches empty intervals that make lower <= pred <= upper vacuous
  assert_correlated       catches values that are present but misplaced (shuffled rows)
  assert_consistent       catches reported metrics that do not follow from the predictions
"""

import math


def _floats(values, name):
    out = [float(v) for v in values]
    if not out:
        raise AssertionError(f"{name}: no values")
    if any(math.isnan(v) or math.isinf(v) for v in out):
        raise AssertionError(f"{name}: contains NaN or infinity")
    return out


def _std(xs):
    m = sum(xs) / len(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / len(xs))


def pearson(x, y):
    xs, ys = _floats(x, "x"), _floats(y, "y")
    if len(xs) != len(ys):
        raise AssertionError(f"length mismatch: {len(xs)} vs {len(ys)}")
    sx, sy = _std(xs), _std(ys)
    if sx == 0 or sy == 0:
        return 0.0
    mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
    return sum((a - mx) * (b - my) for a, b in zip(xs, ys)) / (len(xs) * sx * sy)


def rmse(pred, truth):
    p, t = _floats(pred, "pred"), _floats(truth, "truth")
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(p, t)) / len(p))


def mae(pred, truth):
    p, t = _floats(pred, "pred"), _floats(truth, "truth")
    return sum(abs(a - b) for a, b in zip(p, t)) / len(p)


def assert_non_degenerate(values, name="values", min_unique=2, min_std=0.0):
    """At least `min_unique` distinct values and a standard deviation above `min_std`."""
    xs = _floats(values, name)
    unique = len(set(xs))
    if unique < min_unique:
        raise AssertionError(f"{name}: only {unique} distinct value(s); expected at least {min_unique}")
    if _std(xs) <= min_std:
        raise AssertionError(f"{name}: standard deviation {_std(xs):.3g} <= {min_std}")


def assert_strict_intervals(lower, pred, upper, min_width=1e-9, name="interval"):
    """lower <= pred <= upper AND upper - lower > min_width for every row."""
    lo, p, hi = _floats(lower, "lower"), _floats(pred, "pred"), _floats(upper, "upper")
    bad = [i for i, (a, b, c) in enumerate(zip(lo, p, hi)) if not (a <= b <= c and c - a > min_width)]
    if bad:
        raise AssertionError(f"{name}: {len(bad)} of {len(lo)} rows are empty or do not contain the "
                             f"prediction (first: row {bad[0]})")


def assert_correlated(pred, reference, min_r=0.5, name="predictions"):
    """Pearson correlation with the reference (ground truth or a reference model) of at least min_r."""
    r = pearson(pred, reference)
    if r < min_r:
        raise AssertionError(f"{name}: correlation with the reference is {r:.3f} < {min_r}")


def assert_consistent(reported, recomputed, rel=1e-3, abs_tol=1e-3, name="metrics"):
    """Every recomputed metric matches the reported one: invented numbers do not add up."""
    missing = sorted(set(recomputed) - set(reported))
    if missing:
        raise AssertionError(f"{name}: missing {missing}")
    wrong = {k: (reported[k], v) for k, v in recomputed.items()
             if not math.isclose(float(reported[k]), float(v), rel_tol=rel, abs_tol=abs_tol)}
    if wrong:
        detail = ", ".join(f"{k}: reported {a:.4g} vs recomputed {b:.4g}" for k, (a, b) in sorted(wrong.items()))
        raise AssertionError(f"{name} do not follow from the predictions: {detail}")
