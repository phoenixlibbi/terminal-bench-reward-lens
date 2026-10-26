"""Convert a Terminal-Bench-style Harbor task (binary reward) into a dense milestone-reward task.

The source task is only read. The converted copy keeps the original verifier byte for byte as
tests/test_original.sh and gets a small tests/test.sh wrapper that
  1. runs the original verifier unchanged, with PYTEST_ADDOPTS asking pytest for a JUnit XML report
     (built into pytest: no plugin to install, works under pip, uv run, uvx, venvs, ...),
  2. keeps the original 0/1 verdict as reward_original.txt,
  3. runs tests/reward_lens_grader.py (stdlib only), which writes binary, naive dense and milestone rewards.
"""

import ast
import hashlib
import json
import re
import shutil
from dataclasses import dataclass, field
from importlib import resources
from pathlib import Path

from reward_lens.rubric import Rubric, RubricError, matches, to_dict

SKIP_DIRS = {"runs", "model-run-logs", "jobs", "__pycache__"}
GRADER_MARKER = "# --- reward-lens: dense milestone reward (generated) ---"
JUNIT_PATH = "/logs/verifier/reward_lens_junit.xml"

WRAPPER = """#!/bin/bash
{marker}
# Runs the task's original verifier unchanged (test_original.sh), records per-test results with pytest's
# built-in JUnit XML report, then grades them with the rubric in rubric.json.
mkdir -p /logs/verifier
rm -f {junit} /logs/verifier/reward.txt /logs/verifier/reward.json /logs/verifier/reward_original.txt
export PYTEST_ADDOPTS="${{PYTEST_ADDOPTS:-}} --junitxml={junit}"
bash /tests/test_original.sh
echo "original verifier exited with $?"
[ -f /logs/verifier/reward.txt ] && mv /logs/verifier/reward.txt /logs/verifier/reward_original.txt
PY="$(command -v python3 || command -v python)"
"$PY" /tests/reward_lens_grader.py --results-dir /logs/verifier --rubric /tests/rubric.json \\
  --out-dir /logs/verifier --primary {primary} --original-reward /logs/verifier/reward_original.txt
"""


@dataclass(frozen=True)
class Conversion:
    out_dir: Path
    tests: list[str]
    source_digest: str
    warnings: list[str] = field(default_factory=list)


def collect_test_ids(tests_dir: Path) -> list[str]:
    """Statically list pytest test ids ('test_x' or 'TestClass::test_x') in tests/test_*.py."""
    ids = []
    for path in sorted(tests_dir.glob("test_*.py")):
        tree = ast.parse(path.read_text())
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name.startswith("test"):
                ids.append(node.name)
            elif isinstance(node, ast.ClassDef) and node.name.startswith("Test"):
                ids += [f"{node.name}::{fn.name}" for fn in node.body
                        if isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)) and fn.name.startswith("test")]
    return ids


def check_coverage(rubric: Rubric, test_ids: list[str]) -> None:
    """Every test must belong to exactly one milestone or gate, and every rubric pattern must match a test."""
    errors = []
    for test in test_ids:
        try:
            rubric.owner(test)
        except RubricError as exc:
            errors.append(str(exc))
    for entry in [*rubric.milestones, *rubric.gates]:
        for pattern in entry.tests:
            if not any(matches(pattern, t) for t in test_ids):
                errors.append(f"pattern {pattern!r} in {entry.name!r} matches no test")
    if errors:
        raise RubricError("rubric does not fit the task:\n  " + "\n  ".join(errors))


def script_warnings(script: str) -> list[str]:
    """Verifier patterns the wrapper cannot fully support; conversion still works, with caveats."""
    code = [line for line in script.splitlines() if not line.lstrip().startswith("#")]
    calls = [line for line in code if re.search(r"(^|[\s/;&|(])(pytest|py\.test)(\s|$)", line)
             and not re.search(r"\b(pip|uv pip|apt-get|apk|conda)\b.*\binstall\b", line)]
    warnings = []
    if not calls:
        warnings.append("no pytest call found in test.sh: without per-test results only binary can be recovered")
    if len(calls) > 1:
        warnings.append(f"{len(calls)} pytest calls in test.sh: the JUnit report keeps the last call's results only")
    if "--junitxml" in script or "junit_xml" in script:
        warnings.append("test.sh sets its own --junitxml, which overrides the wrapper's report path")
    return warnings


def tree_digest(root: Path) -> str:
    h = hashlib.sha256()
    for path in sorted(p for p in root.rglob("*") if p.is_file() and not _skipped(p, root)):
        h.update(str(path.relative_to(root)).encode())
        h.update(path.read_bytes())
    return h.hexdigest()


def convert(task_dir: Path, rubric: Rubric, out_dir: Path, primary: str = "milestone",
            force: bool = False, with_checks: bool = False) -> Conversion:
    task_dir, out_dir = task_dir.resolve(), out_dir.resolve()
    test_sh = task_dir / "tests" / "test.sh"
    if not test_sh.exists():
        raise ValueError(f"{task_dir} is not a Harbor task (no tests/test.sh)")
    original = test_sh.read_text()
    if GRADER_MARKER in original:
        raise ValueError(f"{task_dir} is already converted")
    if out_dir == task_dir or task_dir in out_dir.parents:
        raise ValueError("output directory must be outside the source task")
    if out_dir.exists():
        if not force:
            raise FileExistsError(f"{out_dir} exists (use --force to replace it)")
        shutil.rmtree(out_dir)

    test_ids = collect_test_ids(task_dir / "tests")
    check_coverage(rubric, test_ids)
    digest = tree_digest(task_dir)

    shutil.copytree(task_dir, out_dir, ignore=lambda d, names: [
        n for n in names if n in SKIP_DIRS or n == ".DS_Store"])
    tests = out_dir / "tests"
    (tests / "test_original.sh").write_text(original)
    (tests / "test.sh").write_text(WRAPPER.format(marker=GRADER_MARKER, junit=JUNIT_PATH, primary=primary))
    (tests / "test.sh").chmod(0o755)
    (tests / "rubric.json").write_text(json.dumps(to_dict(rubric), indent=2))
    templates = resources.files("reward_lens") / "templates"
    (tests / "reward_lens_grader.py").write_text((templates / "reward_lens_grader.py").read_text())
    if with_checks:  # value-check helpers for hardening the task's own tests (import reward_lens_checks)
        (tests / "reward_lens_checks.py").write_text((resources.files("reward_lens") / "checks.py").read_text())

    if tree_digest(task_dir) != digest:
        raise RuntimeError("source task changed during conversion")
    return Conversion(out_dir, test_ids, digest, script_warnings(original))


def _skipped(path: Path, root: Path) -> bool:
    return path.name == ".DS_Store" or any(part in SKIP_DIRS for part in path.relative_to(root).parts)
