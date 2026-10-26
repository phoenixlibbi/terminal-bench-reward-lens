"""Convert a Terminal-Bench-style Harbor task (binary reward) into a dense milestone-reward task.

The source task is only read. The converted copy gets:
  tests/rubric.json             the rubric, in JSON so the grader needs no third-party parser
  tests/reward_lens_grader.py   stdlib-only grader (see templates/)
  tests/test.sh                 same setup and pytest run, but the 0/1 tail is replaced by the grader
"""

import ast
import hashlib
import json
import re
import shutil
from dataclasses import dataclass
from importlib import resources
from pathlib import Path

from reward_lens.rubric import Rubric, RubricError, matches, to_dict

SKIP_DIRS = {"runs", "model-run-logs", "jobs", "__pycache__"}
CTRF_PATH = "/logs/verifier/ctrf.json"
GRADER_MARKER = "# --- reward-lens: dense milestone reward (generated) ---"


@dataclass(frozen=True)
class Conversion:
    out_dir: Path
    tests: list[str]
    dropped_lines: list[str]
    source_digest: str


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


def rewrite_test_sh(script: str, primary: str) -> tuple[str, list[str]]:
    """Keep everything up to the last pytest call, make sure it writes CTRF, replace the reward tail."""
    if GRADER_MARKER in script:
        raise ValueError("test.sh was already converted")
    lines = script.rstrip("\n").split("\n")
    pytest_lines = [i for i, line in enumerate(lines)
                    if re.search(r"(^|\s|/)(pytest|py\.test)(\s|$)", line) and not line.lstrip().startswith("#")
                    and "install" not in line]
    if not pytest_lines:
        raise ValueError("no pytest invocation found in test.sh")
    last = pytest_lines[-1]
    for i in pytest_lines:
        line = lines[i]
        if "--ctrf" not in line:
            line = re.sub(r"\b(pytest|py\.test)\b", rf"\1 --ctrf {CTRF_PATH}", line, count=1)
        if "||" not in line:
            line += " || true"  # a failing test run must not abort the script before grading
        lines[i] = line

    kept, dropped = lines[: last + 1], [line for line in lines[last + 1:] if line.strip()]
    if "pytest-json-ctrf" not in script:  # the CTRF plugin must be installed before pytest runs
        kept.insert(last, "python3 -m pip install --quiet pytest-json-ctrf==0.3.5 || pip install --quiet pytest-json-ctrf")
    tail = [
        "",
        GRADER_MARKER,
        'PY="$(command -v python3 || command -v python)"',
        f'"$PY" /tests/reward_lens_grader.py --ctrf {CTRF_PATH} --rubric /tests/rubric.json '
        f"--out-dir /logs/verifier --primary {primary}",
        "",
    ]
    return "\n".join(kept + tail), dropped


def tree_digest(root: Path) -> str:
    h = hashlib.sha256()
    for path in sorted(p for p in root.rglob("*") if p.is_file() and not _skipped(p, root)):
        h.update(str(path.relative_to(root)).encode())
        h.update(path.read_bytes())
    return h.hexdigest()


def convert(task_dir: Path, rubric: Rubric, out_dir: Path, primary: str = "milestone",
            force: bool = False) -> Conversion:
    task_dir, out_dir = task_dir.resolve(), out_dir.resolve()
    if not (task_dir / "tests" / "test.sh").exists():
        raise ValueError(f"{task_dir} is not a Harbor task (no tests/test.sh)")
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
    script, dropped = rewrite_test_sh((tests / "test.sh").read_text(), primary)
    (tests / "test.sh").write_text(script)
    (tests / "rubric.json").write_text(json.dumps(to_dict(rubric), indent=2))
    grader = resources.files("reward_lens") / "templates" / "reward_lens_grader.py"
    (tests / "reward_lens_grader.py").write_text(grader.read_text())

    if tree_digest(task_dir) != digest:
        raise RuntimeError("source task changed during conversion")
    return Conversion(out_dir, test_ids, dropped, digest)


def _skipped(path: Path, root: Path) -> bool:
    return path.name == ".DS_Store" or any(part in SKIP_DIRS for part in path.relative_to(root).parts)
