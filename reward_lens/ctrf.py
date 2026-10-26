"""Read per-test verifier results: CTRF JSON (pytest-json-ctrf) or JUnit XML (pytest's built-in --junitxml).

Test ids drop the file: 'test_x.py::TestA::test_b' -> 'TestA::test_b'. When two files define a test with the
same id, both keep their file prefix so that neither result hides the other.
"""

import json
import xml.etree.ElementTree as ET
from pathlib import Path

PASSING = {"passed", "skipped"}  # pytest exits 0 on skips, so a binary verifier treats them as non-failures
RESULT_FILES = ("reward_lens_junit.xml", "ctrf.json")  # looked up in this order inside verifier/
_SEVERITY = {"error": 3, "failed": 2, "other": 1}


def split_outside_brackets(name: str) -> list[str]:
    """Split on '::' except inside '[...]' parameter ids: 'A::test_x[a::b]' -> ['A', 'test_x[a::b]']."""
    parts, depth, start, i = [], 0, 0, 0
    while i < len(name):
        if name[i] == "[":
            depth += 1
        elif name[i] == "]":
            depth = max(depth - 1, 0)
        elif depth == 0 and name.startswith("::", i):
            parts.append(name[start:i])
            start = i + 2
            i += 1
        i += 1
    return parts + [name[start:]]


def bare_name(test_id: str) -> str:
    """The test function's name with its parameters: 'TestA::test_x[1]' -> 'test_x[1]'."""
    return split_outside_brackets(test_id)[-1]


def strip_params(name: str) -> str:
    return name.split("[", 1)[0]


def unique_ids(entries: list[tuple[str, str, str]]) -> dict[str, str]:
    """[(file, test_id, status)] -> {id: status}, prefixing ids defined in more than one file."""
    files: dict[str, set[str]] = {}
    for file, test_id, _ in entries:
        files.setdefault(test_id, set()).add(file)
    results: dict[str, str] = {}
    for file, test_id, status in entries:
        key = f"{file}::{test_id}" if len(files[test_id]) > 1 and file else test_id
        if key not in results or _SEVERITY.get(status, 0) > _SEVERITY.get(results[key], 0):
            results[key] = status  # a rerun never hides a failure
    return results


def load_ctrf(path: str | Path) -> dict[str, str]:
    tests = json.loads(Path(path).read_text())["results"]["tests"]
    entries = []
    for t in tests:
        parts = split_outside_brackets(t["name"])
        file = parts[0] if len(parts) > 1 and parts[0].endswith(".py") else ""
        entries.append((file, "::".join(parts[1:] if file else parts), t["status"]))
    return unique_ids(entries)


def load_junit(path: str | Path) -> dict[str, str]:
    entries = []
    for case in ET.parse(path).getroot().iter("testcase"):
        segments = (case.get("classname") or "").split(".")
        cls = segments.pop() if segments and segments[-1][:1].isupper() else ""
        file = "/".join(s for s in segments if s) + ".py" if any(segments) else ""
        tags = {child.tag for child in case}
        status = ("error" if "error" in tags else "failed" if "failure" in tags
                  else "skipped" if "skipped" in tags else "passed")
        entries.append((file, f"{cls}::{case.get('name')}" if cls else case.get("name", ""), status))
    return unique_ids(entries)


def load_results(verifier_dir: str | Path) -> dict[str, str]:
    """Results from a verifier directory, or {} if the verifier wrote none."""
    for name in RESULT_FILES:
        path = Path(verifier_dir) / name
        if path.exists():
            return load_junit(path) if name.endswith(".xml") else load_ctrf(path)
    return {}


def normalize_name(name: str) -> str:
    """Drop a leading 'file.py::' from a pytest node id."""
    parts = split_outside_brackets(name)
    return "::".join(parts[1:]) if len(parts) > 1 and parts[0].endswith(".py") else name
