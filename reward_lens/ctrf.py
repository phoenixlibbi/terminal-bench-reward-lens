"""Read pytest CTRF reports, the per-test results Harbor verifiers write to verifier/ctrf.json."""

import json
from pathlib import Path

PASSING = {"passed", "skipped"}  # pytest exits 0 on skips, so a binary verifier treats them as non-failures


def load_ctrf(path: str | Path) -> dict[str, str]:
    """Return {test_id: status}. The file prefix is dropped: 'test_x.py::TestA::test_b' -> 'TestA::test_b'."""
    tests = json.loads(Path(path).read_text())["results"]["tests"]
    return {normalize_name(t["name"]): t["status"] for t in tests}


def normalize_name(name: str) -> str:
    parts = name.split("::")
    return "::".join(parts[1:]) if len(parts) > 1 and parts[0].endswith(".py") else name
