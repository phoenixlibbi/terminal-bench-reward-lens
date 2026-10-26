"""Rubrics map a task's tests onto weighted milestones and integrity gates.

A milestone is a meaningful intermediate goal (LHTB's "subtask"); a gate is an
integrity check whose failure zeroes the whole reward (Terminal-Bench's
"integrity" criterion, enforced at grading time).
"""

import tomllib
from dataclasses import dataclass
from fnmatch import fnmatchcase
from pathlib import Path

SCORING_MODES = {"fraction", "all"}


@dataclass(frozen=True)
class Milestone:
    name: str
    weight: float
    tests: tuple[str, ...]
    scoring: str = "fraction"  # "fraction": share of tests passed; "all": 1 only if every test passes


@dataclass(frozen=True)
class Gate:
    name: str
    tests: tuple[str, ...]


@dataclass(frozen=True)
class Rubric:
    name: str
    milestones: tuple[Milestone, ...]
    gates: tuple[Gate, ...] = ()
    ignore: tuple[str, ...] = ()
    pass_threshold: float = 0.95
    near_miss: float = 0.75

    def owner(self, test_id: str) -> str | None:
        """Name of the milestone or gate that owns test_id, None if ignored. Raises if unowned or ambiguous."""
        if any(matches(p, test_id) for p in self.ignore):
            return None
        owners = [m.name for m in self.milestones if any(matches(p, test_id) for p in m.tests)]
        owners += [g.name for g in self.gates if any(matches(p, test_id) for p in g.tests)]
        if len(owners) != 1:
            problem = "is not covered by" if not owners else f"matches several entries ({', '.join(owners)}) in"
            raise RubricError(f"test {test_id!r} {problem} rubric {self.name!r}")
        return owners[0]


class RubricError(ValueError):
    pass


def matches(pattern: str, test_id: str) -> bool:
    """Glob match. Patterns without '::' match the bare test function name; '[params]' suffixes are optional."""
    target = test_id if "::" in pattern else test_id.split("::")[-1]
    return fnmatchcase(target, pattern) or fnmatchcase(target.split("[")[0], pattern)


def to_dict(rubric: Rubric) -> dict:
    """JSON-serializable form, read by the in-container grader."""
    return {
        "name": rubric.name,
        "pass_threshold": rubric.pass_threshold,
        "near_miss": rubric.near_miss,
        "ignore": list(rubric.ignore),
        "milestones": [{"name": m.name, "weight": m.weight, "tests": list(m.tests), "scoring": m.scoring}
                       for m in rubric.milestones],
        "gates": [{"name": g.name, "tests": list(g.tests)} for g in rubric.gates],
    }


def load_rubric(path: str | Path) -> Rubric:
    data = tomllib.loads(Path(path).read_text())
    task = data.get("task", {})
    milestones = tuple(
        Milestone(m["name"], float(m["weight"]), tuple(m["tests"]), m.get("scoring", "fraction"))
        for m in data.get("milestone", [])
    )
    gates = tuple(Gate(g["name"], tuple(g["tests"])) for g in data.get("gate", []))
    rubric = Rubric(
        name=task.get("name", Path(path).stem),
        milestones=milestones,
        gates=gates,
        ignore=tuple(task.get("ignore", [])),
        pass_threshold=float(task.get("pass_threshold", 0.95)),
        near_miss=float(task.get("near_miss", 0.75)),
    )
    validate(rubric)
    return rubric


def validate(rubric: Rubric) -> None:
    if not rubric.milestones:
        raise RubricError(f"rubric {rubric.name!r} has no milestones")
    names = [m.name for m in rubric.milestones] + [g.name for g in rubric.gates]
    if len(set(names)) != len(names):
        raise RubricError(f"rubric {rubric.name!r} has duplicate milestone/gate names")
    for m in rubric.milestones:
        if m.weight <= 0:
            raise RubricError(f"milestone {m.name!r} needs a positive weight")
        if m.scoring not in SCORING_MODES:
            raise RubricError(f"milestone {m.name!r}: scoring must be one of {sorted(SCORING_MODES)}")
        if not m.tests:
            raise RubricError(f"milestone {m.name!r} lists no tests")
    if not 0 < rubric.near_miss <= rubric.pass_threshold <= 1:
        raise RubricError("need 0 < near_miss <= pass_threshold <= 1")
