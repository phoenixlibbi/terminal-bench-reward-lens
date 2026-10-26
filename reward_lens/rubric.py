"""Rubrics map a task's tests onto weighted milestones and integrity gates.

A milestone is a meaningful intermediate goal (LHTB's "subtask"); a gate is an
integrity check whose failure zeroes the whole reward (Terminal-Bench's
"integrity" criterion, enforced at grading time).
"""

import tomllib
from dataclasses import dataclass
from fnmatch import fnmatchcase
from pathlib import Path

from reward_lens.ctrf import bare_name, normalize_name, strip_params

SCORING_MODES = {"fraction", "all"}


@dataclass(frozen=True)
class Milestone:
    name: str
    weight: float
    tests: tuple[str, ...]
    scoring: str = "fraction"  # "fraction": share of tests passed; "all": 1 only if every test passes
    requires: tuple[str, ...] = ()  # credit only if each of these milestones scores > 0 on its own tests
    requires_any: tuple[str, ...] = ()  # credit only if at least one of these milestones scores > 0


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
    """Glob match. Patterns without '::' match the bare test function name; '[params]' suffixes are optional;
    a 'file.py::' prefix (used when two files define the same test) is optional too."""
    if "::" in pattern:
        targets = {test_id, normalize_name(test_id)}
    else:
        targets = {bare_name(test_id)}
    targets |= {strip_params(t) for t in targets}
    return any(fnmatchcase(t, pattern) for t in targets)


def to_dict(rubric: Rubric) -> dict:
    """JSON-serializable form, read by the in-container grader."""
    return {
        "name": rubric.name,
        "pass_threshold": rubric.pass_threshold,
        "near_miss": rubric.near_miss,
        "ignore": list(rubric.ignore),
        "milestones": [{"name": m.name, "weight": m.weight, "tests": list(m.tests), "scoring": m.scoring,
                        "requires": list(m.requires), "requires_any": list(m.requires_any)}
                       for m in rubric.milestones],
        "gates": [{"name": g.name, "tests": list(g.tests)} for g in rubric.gates],
    }


def to_toml(rubric: Rubric) -> str:
    def strings(values) -> str:
        return "[" + ", ".join(f'"{v}"' for v in values) + "]"
    lines = ["[task]", f'name = "{rubric.name}"', f"pass_threshold = {rubric.pass_threshold:g}",
             f"near_miss = {rubric.near_miss:g}"]
    if rubric.ignore:
        lines.append(f"ignore = {strings(rubric.ignore)}")
    for g in rubric.gates:
        lines += ["", "[[gate]]", f'name = "{g.name}"', f"tests = {strings(g.tests)}"]
    for m in rubric.milestones:
        lines += ["", "[[milestone]]", f'name = "{m.name}"', f"weight = {m.weight:g}", f"tests = {strings(m.tests)}"]
        if m.scoring != "fraction":
            lines.append(f'scoring = "{m.scoring}"')
        if m.requires:
            lines.append(f"requires = {strings(m.requires)}")
        if m.requires_any:
            lines.append(f"requires_any = {strings(m.requires_any)}")
    return "\n".join(lines) + "\n"


def load_rubric(path: str | Path) -> Rubric:
    try:
        data = tomllib.loads(Path(path).read_text())
    except tomllib.TOMLDecodeError as exc:
        raise RubricError(f"{path}: invalid TOML: {exc}") from exc
    task = data.get("task", {})
    try:
        milestones = tuple(
            Milestone(m["name"], float(m["weight"]), tuple(m["tests"]), m.get("scoring", "fraction"),
                      tuple(m.get("requires", ())), tuple(m.get("requires_any", ())))
            for m in data.get("milestone", [])
        )
        gates = tuple(Gate(g["name"], tuple(g["tests"])) for g in data.get("gate", []))
    except KeyError as exc:
        raise RubricError(f"{path}: every milestone needs name, weight and tests; every gate name and tests "
                          f"(missing {exc})") from exc
    except (TypeError, ValueError) as exc:
        raise RubricError(f"{path}: {exc}") from exc
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
        milestone_names = {other.name for other in rubric.milestones}
        for req in (*m.requires, *m.requires_any):
            if req == m.name or req not in milestone_names:
                raise RubricError(f"milestone {m.name!r} requires {req!r}, which is not another milestone")
    if not 0 < rubric.near_miss <= rubric.pass_threshold <= 1:
        raise RubricError("need 0 < near_miss <= pass_threshold <= 1")
