"""Reward-hacking floor test: how much reward do non-solutions collect under each verifier?

Each adversary is a replacement solution/solve.sh in a copy of the task. Harbor's
oracle agent executes it, the task's own verifier grades the final container
state, and reward-lens scores the resulting CTRF report under all three rewards.

  nop      do nothing                                    (should score 0 everywhere)
  zeros    reference solution, then every output number -> 0
  perturb  reference solution, then output numbers shuffled / rescaled
  oracle   reference solution unchanged                  (should score 1 everywhere)

zeros and perturb keep every file, schema, row count and ID of a correct answer but
none of its values: exactly the outputs a dense verifier should not pay for.
"""

import shutil
import subprocess
from dataclasses import dataclass, field, replace
from importlib import resources
from pathlib import Path
from statistics import mean

from reward_lens.ctrf import PASSING
from reward_lens.harbor import find_trials
from reward_lens.report import grade
from reward_lens.rewards import milestone_reward
from reward_lens.rubric import Rubric

ADVERSARIES = ("nop", "zeros", "perturb", "oracle")
MARKER = "/tmp/.reward_lens_floor_marker"

NOP_SH = """#!/bin/bash
# reward-lens floor adversary: nop (does nothing)
exit 0
"""

CORRUPT_SH = """#!/bin/bash
# reward-lens floor adversary: {mode}
# Run the reference solution, then keep the outputs' format but destroy their values.
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
touch {marker}
sleep 1.1
bash "$HERE/solve_reference.sh"
PY="$(command -v python3 || command -v python)"
"$PY" "$HERE/corrupt_outputs.py" --mode {mode} --marker {marker} {roots}
"""


@dataclass(frozen=True)
class FloorRow:
    adversary: str
    runs: int
    binary: float
    naive: float
    milestone: float
    milestone_credit: dict[str, float]  # weighted contribution of each milestone to R
    results: list[dict[str, str]] = field(default_factory=list)  # per-run test outcomes

    def pass_rate(self, test: str) -> float | None:
        seen = [r[test] in PASSING for r in self.results if test in r]
        return sum(seen) / len(seen) if seen else None


@dataclass(frozen=True)
class TestVerdict:
    test: str
    owner: str
    is_gate: bool
    passed_by: dict[str, float | None]  # adversary -> pass rate
    verdict: str


@dataclass(frozen=True)
class Diagnosis:
    verdicts: list[TestVerdict]
    weak_milestones: list[str]  # milestones whose tests all pass for some non-solution
    strong_milestones: list[str]  # milestones with at least one test that every non-solution fails
    suggested: Rubric
    worst_before: float
    worst_after: float


def make_variant(task_dir: Path, adversary: str, out_dir: Path, roots: tuple[str, ...] = ("/app",)) -> Path:
    if adversary not in ADVERSARIES:
        raise ValueError(f"unknown adversary {adversary!r}; choose from {ADVERSARIES}")
    if out_dir.exists():
        shutil.rmtree(out_dir)
    shutil.copytree(task_dir, out_dir, ignore=lambda d, names: [
        n for n in names if n in {"runs", "model-run-logs", "jobs", ".DS_Store"}])
    solution = out_dir / "solution"
    solve = solution / "solve.sh"
    if adversary == "nop":
        solve.write_text(NOP_SH)
    elif adversary in ("zeros", "perturb"):
        solve.rename(solution / "solve_reference.sh")
        script = resources.files("reward_lens") / "templates" / "corrupt_outputs.py"
        (solution / "corrupt_outputs.py").write_text(script.read_text())
        solve.write_text(CORRUPT_SH.format(mode=adversary, marker=MARKER,
                                           roots=" ".join(f"--root {r}" for r in roots)))
    solve.chmod(0o755)
    return out_dir


def run_floor(task_dir: Path, work_dir: Path, adversaries=ADVERSARIES, roots=("/app",),
              harbor: str = "harbor", attempts: int = 1) -> Path:
    """Build one variant per adversary and run each through Harbor. Returns the jobs root."""
    jobs = work_dir / "jobs"
    for adversary in adversaries:
        variant = make_variant(task_dir, adversary, work_dir / "variants" / adversary, roots)
        out = jobs / adversary
        if out.exists():
            shutil.rmtree(out)
        subprocess.run([harbor, "run", "-p", str(variant), "-a", "oracle", "-o", str(out),
                        "-k", str(attempts), "-y"], check=True)
    return jobs


def floor_rows(jobs_root: Path, rubric: Rubric) -> list[FloorRow]:
    rows = []
    for adversary_dir in sorted(p for p in jobs_root.iterdir() if p.is_dir()):
        graded = [grade(t, rubric) for t in find_trials(adversary_dir)]
        if not graded:
            continue
        weight = sum(m.weight for m in rubric.milestones)
        credit = {
            m.name: mean(0.0 if g.dense.failed_gate else
                         next(s.score for s in g.dense.milestones if s.name == m.name) * m.weight / weight
                         for g in graded)
            for m in rubric.milestones
        }
        rows.append(FloorRow(adversary_dir.name, len(graded), mean(g.binary for g in graded),
                             mean(g.naive for g in graded), mean(g.dense.reward for g in graded), credit,
                             [g.trial.results for g in graded]))
    order = {a: i for i, a in enumerate(ADVERSARIES)}
    return sorted(rows, key=lambda r: order.get(r.adversary, len(order)))


CHEATERS = ("nop", "zeros", "perturb")


def diagnose(rows: list[FloorRow], rubric: Rubric) -> Diagnosis:
    """Classify every test by which non-solutions pass it, and propose `requires` for weak milestones."""
    by_adv = {r.adversary: r for r in rows}
    cheaters = [a for a in CHEATERS if a in by_adv]
    tests = sorted({t for r in rows for res in r.results for t in res})
    verdicts = []
    for test in tests:
        owner = rubric.owner(test)
        if owner is None:
            continue
        is_gate = owner in {g.name for g in rubric.gates}
        passed_by = {a: by_adv[a].pass_rate(test) for a in [*cheaters, *(["oracle"] if "oracle" in by_adv else [])]}
        verdicts.append(TestVerdict(test, owner, is_gate, passed_by, _verdict(passed_by, is_gate)))

    milestone_tests = {m.name: [v for v in verdicts if v.owner == m.name] for m in rubric.milestones}
    strong = [m for m, vs in milestone_tests.items() if any(v.verdict.startswith("strong") for v in vs)]
    weak = [m for m, vs in milestone_tests.items()
            if vs and all(v.verdict.startswith(("vacuous", "format-only", "partial")) for v in vs)]
    suggested = replace(rubric, milestones=tuple(
        replace(m, requires_any=tuple(dict.fromkeys([*m.requires_any, *strong]))) if m.name in weak and strong else m
        for m in rubric.milestones))

    def worst(r: Rubric) -> float:
        return max((mean(milestone_reward(res, r).reward for res in by_adv[a].results)
                    for a in cheaters if by_adv[a].results), default=0.0)
    return Diagnosis(verdicts, weak, strong, suggested, worst(rubric), worst(suggested))


def _verdict(passed_by: dict[str, float | None], is_gate: bool) -> str:
    def passes(adversary):
        rate = passed_by.get(adversary)
        return rate is not None and rate > 0
    if "oracle" in passed_by and passed_by["oracle"] is not None and passed_by["oracle"] < 1:
        return "broken: the reference solution fails it"
    if is_gate:
        return "integrity gate: expected to pass"
    if passes("nop"):
        return "vacuous: passes with no work done"
    if passes("zeros") and passes("perturb"):
        return "format-only: passes with wrong values"
    if passes("perturb"):
        return "partial: catches constant outputs, not misplaced values"
    if passes("zeros"):
        return "partial: catches misplaced values, not constant outputs"
    return "strong: every non-solution fails it"


def floor_markdown(rows: list[FloorRow], rubric: Rubric) -> str:
    out = [f"# Reward-hacking floor: `{rubric.name}`", "",
           "Reward collected by non-solutions. A robust verifier gives every adversary except `oracle` ~0.", "",
           "| Adversary | Runs | Binary | Naive dense | Milestone R |", "|---|---|---|---|---|"]
    for r in rows:
        out.append(f"| {r.adversary} | {r.runs} | {r.binary:.2f} | {r.naive:.2f} | **{r.milestone:.2f}** |")
    out += ["", "## Where the milestone credit comes from", "",
            "| Milestone | Weight | " + " | ".join(r.adversary for r in rows) + " |",
            "|---|---|" + "---|" * len(rows)]
    for m in rubric.milestones:
        out.append(f"| {m.name} | {m.weight:g} | " + " | ".join(f"{r.milestone_credit[m.name]:.2f}" for r in rows) + " |")
    cheaters = [r for r in rows if r.adversary not in ("oracle",)]
    if cheaters:
        worst = max(cheaters, key=lambda r: r.naive)
        out += ["", f"Worst case: `{worst.adversary}` earns naive dense {worst.naive:.2f} "
                    f"vs milestone R {worst.milestone:.2f} without solving the task."]
    if any(r.results for r in rows):
        out += ["", diagnosis_markdown(diagnose(rows, rubric))]
    return "\n".join(out) + "\n"


def diagnosis_markdown(d: Diagnosis) -> str:
    adversaries = list(d.verdicts[0].passed_by) if d.verdicts else []
    out = ["## Weak-test report", "",
           "Which tests each non-solution passes. A test a non-solution passes cannot be evidence of solving the task.",
           "", "| Test | Credited to | " + " | ".join(adversaries) + " | Verdict |",
           "|---|---|" + "---|" * len(adversaries) + "---|"]
    for v in sorted(d.verdicts, key=lambda v: (v.verdict, v.owner, v.test)):
        cells = ["-" if r is None else ("pass" if r == 1 else "FAIL" if r == 0 else f"{r:.0%}")
                 for r in v.passed_by.values()]
        owner = f"{v.owner} (gate)" if v.is_gate else v.owner
        out.append(f"| `{v.test}` | {owner} | " + " | ".join(cells) + f" | {v.verdict} |")
    counts = {}
    for v in d.verdicts:
        counts[v.verdict.split(":")[0]] = counts.get(v.verdict.split(":")[0], 0) + 1
    out += ["", "Summary: " + ", ".join(f"{n} {k}" for k, n in sorted(counts.items())), "", "## Suggested fixes", ""]
    if d.weak_milestones and d.strong_milestones:
        out.append(f"- Milestones whose tests all pass for some non-solution: {', '.join(d.weak_milestones)}. "
                   f"Make their credit conditional on real progress:")
        for m in d.suggested.milestones:
            if m.name in d.weak_milestones:
                out.append(f"  - `{m.name}`: `requires_any = {list(m.requires_any)}`".replace("'", '"'))
        out.append(f"- Worst non-solution milestone R: **{d.worst_before:.2f} -> {d.worst_after:.2f}** "
                   "with these conditions (write the rubric with `--suggest-rubric`).")
    elif d.weak_milestones:
        out.append("- No milestone contains a test that every non-solution fails: the verifier cannot tell a correct "
                   "answer from a well-formatted wrong one. Add value checks (see `reward_lens.checks`).")
    else:
        out.append("- Every milestone contains at least one test no non-solution passes.")
    weak_tests = [v for v in d.verdicts if v.verdict.startswith(("vacuous", "format-only", "partial"))]
    if weak_tests:
        out.append(f"- Strengthen the {len(weak_tests)} weak tests themselves: compare values with the reference, "
                   "require non-degenerate outputs, recompute reported metrics from the predictions, or test the "
                   "agent's code on hidden inputs.")
    return "\n".join(out)
