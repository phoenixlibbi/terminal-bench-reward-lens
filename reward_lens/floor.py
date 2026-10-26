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
from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from statistics import mean

from reward_lens.harbor import find_trials
from reward_lens.report import grade
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
                             mean(g.naive for g in graded), mean(g.dense.reward for g in graded), credit))
    order = {a: i for i, a in enumerate(ADVERSARIES)}
    return sorted(rows, key=lambda r: order.get(r.adversary, len(order)))


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
    return "\n".join(out) + "\n"
