"""`reward-lens run`: convert a task to dense grading, run an agent on it through Harbor, report every reward."""

import json
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from reward_lens.convert import convert
from reward_lens.ctrf import PASSING, load_ctrf
from reward_lens.harbor import Trial, find_trials
from reward_lens.rubric import Rubric

AGENTS = {
    "replay": "reward_lens.harness.agents:ReplayAgent",
    "loop": "reward_lens.harness.agents:LoopAgent",
    "oracle": "oracle",
    "nop": "nop",
}


@dataclass(frozen=True)
class Fidelity:
    agree: int
    total: int
    flipped: dict[str, tuple[str, str]]  # test -> (original status, replayed status)


def harbor_binary() -> str:
    local = Path(sys.executable).with_name("harbor")
    return str(local) if local.exists() else (shutil.which("harbor") or "harbor")


def run_task(task: Path, rubric: Rubric, agent: str, work: Path, *, model: str | None = None,
             agent_kwargs: dict[str, str] | None = None, attempts: int = 1) -> Path:
    converted = convert(task, rubric, work / "task", force=True).out_dir
    jobs = work / "jobs"
    cmd = [harbor_binary(), "run", "-p", str(converted), "-a", AGENTS.get(agent, agent),
           "-o", str(jobs), "-k", str(attempts), "-y"]
    if model:
        cmd += ["-m", model]
    for key, value in (agent_kwargs or {}).items():
        cmd += ["--ak", f"{key}={value}"]
    subprocess.run(cmd, check=True)
    return jobs


def replay_fidelity(original: Path, replayed: Trial) -> Fidelity:
    """How many tests end in the same pass/fail state when a recorded run is replayed."""
    before = load_ctrf(original / "verifier" / "ctrf.json")
    after = replayed.results
    flipped = {t: (s, after.get(t, "missing")) for t, s in before.items()
               if (s in PASSING) != (after.get(t) in PASSING)}
    return Fidelity(len(before) - len(flipped), len(before), flipped)


def summary_markdown(trials: list[Trial], original: Path | None = None) -> str:
    out = ["| Trial | Agent | Binary | Naive dense | Milestone R | Ended |", "|---|---|---|---|---|---|"]
    details = []
    for t in trials:
        r = t.rewards
        out.append(f"| {t.path.name} | {t.model} | {r.get('binary', 0):.0f} | {r.get('naive_dense', 0):.2f} "
                   f"| **{r.get('milestone', 0):.2f}** | {t.termination} |")
        breakdown = t.path / "verifier" / "reward_breakdown.json"
        if breakdown.exists():
            data = json.loads(breakdown.read_text())
            details.append(f"\n`{t.path.name}` milestones" + (f" (gate failed: {data['failed_gate']})"
                                                              if data.get("failed_gate") else "") + ":")
            details += [f"- {m['name']} (w={m['weight']:g}): {m['passed']}/{m['total']} -> {m['score']:.2f}"
                        for m in data.get("milestones", [])]
        if original is not None:
            f = replay_fidelity(original, t)
            details.append(f"\nReplay fidelity: {f.agree}/{f.total} tests end in the same state as the recording.")
            details += [f"- {test}: {a} -> {b}" for test, (a, b) in sorted(f.flipped.items())]
    return "\n".join(out + details) + "\n"


def parse_kwargs(pairs: list[str] | None) -> dict[str, str]:
    kwargs = {}
    for pair in pairs or []:
        key, sep, value = pair.partition("=")
        if not sep:
            raise ValueError(f"agent kwarg {pair!r} must look like key=value")
        kwargs[key] = value
    return kwargs

