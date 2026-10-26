"""Grade trials under all three reward functions and summarize what each one hides."""

from collections import Counter
from dataclasses import asdict, dataclass, field
from statistics import mean

from reward_lens.harbor import Trial
from reward_lens.rewards import DenseResult, binary_reward, milestone_reward, naive_dense_reward
from reward_lens.rubric import Rubric

REFERENCE_AGENTS = {"oracle", "nop"}
NO_PROGRESS = 0.05
# Bucket edges follow Figure 4 of Long-Horizon-Terminal-Bench.
BUCKETS = [(0.0, 0.05), *[(round(a, 2), round(a + 0.1, 2)) for a in [0.05 + 0.1 * i for i in range(9)]], (0.95, 1.01)]


@dataclass(frozen=True)
class Graded:
    trial: Trial
    binary: float
    naive: float
    dense: DenseResult


@dataclass
class ModelSummary:
    model: str
    runs: int
    binary_pass: float
    naive_mean: float
    r_mean: float
    r_pass: float
    near_misses: int
    no_progress: int
    gate_failures: int
    milestone_means: dict[str, float]
    terminations: dict[str, int]
    episodes_mean: float | None
    binary_rank: int = 0
    r_rank: int = 0


@dataclass
class Report:
    rubric: Rubric
    graded: list[Graded]
    models: list[ModelSummary]
    reference: list[Graded] = field(default_factory=list)

    @property
    def agent_runs(self) -> list[Graded]:
        return [g for g in self.graded if g.trial.model not in REFERENCE_AGENTS]


def grade(trial: Trial, rubric: Rubric) -> Graded:
    return Graded(trial, binary_reward(trial.results), naive_dense_reward(trial.results),
                  milestone_reward(trial.results, rubric))


def build_report(trials: list[Trial], rubric: Rubric) -> Report:
    graded = [grade(t, rubric) for t in trials]
    agents = [g for g in graded if g.trial.model not in REFERENCE_AGENTS]
    models = [summarize(m, [g for g in agents if g.trial.model == m], rubric)
              for m in sorted({g.trial.model for g in agents})]
    for m, rank in zip(models, competition_ranks([m.binary_pass for m in models])):
        m.binary_rank = rank
    for m, rank in zip(models, competition_ranks([m.r_mean for m in models])):
        m.r_rank = rank
    models.sort(key=lambda m: (-m.r_mean, m.model))
    reference = [g for g in graded if g.trial.model in REFERENCE_AGENTS]
    return Report(rubric, graded, models, reference)


def summarize(model: str, runs: list[Graded], rubric: Rubric) -> ModelSummary:
    rs = [g.dense.reward for g in runs]
    episodes = [g.trial.n_episodes for g in runs if g.trial.n_episodes is not None]
    return ModelSummary(
        model=model,
        runs=len(runs),
        binary_pass=mean(g.binary for g in runs),
        naive_mean=mean(g.naive for g in runs),
        r_mean=mean(rs),
        r_pass=mean(r >= rubric.pass_threshold for r in rs),
        near_misses=sum(rubric.near_miss <= r < rubric.pass_threshold for r in rs),
        no_progress=sum(r < NO_PROGRESS for r in rs),
        gate_failures=sum(g.dense.failed_gate is not None for g in runs),
        milestone_means={m.name: mean(_milestone(g, m.name) for g in runs) for m in rubric.milestones},
        terminations=dict(Counter(g.trial.termination for g in runs)),
        episodes_mean=mean(episodes) if episodes else None,
    )


def sanity_problems(report: Report) -> list[str]:
    """Conditions under which the rubric or the re-grade should not be trusted."""
    problems = [f"{g.trial.path.name}: oracle scores R={g.dense.reward:.2f}, binary={g.binary:.0f}"
                for g in report.reference if g.trial.model == "oracle" and (g.dense.reward < 1.0 or g.binary < 1.0)]
    problems += [f"{g.trial.path.name}: recomputed binary {g.binary:.0f} != verifier reward {g.trial.original_reward}"
                 for g in report.graded
                 if g.trial.original_reward is not None and g.binary != g.trial.original_reward]
    return problems


def competition_ranks(values: list[float]) -> list[int]:
    """Rank descending; ties share a rank (1, 2, 2, 4)."""
    return [1 + sum(other > v for other in values) for v in values]


def histogram(rewards: list[float]) -> list[tuple[str, int]]:
    rows = []
    for lo, hi in BUCKETS:
        label = "< 0.05" if lo == 0 else ">= 0.95" if hi > 1 else f"{lo:.2f}-{hi:.2f}"
        rows.append((label, sum(lo <= r < hi for r in rewards)))
    return rows


def _milestone(g: Graded, name: str) -> float:
    return next(s.score for s in g.dense.milestones if s.name == name)


def to_markdown(report: Report) -> str:
    r, runs = report.rubric, report.agent_runs
    out = [f"# reward-lens report: `{r.name}`", "",
           f"{len(runs)} agent runs, {len(report.models)} models. Pass threshold R >= {r.pass_threshold}, "
           f"near-miss {r.near_miss} <= R < {r.pass_threshold}.", ""]

    out += ["## Sanity checks", ""]
    for g in report.reference:
        out.append(f"- `{g.trial.model}` run: binary {g.binary:.0f}, milestone R {g.dense.reward:.2f} "
                   f"({'ok' if g.dense.reward == 1.0 or g.trial.model == 'nop' else 'CHECK RUBRIC'})")
    known = [g for g in report.graded if g.trial.original_reward is not None]
    agree = sum(g.binary == g.trial.original_reward for g in known)
    out += [f"- Re-computed binary reward matches the verifier's own reward.txt in {agree}/{len(known)} runs.", ""]

    out += ["## Leaderboard", "",
            "| Model | Runs | Binary pass | Naive dense | Milestone R | Pass @ R | Near-misses | No progress "
            "| Rank (binary -> R) |",
            "|---|---|---|---|---|---|---|---|---|"]
    for m in report.models:
        out.append(f"| {m.model} | {m.runs} | {m.binary_pass:.0%} | {m.naive_mean:.2f} | **{m.r_mean:.2f}** "
                   f"| {m.r_pass:.0%} | {m.near_misses} | {m.no_progress} | {m.binary_rank} -> {m.r_rank} |")
    out.append("")

    failed = [g for g in runs if g.binary == 0]
    if failed:
        hidden = sum(g.dense.reward >= r.near_miss for g in failed)
        out += ["## What each reward hides", "",
                f"- **Binary hides progress:** {len(failed)} runs score 0 under binary grading; "
                f"their mean milestone R is {mean(g.dense.reward for g in failed):.2f}, "
                f"and {hidden} of them reach R >= {r.near_miss}.",
                f"- **Naive dense inflates:** on those same runs the unweighted share of tests passed averages "
                f"{mean(g.naive for g in failed):.2f}, i.e. "
                f"{mean(g.naive - g.dense.reward for g in failed):+.2f} of credit per run that the rubric "
                f"does not grant.", ""]

    out += ["## Reward distribution (milestone R, all agent runs)", "", "```"]
    width = max((n for _, n in histogram([g.dense.reward for g in runs])), default=1) or 1
    for label, n in histogram([g.dense.reward for g in runs]):
        out.append(f"{label:>10} | {'#' * round(30 * n / width):<30} {n}")
    out += ["```", ""]

    out += ["## Milestone completion (mean score per model)", "",
            "| Milestone | Weight | " + " | ".join(m.model for m in report.models) + " |",
            "|---|---|" + "---|" * len(report.models)]
    for ms in r.milestones:
        out.append(f"| {ms.name} | {ms.weight:g} | "
                   + " | ".join(f"{m.milestone_means[ms.name]:.2f}" for m in report.models) + " |")
    if r.gates:
        out.append(f"| gate failures ({', '.join(g.name for g in r.gates)}) | R = 0 | "
                   + " | ".join(str(m.gate_failures) for m in report.models) + " |")
    out.append("")

    out += ["## How runs ended", "", "| Model | Exited | Timeout | Error | Mean episodes |", "|---|---|---|---|---|"]
    for m in report.models:
        t = m.terminations
        eps = f"{m.episodes_mean:.0f}" if m.episodes_mean is not None else "-"
        out.append(f"| {m.model} | {t.get('exited', 0)} | {t.get('timeout', 0)} | {t.get('error', 0)} | {eps} |")
    out.append("")
    return "\n".join(out)


def to_json(report: Report) -> dict:
    return {
        "rubric": report.rubric.name,
        "models": [asdict(m) for m in report.models],
        "runs": [
            {"trial": g.trial.path.name, "model": g.trial.model, "binary": g.binary, "naive": g.naive,
             "milestone_r": g.dense.reward, "failed_gate": g.dense.failed_gate,
             "termination": g.trial.termination,
             "milestones": {s.name: s.score for s in g.dense.milestones}}
            for g in report.graded
        ],
    }
