"""How much do conclusions depend on the rubric's weights and on which runs we happened to sample?

weight_sensitivity  Resample milestone weights from a Dirichlet centred on the rubric's weights
                    (concentration alpha; large alpha = small perturbations) and from a flat
                    Dirichlet (any weighting at all). Report how often the model ranking survives
                    and the spread of each model's mean R.
bootstrap_ci        Percentile bootstrap confidence interval of a model's mean R over its runs,
                    the analogue of Terminal-Bench's 95% error bars.
"""

import random
from dataclasses import dataclass, replace
from statistics import mean

from reward_lens.report import Graded
from reward_lens.rewards import milestone_reward
from reward_lens.rubric import Rubric


@dataclass(frozen=True)
class Sensitivity:
    alpha: float | None  # None = flat Dirichlet
    samples: int
    ranking_kept: float  # share of samples with the same model order as the rubric's weights
    r_range: dict[str, tuple[float, float]]  # per model: 5th and 95th percentile of mean R


def dirichlet(alphas: list[float], rng: random.Random) -> list[float]:
    draws = [rng.gammavariate(a, 1.0) for a in alphas]
    total = sum(draws)
    return [d / total for d in draws]


def reweight(rubric: Rubric, weights: list[float]) -> Rubric:
    return replace(rubric, milestones=tuple(replace(m, weight=w) for m, w in zip(rubric.milestones, weights)))


def model_means(runs: list[Graded], rubric: Rubric) -> dict[str, float]:
    by_model: dict[str, list[float]] = {}
    for g in runs:
        by_model.setdefault(g.trial.model, []).append(milestone_reward(g.trial.results, rubric).reward)
    return {m: mean(rs) for m, rs in by_model.items()}


def ranking(means: dict[str, float]) -> tuple[str, ...]:
    return tuple(sorted(means, key=lambda m: (-means[m], m)))


def weight_sensitivity(runs: list[Graded], rubric: Rubric, alpha: float | None = 20.0,
                       samples: int = 500, seed: int = 0) -> Sensitivity:
    rng = random.Random(seed)
    base = [m.weight for m in rubric.milestones]
    total = sum(base)
    concentration = [1.0] * len(base) if alpha is None else [alpha * w / total for w in base]
    baseline = ranking(model_means(runs, rubric))
    kept, per_model = 0, {}
    for _ in range(samples):
        means = model_means(runs, reweight(rubric, dirichlet(concentration, rng)))
        kept += ranking(means) == baseline
        for model, r in means.items():
            per_model.setdefault(model, []).append(r)
    return Sensitivity(alpha, samples, kept / samples,
                       {m: (_percentile(rs, 5), _percentile(rs, 95)) for m, rs in per_model.items()})


def bootstrap_ci(values: list[float], samples: int = 2000, level: float = 0.95, seed: int = 0) -> tuple[float, float]:
    if not values:
        return (0.0, 0.0)
    rng = random.Random(seed)
    means = [mean(rng.choices(values, k=len(values))) for _ in range(samples)]
    tail = (1 - level) / 2 * 100
    return _percentile(means, tail), _percentile(means, 100 - tail)


def _percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    k = (len(ordered) - 1) * q / 100
    lo, hi = int(k), min(int(k) + 1, len(ordered) - 1)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (k - lo)


def sensitivity_markdown(runs: list[Graded], rubric: Rubric, results: list[Sensitivity]) -> str:
    models = sorted({g.trial.model for g in runs})
    out = [f"# Robustness: `{rubric.name}`", "", "## Mean milestone R with 95% bootstrap CI", "",
           "| Model | Runs | Mean R | 95% CI | Binary pass | 95% CI |", "|---|---|---|---|---|---|"]
    for m in models:
        rs = [g.dense.reward for g in runs if g.trial.model == m]
        bs = [g.binary for g in runs if g.trial.model == m]
        lo, hi = bootstrap_ci(rs)
        blo, bhi = bootstrap_ci(bs)
        out.append(f"| {m} | {len(rs)} | {mean(rs):.2f} | [{lo:.2f}, {hi:.2f}] | {mean(bs):.0%} | [{blo:.0%}, {bhi:.0%}] |")
    out += ["", "## Sensitivity to milestone weights", "",
            "| Weights sampled from | Samples | Ranking unchanged | " + " | ".join(f"{m} R (5-95%)" for m in models) + " |",
            "|---|---|---|" + "---|" * len(models)]
    for s in results:
        label = "flat Dirichlet (any weights)" if s.alpha is None else f"Dirichlet around rubric, alpha={s.alpha:g}"
        out.append(f"| {label} | {s.samples} | {s.ranking_kept:.0%} | "
                   + " | ".join(f"{s.r_range[m][0]:.2f}-{s.r_range[m][1]:.2f}" for m in models) + " |")
    return "\n".join(out) + "\n"
