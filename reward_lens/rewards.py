"""The three reward functions compared throughout reward-lens.

binary_reward       Terminal-Bench: 1 only if no test fails.
naive_dense_reward  Unweighted share of tests passed (what a careless dense verifier does).
milestone_reward    Long-Horizon-Terminal-Bench style: R = sum(w_k * r_k) / sum(w_k), plus gates.
"""

from dataclasses import dataclass

from reward_lens.ctrf import PASSING
from reward_lens.rubric import Rubric


@dataclass(frozen=True)
class MilestoneScore:
    name: str
    weight: float
    passed: int
    total: int
    score: float


@dataclass(frozen=True)
class DenseResult:
    reward: float
    milestones: tuple[MilestoneScore, ...]
    failed_gate: str | None = None


def binary_reward(results: dict[str, str]) -> float:
    return float(bool(results) and all(s in PASSING for s in results.values()))


def naive_dense_reward(results: dict[str, str]) -> float:
    if not results:
        return 0.0
    return sum(s in PASSING for s in results.values()) / len(results)


def milestone_reward(results: dict[str, str], rubric: Rubric) -> DenseResult:
    by_owner: dict[str, list[bool]] = {}
    for test, status in results.items():
        owner = rubric.owner(test)
        if owner is not None:
            by_owner.setdefault(owner, []).append(status in PASSING)

    scores = []
    for m in rubric.milestones:
        outcomes = by_owner.get(m.name, [])  # a milestone with no reported tests (verifier crashed) scores 0
        passed = sum(outcomes)
        if not outcomes:
            score = 0.0
        elif m.scoring == "all":
            score = float(passed == len(outcomes))
        else:
            score = passed / len(outcomes)
        scores.append(MilestoneScore(m.name, m.weight, passed, len(outcomes), score))

    for g in rubric.gates:
        outcomes = by_owner.get(g.name, [])
        if not outcomes or not all(outcomes):
            return DenseResult(0.0, tuple(scores), failed_gate=g.name)

    total_weight = sum(s.weight for s in scores)
    return DenseResult(sum(s.weight * s.score for s in scores) / total_weight, tuple(scores))
