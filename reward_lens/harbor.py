"""Discover Harbor trials on disk.

A Harbor trial directory holds result.json (with a "trial_name" key), config.json,
agent/ and verifier/. Job-level result.json files aggregate trials and are skipped.
"""

import json
import sys
from dataclasses import dataclass
from pathlib import Path

from reward_lens.ctrf import load_results


@dataclass(frozen=True)
class Trial:
    path: Path
    task: str
    model: str
    results: dict[str, str]  # test_id -> status; empty if the verifier never wrote ctrf.json
    original_reward: float | None
    rewards: dict[str, float]  # everything the verifier reported, e.g. binary/naive_dense/milestone
    termination: str  # "exited" | "timeout" | "error"
    n_episodes: int | None


def find_trials(root: str | Path) -> list[Trial]:
    """Every readable trial under root. Unreadable result.json files (e.g. a run still in progress) are skipped."""
    trials = []
    for result_path in sorted(Path(root).rglob("result.json")):
        try:
            result = json.loads(result_path.read_text())
        except (OSError, ValueError) as exc:
            print(f"reward-lens: skipping {result_path}: {exc}", file=sys.stderr)
            continue
        if isinstance(result, dict) and "trial_name" in result:
            trials.append(load_trial(result_path.parent, result))
    return trials


def load_trial(path: Path, result: dict) -> Trial:
    config = result.get("config") or _read_json(path / "config.json")
    agent = config.get("agent", {})
    rewards = (result.get("verifier_result") or {}).get("rewards") or {}
    metadata = (result.get("agent_result") or {}).get("metadata") or {}
    return Trial(
        path=path,
        task=result.get("task_name", path.parent.name),
        model=_model_name(agent, path),
        results=load_results(path / "verifier"),
        original_reward=rewards.get("binary", rewards.get("reward")),
        rewards=dict(rewards),
        termination=_termination(result.get("exception_info")),
        n_episodes=metadata.get("n_episodes"),
    )


def _model_name(agent: dict, path: Path) -> str:
    if agent.get("name") in ("oracle", "nop"):
        return agent["name"]
    if agent.get("model_name"):
        return agent["model_name"].split("/", 1)[-1]  # "ollama/llama3.1:8b" -> "llama3.1:8b"
    name = agent.get("name") or path.parent.name
    return name.split(":")[-1] if ":" in name and "." in name.split(":")[0] else name  # "pkg.mod:Agent" -> Agent


def _termination(exception_info: dict | None) -> str:
    if not exception_info:
        return "exited"
    return "timeout" if "Timeout" in (exception_info.get("exception_type") or "") else "error"


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text()) if path.exists() else {}
