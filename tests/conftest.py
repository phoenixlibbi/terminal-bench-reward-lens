import json
from pathlib import Path

import pytest


def write_trial(root: Path, name: str, *, model: str, tests: dict[str, str], agent: str = "terminus-2",
                exception_type: str | None = None, reward: float | None = None) -> Path:
    """Write a minimal Harbor trial directory with a CTRF report."""
    trial = root / model / name
    (trial / "verifier").mkdir(parents=True)
    if reward is None:
        reward = float(all(s == "passed" for s in tests.values()))
    (trial / "result.json").write_text(json.dumps({
        "task_name": "demo-task",
        "trial_name": name,
        "config": {"agent": {"name": agent, "model_name": f"provider/{model}"}},
        "verifier_result": {"rewards": {"reward": reward}},
        "exception_info": {"exception_type": exception_type} if exception_type else None,
        "agent_result": {"metadata": {"n_episodes": 7}},
    }))
    (trial / "verifier" / "ctrf.json").write_text(json.dumps({"results": {"tests": [
        {"name": f"test_state.py::{t}", "status": s} for t, s in tests.items()
    ]}}))
    return trial


@pytest.fixture
def make_trial(tmp_path):
    return lambda name, **kw: write_trial(tmp_path, name, **kw)
