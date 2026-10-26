import json
import random
import subprocess
import sys
from importlib import resources

import pytest

from reward_lens.convert import GRADER_MARKER, collect_test_ids, convert, rewrite_test_sh, tree_digest
from reward_lens.rewards import binary_reward, milestone_reward, naive_dense_reward
from reward_lens.rubric import Gate, Milestone, Rubric, RubricError, to_dict

TB_TEST_SH = """#!/bin/bash
set -euo pipefail
python -m pip install --no-cache-dir pytest==8.4.1 pytest-json-ctrf==0.3.5
pytest --ctrf /logs/verifier/ctrf.json /tests/test_state.py -rA || TEST_FAIL=1

if [ "${TEST_FAIL:-0}" -eq 0 ]; then
  echo 1 > /logs/verifier/reward.txt
else
  echo 0 > /logs/verifier/reward.txt
fi
"""

TESTS_PY = '''
def test_outputs_exist(): pass
def test_inputs_unchanged(): pass

class TestMetrics:
    def test_rmse(self): pass
    def test_mae(self): pass
    def helper(self): pass

def helper(): pass
'''

RUBRIC = Rubric(
    "toy",
    milestones=(Milestone("deliverables", 0.2, ("test_outputs_exist",)), Milestone("model", 0.8, ("TestMetrics::*",))),
    gates=(Gate("integrity", ("test_inputs_unchanged",)),),
)


@pytest.fixture
def task(tmp_path):
    root = tmp_path / "toy-task"
    (root / "tests").mkdir(parents=True)
    (root / "environment").mkdir()
    (root / "tests" / "test.sh").write_text(TB_TEST_SH)
    (root / "tests" / "test_state.py").write_text(TESTS_PY)
    (root / "instruction.md").write_text("Fix the pipeline.")
    (root / "environment" / "Dockerfile").write_text("FROM python:3.12-slim\n")
    (root / "runs" / "model-x").mkdir(parents=True)
    (root / "runs" / "model-x" / "trajectory.json").write_text("{}")
    return root


def test_collect_test_ids(task):
    assert collect_test_ids(task / "tests") == [
        "test_outputs_exist", "test_inputs_unchanged", "TestMetrics::test_rmse", "TestMetrics::test_mae"]


def test_rewrite_keeps_setup_and_replaces_binary_tail():
    script, dropped = rewrite_test_sh(TB_TEST_SH, "milestone")
    assert "pip install --no-cache-dir pytest==8.4.1" in script
    assert "pytest --ctrf /logs/verifier/ctrf.json /tests/test_state.py -rA || TEST_FAIL=1" in script
    assert "reward.txt" not in script
    assert GRADER_MARKER in script and "--primary milestone" in script
    assert any("echo 1 > /logs/verifier/reward.txt" in line for line in dropped)


def test_rewrite_adds_ctrf_flag_plugin_and_failure_guard():
    script, _ = rewrite_test_sh("#!/bin/bash\nset -e\nuv run pytest /tests -q\n", "binary")
    lines = script.splitlines()
    run = next(line for line in lines if "uv run pytest" in line)
    assert "pytest --ctrf /logs/verifier/ctrf.json /tests -q || true" in run
    assert lines.index(run) > next(i for i, line in enumerate(lines) if "pytest-json-ctrf" in line)


def test_rewrite_rejects_scripts_without_pytest_or_already_converted():
    with pytest.raises(ValueError, match="no pytest"):
        rewrite_test_sh("echo hi\n", "milestone")
    script, _ = rewrite_test_sh(TB_TEST_SH, "milestone")
    with pytest.raises(ValueError, match="already converted"):
        rewrite_test_sh(script, "milestone")


def test_convert_writes_dense_task_and_leaves_source_untouched(task, tmp_path):
    before = tree_digest(task)
    result = convert(task, RUBRIC, tmp_path / "out")

    out = result.out_dir
    assert tree_digest(task) == before
    assert not (out / "runs").exists()  # trajectories are never copied
    assert (out / "instruction.md").read_text() == "Fix the pipeline."
    assert json.loads((out / "tests" / "rubric.json").read_text()) == to_dict(RUBRIC)
    assert (out / "tests" / "reward_lens_grader.py").exists()
    assert GRADER_MARKER in (out / "tests" / "test.sh").read_text()


def test_convert_refuses_bad_rubrics_and_unsafe_outputs(task, tmp_path):
    partial = Rubric("toy", (Milestone("m", 1.0, ("test_outputs_exist",)),))
    with pytest.raises(RubricError, match="not covered"):
        convert(task, partial, tmp_path / "out")
    stale = Rubric("toy", (*RUBRIC.milestones, Milestone("extra", 1.0, ("test_gone",))), RUBRIC.gates)
    with pytest.raises(RubricError, match="matches no test"):
        convert(task, stale, tmp_path / "out")
    with pytest.raises(ValueError, match="outside the source"):
        convert(task, RUBRIC, task / "converted")
    convert(task, RUBRIC, tmp_path / "out")
    with pytest.raises(FileExistsError):
        convert(task, RUBRIC, tmp_path / "out")
    convert(task, RUBRIC, tmp_path / "out", force=True)


def _run_grader(tmp_path, rubric, results, primary="milestone"):
    grader = resources.files("reward_lens") / "templates" / "reward_lens_grader.py"
    tmp_path.mkdir(parents=True, exist_ok=True)
    (tmp_path / "rubric.json").write_text(json.dumps(to_dict(rubric)))
    if results is not None:
        (tmp_path / "ctrf.json").write_text(json.dumps({"results": {"tests": [
            {"name": f"test_state.py::{t}", "status": s} for t, s in results.items()]}}))
    subprocess.run([sys.executable, str(grader), "--ctrf", str(tmp_path / "ctrf.json"),
                    "--rubric", str(tmp_path / "rubric.json"), "--out-dir", str(tmp_path / "logs"),
                    "--primary", primary], check=True, capture_output=True)
    return json.loads((tmp_path / "logs" / "reward.json").read_text())


def test_standalone_grader_matches_library_on_random_runs(tmp_path):
    rng = random.Random(0)
    tests = ["test_outputs_exist", "test_inputs_unchanged", "TestMetrics::test_rmse", "TestMetrics::test_mae"]
    for i in range(40):
        results = {t: rng.choice(["passed", "passed", "failed", "skipped", "error"]) for t in tests if rng.random() > 0.1}
        out = _run_grader(tmp_path / str(i), RUBRIC, results)
        assert out["binary"] == binary_reward(results)
        assert out["naive_dense"] == pytest.approx(naive_dense_reward(results))
        assert out["milestone"] == pytest.approx(milestone_reward(results, RUBRIC).reward)
        assert out["reward"] == out["milestone"]


def test_standalone_grader_scores_zero_without_ctrf(tmp_path):
    out = _run_grader(tmp_path, RUBRIC, None, primary="binary")
    assert out == {"reward": 0.0, "binary": 0.0, "naive_dense": 0.0, "milestone": 0.0}
