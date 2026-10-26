import subprocess
import sys

import pytest

from reward_lens.cli import main
from tests.conftest import write_trial

RUBRIC = '[[milestone]]\nname="format"\nweight=1\ntests=["test_schema"]\n' \
         '[[milestone]]\nname="values"\nweight=3\ntests=["test_value"]\n'


def exit_code(argv):
    with pytest.raises(SystemExit) as info:
        main(argv)
    return info.value.code


@pytest.fixture
def rubric(tmp_path):
    path = tmp_path / "rubric.toml"
    path.write_text(RUBRIC)
    return path


def test_regrade_strict_passes_on_consistent_runs(tmp_path, rubric):
    runs = tmp_path / "runs"
    write_trial(runs, "o", model="oracle", agent="oracle", tests={"test_schema": "passed", "test_value": "passed"})
    write_trial(runs, "a", model="m", tests={"test_schema": "passed", "test_value": "failed"})
    assert exit_code(["regrade", str(runs), "--rubric", str(rubric), "--out", str(tmp_path / "r.md"), "--strict"]) == 0


def test_regrade_strict_fails_when_oracle_is_not_perfect(tmp_path, rubric, capsys):
    runs = tmp_path / "runs"
    write_trial(runs, "o", model="oracle", agent="oracle", tests={"test_schema": "passed", "test_value": "failed"})
    assert exit_code(["regrade", str(runs), "--rubric", str(rubric), "--out", str(tmp_path / "r.md"), "--strict"]) == 1
    assert "oracle scores R=0.25" in capsys.readouterr().err


def test_regrade_strict_fails_when_binary_disagrees_with_verifier(tmp_path, rubric, capsys):
    runs = tmp_path / "runs"
    write_trial(runs, "a", model="m", tests={"test_schema": "passed", "test_value": "passed"}, reward=0.0)
    assert exit_code(["regrade", str(runs), "--rubric", str(rubric), "--out", str(tmp_path / "r.md"), "--strict"]) == 1
    assert "!= verifier reward" in capsys.readouterr().err


def test_floor_gate_on_free_credit(tmp_path, rubric, capsys):
    jobs = tmp_path / "work" / "jobs"
    write_trial(jobs / "zeros", "z", model="x", agent="oracle", tests={"test_schema": "passed", "test_value": "failed"})
    write_trial(jobs / "oracle", "o", model="x", agent="oracle", tests={"test_schema": "passed", "test_value": "passed"})
    base = ["floor", "unused-task", "--rubric", str(rubric), "--work", str(tmp_path / "work"), "--report-only",
            "--out", str(tmp_path / "floor.md")]
    assert exit_code(base + ["--max-free-credit", "0.3"]) == 0
    assert exit_code(base + ["--max-free-credit", "0.2"]) == 1
    assert "zeros R=0.25" in capsys.readouterr().err


def test_user_errors_are_reported_without_traceback(tmp_path, capsys):
    bad = tmp_path / "bad.toml"
    bad.write_text("")
    assert exit_code(["regrade", str(tmp_path), "--rubric", str(bad)]) == 2
    assert capsys.readouterr().err.startswith("reward-lens: error:")
    assert exit_code(["regrade", str(tmp_path / "empty"), "--rubric", str(bad)]) == 2


def test_console_entry_point_runs():
    out = subprocess.run([sys.executable, "-m", "reward_lens.cli", "--version"], capture_output=True, text=True)
    assert out.returncode == 0 and out.stdout.strip()
