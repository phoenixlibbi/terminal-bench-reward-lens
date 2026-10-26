import json

from reward_lens.harbor import find_trials


def test_finds_trials_and_skips_job_level_results(tmp_path, make_trial):
    make_trial("t1", model="model-x", tests={"test_a": "passed"})
    make_trial("t2", model="model-y", tests={"test_a": "failed"}, exception_type="AgentTimeoutError")
    (tmp_path / "result.json").write_text(json.dumps({"n_total_trials": 2}))

    trials = find_trials(tmp_path)

    assert [t.model for t in trials] == ["model-x", "model-y"]
    assert trials[0].results == {"test_a": "passed"}
    assert trials[0].original_reward == 1.0
    assert trials[0].termination == "exited"
    assert trials[1].termination == "timeout"
    assert trials[0].n_episodes == 7


def test_oracle_and_error_trials(tmp_path, make_trial):
    make_trial("t1", model="whatever", agent="oracle", tests={"test_a": "passed"})
    make_trial("t2", model="model-x", tests={"test_a": "failed"}, exception_type="RuntimeError")
    by_model = {t.model: t for t in find_trials(tmp_path)}
    assert set(by_model) == {"oracle", "model-x"}
    assert by_model["model-x"].termination == "error"


def test_trial_without_ctrf_has_empty_results(tmp_path, make_trial):
    trial = make_trial("t1", model="model-x", tests={"test_a": "passed"})
    (trial / "verifier" / "ctrf.json").unlink()
    assert find_trials(tmp_path)[0].results == {}


def test_model_name_from_custom_agent_import_path(tmp_path, make_trial):
    trial = make_trial("t1", model="x", tests={"test_a": "passed"})
    result = json.loads((trial / "result.json").read_text())
    result["config"]["agent"] = {"name": "reward_lens.harness.agents:ReplayAgent", "model_name": None}
    (trial / "result.json").write_text(json.dumps(result))
    assert find_trials(tmp_path)[0].model == "ReplayAgent"
