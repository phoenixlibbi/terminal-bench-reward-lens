import json

import pytest

from reward_lens.ctrf import load_ctrf, normalize_name
from reward_lens.rewards import binary_reward, milestone_reward, naive_dense_reward
from reward_lens.rubric import Gate, Milestone, Rubric, RubricError, load_rubric, matches

RUBRIC = Rubric(
    name="demo",
    milestones=(
        Milestone("hygiene", 1.0, ("TestSchema::*",)),
        Milestone("core", 3.0, ("test_rmse", "test_coverage")),
    ),
    gates=(Gate("integrity", ("test_inputs_unchanged",)),),
)

ALL_PASS = {
    "TestSchema::test_columns": "passed",
    "TestSchema::test_rows": "passed",
    "test_rmse": "passed",
    "test_coverage": "passed",
    "test_inputs_unchanged": "passed",
}


def test_normalize_name_drops_file_prefix():
    assert normalize_name("test_state.py::TestA::test_b") == "TestA::test_b"
    assert normalize_name("test_state.py::test_b") == "test_b"
    assert normalize_name("test_b") == "test_b"


def test_load_ctrf(tmp_path):
    path = tmp_path / "ctrf.json"
    path.write_text(json.dumps({"results": {"tests": [
        {"name": "test_state.py::TestA::test_b", "status": "passed"},
        {"name": "test_state.py::test_c", "status": "failed"},
    ]}}))
    assert load_ctrf(path) == {"TestA::test_b": "passed", "test_c": "failed"}


def test_matches_bare_and_qualified_patterns():
    assert matches("test_rmse", "TestMetrics::test_rmse")
    assert matches("TestMetrics::*", "TestMetrics::test_rmse")
    assert not matches("TestOther::*", "TestMetrics::test_rmse")


def test_all_pass_scores_one_everywhere():
    assert binary_reward(ALL_PASS) == 1.0
    assert naive_dense_reward(ALL_PASS) == 1.0
    assert milestone_reward(ALL_PASS, RUBRIC).reward == 1.0


def test_skipped_counts_as_non_failure():
    results = {**ALL_PASS, "test_rmse": "skipped"}
    assert binary_reward(results) == 1.0


def test_empty_results_score_zero():
    assert binary_reward({}) == 0.0
    assert naive_dense_reward({}) == 0.0
    assert milestone_reward({}, RUBRIC).reward == 0.0


def test_hygiene_only_run_gets_free_credit_from_naive_dense_but_not_from_rubric():
    results = {**ALL_PASS, "test_rmse": "failed", "test_coverage": "failed"}
    assert binary_reward(results) == 0.0
    assert naive_dense_reward(results) == pytest.approx(3 / 5)
    assert milestone_reward(results, RUBRIC).reward == pytest.approx(1 / 4)  # hygiene weight 1 of 4


def test_partial_core_progress_is_visible():
    results = {**ALL_PASS, "test_coverage": "failed"}
    dense = milestone_reward(results, RUBRIC)
    assert binary_reward(results) == 0.0
    assert dense.reward == pytest.approx((1 * 1 + 3 * 0.5) / 4)
    assert [s.score for s in dense.milestones] == [1.0, 0.5]


def test_failed_gate_zeroes_reward():
    results = {**ALL_PASS, "test_inputs_unchanged": "failed"}
    dense = milestone_reward(results, RUBRIC)
    assert dense.reward == 0.0
    assert dense.failed_gate == "integrity"


def test_missing_gate_test_zeroes_reward():
    results = dict(ALL_PASS)
    del results["test_inputs_unchanged"]
    assert milestone_reward(results, RUBRIC).failed_gate == "integrity"


def test_all_scoring_mode():
    rubric = Rubric("demo", (Milestone("core", 1.0, ("test_*",), scoring="all"),))
    assert milestone_reward({"test_a": "passed", "test_b": "failed"}, rubric).reward == 0.0


def test_uncovered_test_is_an_error():
    with pytest.raises(RubricError, match="not covered"):
        milestone_reward({**ALL_PASS, "test_surprise": "passed"}, RUBRIC)


def test_ambiguous_test_is_an_error():
    rubric = Rubric("demo", (Milestone("a", 1.0, ("test_x",)), Milestone("b", 1.0, ("test_*",))))
    with pytest.raises(RubricError, match="several"):
        milestone_reward({"test_x": "passed"}, rubric)


def test_ignored_tests_do_not_count():
    rubric = Rubric("demo", (Milestone("a", 1.0, ("test_a",)),), ignore=("test_flaky",))
    assert milestone_reward({"test_a": "passed", "test_flaky": "failed"}, rubric).reward == 1.0


def test_load_rubric(tmp_path):
    path = tmp_path / "rubric.toml"
    path.write_text("""
[task]
name = "demo"
pass_threshold = 0.9
ignore = ["test_flaky"]

[[milestone]]
name = "core"
weight = 2
tests = ["test_a"]
scoring = "all"

[[gate]]
name = "integrity"
tests = ["test_b"]
""")
    rubric = load_rubric(path)
    assert rubric.pass_threshold == 0.9
    assert rubric.milestones[0] == Milestone("core", 2.0, ("test_a",), "all")
    assert rubric.gates[0].name == "integrity"


@pytest.mark.parametrize("body, message", [
    ("", "no milestones"),
    ('[[milestone]]\nname="a"\nweight=0\ntests=["t"]', "positive weight"),
    ('[[milestone]]\nname="a"\nweight=1\ntests=["t"]\nscoring="max"', "scoring"),
    ('[[milestone]]\nname="a"\nweight=1\ntests=[]', "no tests"),
    ('[[milestone]]\nname="a"\nweight=1\ntests=["t"]\n[[gate]]\nname="a"\ntests=["u"]', "duplicate"),
])
def test_invalid_rubrics(tmp_path, body, message):
    path = tmp_path / "rubric.toml"
    path.write_text(body)
    with pytest.raises(RubricError, match=message):
        load_rubric(path)


CONDITIONAL = Rubric(
    "demo",
    milestones=(
        Milestone("hygiene", 1.0, ("TestSchema::*",), requires=("core",)),
        Milestone("core", 3.0, ("test_rmse", "test_coverage")),
    ),
    gates=(Gate("integrity", ("test_inputs_unchanged",)),),
)


def test_requires_withholds_format_credit_from_runs_with_no_real_progress():
    hygiene_only = {**ALL_PASS, "test_rmse": "failed", "test_coverage": "failed"}
    result = milestone_reward(hygiene_only, CONDITIONAL)
    assert result.reward == 0.0
    hygiene = result.milestones[0]
    assert (hygiene.passed, hygiene.total, hygiene.score, hygiene.locked) == (2, 2, 0.0, True)


def test_requires_unlocks_once_the_required_milestone_scores():
    partial = {**ALL_PASS, "test_coverage": "failed"}
    result = milestone_reward(partial, CONDITIONAL)
    assert not result.milestones[0].locked
    assert result.reward == pytest.approx((1 * 1 + 3 * 0.5) / 4)


@pytest.mark.parametrize("requires", [("hygiene",), ("missing",)])
def test_requires_must_name_another_milestone(tmp_path, requires):
    path = tmp_path / "rubric.toml"
    path.write_text(f'[[milestone]]\nname="hygiene"\nweight=1\ntests=["t"]\nrequires={list(requires)!r}\n'.replace("'", '"'))
    with pytest.raises(RubricError, match="not another milestone"):
        load_rubric(path)


def test_load_rubric_reads_requires(tmp_path):
    path = tmp_path / "rubric.toml"
    path.write_text('[[milestone]]\nname="a"\nweight=1\ntests=["t1"]\nrequires=["b"]\n'
                    '[[milestone]]\nname="b"\nweight=1\ntests=["t2"]\n')
    assert load_rubric(path).milestones[0].requires == ("b",)


def test_requires_any_unlocks_on_any_listed_milestone():
    rubric = Rubric("demo", (Milestone("format", 1.0, ("test_schema",), requires_any=("a", "b")),
                             Milestone("a", 1.0, ("test_a",)), Milestone("b", 1.0, ("test_b",))))
    none = milestone_reward({"test_schema": "passed", "test_a": "failed", "test_b": "failed"}, rubric)
    one = milestone_reward({"test_schema": "passed", "test_a": "failed", "test_b": "passed"}, rubric)
    assert none.reward == 0.0 and none.milestones[0].locked
    assert one.reward == pytest.approx(2 / 3) and not one.milestones[0].locked


def test_to_toml_round_trips(tmp_path):
    from reward_lens.rubric import to_toml
    rubric = Rubric("demo", (Milestone("format", 0.25, ("TestSchema::*",), "all", ("core",), ("core",)),
                             Milestone("core", 0.75, ("test_rmse",))),
                    gates=(Gate("integrity", ("test_inputs_unchanged",)),), ignore=("test_flaky",),
                    pass_threshold=0.9, near_miss=0.7)
    path = tmp_path / "r.toml"
    path.write_text(to_toml(rubric))
    assert load_rubric(path) == rubric
