import json

import pytest

from reward_lens.cli import draft_rubric, main
from reward_lens.harbor import find_trials
from reward_lens.report import build_report, competition_ranks, histogram, to_markdown
from reward_lens.rubric import Milestone, Rubric, load_rubric

RUBRIC = Rubric("demo", (Milestone("hygiene", 1.0, ("test_schema",)), Milestone("core", 3.0, ("test_core_*",))))
PASS = {"test_schema": "passed", "test_core_a": "passed", "test_core_b": "passed"}
HALF = {"test_schema": "passed", "test_core_a": "passed", "test_core_b": "failed"}
HYGIENE_ONLY = {"test_schema": "passed", "test_core_a": "failed", "test_core_b": "failed"}


@pytest.fixture
def runs(tmp_path, make_trial):
    make_trial("o1", model="oracle", agent="oracle", tests=PASS)
    make_trial("a1", model="model-a", tests=PASS)
    make_trial("a2", model="model-a", tests=HYGIENE_ONLY)
    make_trial("b1", model="model-b", tests=HALF)
    make_trial("b2", model="model-b", tests=HALF)
    return tmp_path


def test_ranks_and_summaries(runs):
    report = build_report(find_trials(runs), RUBRIC)
    a, b = sorted(report.models, key=lambda m: m.model)

    assert a.binary_pass == 0.5 and b.binary_pass == 0.0
    assert a.r_mean == pytest.approx((1.0 + 0.25) / 2)
    assert b.r_mean == pytest.approx(0.625)
    assert (a.binary_rank, b.binary_rank) == (1, 2)  # binary prefers model-a ...
    assert (a.r_rank, b.r_rank) == (1, 1)  # ... but both make the same weighted progress
    assert [g.trial.model for g in report.reference] == ["oracle"]


def test_markdown_mentions_key_sections(runs):
    md = to_markdown(build_report(find_trials(runs), RUBRIC))
    for section in ("Sanity checks", "Leaderboard", "What each reward hides", "Reward distribution",
                    "Milestone completion", "How runs ended"):
        assert section in md
    assert "`oracle` run: binary 1, milestone R 1.00 (ok)" in md
    assert "matches the verifier's own reward.txt in 5/5 runs" in md


def test_competition_ranks():
    assert competition_ranks([0.5, 0.9, 0.5, 0.1]) == [2, 1, 2, 4]


def test_histogram_buckets_cover_unit_interval():
    rows = histogram([0.0, 0.04, 0.05, 0.5, 0.94, 0.95, 1.0])
    assert rows[0] == ("< 0.05", 2)
    assert rows[-1] == (">= 0.95", 2)
    assert sum(n for _, n in rows) == 7


def test_cli_regrade_writes_reports(runs, tmp_path):
    rubric = tmp_path / "rubric.toml"
    rubric.write_text('[[milestone]]\nname="all"\nweight=1\ntests=["test_*"]\n')
    with pytest.raises(SystemExit) as info:
        main(["regrade", str(runs), "--rubric", str(rubric),
              "--out", str(tmp_path / "r.md"), "--json", str(tmp_path / "r.json")])
    assert info.value.code == 0
    assert "Leaderboard" in (tmp_path / "r.md").read_text()
    data = json.loads((tmp_path / "r.json").read_text())
    assert len(data["runs"]) == 5


def test_draft_rubric_is_loadable_and_covers_every_test(tmp_path, make_trial):
    make_trial("t1", model="m", tests={"TestSchema::test_cols": "passed", "TestModelFit::test_rmse": "failed",
                                       "test_loose": "passed"})
    path = tmp_path / "draft.toml"
    path.write_text(draft_rubric(tmp_path, "demo"))
    rubric = load_rubric(path)
    assert {m.name for m in rubric.milestones} == {"schema", "model_fit", "module_tests"}
    for test in ("TestSchema::test_cols", "TestModelFit::test_rmse", "test_loose"):
        rubric.owner(test)
