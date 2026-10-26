import csv
import io
import json
import os
import random
import subprocess
import sys
import time
from importlib import resources

import pytest

from reward_lens.floor import floor_markdown, floor_rows, make_variant
from reward_lens.harbor import find_trials
from reward_lens.report import build_report
from reward_lens.rubric import Milestone, Rubric
from reward_lens.sensitivity import bootstrap_ci, dirichlet, weight_sensitivity
from tests.conftest import write_trial

CORRUPT = resources.files("reward_lens") / "templates" / "corrupt_outputs.py"


def _corrupt(root, marker, mode):
    subprocess.run([sys.executable, str(CORRUPT), "--mode", mode, "--marker", str(marker), "--root", str(root)],
                   check=True, capture_output=True)


@pytest.fixture
def outputs(tmp_path):
    root = tmp_path / "app"
    (root / "data").mkdir(parents=True)
    (root / "data" / "input.csv").write_text("id,x\n1,5\n")  # older than the marker: must stay untouched
    old = time.time() - 100
    os.utime(root / "data" / "input.csv", (old, old))
    marker = tmp_path / "marker"
    marker.touch()
    os.utime(marker, (old + 50, old + 50))
    (root / "out").mkdir()
    (root / "out" / "metrics.json").write_text(json.dumps(
        {"rmse": 1.5, "n_rows": 3, "model": "gbm", "fold_scores": [0.7, 0.8, 0.9], "run_id": 42, "ok": True}))
    (root / "out" / "preds.csv").write_text("delivery_id,pred,label\n7,1.5,a\n8,2.5,b\n9,3.5,c\n")
    (root / "src.py").write_text("x = 1\n")
    return root, marker


def test_zeros_keeps_format_and_ids(outputs):
    root, marker = outputs
    _corrupt(root, marker, "zeros")
    metrics = json.loads((root / "out" / "metrics.json").read_text())
    assert metrics == {"rmse": 0.0, "n_rows": 0, "model": "gbm", "fold_scores": [0.0, 0.0, 0.0],
                       "run_id": 42, "ok": True}
    rows = list(csv.DictReader(io.StringIO((root / "out" / "preds.csv").read_text())))
    assert [r["delivery_id"] for r in rows] == ["7", "8", "9"]
    assert [r["pred"] for r in rows] == ["0", "0", "0"]
    assert [r["label"] for r in rows] == ["a", "b", "c"]
    assert (root / "data" / "input.csv").read_text() == "id,x\n1,5\n"
    assert (root / "src.py").read_text() == "x = 1\n"


def test_perturb_changes_every_numeric_field(outputs):
    root, marker = outputs
    _corrupt(root, marker, "perturb")
    metrics = json.loads((root / "out" / "metrics.json").read_text())
    assert metrics["rmse"] != 1.5 and metrics["n_rows"] != 3
    assert sorted(metrics["fold_scores"]) == [0.7, 0.8, 0.9] and metrics["fold_scores"] != [0.7, 0.8, 0.9]
    rows = list(csv.DictReader(io.StringIO((root / "out" / "preds.csv").read_text())))
    assert sorted(r["pred"] for r in rows) == ["1.5", "2.5", "3.5"]
    assert [r["pred"] for r in rows] != ["1.5", "2.5", "3.5"]
    assert [r["delivery_id"] for r in rows] == ["7", "8", "9"]


@pytest.fixture
def task(tmp_path):
    root = tmp_path / "task"
    (root / "solution").mkdir(parents=True)
    (root / "solution" / "solve.sh").write_text("#!/bin/bash\necho solved\n")
    (root / "runs").mkdir()
    return root


def test_variants(task, tmp_path):
    nop = make_variant(task, "nop", tmp_path / "v" / "nop")
    assert "exit 0" in (nop / "solution" / "solve.sh").read_text()
    assert not (nop / "runs").exists()

    zeros = make_variant(task, "zeros", tmp_path / "v" / "zeros", roots=("/app", "/output"))
    solve = (zeros / "solution" / "solve.sh").read_text()
    assert (zeros / "solution" / "solve_reference.sh").read_text() == "#!/bin/bash\necho solved\n"
    assert (zeros / "solution" / "corrupt_outputs.py").exists()
    assert "--mode zeros" in solve and "--root /app --root /output" in solve
    assert os.access(zeros / "solution" / "solve.sh", os.X_OK)

    oracle = make_variant(task, "oracle", tmp_path / "v" / "oracle")
    assert (oracle / "solution" / "solve.sh").read_text() == "#!/bin/bash\necho solved\n"
    assert (task / "solution" / "solve.sh").read_text() == "#!/bin/bash\necho solved\n"
    with pytest.raises(ValueError):
        make_variant(task, "magic", tmp_path / "v" / "magic")


RUBRIC = Rubric("toy", (Milestone("format", 1.0, ("test_schema",)), Milestone("values", 3.0, ("test_value_*",))))


def test_floor_rows_score_each_adversary(tmp_path):
    jobs = tmp_path / "jobs"
    write_trial(jobs / "oracle", "t1", model="x", agent="oracle",
                tests={"test_schema": "passed", "test_value_a": "passed", "test_value_b": "passed"})
    write_trial(jobs / "zeros", "t2", model="x", agent="oracle",
                tests={"test_schema": "passed", "test_value_a": "failed", "test_value_b": "failed"})
    write_trial(jobs / "nop", "t3", model="x", agent="oracle",
                tests={"test_schema": "failed", "test_value_a": "failed", "test_value_b": "failed"})
    rows = {r.adversary: r for r in floor_rows(jobs, RUBRIC)}

    assert list(rows) == ["nop", "zeros", "oracle"]
    assert (rows["zeros"].binary, rows["zeros"].naive, rows["zeros"].milestone) == (0.0, pytest.approx(1 / 3), 0.25)
    assert rows["zeros"].milestone_credit == {"format": 0.25, "values": 0.0}
    assert rows["oracle"].milestone == 1.0
    md = floor_markdown(list(rows.values()), RUBRIC)
    assert "| zeros | 1 | 0.00 | 0.33 | **0.25** |" in md


def test_dirichlet_is_a_distribution():
    rng = random.Random(1)
    w = dirichlet([2.0, 5.0, 1.0], rng)
    assert len(w) == 3 and sum(w) == pytest.approx(1.0) and all(x > 0 for x in w)


def test_bootstrap_ci_contains_mean_and_shrinks_with_n():
    small = bootstrap_ci([0.0, 1.0] * 5)
    large = bootstrap_ci([0.0, 1.0] * 200)
    assert small[0] <= 0.5 <= small[1]
    assert (large[1] - large[0]) < (small[1] - small[0])
    assert bootstrap_ci([]) == (0.0, 0.0)


def test_weight_sensitivity_detects_fragile_and_robust_rankings(tmp_path):
    # model-a is better on 'format', model-b on 'values': the ranking depends on the weights.
    for i in range(3):
        write_trial(tmp_path, f"a{i}", model="model-a",
                    tests={"test_schema": "passed", "test_value_a": "failed", "test_value_b": "failed"})
        write_trial(tmp_path, f"b{i}", model="model-b",
                    tests={"test_schema": "failed", "test_value_a": "passed", "test_value_b": "failed"})
    runs = build_report(find_trials(tmp_path), RUBRIC).agent_runs
    flat = weight_sensitivity(runs, RUBRIC, alpha=None, samples=300)
    tight = weight_sensitivity(runs, RUBRIC, alpha=1000.0, samples=300)
    assert 0.2 < flat.ranking_kept < 0.8
    assert tight.ranking_kept > 0.95
    lo, hi = tight.r_range["model-b"]
    assert lo <= 3 / 8 <= hi


def test_perturb_shuffles_record_fields_including_binary_labels(tmp_path):
    root = tmp_path / "app"
    root.mkdir()
    marker = tmp_path / "marker"
    marker.touch()
    old = time.time() - 100
    os.utime(marker, (old, old))
    records = [{"row_id": i, "label": i % 2, "prob": i / 10, "name": f"r{i}", "const": 3} for i in range(8)]
    (root / "preds.json").write_text(json.dumps(records))
    _corrupt(root, marker, "perturb")
    out = json.loads((root / "preds.json").read_text())
    assert [r["row_id"] for r in out] == list(range(8))           # IDs untouched
    assert [r["name"] for r in out] == [f"r{i}" for i in range(8)]  # strings untouched
    assert sorted(r["label"] for r in out) == sorted(i % 2 for i in range(8))  # balance kept
    assert [r["label"] for r in out] != [i % 2 for i in range(8)]   # correspondence broken
    assert [r["prob"] for r in out] != [i / 10 for i in range(8)]
    assert all(r["const"] == 4 for r in out)                        # constant field rescaled


def test_diagnose_classifies_tests_and_suggests_conditions(tmp_path):
    from reward_lens.floor import diagnose
    from reward_lens.rubric import Gate
    rubric = Rubric("toy", (Milestone("format", 1.0, ("test_schema", "test_sane", "test_varied")),
                            Milestone("values", 3.0, ("test_value_*",))),
                    gates=(Gate("integrity", ("test_inputs",)),))
    outcomes = {  # test -> (nop, zeros, perturb, oracle)
        "test_inputs": ("passed",) * 4,
        "test_schema": ("failed", "passed", "passed", "passed"),
        "test_sane": ("failed", "passed", "failed", "passed"),
        "test_varied": ("failed", "failed", "passed", "passed"),
        "test_value_a": ("failed", "failed", "failed", "passed"),
        "test_value_b": ("failed", "failed", "failed", "failed"),
    }
    jobs = tmp_path / "jobs"
    for i, adversary in enumerate(("nop", "zeros", "perturb", "oracle")):
        write_trial(jobs / adversary, f"t{i}", model="x", agent="oracle",
                    tests={t: o[i] for t, o in outcomes.items()})

    d = diagnose(floor_rows(jobs, rubric), rubric)
    verdicts = {v.test: v.verdict.split(":")[0] for v in d.verdicts}
    assert verdicts == {"test_inputs": "integrity gate", "test_schema": "format-only", "test_sane": "partial",
                        "test_varied": "partial", "test_value_a": "strong", "test_value_b": "broken"}
    assert d.weak_milestones == ["format"] and d.strong_milestones == ["values"]
    assert d.suggested.milestones[0].requires_any == ("values",)
    assert d.worst_before == pytest.approx(2 / 3 * 1 / 4) and d.worst_after == 0.0
    md = floor_markdown(floor_rows(jobs, rubric), rubric)
    assert "Weak-test report" in md and "requires_any = [\"values\"]" in md and "0.17 -> 0.00" in md
