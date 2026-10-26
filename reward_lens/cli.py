import argparse
import json
import re
import sys
from pathlib import Path

from reward_lens import __version__
from reward_lens.convert import check_coverage, collect_test_ids, convert
from reward_lens.floor import ADVERSARIES, floor_markdown, floor_rows, run_floor
from reward_lens.harbor import find_trials
from reward_lens.report import build_report, to_json, to_markdown
from reward_lens.sensitivity import sensitivity_markdown, weight_sensitivity
from reward_lens.rubric import load_rubric


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="reward-lens", description=__doc__)
    parser.add_argument("--version", action="version", version=__version__)
    sub = parser.add_subparsers(dest="command", required=True)

    regrade = sub.add_parser("regrade", help="grade Harbor trials under binary, naive dense and milestone rewards")
    regrade.add_argument("runs", type=Path, help="directory containing Harbor trial folders")
    regrade.add_argument("--rubric", type=Path, required=True)
    regrade.add_argument("--out", type=Path, help="write the markdown report here instead of stdout")
    regrade.add_argument("--json", type=Path, help="also write per-run and per-model results as JSON")

    init = sub.add_parser("rubric-init", help="draft a rubric from the tests seen in Harbor trials")
    init.add_argument("runs", type=Path)
    init.add_argument("--name", default="my-task")

    conv = sub.add_parser("convert", help="turn a binary-reward Harbor task into a dense milestone-reward task")
    conv.add_argument("task", type=Path, help="source Harbor task directory (only read)")
    conv.add_argument("--rubric", type=Path, required=True)
    conv.add_argument("--out", type=Path, required=True, help="where to write the converted task")
    conv.add_argument("--primary", choices=["milestone", "binary", "naive_dense"], default="milestone",
                      help="which reward Harbor reports as 'reward' (all three are always written)")
    conv.add_argument("--force", action="store_true", help="replace --out if it exists")

    check = sub.add_parser("check-rubric", help="statically check that a rubric fits a task's tests")
    check.add_argument("task", type=Path)
    check.add_argument("--rubric", type=Path, required=True)

    floor = sub.add_parser("floor", help="reward-hacking floor test: grade adversarial non-solutions in Docker")
    floor.add_argument("task", type=Path, help="Harbor task directory (only read)")
    floor.add_argument("--rubric", type=Path, required=True)
    floor.add_argument("--work", type=Path, required=True, help="where variants and Harbor jobs are written")
    floor.add_argument("--adversaries", default=",".join(ADVERSARIES))
    floor.add_argument("--root", action="append", help="directories holding outputs (default /app)")
    floor.add_argument("--attempts", type=int, default=1)
    floor.add_argument("--report-only", action="store_true", help="skip Harbor, re-score existing jobs")
    floor.add_argument("--out", type=Path)

    sens = sub.add_parser("sensitivity", help="bootstrap CIs and robustness of rankings to rubric weights")
    sens.add_argument("runs", type=Path)
    sens.add_argument("--rubric", type=Path, required=True)
    sens.add_argument("--samples", type=int, default=500)
    sens.add_argument("--alpha", type=float, action="append",
                      help="Dirichlet concentration around the rubric weights (repeatable; default 50 and 10)")
    sens.add_argument("--out", type=Path)

    args = parser.parse_args(argv)
    if args.command == "regrade":
        run_regrade(args)
    elif args.command == "rubric-init":
        print(draft_rubric(args.runs, args.name))
    elif args.command == "convert":
        result = convert(args.task, load_rubric(args.rubric), args.out, args.primary, args.force)
        print(f"converted {len(result.tests)} tests -> {result.out_dir}")
        if result.dropped_lines:
            print("replaced reward-writing lines in test.sh:\n  " + "\n  ".join(result.dropped_lines))
    elif args.command == "floor":
        rubric = load_rubric(args.rubric)
        jobs = args.work / "jobs"
        if not args.report_only:
            jobs = run_floor(args.task, args.work, tuple(args.adversaries.split(",")),
                             tuple(args.root or ["/app"]), attempts=args.attempts)
        emit(floor_markdown(floor_rows(jobs, rubric), rubric), args.out)
    elif args.command == "sensitivity":
        rubric = load_rubric(args.rubric)
        runs = build_report(find_trials(args.runs), rubric).agent_runs
        results = [weight_sensitivity(runs, rubric, a, args.samples) for a in (args.alpha or [50.0, 10.0])]
        results.append(weight_sensitivity(runs, rubric, None, args.samples))
        emit(sensitivity_markdown(runs, rubric, results), args.out)
    else:
        tests = collect_test_ids(args.task / "tests")
        check_coverage(load_rubric(args.rubric), tests)
        print(f"ok: rubric covers all {len(tests)} tests exactly once")


def run_regrade(args: argparse.Namespace) -> None:
    trials = find_trials(args.runs)
    if not trials:
        sys.exit(f"no Harbor trials found under {args.runs}")
    report = build_report(trials, load_rubric(args.rubric))
    emit(to_markdown(report), args.out)
    if args.json:
        args.json.write_text(json.dumps(to_json(report), indent=2))


def emit(text: str, out: Path | None) -> None:
    if out:
        out.write_text(text)
    else:
        print(text)


def draft_rubric(runs: Path, name: str) -> str:
    """One milestone per test class (or one for module-level tests), equal weights, to be edited by hand."""
    groups: dict[str, set[str]] = {}
    for trial in find_trials(runs):
        for test in trial.results:
            cls, _, fn = test.rpartition("::")
            groups.setdefault(cls or "module_tests", set()).add(test if cls else fn)
    lines = ["[task]", f'name = "{name}"', "pass_threshold = 0.95", "near_miss = 0.75", ""]
    for group, tests in sorted(groups.items()):
        milestone = re.sub(r"(?<!^)(?=[A-Z])", "_", group.removeprefix("Test")).lower()
        lines += ["[[milestone]]", f'name = "{milestone}"', "weight = 1",
                  "tests = [" + ", ".join(f'"{t}"' for t in sorted(tests)) + "]", ""]
    return "\n".join(lines)


if __name__ == "__main__":
    main()
