import argparse
import json
import re
import sys
from pathlib import Path

from reward_lens import __version__
from reward_lens.convert import check_coverage, collect_test_ids, convert
from reward_lens.floor import ADVERSARIES, diagnose, floor_markdown, floor_rows, run_floor
from reward_lens.harbor import find_trials
from reward_lens.report import build_report, sanity_problems, to_json, to_markdown
from reward_lens.sensitivity import sensitivity_markdown, weight_sensitivity
from reward_lens.rubric import RubricError, load_rubric, to_toml
from reward_lens.run import AGENTS, parse_kwargs, run_task, summary_markdown


def main(argv: list[str] | None = None) -> None:
    try:
        sys.exit(_main(argv))
    except (RubricError, ValueError, FileExistsError, FileNotFoundError, NotADirectoryError) as exc:
        print(f"reward-lens: error: {exc}", file=sys.stderr)
        sys.exit(2)


def _main(argv: list[str] | None) -> int:
    parser = argparse.ArgumentParser(prog="reward-lens", description=__doc__)
    parser.add_argument("--version", action="version", version=__version__)
    sub = parser.add_subparsers(dest="command", required=True)

    regrade = sub.add_parser("regrade", help="grade Harbor trials under binary, naive dense and milestone rewards")
    regrade.add_argument("runs", type=Path, help="directory containing Harbor trial folders")
    regrade.add_argument("--rubric", type=Path, required=True)
    regrade.add_argument("--out", type=Path, help="write the markdown report here instead of stdout")
    regrade.add_argument("--json", type=Path, help="also write per-run and per-model results as JSON")
    regrade.add_argument("--strict", action="store_true",
                         help="exit 1 if an oracle run scores below 1.0 or binary disagrees with the verifier")

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
    conv.add_argument("--with-checks", action="store_true",
                      help="also copy reward_lens_checks.py (value-check helpers) into the task's tests/")

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
    floor.add_argument("--suggest-rubric", type=Path, help="write the rubric with suggested requires here")
    floor.add_argument("--max-free-credit", type=float,
                       help="exit 1 if any adversary other than oracle earns milestone R above this (CI gate)")

    sens = sub.add_parser("sensitivity", help="bootstrap CIs and robustness of rankings to rubric weights")
    sens.add_argument("runs", type=Path)
    sens.add_argument("--rubric", type=Path, required=True)
    sens.add_argument("--samples", type=int, default=500)
    sens.add_argument("--alpha", type=float, action="append",
                      help="Dirichlet concentration around the rubric weights (repeatable; default 50 and 10)")
    sens.add_argument("--out", type=Path)

    run = sub.add_parser("run", help="convert a task, run an agent on it via Harbor, report all rewards")
    run.add_argument("task", type=Path)
    run.add_argument("--rubric", type=Path, required=True)
    run.add_argument("--agent", default="oracle", help=f"one of {sorted(AGENTS)} or a module:Class import path")
    run.add_argument("--work", type=Path, required=True, help="where the converted task and jobs are written")
    run.add_argument("-m", "--model", help="model for LLM agents, e.g. ollama/llama3.1:8b")
    run.add_argument("--trajectory", type=Path, help="recorded trial to replay (with --agent replay)")
    run.add_argument("--ak", action="append", help="extra agent kwarg key=value (repeatable)")
    run.add_argument("--attempts", type=int, default=1)
    run.add_argument("--env-file", type=Path, help="dotenv file with agent credentials, passed to Harbor")
    run.add_argument("--out", type=Path)

    args = parser.parse_args(argv)
    if args.command == "regrade":
        return run_regrade(args)
    elif args.command == "rubric-init":
        print(draft_rubric(args.runs, args.name))
    elif args.command == "convert":
        result = convert(args.task, load_rubric(args.rubric), args.out, args.primary, args.force, args.with_checks)
        print(f"converted {len(result.tests)} tests -> {result.out_dir} (original verifier kept as tests/test_original.sh)")
        for warning in result.warnings:
            print(f"warning: {warning}", file=sys.stderr)
    elif args.command == "floor":
        rubric = load_rubric(args.rubric)
        jobs = args.work / "jobs"
        if not args.report_only:
            jobs = run_floor(args.task, args.work, tuple(args.adversaries.split(",")),
                             tuple(args.root or ["/app"]), attempts=args.attempts)
        rows = floor_rows(jobs, rubric)
        emit(floor_markdown(rows, rubric), args.out)
        if args.suggest_rubric:
            args.suggest_rubric.write_text(to_toml(diagnose(rows, rubric).suggested))
        leaks = [r for r in rows if r.adversary != "oracle" and args.max_free_credit is not None
                 and r.milestone > args.max_free_credit]
        if leaks:
            print("reward-lens: free credit above threshold: "
                  + ", ".join(f"{r.adversary} R={r.milestone:.2f}" for r in leaks), file=sys.stderr)
            return 1
    elif args.command == "run":
        kwargs = parse_kwargs(args.ak)
        if args.trajectory:
            kwargs["trajectory"] = str(args.trajectory.resolve())
        jobs = run_task(args.task, load_rubric(args.rubric), args.agent, args.work, model=args.model,
                        agent_kwargs=kwargs, attempts=args.attempts, env_file=args.env_file)
        original = args.trajectory if args.trajectory and (args.trajectory / "verifier").exists() else None
        emit(summary_markdown(find_trials(jobs), original), args.out)
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
    return 0


def run_regrade(args: argparse.Namespace) -> int:
    trials = find_trials(args.runs)
    if not trials:
        raise FileNotFoundError(f"no Harbor trials found under {args.runs}")
    report = build_report(trials, load_rubric(args.rubric))
    emit(to_markdown(report), args.out)
    if args.json:
        args.json.write_text(json.dumps(to_json(report), indent=2))
    problems = sanity_problems(report)
    if args.strict and problems:
        print("reward-lens: sanity check failed:\n  " + "\n  ".join(problems), file=sys.stderr)
        return 1
    return 0


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
