#!/usr/bin/env python3
"""Dense milestone grader, generated into a task's tests/ by `reward-lens convert`.

Runs inside the task container after pytest. Standard library only, Python 3.8+.
Reads the CTRF report and tests/rubric.json, then writes:
  /logs/verifier/reward.json            {"reward": <primary>, "binary": .., "naive_dense": .., "milestone": ..}
  /logs/verifier/reward_breakdown.json  per-milestone scores and the failed gate, if any
It always writes a reward, even when pytest crashed and no CTRF report exists.
"""
import argparse
import fnmatch
import json
import os
import sys

PASSING = {"passed", "skipped"}


def normalize_name(name):
    parts = name.split("::")
    return "::".join(parts[1:]) if len(parts) > 1 and parts[0].endswith(".py") else name


def matches(pattern, test_id):
    target = test_id if "::" in pattern else test_id.split("::")[-1]
    return fnmatch.fnmatchcase(target, pattern) or fnmatch.fnmatchcase(target.split("[")[0], pattern)


def owner(rubric, test_id):
    if any(matches(p, test_id) for p in rubric.get("ignore", [])):
        return None
    owners = [m["name"] for m in rubric["milestones"] if any(matches(p, test_id) for p in m["tests"])]
    owners += [g["name"] for g in rubric.get("gates", []) if any(matches(p, test_id) for p in g["tests"])]
    if len(owners) != 1:
        raise ValueError("test %r is owned by %d rubric entries" % (test_id, len(owners)))
    return owners[0]


def grade(results, rubric):
    binary = float(bool(results) and all(s in PASSING for s in results.values()))
    naive = sum(s in PASSING for s in results.values()) / len(results) if results else 0.0

    by_owner = {}
    for test, status in results.items():
        name = owner(rubric, test)
        if name is not None:
            by_owner.setdefault(name, []).append(status in PASSING)

    raw = {}
    for m in rubric["milestones"]:
        outcomes = by_owner.get(m["name"], [])
        passed = sum(outcomes)
        if not outcomes:
            score = 0.0
        elif m.get("scoring", "fraction") == "all":
            score = float(passed == len(outcomes))
        else:
            score = passed / len(outcomes)
        raw[m["name"]] = (passed, len(outcomes), score)

    milestones = []
    for m in rubric["milestones"]:
        passed, total, score = raw[m["name"]]
        any_of = m.get("requires_any", [])
        locked = (any(raw[req][2] == 0 for req in m.get("requires", []))
                  or (bool(any_of) and all(raw[req][2] == 0 for req in any_of)))
        milestones.append({"name": m["name"], "weight": m["weight"], "passed": passed, "total": total,
                           "score": 0.0 if locked else score, "locked": locked})

    failed_gate = None
    for g in rubric.get("gates", []):
        outcomes = by_owner.get(g["name"], [])
        if not outcomes or not all(outcomes):
            failed_gate = g["name"]
            break

    total_weight = sum(m["weight"] for m in milestones)
    milestone = 0.0 if failed_gate else sum(m["weight"] * m["score"] for m in milestones) / total_weight
    return {"binary": binary, "naive_dense": naive, "milestone": milestone,
            "failed_gate": failed_gate, "milestones": milestones}


def load_results(ctrf_path):
    if not os.path.exists(ctrf_path):
        return {}
    with open(ctrf_path) as f:
        tests = json.load(f)["results"]["tests"]
    return {normalize_name(t["name"]): t["status"] for t in tests}


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--ctrf", default="/logs/verifier/ctrf.json")
    parser.add_argument("--rubric", default="/tests/rubric.json")
    parser.add_argument("--out-dir", default="/logs/verifier")
    parser.add_argument("--primary", choices=["milestone", "binary", "naive_dense"], default="milestone")
    args = parser.parse_args(argv)

    with open(args.rubric) as f:
        rubric = json.load(f)
    try:
        result = grade(load_results(args.ctrf), rubric)
    except Exception as exc:  # a broken report must still produce a reward
        print("reward-lens grader: %s; scoring 0" % exc, file=sys.stderr)
        result = {"binary": 0.0, "naive_dense": 0.0, "milestone": 0.0, "failed_gate": None,
                  "milestones": [], "error": str(exc)}

    rewards = {"reward": result[args.primary], "binary": result["binary"],
               "naive_dense": result["naive_dense"], "milestone": result["milestone"]}
    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "reward.json"), "w") as f:
        json.dump(rewards, f)
    with open(os.path.join(args.out_dir, "reward_breakdown.json"), "w") as f:
        json.dump(result, f, indent=2)
    print("reward-lens: " + ", ".join("%s=%.3f" % kv for kv in rewards.items()))


if __name__ == "__main__":
    main()
