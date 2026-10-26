#!/usr/bin/env python3
"""Dense milestone grader, generated into a task's tests/ by `reward-lens convert`.

Runs inside the task container after the task's original verifier. Standard library only, Python 3.8+.
Reads per-test results (JUnit XML written via PYTEST_ADDOPTS, or a CTRF report) and tests/rubric.json, then writes:
  /logs/verifier/reward.json            {"reward": <primary>, "binary": .., "naive_dense": .., "milestone": ..}
  /logs/verifier/reward_breakdown.json  per-milestone scores, the failed gate, any grading error
It always writes a reward, even when the verifier crashed and produced no results.
Must compute exactly what reward_lens.ctrf / rubric / rewards compute (tests/test_convert.py checks parity).
"""
import argparse
import fnmatch
import json
import os
import sys
import xml.etree.ElementTree as ET

PASSING = {"passed", "skipped"}
RESULT_FILES = ("reward_lens_junit.xml", "ctrf.json")
SEVERITY = {"error": 3, "failed": 2, "other": 1}


def split_outside_brackets(name):
    parts, depth, start, i = [], 0, 0, 0
    while i < len(name):
        if name[i] == "[":
            depth += 1
        elif name[i] == "]":
            depth = max(depth - 1, 0)
        elif depth == 0 and name.startswith("::", i):
            parts.append(name[start:i])
            start = i + 2
            i += 1
        i += 1
    return parts + [name[start:]]


def normalize_name(name):
    parts = split_outside_brackets(name)
    return "::".join(parts[1:]) if len(parts) > 1 and parts[0].endswith(".py") else name


def unique_ids(entries):
    files = {}
    for file, test_id, _ in entries:
        files.setdefault(test_id, set()).add(file)
    results = {}
    for file, test_id, status in entries:
        key = "%s::%s" % (file, test_id) if len(files[test_id]) > 1 and file else test_id
        if key not in results or SEVERITY.get(status, 0) > SEVERITY.get(results[key], 0):
            results[key] = status
    return results


def load_ctrf(path):
    with open(path) as f:
        tests = json.load(f)["results"]["tests"]
    entries = []
    for t in tests:
        parts = split_outside_brackets(t["name"])
        file = parts[0] if len(parts) > 1 and parts[0].endswith(".py") else ""
        entries.append((file, "::".join(parts[1:] if file else parts), t["status"]))
    return unique_ids(entries)


def load_junit(path):
    entries = []
    for case in ET.parse(path).getroot().iter("testcase"):
        segments = (case.get("classname") or "").split(".")
        cls = segments.pop() if segments and segments[-1][:1].isupper() else ""
        file = "/".join(s for s in segments if s) + ".py" if any(segments) else ""
        tags = {child.tag for child in case}
        status = ("error" if "error" in tags else "failed" if "failure" in tags
                  else "skipped" if "skipped" in tags else "passed")
        entries.append((file, "%s::%s" % (cls, case.get("name")) if cls else case.get("name", ""), status))
    return unique_ids(entries)


def load_results(results_dir):
    for name in RESULT_FILES:
        path = os.path.join(results_dir, name)
        if os.path.exists(path):
            return load_junit(path) if name.endswith(".xml") else load_ctrf(path)
    return {}


def matches(pattern, test_id):
    if "::" in pattern:
        targets = {test_id, normalize_name(test_id)}
    else:
        targets = {split_outside_brackets(test_id)[-1]}
    targets |= {t.split("[", 1)[0] for t in targets}
    return any(fnmatch.fnmatchcase(t, pattern) for t in targets)


def owner(rubric, test_id):
    if any(matches(p, test_id) for p in rubric.get("ignore", [])):
        return None
    owners = [m["name"] for m in rubric["milestones"] if any(matches(p, test_id) for p in m["tests"])]
    owners += [g["name"] for g in rubric.get("gates", []) if any(matches(p, test_id) for p in g["tests"])]
    if len(owners) != 1:
        raise ValueError("test %r is owned by %d rubric entries" % (test_id, len(owners)))
    return owners[0]


def milestone_score(results, rubric):
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

    for g in rubric.get("gates", []):
        outcomes = by_owner.get(g["name"], [])
        if not outcomes or not all(outcomes):
            return 0.0, milestones, g["name"]
    total_weight = sum(m["weight"] for m in milestones)
    return sum(m["weight"] * m["score"] for m in milestones) / total_weight, milestones, None


def grade(results, rubric):
    # binary and naive dense need no rubric, so a rubric problem can only affect the milestone reward
    binary = float(bool(results) and all(s in PASSING for s in results.values()))
    naive = sum(s in PASSING for s in results.values()) / len(results) if results else 0.0
    out = {"binary": binary, "naive_dense": naive, "milestone": 0.0, "failed_gate": None, "milestones": []}
    try:
        out["milestone"], out["milestones"], out["failed_gate"] = milestone_score(results, rubric)
    except Exception as exc:
        out["error"] = "milestone reward set to 0: %s" % exc
    return out


def read_float(path):
    try:
        with open(path) as f:
            return float(f.read().strip())
    except (OSError, ValueError):
        return None


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="/logs/verifier")
    parser.add_argument("--rubric", default="/tests/rubric.json")
    parser.add_argument("--out-dir", default="/logs/verifier")
    parser.add_argument("--primary", choices=["milestone", "binary", "naive_dense"], default="milestone")
    parser.add_argument("--original-reward", help="file holding the original verifier's own reward")
    args = parser.parse_args(argv)

    with open(args.rubric) as f:
        rubric = json.load(f)
    try:
        results = load_results(args.results_dir)
        result = grade(results, rubric)
    except Exception as exc:  # an unreadable report must still produce a reward
        result = {"binary": 0.0, "naive_dense": 0.0, "milestone": 0.0, "failed_gate": None, "milestones": [],
                  "error": "could not read test results: %s" % exc}
    if result.get("error"):
        print("reward-lens grader: %s" % result["error"], file=sys.stderr)

    rewards = {"reward": result[args.primary], "binary": result["binary"],
               "naive_dense": result["naive_dense"], "milestone": result["milestone"]}
    original = read_float(args.original_reward) if args.original_reward else None
    if original is not None:
        rewards["original_verifier"] = original
    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "reward.json"), "w") as f:
        json.dump(rewards, f)
    with open(os.path.join(args.out_dir, "reward_breakdown.json"), "w") as f:
        json.dump(result, f, indent=2)
    print("reward-lens: " + ", ".join("%s=%.3f" % kv for kv in rewards.items()))


if __name__ == "__main__":
    main()
