#!/usr/bin/env python3
"""Floor-test adversary: keep the format of a solution's outputs, destroy their values.

Used by `reward-lens floor`. Runs inside the task container after the reference
solution. Standard library only, Python 3.8+. Every .json/.jsonl/.csv/.tsv file
under --root modified after --marker is rewritten:
  zeros    every number becomes 0
  perturb  numeric lists and columns are shuffled; scalars are rescaled
String fields and ID-like columns are left alone, so schema and row checks still pass.
"""
import argparse
import csv
import io
import json
import os
import random
import re

DATA_EXTENSIONS = {".json", ".jsonl", ".csv", ".tsv"}
SKIP_DIRS = {"/proc", "/sys", "/dev", "/tests", "/logs", "/solution", "/usr", "/lib", "/bin", "/etc"}
ID_COLUMN = re.compile(r"(^|_)(id|index|key)$", re.IGNORECASE)


def is_number(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def scale(x):
    y = x * 1.37 + 0.11
    return int(round(y)) if isinstance(x, int) else y


def corrupt_json(value, mode, rng, key=""):
    if is_number(value):
        if ID_COLUMN.search(key):
            return value
        return type(value)(0) if mode == "zeros" else scale(value)
    if isinstance(value, dict):
        return {k: corrupt_json(v, mode, rng, k) for k, v in value.items()}
    if isinstance(value, list):
        if mode == "perturb" and len(value) > 1 and all(isinstance(v, dict) for v in value):
            return corrupt_records(value, rng)
        if mode == "perturb" and len(value) > 1 and all(is_number(v) for v in value) and len(set(value)) > 1:
            shuffled = list(value)
            while shuffled == value:
                rng.shuffle(shuffled)
            return shuffled
        return [corrupt_json(v, mode, rng, key) for v in value]
    return value


def corrupt_records(records, rng):
    """A list of JSON objects is a table: shuffle each numeric field across records.

    Rescaling alone leaves small integers such as 0/1 labels unchanged; shuffling keeps
    their distribution (so range and balance checks still pass) but breaks which record
    gets which value. Fields that never vary are rescaled instead.
    """
    keys = sorted(set().union(*(r.keys() for r in records)))
    out = [dict(r) for r in records]
    for key in keys:
        values = [r.get(key) for r in records]
        if ID_COLUMN.search(key) or not all(is_number(v) for v in values):
            for r in out:
                if key in r:
                    r[key] = corrupt_json(r[key], "perturb", rng, key)
            continue
        if len(set(values)) > 1:
            shuffled = list(values)
            while shuffled == values:
                rng.shuffle(shuffled)
        else:
            shuffled = [scale(v) for v in values]
        for r, v in zip(out, shuffled):
            r[key] = v
    return out


def _is_float(s):
    try:
        float(s)
        return True
    except ValueError:
        return False


def corrupt_table(text, delimiter, mode, rng):
    rows = list(csv.reader(io.StringIO(text), delimiter=delimiter))
    if len(rows) < 2:
        return text
    header, body = rows[0], rows[1:]
    for col, name in enumerate(header):
        values = [r[col] for r in body if col < len(r)]
        if ID_COLUMN.search(name) or not values or not all(v == "" or _is_float(v) for v in values):
            continue
        if mode == "zeros":
            new = ["0" if v != "" else v for v in values]
        elif len(set(values)) > 1:
            new = list(values)
            while new == values:
                rng.shuffle(new)
        else:
            new = [repr(scale(float(v))) if v != "" else v for v in values]
        for r, v in zip((r for r in body if col < len(r)), new):
            r[col] = v
    out = io.StringIO()
    csv.writer(out, delimiter=delimiter, lineterminator="\n").writerows([header] + body)
    return out.getvalue()


def corrupt_file(path, mode, rng):
    ext = os.path.splitext(path)[1].lower()
    with open(path) as f:
        text = f.read()
    if ext == ".json":
        new = json.dumps(corrupt_json(json.loads(text), mode, rng), indent=2)
    elif ext == ".jsonl":
        new = "\n".join(json.dumps(corrupt_json(json.loads(line), mode, rng))
                        for line in text.splitlines() if line.strip()) + "\n"
    else:
        new = corrupt_table(text, "\t" if ext == ".tsv" else ",", mode, rng)
    with open(path, "w") as f:
        f.write(new)


def find_outputs(roots, marker_mtime):
    for root in roots:
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [d for d in dirnames if os.path.join(dirpath, d) not in SKIP_DIRS
                           and not d.startswith(".") and d not in ("__pycache__", "node_modules", "site-packages")]
            for name in filenames:
                path = os.path.join(dirpath, name)
                if os.path.splitext(name)[1].lower() in DATA_EXTENSIONS and os.path.getmtime(path) > marker_mtime:
                    yield path


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["zeros", "perturb"], required=True)
    parser.add_argument("--marker", required=True, help="file touched before the reference solution ran")
    parser.add_argument("--root", action="append", default=None, help="where outputs live (default /app)")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    rng = random.Random(args.seed)
    marker_mtime = os.path.getmtime(args.marker)
    for path in sorted(find_outputs(args.root or ["/app"], marker_mtime)):
        try:
            corrupt_file(path, args.mode, rng)
            print("reward-lens floor: %s %s" % (args.mode, path))
        except Exception as exc:  # unparseable outputs are left as they are
            print("reward-lens floor: skipped %s (%s)" % (path, exc))


if __name__ == "__main__":
    main()
