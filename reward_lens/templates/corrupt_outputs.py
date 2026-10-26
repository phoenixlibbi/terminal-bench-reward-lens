#!/usr/bin/env python3
"""Floor-test adversary: keep the format of a solution's outputs, destroy their values.

Used by `reward-lens floor`. Runs inside the task container: once with --snapshot before the
reference solution, then with --mode after it. Standard library only, Python 3.8+. Every
.json/.jsonl/.csv/.tsv file under --root that the solution created or changed (by content hash,
so copies with preserved timestamps are caught too) is rewritten:
  zeros    every number becomes 0
  perturb  numeric lists and columns are shuffled; scalars are rescaled (integers always change)
String fields and ID-like columns are left alone, so schema and row checks still pass.
"""
import argparse
import csv
import hashlib
import io
import json
import os
import random
import re

DATA_EXTENSIONS = {".json", ".jsonl", ".csv", ".tsv"}
SKIP_DIRS = {"/proc", "/sys", "/dev", "/tests", "/logs", "/solution", "/usr", "/lib", "/bin", "/etc"}
ID_COLUMN = re.compile(r"(^|[_\-\s.])(id|index|key|uuid)$", re.IGNORECASE)
CAMEL_ID = re.compile(r"[a-z0-9](Id|ID|Key|Index|Uuid|UUID)$")


def is_id(name):
    name = name.lstrip("\ufeff").strip()
    return bool(ID_COLUMN.search(name) or CAMEL_ID.search(name))


def is_number(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def scale(x):
    if isinstance(x, int):
        y = int(round(x * 1.37 + 0.11))
        return y if y != x else x + 1  # 0 and 1 would otherwise survive unchanged
    return x * 1.37 + 0.11


def corrupt_json(value, mode, rng, key=""):
    if is_number(value):
        if is_id(key):
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
        if is_id(key) or not all(is_number(v) for v in values):
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
        if is_id(name) or not values or not all(v == "" or _is_float(v) for v in values):
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
    with open(path, "rb") as f:
        bom = f.read(3) == b"\xef\xbb\xbf"
    encoding = "utf-8-sig" if bom else "utf-8"
    with open(path, encoding=encoding, newline="") as f:
        text = f.read()
    if ext == ".json":
        new = json.dumps(corrupt_json(json.loads(text), mode, rng), indent=2)
    elif ext == ".jsonl":
        new = "\n".join(json.dumps(corrupt_json(json.loads(line), mode, rng))
                        for line in text.splitlines() if line.strip()) + "\n"
    else:
        new = corrupt_table(text, "\t" if ext == ".tsv" else ",", mode, rng)
    with open(path, "w", encoding=encoding, newline="") as f:
        f.write(new)


def data_files(roots):
    for root in roots:
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [d for d in dirnames if os.path.join(dirpath, d) not in SKIP_DIRS
                           and not d.startswith(".") and d not in ("__pycache__", "node_modules", "site-packages")]
            for name in filenames:
                if os.path.splitext(name)[1].lower() in DATA_EXTENSIONS:
                    yield os.path.join(dirpath, name)


def digest(path):
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def snapshot(roots):
    return {path: digest(path) for path in data_files(roots)}


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot", help="record data-file hashes here (run before the reference solution)")
    parser.add_argument("--since", help="snapshot file: corrupt files created or changed since it was taken")
    parser.add_argument("--mode", choices=["zeros", "perturb"])
    parser.add_argument("--root", action="append", default=None, help="where outputs live (default /app)")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    roots = args.root or ["/app"]

    if args.snapshot:
        with open(args.snapshot, "w") as f:
            json.dump(snapshot(roots), f)
        return
    if not (args.mode and args.since):
        parser.error("use --snapshot FILE, or --mode MODE --since FILE")
    with open(args.since) as f:
        before = json.load(f)
    rng = random.Random(args.seed)
    changed = sorted(path for path, h in snapshot(roots).items() if before.get(path) != h)
    if not changed:
        print("reward-lens floor: no new or changed data files under %s" % ", ".join(roots))
    for path in changed:
        try:
            corrupt_file(path, args.mode, rng)
            print("reward-lens floor: %s %s" % (args.mode, path))
        except Exception as exc:  # unparseable outputs are left as they are
            print("reward-lens floor: skipped %s (%s)" % (path, exc))


if __name__ == "__main__":
    main()
