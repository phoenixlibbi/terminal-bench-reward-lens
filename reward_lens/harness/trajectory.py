"""Extract the keystrokes an agent typed from a recorded trajectory (Harbor ATIF / Terminus format)."""

import ast
import json
from dataclasses import dataclass
from pathlib import Path

KEYSTROKE_TOOLS = {"bash_command", "send_keys", "keystrokes"}


@dataclass(frozen=True)
class Command:
    keystrokes: str
    duration: float = 1.0


def load_commands(path: str | Path) -> list[Command]:
    """Commands in the order the agent issued them. Accepts a trial dir or a trajectory.json path."""
    path = Path(path)
    if path.is_dir():
        path = path / "agent" / "trajectory.json"
    data = json.loads(path.read_text())
    steps = data.get("steps", []) if isinstance(data, dict) else data
    commands = []
    for step in steps:
        if step.get("source", "agent") != "agent":
            continue
        calls = step.get("tool_calls") or []
        if isinstance(calls, str):
            calls = ast.literal_eval(calls)
        for call in calls:
            name = call.get("function_name") or call.get("name")
            args = call.get("arguments") or {}
            if isinstance(args, str):
                args = json.loads(args)
            if name in KEYSTROKE_TOOLS or "keystrokes" in args:
                commands.append(Command(args.get("keystrokes", ""), float(args.get("duration", 1.0))))
    return commands
