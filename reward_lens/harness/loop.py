"""The two agent loops, independent of Harbor so they can be tested with a fake terminal."""

import asyncio
import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from reward_lens.harness.llm import parse_reply
from reward_lens.harness.terminal import Terminal
from reward_lens.harness.trajectory import Command

SYSTEM_PROMPT = """You are an autonomous agent solving a task inside a Linux container through a tmux terminal.
Each turn you see the current terminal screen. Reply with ONLY one JSON object:
{"analysis": "what the screen shows and what is still missing",
 "plan": "your next steps",
 "commands": [{"keystrokes": "ls -la /app\\n", "duration": 1.0}],
 "task_complete": false}
Rules:
- keystrokes are typed verbatim into the terminal; end a shell command with \\n to run it.
- A keystrokes value that is exactly a tmux key name (C-c, C-d, Escape, Up, ...) presses that key.
- duration is how many seconds to wait after typing (0.1 to 60). Long jobs: wait, then check again.
- Prefer non-interactive commands. Write files with heredocs (cat > file << 'EOF').
- If the screen ends with a ">" continuation prompt, your last command was incomplete (unclosed quote or
  heredoc): send "C-c" first, then retry with a complete command.
- Verify your work against the instruction before finishing. Set task_complete to true only when done;
  you will be asked to confirm once."""


@dataclass
class LoopResult:
    turns: int
    completed: bool
    parse_errors: int


def _log(path: Path | None, record: dict) -> None:
    if path is not None:
        with path.open("a") as f:
            f.write(json.dumps(record) + "\n")


async def replay(terminal: Terminal, commands: list[Command], *, speedup: float = 1.0,
                 max_wait: float = 120.0, log_path: Path | None = None) -> int:
    """Type recorded commands in order; after each, wait its duration (and until the screen settles)."""
    for i, command in enumerate(commands):
        await terminal.send(command.keystrokes)
        wait = command.duration / speedup
        await terminal.wait(min_sec=wait, max_sec=max(wait, max_wait))
        _log(log_path, {"step": i, "keystrokes": command.keystrokes, "duration": command.duration,
                        "screen": await terminal.screen()})
    return len(commands)


async def agent_loop(terminal: Terminal, instruction: str, complete: Callable[[list[dict]], str], *,
                     max_turns: int = 50, history_turns: int = 10, max_wait: float = 60.0,
                     max_failed_calls: int = 3, log_path: Path | None = None) -> LoopResult:
    """Observe -> ask the model -> type its commands, until it confirms completion or runs out of turns.

    Context is a sliding window: the system prompt, the task, the last `history_turns` exchanges and the
    current screen.
    """
    task_message = {"role": "user", "content": f"Task:\n{instruction}\n\nThe terminal is ready."}
    history: list[dict] = []
    feedback, awaiting_confirmation, errors, failed_calls = "", False, 0, 0
    for turn in range(1, max_turns + 1):
        screen = await terminal.screen()
        observation = f"{feedback}Current terminal screen:\n{screen or '(empty)'}"
        history.append({"role": "user", "content": observation})
        messages = [{"role": "system", "content": SYSTEM_PROMPT}, task_message, *history[-(2 * history_turns + 1):]]
        try:
            text = await asyncio.to_thread(complete, messages)
        except Exception as exc:  # timeouts, connection errors, server errors: retry, then give up
            failed_calls += 1
            history.pop()
            _log(log_path, {"turn": turn, "screen": screen, "reply": None, "error": f"model call failed: {exc}"})
            if failed_calls >= max_failed_calls:
                return LoopResult(turn, False, errors)
            continue
        failed_calls = 0
        history.append({"role": "assistant", "content": text})
        reply = parse_reply(text, max_duration=max_wait)
        _log(log_path, {"turn": turn, "screen": screen, "reply": text, "error": reply.error})

        if reply.error:
            errors += 1
            feedback = f"Your previous reply could not be used ({reply.error}). Reply with one JSON object.\n\n"
            continue
        for command in reply.commands:
            await terminal.send(command.keystrokes)
            await terminal.wait(min_sec=command.duration, max_sec=max(command.duration, max_wait))
        if reply.task_complete and awaiting_confirmation:
            return LoopResult(turn, True, errors)
        awaiting_confirmation = reply.task_complete
        feedback = ("You marked the task complete. Check the screen below; reply with task_complete true again "
                    "to finish, or keep working.\n\n") if reply.task_complete else ""
    return LoopResult(max_turns, False, errors)
