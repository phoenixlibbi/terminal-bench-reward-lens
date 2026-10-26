"""A tmux terminal driven through a container's exec().

The agent sees exactly what a human would see in the pane, and types exactly what
it decides to type. Literal text goes through base64 -> tmux load-buffer ->
paste-buffer, so no keystroke string can break shell quoting; a keystroke string
that is a bare tmux key name ("C-c", "Escape", "Up", ...) is sent as that key.
"""

import asyncio
import base64
import re
import shlex
from collections.abc import Awaitable, Callable
from dataclasses import dataclass

SPECIAL_KEY = re.compile(r"^(C-[a-z\\\]\[@^_]|M-[a-z]|Enter|Escape|Tab|BSpace|Up|Down|Left|Right|Home|End"
                         r"|PageUp|PageDown|PPage|NPage|F([1-9]|1[0-2]))$")

INSTALL_TMUX = (
    "command -v tmux >/dev/null 2>&1 || "
    "(command -v apt-get >/dev/null && (apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq tmux)) || "
    "(command -v apk >/dev/null && apk add --no-cache tmux) || "
    "(command -v dnf >/dev/null && dnf install -y tmux) || "
    "(command -v yum >/dev/null && yum install -y tmux)"
)


@dataclass(frozen=True)
class ExecOutput:
    stdout: str
    return_code: int


ExecFn = Callable[[str], Awaitable[ExecOutput]]


CHUNK = 60_000  # stay well below Linux's 128 KiB limit for a single command-line argument


def send_commands(session: str, keystrokes: str) -> list[str]:
    """Shell commands that type `keystrokes` into the tmux pane (several for very long payloads)."""
    if SPECIAL_KEY.match(keystrokes):
        return [f"tmux send-keys -t {session} {keystrokes}"]
    payload = base64.b64encode(keystrokes.encode()).decode()
    b64, buf = f"/tmp/.{session}_keys.b64", f"/tmp/.{session}_keys"
    commands = [f": > {b64}"] + [f"printf %s {payload[i:i + CHUNK]} >> {b64}" for i in range(0, len(payload), CHUNK)]
    # paste-buffer turns "\n" into Enter, so "ls -la\n" runs the command
    commands.append(f"base64 -d {b64} > {buf} && tmux load-buffer -b {session} {buf} "
                    f"&& tmux paste-buffer -d -b {session} -t {session} && rm -f {buf} {b64}")
    return commands


def send_command(session: str, keystrokes: str) -> str:
    """Single shell command for short keystrokes (see send_commands for long ones)."""
    return " && ".join(send_commands(session, keystrokes))


def capture_command(session: str, history_lines: int = 0) -> str:
    start = f" -S -{history_lines}" if history_lines else ""
    return f"tmux capture-pane -p -J -t {session}{start}"


class Terminal:
    def __init__(self, exec_fn: ExecFn, session: str = "rl", width: int = 160, height: int = 40):
        self._exec = exec_fn
        self.session = session
        self.width, self.height = width, height

    async def start(self) -> None:
        installed = await self._exec(INSTALL_TMUX)
        if installed.return_code != 0:
            raise RuntimeError("tmux is not available and could not be installed in the container")
        await self._exec(f"tmux kill-session -t {self.session} 2>/dev/null; "
                         f"tmux new-session -d -s {self.session} -x {self.width} -y {self.height} "
                         f"&& tmux set-option -t {self.session} history-limit 50000")

    async def send(self, keystrokes: str) -> None:
        for command in send_commands(self.session, keystrokes) if keystrokes else []:
            result = await self._exec(command)
            if result.return_code != 0:
                raise RuntimeError(f"failed to send keys: {result.stdout[-500:]}")

    async def screen(self, history_lines: int = 0) -> str:
        return (await self._exec(capture_command(self.session, history_lines))).stdout.rstrip()

    async def wait(self, min_sec: float, max_sec: float, poll_sec: float = 1.0) -> None:
        """Sleep min_sec, then keep waiting (up to max_sec) while the screen is still changing."""
        await asyncio.sleep(min_sec)
        waited, previous = min_sec, await self.screen()
        while waited < max_sec:
            await asyncio.sleep(poll_sec)
            waited += poll_sec
            current = await self.screen()
            if current == previous:
                return
            previous = current


def harbor_exec(environment, user=None) -> ExecFn:
    """Adapt a Harbor BaseEnvironment to the ExecFn the terminal needs."""
    async def run(command: str) -> ExecOutput:
        # plain sh: works in images without bash (e.g. Alpine) and keeps login-profile output off the screen
        result = await environment.exec(command=f"sh -c {shlex.quote(command)}", user=user)
        return ExecOutput((result.stdout or "") + (result.stderr or ""), result.return_code)
    return run
