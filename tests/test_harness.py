import asyncio
import base64
import json
import re

import pytest

from reward_lens.harness.llm import ChatClient, parse_reply
from reward_lens.harness.loop import agent_loop, replay
from reward_lens.harness.terminal import ExecOutput, Terminal, capture_command, send_command
from reward_lens.harness.trajectory import Command, load_commands
from reward_lens.run import parse_kwargs, replay_fidelity
from tests.conftest import write_trial


class FakeShell:
    """Records commands sent to tmux and plays back screens."""

    def __init__(self, screens=None):
        self.typed: list[str] = []
        self.commands: list[str] = []
        self.screens = list(screens or [])

    async def __call__(self, command: str) -> ExecOutput:
        self.commands.append(command)
        if "tmux send-keys" in command:
            self.typed.append(command.split()[-1])
        elif "base64 -d" in command:
            payload = re.search(r"printf %s (\S+)", command).group(1)
            self.typed.append(base64.b64decode(payload).decode())
        elif "capture-pane" in command:
            return ExecOutput(self.screens.pop(0) if len(self.screens) > 1 else (self.screens or [""])[0], 0)
        return ExecOutput("", 0)


def fast_terminal(shell: FakeShell) -> Terminal:
    terminal = Terminal(shell)

    async def no_wait(min_sec, max_sec, poll_sec=1.0):
        return None
    terminal.wait = no_wait
    return terminal


@pytest.mark.parametrize("keys", ["C-c", "Escape", "Up", "C-d", "Enter"])
def test_special_keys_use_send_keys(keys):
    assert send_command("rl", keys) == f"tmux send-keys -t rl {keys}"


@pytest.mark.parametrize("keys", ["ls -la\n", "echo 'quote\" $HOME `x`'\n", "C-c is not a key here\n", "cat << 'EOF'\nx\nEOF\n"])
def test_literal_text_round_trips_through_base64(keys):
    command = send_command("rl", keys)
    payload = re.search(r"printf %s (\S+)", command).group(1)
    assert base64.b64decode(payload).decode() == keys
    assert "paste-buffer" in command


def test_capture_command():
    assert capture_command("rl") == "tmux capture-pane -p -J -t rl"
    assert capture_command("rl", 500) == "tmux capture-pane -p -J -t rl -S -500"


def test_terminal_start_fails_without_tmux():
    async def no_tmux(command):
        return ExecOutput("not found", 127)
    with pytest.raises(RuntimeError, match="tmux"):
        asyncio.run(Terminal(no_tmux).start())


def test_load_commands_from_atif_trajectory(tmp_path):
    trial = tmp_path / "trial"
    (trial / "agent").mkdir(parents=True)
    (trial / "agent" / "trajectory.json").write_text(json.dumps({"steps": [
        {"source": "user", "message": "task"},
        {"source": "agent", "tool_calls": [{"function_name": "bash_command",
                                            "arguments": {"keystrokes": "ls\n", "duration": 0.5}}]},
        {"source": "agent", "tool_calls": None},
        {"source": "agent", "tool_calls": "[{'function_name': 'bash_command', 'arguments': {'keystrokes': 'C-c'}}]"},
        {"source": "agent", "tool_calls": [{"function_name": "mark_task_complete", "arguments": {}}]},
    ]}))
    assert load_commands(trial) == [Command("ls\n", 0.5), Command("C-c", 1.0)]


def test_replay_types_every_command_in_order(tmp_path):
    shell = FakeShell(["$ "])
    steps = asyncio.run(replay(fast_terminal(shell), [Command("cd /app\n"), Command("C-c"), Command("ls\n")],
                               log_path=tmp_path / "replay.jsonl"))
    assert steps == 3
    assert shell.typed == ["cd /app\n", "C-c", "ls\n"]
    assert len((tmp_path / "replay.jsonl").read_text().splitlines()) == 3


def test_parse_reply_variants():
    good = parse_reply('Sure!\n```json\n{"analysis": "a", "plan": "p", "commands": '
                       '[{"keystrokes": "ls\\n", "duration": 500}], "task_complete": false}\n```', max_duration=60)
    assert good.error is None and good.commands == [Command("ls\n", 60)]
    assert parse_reply("no json here").error
    assert parse_reply('{"commands": [{"duration": 1}]}').error
    assert parse_reply('{"commands": [], "task_complete": true}').task_complete


def test_agent_loop_needs_confirmation_and_recovers_from_bad_json(tmp_path):
    replies = iter([
        "not json",
        json.dumps({"analysis": "", "plan": "", "commands": [{"keystrokes": "make\n", "duration": 1}],
                    "task_complete": True}),
        json.dumps({"commands": [], "task_complete": True}),
    ])
    seen = []

    def complete(messages):
        seen.append(messages)
        return next(replies)

    shell = FakeShell(["$ "])
    result = asyncio.run(agent_loop(fast_terminal(shell), "build it", complete, max_turns=10,
                                    log_path=tmp_path / "loop.jsonl"))
    assert (result.turns, result.completed, result.parse_errors) == (3, True, 1)
    assert shell.typed == ["make\n"]
    assert "could not be used" in seen[1][-1]["content"]
    assert "marked the task complete" in seen[2][-1]["content"]
    assert seen[0][0]["role"] == "system" and "build it" in seen[0][1]["content"]


def test_agent_loop_stops_at_max_turns_and_bounds_context():
    calls = []

    def complete(messages):
        calls.append(len(messages))
        return json.dumps({"commands": [{"keystrokes": "echo hi\n", "duration": 0}]})

    result = asyncio.run(agent_loop(fast_terminal(FakeShell(["$ "])), "t", complete, max_turns=12, history_turns=3))
    assert (result.turns, result.completed) == (12, False)
    assert max(calls) == 2 + 2 * 3 - 1  # system + task + last exchanges ending with the new observation


def test_chat_client_provider_defaults(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "k")
    ollama = ChatClient("ollama/llama3.1:8b")
    assert (ollama.model, ollama.api_base, ollama.api_key) == ("llama3.1:8b", "http://localhost:11434/v1", None)
    openai = ChatClient("openai/gpt-4o-mini")
    assert (openai.model, openai.api_key) == ("gpt-4o-mini", "k")
    custom = ChatClient("my-model", api_base="http://vllm:8000/v1/")
    assert custom.api_base == "http://vllm:8000/v1"
    with pytest.raises(ValueError):
        ChatClient("mystery-model")


def test_replay_fidelity(tmp_path):
    original = write_trial(tmp_path / "orig", "t1", model="m", tests={"test_a": "passed", "test_b": "failed"})
    replayed = write_trial(tmp_path / "new", "t2", model="m", tests={"test_a": "passed", "test_b": "passed"})
    from reward_lens.harbor import find_trials
    f = replay_fidelity(original, find_trials(tmp_path / "new")[0])
    assert (f.agree, f.total, f.flipped) == (1, 2, {"test_b": ("failed", "passed")})


def test_parse_kwargs():
    assert parse_kwargs(["a=1", "b=x=y"]) == {"a": "1", "b": "x=y"}
    with pytest.raises(ValueError):
        parse_kwargs(["oops"])


def test_harbor_agents_import_and_validate(tmp_path):
    pytest.importorskip("harbor")
    from reward_lens.harness.agents import LoopAgent, ReplayAgent
    with pytest.raises(ValueError, match="trajectory"):
        ReplayAgent(tmp_path)
    with pytest.raises(ValueError, match="model"):
        LoopAgent(tmp_path)
    agent = LoopAgent(tmp_path, model_name="ollama/llama3.1:8b", max_turns="5")
    assert agent.max_turns == 5 and agent.name() == "reward-lens-loop"


def test_run_task_builds_harbor_command(tmp_path, monkeypatch):
    from reward_lens import run as run_module
    from tests.test_convert import RUBRIC, TB_TEST_SH, TESTS_PY
    task = tmp_path / "task"
    (task / "tests").mkdir(parents=True)
    (task / "tests" / "test.sh").write_text(TB_TEST_SH)
    (task / "tests" / "test_state.py").write_text(TESTS_PY)
    calls = []
    monkeypatch.setattr(run_module.subprocess, "run", lambda cmd, check: calls.append(cmd))
    jobs = run_module.run_task(task, RUBRIC, "claude-code", tmp_path / "work", model="anthropic/claude-sonnet-5-5",
                               agent_kwargs={"max_turns": "5"}, env_file=tmp_path / ".env")
    cmd = calls[0]
    assert cmd[cmd.index("-a") + 1] == "claude-code"
    assert cmd[cmd.index("-m") + 1] == "anthropic/claude-sonnet-5-5"
    assert cmd[cmd.index("--env-file") + 1] == str(tmp_path / ".env")
    assert cmd[cmd.index("--ak") + 1] == "max_turns=5"
    assert cmd[cmd.index("-p") + 1] == str((tmp_path / "work" / "task").resolve())
    assert jobs == tmp_path / "work" / "jobs"


def test_agent_loop_survives_failed_model_calls_and_gives_up_after_three():
    calls = iter([TimeoutError("slow"), json.dumps({"commands": [{"keystrokes": "ls\n", "duration": 0}]}),
                  ConnectionError("down"), ConnectionError("down"), ConnectionError("down")])

    def complete(messages):
        item = next(calls)
        if isinstance(item, Exception):
            raise item
        return item

    shell = FakeShell(["$ "])
    result = asyncio.run(agent_loop(fast_terminal(shell), "t", complete, max_turns=10))
    assert shell.typed == ["ls\n"]
    assert (result.turns, result.completed) == (5, False)


def test_chat_client_sends_max_tokens(monkeypatch):
    sent = {}

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self):
            return json.dumps({"choices": [{"message": {"content": "{}"}}],
                               "usage": {"prompt_tokens": 3, "completion_tokens": 1}}).encode()

    def fake_urlopen(request, timeout):
        sent.update(json.loads(request.data), timeout=timeout)
        return Response()

    import reward_lens.harness.llm as llm
    monkeypatch.setattr(llm.urllib.request, "urlopen", fake_urlopen)
    client = ChatClient("ollama/llama3.1:8b", max_tokens=256, timeout_sec=30)
    assert client.complete([{"role": "user", "content": "hi"}]) == "{}"
    assert (sent["max_tokens"], sent["timeout"], sent["model"]) == (256, 30, "llama3.1:8b")
    assert (client.usage.prompt_tokens, client.usage.completion_tokens) == (3, 1)
