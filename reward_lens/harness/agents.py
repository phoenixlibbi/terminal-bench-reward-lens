"""Harbor agents backed by the reward-lens terminal.

  harbor run -p TASK -a reward_lens.harness.agents:ReplayAgent --ak trajectory=/path/to/trial
  harbor run -p TASK -a reward_lens.harness.agents:LoopAgent -m ollama/llama3.1:8b
"""

import os
from pathlib import Path

from harbor.agents.base import BaseAgent
from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext

from reward_lens.harness.llm import ChatClient
from reward_lens.harness.loop import agent_loop, replay
from reward_lens.harness.terminal import Terminal, harbor_exec
from reward_lens.harness.trajectory import load_commands


class _TerminalAgent(BaseAgent):
    def __init__(self, logs_dir: Path, model_name: str | None = None, *args, width: int = 160,
                 height: int = 40, **kwargs):
        super().__init__(logs_dir, model_name, *args, **kwargs)
        self.width, self.height = int(width), int(height)
        self.terminal: Terminal | None = None

    def version(self) -> str:
        return "0.1.0"

    async def setup(self, environment: BaseEnvironment) -> None:
        self.terminal = Terminal(harbor_exec(environment), width=self.width, height=self.height)
        await self.terminal.start()


class ReplayAgent(_TerminalAgent):
    """Re-type a recorded trajectory in a fresh container. Zero model cost; measures replay fidelity."""

    def __init__(self, logs_dir: Path, model_name: str | None = None, *args, trajectory: str = "",
                 speedup: float = 1.0, max_wait: float = 120.0, **kwargs):
        super().__init__(logs_dir, model_name, *args, **kwargs)
        if not trajectory:
            raise ValueError("ReplayAgent needs --ak trajectory=<trial dir or trajectory.json>")
        self.commands = load_commands(trajectory)
        self.speedup, self.max_wait = float(speedup), float(max_wait)

    @staticmethod
    def name() -> str:
        return "reward-lens-replay"

    async def run(self, instruction: str, environment: BaseEnvironment, context: AgentContext) -> None:
        steps = await replay(self.terminal, self.commands, speedup=self.speedup, max_wait=self.max_wait,
                             log_path=self.logs_dir / "replay.jsonl")
        context.metadata = {"n_episodes": steps, "replayed_commands": steps}


class LoopAgent(_TerminalAgent):
    """Minimal Terminus-style agent: screen in, JSON commands out, over any OpenAI-compatible endpoint."""

    def __init__(self, logs_dir: Path, model_name: str | None = None, *args, api_base: str | None = None,
                 api_key_env: str | None = None, max_turns: int = 50, history_turns: int = 10,
                 temperature: float = 0.2, max_wait: float = 60.0, max_tokens: int = 2048,
                 timeout_sec: float = 300, **kwargs):
        super().__init__(logs_dir, model_name, *args, **kwargs)
        if not model_name:
            raise ValueError("LoopAgent needs a model, e.g. -m ollama/llama3.1:8b")
        self.client = ChatClient(model_name, api_base, os.environ.get(api_key_env) if api_key_env else None,
                                 float(temperature), float(timeout_sec), int(max_tokens))
        self.max_turns, self.history_turns, self.max_wait = int(max_turns), int(history_turns), float(max_wait)

    @staticmethod
    def name() -> str:
        return "reward-lens-loop"

    async def run(self, instruction: str, environment: BaseEnvironment, context: AgentContext) -> None:
        try:
            result = await agent_loop(self.terminal, instruction, self.client.complete, max_turns=self.max_turns,
                                      history_turns=self.history_turns, max_wait=self.max_wait,
                                      log_path=self.logs_dir / "loop.jsonl")
            context.metadata = {"n_episodes": result.turns, "completed": result.completed,
                                "parse_errors": result.parse_errors}
        finally:
            context.n_input_tokens = self.client.usage.prompt_tokens
            context.n_output_tokens = self.client.usage.completion_tokens
