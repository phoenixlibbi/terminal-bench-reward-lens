"""OpenAI-compatible chat client (stdlib only) and the agent's JSON reply format.

Works with any /v1/chat/completions endpoint: Ollama, vLLM, OpenAI, Together, OpenRouter, ...
"""

import json
import os
import re
import urllib.request
from dataclasses import dataclass, field

from reward_lens.harness.trajectory import Command

PROVIDERS = {  # provider prefix -> (default base URL, API key env var)
    "ollama": ("http://localhost:11434/v1", None),
    "openai": ("https://api.openai.com/v1", "OPENAI_API_KEY"),
    "together": ("https://api.together.xyz/v1", "TOGETHER_API_KEY"),
    "openrouter": ("https://openrouter.ai/api/v1", "OPENROUTER_API_KEY"),
}


@dataclass
class Usage:
    prompt_tokens: int = 0
    completion_tokens: int = 0


@dataclass
class Reply:
    analysis: str = ""
    plan: str = ""
    commands: list[Command] = field(default_factory=list)
    task_complete: bool = False
    error: str | None = None


class ChatClient:
    def __init__(self, model: str, api_base: str | None = None, api_key: str | None = None,
                 temperature: float = 0.2, timeout_sec: float = 300, max_tokens: int = 2048):
        provider, _, name = model.partition("/") if "/" in model else ("", "", model)
        default_base, key_env = PROVIDERS.get(provider, (None, None))
        self.model = name if provider in PROVIDERS else model
        self.api_base = (api_base or default_base or "").rstrip("/")
        if not self.api_base:
            raise ValueError(f"no api_base for model {model!r}; pass api_base=...")
        self.api_key = api_key or (os.environ.get(key_env) if key_env else None)
        self.temperature, self.timeout_sec, self.max_tokens = temperature, timeout_sec, max_tokens
        self.usage = Usage()

    def complete(self, messages: list[dict]) -> str:
        # max_tokens bounds generation: without it some servers let a model ramble until its context is full
        body = json.dumps({"model": self.model, "messages": messages, "temperature": self.temperature,
                           "max_tokens": self.max_tokens}).encode()
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        request = urllib.request.Request(f"{self.api_base}/chat/completions", body, headers)
        with urllib.request.urlopen(request, timeout=self.timeout_sec) as response:
            data = json.loads(response.read())
        usage = data.get("usage") or {}
        self.usage.prompt_tokens += usage.get("prompt_tokens", 0)
        self.usage.completion_tokens += usage.get("completion_tokens", 0)
        return data["choices"][0]["message"]["content"] or ""


def parse_reply(text: str, max_duration: float = 60.0) -> Reply:
    """Parse the agent's JSON reply, tolerating code fences and prose around the object."""
    match = re.search(r"\{.*\}", re.sub(r"```(?:json)?", "", text), re.DOTALL)
    if not match:
        return Reply(error="no JSON object found in your reply")
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError as exc:
        return Reply(error=f"invalid JSON: {exc}")
    if not isinstance(data, dict):
        return Reply(error="the reply must be a JSON object")
    commands = []
    for c in data.get("commands") or []:
        if not isinstance(c, dict) or not isinstance(c.get("keystrokes"), str):
            return Reply(error="each command needs a string 'keystrokes' field")
        try:
            duration = float(c.get("duration", 1.0))
        except (TypeError, ValueError):
            duration = 1.0
        commands.append(Command(c["keystrokes"], max(0.0, min(duration, max_duration))))
    return Reply(str(data.get("analysis", "")), str(data.get("plan", "")), commands,
                 bool(data.get("task_complete", False)))
