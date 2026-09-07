"""Exercise recipe control flow with scripted interfaces, not verl conformance."""

import asyncio
import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest


@pytest.mark.parametrize("reward", [-1.0, 0.0, 2.0])
@pytest.mark.parametrize("tokens, truncated", [([1], False), ([1, 2], True)])
def test_truncation_does_not_improve_reward_or_execute_discarded_action(
    monkeypatch, tmp_path, reward, tokens, truncated
):
    stub = ModuleType("verl.experimental.agent_loop.agent_loop")
    stub.AgentLoopBase = object
    stub.AgentLoopOutput = SimpleNamespace
    stub.register = lambda _name: lambda cls: cls
    monkeypatch.setitem(sys.modules, stub.__name__, stub)
    path = Path(__file__).parents[1] / "recipes/verl_bridge/bridge_agent_loop.py"
    spec = importlib.util.spec_from_file_location("tested_bridge_loop", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    log = tmp_path / "episodes.jsonl"
    monkeypatch.setenv(module.EPISODES_PATH_ENV, str(log))
    loop = object.__new__(module.CreditBenchBridgeLoop)
    loop.response_length = 1
    loop.turn_separator = []

    async def template(*_args, **_kwargs):
        return [101]

    async def generate(**_kwargs):
        return SimpleNamespace(token_ids=tokens)

    loop.apply_chat_template = template
    loop.server_manager = SimpleNamespace(generate=generate)
    loop.tokenizer = SimpleNamespace(decode=lambda *_args, **_kwargs: "STOP")
    result = asyncio.run(loop.run({}, extra_info={
        "env": "variable_horizon",
        "env_params": {"stop_rewards": [reward, reward], "continue_reward": 0.0},
    }))
    assert result.reward_score == reward
    assert result.response_ids == [1]
    assert result.response_mask == [1]
    assert result.metrics["credit_bench_truncated"] == float(truncated)
    assert result.num_turns == (1 if truncated else 3)
    assert log.exists() is not truncated
