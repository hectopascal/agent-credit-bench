"""Suite environments as a verifiers MultiTurnEnv (verifiers 0.1.x).

The model plays a verbalized suite MDP as a multi-turn chat game, exactly as
in the verl bridge: the framework-free machinery (rendering, parsing,
transition sampling, fallback) all lives in
``agent_credit_bench.integrations.bridge`` — this file only adapts it to
verifiers' rollout loop, so the exactness argument from
``recipes/verl_bridge/README.md`` carries over unchanged.

Per rollout: ``setup_state`` opens a BridgeSession, ``env_response`` feeds
each assistant reply through ``session.act`` and returns the next observation
as the user message; when the episode ends it sets ``final_env_response``
(verifiers' env-side termination idiom) and appends the finished episode to
the JSONL log that ``recipes/verl_bridge/analyze_checkpoint.py`` consumes.
The rubric reward is the episode's total return.

Usage (also the Environments-Hub ``load_environment`` convention)::

    from credit_bench_env import load_environment
    env = load_environment(env="recovery", episodes_path="episodes/run1.jsonl")

Rollout seeds default to None (fresh entropy per episode): a fixed seed would
replay identical transition outcomes in every rollout of a group.
"""

import json
import os
import warnings
from pathlib import Path
from typing import Any

import verifiers as vf
from datasets import Dataset

from agent_credit_bench.integrations.bridge import (
    BridgeSession,
    episode_record,
    make_env,
    render_state,
    render_system_prompt,
)

EPISODES_PATH_ENV = "CREDIT_BENCH_EPISODES_PATH"


def _field(message: Any, key: str, default: Any = None) -> Any:
    """Messages arrive as dicts (dataset rows) or pydantic models (rollout)."""
    if isinstance(message, dict):
        return message.get(key, default)
    return getattr(message, key, default)


def _message_text(message: Any) -> str:
    content = _field(message, "content", "")
    if isinstance(content, str):
        return content
    if content is None:
        return ""
    return "".join(_field(part, "text", "") or "" for part in content)


def _last_assistant_text(messages: list[Any]) -> str:
    for message in reversed(messages):
        if _field(message, "role") == "assistant":
            return _message_text(message)
    raise ValueError("no assistant message to act on")


def episode_return(state: dict[str, Any]) -> float:
    """Rubric reward: the episode's total return (partial if truncated)."""
    return float(state["credit_bench_session"].total_return)


class CreditBenchEnv(vf.MultiTurnEnv):
    def __init__(
        self,
        env_name: str = "recovery",
        env_params: dict[str, Any] | None = None,
        seed: int | None = None,
        episodes_path: str | None = None,
        **kwargs: Any,
    ):
        self.env_name = env_name
        self.env_params = dict(env_params or {})
        self.seed = seed
        self.episodes_path = episodes_path or os.environ.get(EPISODES_PATH_ENV)
        horizon = make_env(env_name, **self.env_params).horizon
        # Env-side termination always fires within `horizon` assistant turns
        # (the parse fallback keeps every reply actionable); the +2 backstop
        # only catches rollouts that error out mid-episode.
        kwargs.setdefault("max_turns", horizon + 2)
        super().__init__(**kwargs)

    async def setup_state(self, state: dict[str, Any]) -> None:
        mdp = make_env(self.env_name, **self.env_params)
        state["credit_bench_session"] = BridgeSession(mdp, seed=self.seed)

    async def env_response(
        self, messages: list[dict[str, Any]], state: dict[str, Any], **kwargs: Any
    ) -> list[dict[str, Any]]:
        session = state["credit_bench_session"]
        session.act(_last_assistant_text(messages))
        if session.done:
            self._log_episode(session)
            closing = [
                vf.UserMessage(
                    content=f"Episode over. Total return: {session.total_return:g}."
                )
            ]
            state["final_env_response"] = closing
            return closing
        return [vf.UserMessage(content=session.observe())]

    def _log_episode(self, session: BridgeSession) -> None:
        if self.episodes_path is None:
            return
        record = episode_record(
            session,
            env=self.env_name,
            extra={"env_params": self.env_params},
        )
        path = Path(self.episodes_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a") as f:
            f.write(json.dumps(record) + "\n")


def _dataset(env_name: str, env_params: dict[str, Any], n: int) -> Dataset:
    mdp = make_env(env_name, **env_params)
    prompt = [
        {"role": "system", "content": render_system_prompt(mdp)},
        {
            "role": "user",
            "content": render_state(
                0, mdp.initial_state, list(mdp.actions(0, mdp.initial_state))
            ),
        },
    ]
    return Dataset.from_list(
        [
            {
                "prompt": prompt,
                "answer": "",
                "info": {"env": env_name, "env_params": env_params},
                "task": "credit-bench",
            }
            for _ in range(n)
        ]
    )


def load_environment(
    env: str = "recovery",
    env_params: dict[str, Any] | None = None,
    num_train_examples: int = 2048,
    num_eval_examples: int = 256,
    seed: int | None = None,
    episodes_path: str | None = None,
    **kwargs: Any,
) -> vf.Environment:
    """Environments-Hub entry point."""
    if seed is not None:
        warnings.warn(
            "a fixed seed replays identical transition outcomes in every "
            "rollout — use only for debugging, never for training",
            stacklevel=2,
        )
    params = dict(env_params or {})
    return CreditBenchEnv(
        env_name=env,
        env_params=params,
        seed=seed,
        episodes_path=episodes_path,
        dataset=_dataset(env, params, num_train_examples),
        eval_dataset=_dataset(env, params, num_eval_examples),
        rubric=vf.Rubric(funcs=[episode_return]),
        **kwargs,
    )
