"""verl agent loop that plays agent-credit-bench environments as chat games.

Wiring (verl 0.9.0, experimental agent-loop API):

1. Put this directory on PYTHONPATH.
2. Point the trainer at agent_loops.yaml:
       actor_rollout_ref.rollout.agent.agent_loop_config_path=.../agent_loops.yaml
3. Give dataset rows agent_name="credit_bench_bridge" (see make_dataset.py),
   with extra_info={"env": "recovery", "env_params": {...}}. Optionally set
   parse_failure_policy to "raise" instead of the default "minimum_return".
4. Export CREDIT_BENCH_EPISODES_PATH=/path/to/episodes.jsonl to log every
   finished episode for analyze_checkpoint.py.

Rewards reach the trainer through AgentLoopOutput.reward_score (the episode
return, placed on the last response token by verl), so no reward-manager
function is needed. Token bookkeeping mirrors verl's own tool_agent_loop:
one running token sequence, mask 1 for generated tokens, observation turns
rendered with remove_system_prompt=True plus the turn separator, mask 0.

Episodes that exhaust response_length mid-game are dropped from the JSONL
log (their return is not an episode return); size response_length so this
stays rare.
"""

import json
import logging
import os
from typing import Any
from uuid import uuid4

from verl.experimental.agent_loop.agent_loop import (
    AgentLoopBase,
    AgentLoopOutput,
    register,
)

from agent_credit_bench.integrations.bridge import (
    BridgeSession,
    episode_record,
    make_env,
    render_system_prompt,
)

logger = logging.getLogger(__file__)

EPISODES_PATH_ENV = "CREDIT_BENCH_EPISODES_PATH"


@register("credit_bench_bridge")
class CreditBenchBridgeLoop(AgentLoopBase):
    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.response_length = self.rollout_config.response_length

    async def run(
        self, sampling_params: dict[str, Any], **kwargs: Any
    ) -> AgentLoopOutput:
        info = dict(kwargs.get("extra_info") or {})
        env_name = info.get("env", "recovery")
        env_params = dict(info.get("env_params") or {})
        parse_failure_policy = info.get("parse_failure_policy", "minimum_return")
        # No seed during training: a per-prompt seed would correlate
        # transition outcomes inside a GRPO group. extra_info["seed"] exists
        # for reproducible offline evaluation only.
        seed = info.get("seed")
        session = BridgeSession(
            make_env(env_name, **env_params),
            seed=None if seed is None else int(seed),
            parse_failure_policy=parse_failure_policy,
        )

        messages = [
            {"role": "system", "content": render_system_prompt(session.mdp)},
            {"role": "user", "content": session.observe()},
        ]
        prompt_ids: list[int] = await self.apply_chat_template(messages)
        prompt_length = len(prompt_ids)
        response_mask: list[int] = []
        truncated = False

        while not session.done:
            output = await self.server_manager.generate(
                request_id=uuid4().hex,
                prompt_ids=prompt_ids,
                sampling_params=sampling_params,
            )
            generated_ids = list(output.token_ids)
            prompt_ids = prompt_ids + generated_ids
            response_mask = response_mask + [1] * len(generated_ids)
            # Do not execute an action from tokens the trainer will slice away.
            # In particular, a terminal action that crosses response_length
            # must not keep its reward while disappearing from response_ids.
            if len(response_mask) > self.response_length:
                truncated = True
                break
            reply = self.tokenizer.decode(output.token_ids, skip_special_tokens=True)
            session.act(reply)
            if session.done:
                break
            if len(response_mask) >= self.response_length:
                truncated = True
                break
            observation = {"role": "user", "content": session.observe()}
            observation_ids = await self.apply_chat_template(
                [observation], remove_system_prompt=True
            )
            observation_ids = self.turn_separator + observation_ids
            prompt_ids = prompt_ids + observation_ids
            response_mask = response_mask + [0] * len(observation_ids)
            if len(response_mask) >= self.response_length:
                truncated = True
                break

        self._log_episode(session, env_name, env_params, truncated)

        response_ids = prompt_ids[prompt_length:]
        return AgentLoopOutput(
            prompt_ids=prompt_ids[:prompt_length],
            response_ids=response_ids[: self.response_length],
            response_mask=response_mask[: self.response_length],
            reward_score=0.0 if truncated else session.total_return,
            num_turns=2 * len(session.turns) + 1,
            metrics={},
        )

    def _log_episode(
        self,
        session: BridgeSession,
        env_name: str,
        env_params: dict[str, Any],
        truncated: bool,
    ) -> None:
        path = os.getenv(EPISODES_PATH_ENV)
        if not path:
            return
        if truncated or not session.done:
            logger.warning("dropping truncated episode (raise response_length)")
            return
        record = episode_record(
            session, env=env_name, extra={"env_params": env_params}
        )
        # Line-buffered appends from concurrent actors: fine at these line
        # sizes on POSIX; point each run at its own file regardless.
        with open(path, "a") as handle:
            handle.write(json.dumps(record) + "\n")
