"""Conformance tests for the verifiers recipe (recipes/verifiers_bridge).

The whole module skips when verifiers is not installed. The rollout tests
drive verifiers' real MultiTurnEnv rollout loop end-to-end with a scripted
client — no network, no model — so they exercise setup_state, env_response,
env-side termination, episode logging, and rubric scoring exactly as a
training run would.
"""

import asyncio
import importlib.util
import json
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("verifiers") is None, reason="verifiers is not installed"
)

RECIPE_DIR = Path(__file__).resolve().parent.parent / "recipes" / "verifiers_bridge"


@pytest.fixture(scope="module")
def recipe():
    sys.path.insert(0, str(RECIPE_DIR))
    try:
        import credit_bench_env

        yield credit_bench_env
    finally:
        sys.path.remove(str(RECIPE_DIR))


def scripted_client(replies):
    """A vf.Client that plays back canned replies, one per model call."""
    import verifiers as vf
    from verifiers.types import Response, ResponseMessage

    queue = list(replies)

    class ScriptedClient(vf.Client):
        def setup_client(self, config):
            return None

        async def to_native_tool(self, tool):
            return tool

        async def to_native_prompt(self, messages):
            return messages, {}

        async def get_native_response(
            self, prompt, model, sampling_args, tools=None, **kwargs
        ):
            if not queue:
                raise AssertionError("scripted client ran out of replies")
            return Response(
                id="scripted",
                created=0,
                model=model,
                message=ResponseMessage(
                    content=queue.pop(0), finish_reason="stop", is_truncated=False
                ),
            )

        async def raise_from_native_response(self, response):
            return None

        async def from_native_response(self, response):
            return response

        async def close(self):
            return None

    return ScriptedClient(None)


def run_rollout(env, replies):
    prompt = env.dataset[0]["prompt"]
    return asyncio.run(
        env.rollout(
            input={"prompt": prompt, "example_id": 0, "answer": "", "info": {}},
            client=scripted_client(replies),
            model="scripted",
        )
    )


def test_load_environment_dataset_and_rubric(recipe) -> None:
    env = recipe.load_environment(
        env="recovery", num_train_examples=4, num_eval_examples=2
    )
    assert len(env.dataset) == 4
    assert len(env.eval_dataset) == 2
    prompt = env.dataset[0]["prompt"]
    assert prompt[0]["role"] == "system"
    assert prompt[1]["role"] == "user"
    assert "GOOD" in prompt[1]["content"] and "BAD" in prompt[1]["content"]
    # MultiTurnEnv appends its monitor rubric, so ours lives inside a RubricGroup
    rubrics = getattr(env.rubric, "rubrics", [env.rubric])
    assert any(recipe.episode_return in rubric.funcs for rubric in rubrics)
    assert env.max_turns == 4  # horizon 2 + backstop 2


def test_rollout_good_terminates_in_one_turn(recipe) -> None:
    env = recipe.load_environment(
        env="recovery", num_train_examples=1, num_eval_examples=1
    )
    state = run_rollout(env, ["I pick GOOD."])
    # score_rollout writes into the state rather than returning
    asyncio.run(env.rubric.score_rollout(state))
    assert state["reward"] == pytest.approx(1.0)
    assert state["metrics"]["episode_return"] == pytest.approx(1.0)


def test_rollout_bad_recover_full_episode(recipe, tmp_path) -> None:
    episodes = tmp_path / "episodes.jsonl"
    env = recipe.load_environment(
        env="recovery",
        num_train_examples=1,
        num_eval_examples=1,
        episodes_path=str(episodes),
    )
    state = run_rollout(env, ["BAD, deliberately.", "now RECOVER"])
    assert state["is_completed"]
    assert state["stop_condition"] == "has_final_env_response"
    session = state["credit_bench_session"]
    assert [t.step.action for t in session.turns] == ["BAD", "RECOVER"]
    assert session.total_return == pytest.approx(1.0)
    assert recipe.episode_return(state) == pytest.approx(1.0)
    # the second user turn must be the rendered mistake-state observation
    assert len(state["trajectory"]) == 2

    record = json.loads(episodes.read_text().strip())
    assert record["env"] == "recovery"
    assert record["total_return"] == pytest.approx(1.0)
    assert [turn["action"] for turn in record["turns"]] == ["BAD", "RECOVER"]
    assert all(turn["parsed"] for turn in record["turns"])


def test_rollout_unparseable_reply_uses_minimum_return_fallback(recipe) -> None:
    env = recipe.load_environment(
        env="recovery", num_train_examples=1, num_eval_examples=1
    )
    state = run_rollout(env, ["hmm, let me think...", "still undecided"])
    session = state["credit_bench_session"]
    assert [t.step.action for t in session.turns] == ["BAD", "GIVE_UP"]
    assert not any(turn.parsed for turn in session.turns)
    assert session.total_return == 0.0
    assert state["is_completed"]


def test_episodes_log_round_trips_to_suite_trajectories(recipe, tmp_path) -> None:
    from agent_credit_bench.integrations.bridge import (
        load_episodes,
        trajectory_from_record,
    )

    episodes = tmp_path / "episodes.jsonl"
    env = recipe.load_environment(
        env="recovery",
        num_train_examples=1,
        num_eval_examples=1,
        episodes_path=str(episodes),
    )
    run_rollout(env, ["BAD please", "RECOVER"])
    run_rollout(env, ["GOOD"])
    records = load_episodes([episodes])
    assert len(records) == 2
    trajectories = [trajectory_from_record(r) for r in records]
    assert [len(t.steps) for t in trajectories] == [2, 1]
    assert all(t.total_return == pytest.approx(1.0) for t in trajectories)


def test_fixed_seed_warns(recipe) -> None:
    with pytest.warns(UserWarning, match="fixed seed"):
        recipe.load_environment(
            env="recovery", num_train_examples=1, num_eval_examples=1, seed=0
        )
