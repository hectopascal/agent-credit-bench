"""Tests for the verbalized-MDP bridge (integrations/bridge.py).

Framework-free: everything here runs in the zero-dependency environment.
"""

import pytest

from agent_credit_bench.envs import (
    DelayedEffectEnv,
    RecoveryEnv,
    VariableHorizonEnv,
)
from agent_credit_bench.integrations.bridge import (
    BridgeSession,
    EmpiricalTabularPolicy,
    episode_record,
    load_episodes,
    make_env,
    parse_action,
    play_episode,
    trajectory_from_record,
)
from agent_credit_bench.oracle import solve_exact_values
from agent_credit_bench.types import Step, Trajectory


def test_parse_action_last_mention_wins() -> None:
    actions = ["GOOD", "BAD"]
    text = "I could pick GOOD here, but I will pick BAD."
    assert parse_action(text, actions) == "BAD"


def test_parse_action_case_insensitive_and_word_bounded() -> None:
    assert parse_action("let's continue", ["STOP", "CONTINUE"]) == "CONTINUE"
    assert parse_action("STOPPING is not an action", ["STOP", "CONTINUE"]) is None
    assert parse_action("I GIVE_UP.", ["RECOVER", "GIVE_UP"]) == "GIVE_UP"
    assert parse_action("no action here", ["GOOD", "BAD"]) is None


def scripted(replies_by_turn: dict[int, str]):
    def respond(messages: list[dict[str, str]]) -> str:
        turn = sum(m["role"] == "user" for m in messages) - 1
        return replies_by_turn[turn]

    return respond


def test_play_episode_good_immediately_terminates() -> None:
    session = play_episode(RecoveryEnv(), scripted({0: "GOOD"}), seed=0)
    assert session.done
    assert session.total_return == 1.0
    steps = session.trajectory.steps
    assert len(steps) == 1
    assert steps[0].action == "GOOD"
    assert steps[0].terminated


def test_play_episode_bad_then_recover() -> None:
    session = play_episode(
        RecoveryEnv(), scripted({0: "Hmm. BAD", 1: "then RECOVER"}), seed=0
    )
    steps = session.trajectory.steps
    assert [s.action for s in steps] == ["BAD", "RECOVER"]
    assert session.total_return == 1.0
    assert all(turn.parsed for turn in session.turns)


def test_unparseable_reply_uses_deterministic_minimum_return_fallback() -> None:
    session = play_episode(
        RecoveryEnv(), scripted({0: "no idea!", 1: "still no idea!"}), seed=0
    )
    assert [step.action for step in session.trajectory.steps] == ["BAD", "GIVE_UP"]
    assert not any(turn.parsed for turn in session.turns)
    assert session.total_return == 0.0
    assert session.done


def test_minimum_return_fallback_looks_through_delayed_reward() -> None:
    mdp = DelayedEffectEnv(
        horizon=2,
        good_success_probability=1.0,
        bad_success_probability=0.0,
    )
    session = play_episode(
        mdp,
        scripted({0: "unparseable", 1: "DISTRACT_0"}),
        seed=0,
    )
    # Both initial actions pay zero immediately. Backward induction still
    # selects BAD because its expected terminal return is lower.
    assert session.trajectory.steps[0].action == "BAD"
    assert session.total_return == 0.0


def test_unparseable_reply_can_raise_instead_of_falling_back() -> None:
    session = BridgeSession(RecoveryEnv(), parse_failure_policy="raise")
    with pytest.raises(ValueError, match="did not name a legal action"):
        session.act("no action here")
    assert session.turns == ()


def test_horizon_caps_episode_length() -> None:
    mdp = VariableHorizonEnv()
    replies = {t: "CONTINUE" for t in range(mdp.horizon)}
    session = play_episode(mdp, scripted(replies), seed=0)
    assert len(session.trajectory.steps) == mdp.horizon


def test_same_seed_reproduces_stochastic_transitions() -> None:
    script = {0: "BAD", 1: "RECOVER"}
    mdp = RecoveryEnv(recover_success_probability=0.5)
    first = play_episode(mdp, scripted(script), seed=7)
    second = play_episode(mdp, scripted(script), seed=7)
    assert first.trajectory == second.trajectory


def test_episode_record_round_trip(tmp_path) -> None:
    session = play_episode(RecoveryEnv(), scripted({0: "BAD", 1: "RECOVER"}), seed=0)
    record = episode_record(session, env="recovery", extra={"checkpoint": 3})
    path = tmp_path / "episodes.jsonl"
    import json

    path.write_text(json.dumps(record) + "\n")
    loaded = load_episodes([path])
    assert loaded[0]["checkpoint"] == 3
    assert loaded[0]["parse_failure_policy"] == "minimum_return"
    assert loaded[0]["turns"][0]["parsed"] is True
    assert trajectory_from_record(loaded[0]) == session.trajectory


def test_episode_record_requires_finished_episode() -> None:
    session = BridgeSession(RecoveryEnv(), seed=0)
    with pytest.raises(ValueError):
        episode_record(session, env="recovery")


def test_empirical_policy_counts_and_fallback() -> None:
    sessions = [
        play_episode(RecoveryEnv(), scripted({0: "GOOD"}), seed=0),
        play_episode(RecoveryEnv(), scripted({0: "GOOD"}), seed=1),
        play_episode(RecoveryEnv(), scripted({0: "BAD", 1: "RECOVER"}), seed=2),
    ]
    policy = EmpiricalTabularPolicy.from_trajectories(
        [s.trajectory for s in sessions]
    )
    assert policy.sample_size == 3
    probs = policy.action_probabilities(0, "s0", ["GOOD", "BAD"])
    assert probs["GOOD"] == pytest.approx(2 / 3)
    assert probs["BAD"] == pytest.approx(1 / 3)
    assert policy.visit_count(0, "s0") == 3
    assert policy.visit_count(1, "mistake") == 1
    unseen = policy.action_probabilities(5, "nowhere", ["A", "B"])
    assert unseen == {"A": 0.5, "B": 0.5}


def test_empirical_policy_collapses_histories_at_same_markov_key() -> None:
    trajectories = [
        Trajectory(
            (
                Step(0, "root", "LEFT", 0.0, "merged", False),
                Step(1, "merged", "A", 0.0, "done", True),
            )
        ),
        Trajectory(
            (
                Step(0, "root", "RIGHT", 0.0, "merged", False),
                Step(1, "merged", "B", 0.0, "done", True),
            )
        ),
    ]
    policy = EmpiricalTabularPolicy.from_trajectories(trajectories)
    assert policy.sample_size == 2
    assert policy.action_probabilities(1, "merged", ["A", "B"]) == {
        "A": 0.5,
        "B": 0.5,
    }


def test_exact_values_solve_under_empirical_markov_projection() -> None:
    mdp = RecoveryEnv()
    sessions = [
        play_episode(mdp, scripted({0: "GOOD"}), seed=0),
        play_episode(mdp, scripted({0: "BAD", 1: "RECOVER"}), seed=1),
        play_episode(mdp, scripted({0: "BAD", 1: "GIVE_UP"}), seed=2),
    ]
    policy = EmpiricalTabularPolicy.from_trajectories(
        [s.trajectory for s in sessions]
    )
    exact = solve_exact_values(mdp, policy)
    # V(0, s0) under the fitted projection:
    # 1/3 * 1 + 2/3 * (1/2 * 1) = 2/3
    assert exact.state_values[(0, "s0")] == pytest.approx(2 / 3)


def test_make_env_rejects_unknown_name() -> None:
    with pytest.raises(ValueError):
        make_env("cartpole")
