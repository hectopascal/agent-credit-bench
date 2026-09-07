"""Tests for the verbalized-MDP bridge (integrations/bridge.py).

Framework-free: everything here runs in the zero-dependency environment.
"""

import math

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
    minimum_episode_return,
    parse_action,
    play_episode,
    trajectory_from_record,
)
from agent_credit_bench.oracle import solve_exact_values
from agent_credit_bench.types import Step, Trajectory, Transition
from helpers import TableMDP


def test_parse_action_last_mention_wins() -> None:
    actions = ["GOOD", "BAD"]
    text = "I could pick GOOD here, but I will pick BAD."
    assert parse_action(text, actions) == "BAD"


def test_parse_action_case_insensitive_and_word_bounded() -> None:
    assert parse_action("let's continue", ["STOP", "CONTINUE"]) == "CONTINUE"
    assert parse_action("STOPPING is not an action", ["STOP", "CONTINUE"]) is None
    assert parse_action("I GIVE_UP.", ["RECOVER", "GIVE_UP"]) == "GIVE_UP"
    assert parse_action("no action here", ["GOOD", "BAD"]) is None


@pytest.mark.parametrize("action", ["GO NOW", "C++", "[GO]"])
def test_parser_accepts_exact_and_embedded_punctuation_and_overlapping_actions(action):
    actions = ["GO", "NOW", "C", action, "STOP"]
    assert parse_action(action, actions) == action
    assert parse_action(f"I considered STOP. I choose {action}.", actions) == action
    assert parse_action(f"{action}, then STOP", actions) == "STOP"


def test_parser_preserves_exact_case_when_legal_names_differ_only_by_case():
    assert parse_action("go", ["GO", "go"]) == "go"


def test_record_total_is_stable_and_accepts_legacy_python_sums():
    mdp = VariableHorizonEnv(stop_rewards=(0.1,) * 10, continue_reward=0.1)
    session = play_episode(mdp, lambda _: "CONTINUE", seed=0)
    record = episode_record(session, "variable_horizon")
    assert record["total_return"] == session.trajectory.total_return == 1.0
    for total in (1.0, 0.9999999999999999):
        record["total_return"] = total
        assert trajectory_from_record(record, mdp=mdp) == session.trajectory
    record["total_return"] = 1.000000001
    with pytest.raises(ValueError, match="total_return"):
        trajectory_from_record(record, mdp=mdp)


def test_minimum_episode_return_accounts_for_dense_rewards_and_worst_outcome():
    mdp = TableMDP(2, "s", {
        (0, "s", "STOP"): (Transition("done", -1.0, 1.0, True),),
        (0, "s", "GO"): (
            Transition("next", -2.0, 0.5, False),
            Transition("done", 10.0, 0.5, True),
            Transition("missing", -1000.0, 0.0, False),
        ),
        (1, "next", "END"): (Transition("done", -3.0, 1.0, True),),
    })
    assert minimum_episode_return(mdp) == -5.0


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


@pytest.mark.parametrize(
    "name, mdp, replies",
    [
        ("recovery", RecoveryEnv(recover_success_probability=0.0), {0: "GOOD"}),
        (
            "delayed_effect",
            DelayedEffectEnv(horizon=2, good_success_probability=0.1),
            {0: "GOOD", 1: "DISTRACT_0"},
        ),
        (
            "variable_horizon",
            VariableHorizonEnv(stop_rewards=(1.0, 4.0), continue_reward=0.3),
            {0: "STOP"},
        ),
    ],
)
def test_episode_record_preserves_environment_parameters(name, mdp, replies):
    import json

    from agent_credit_bench.policy import UniformPolicy

    session = play_episode(mdp, scripted(replies), seed=0)
    record = json.loads(json.dumps(episode_record(session, name)))
    restored = make_env(record["env"], **record["env_params"])
    assert solve_exact_values(restored, UniformPolicy()) == solve_exact_values(
        mdp, UniformPolicy()
    )


def test_episode_record_accepts_matching_legacy_parameters():
    session = play_episode(RecoveryEnv(), scripted({0: "GOOD"}), seed=0)
    record = episode_record(session, "recovery", extra={"env_params": {}})
    assert record["env_params"] == {"recover_success_probability": 1.0, "horizon": 2}


def test_episode_record_rejects_mislabelled_environment_and_parameters():
    session = play_episode(RecoveryEnv(), scripted({0: "GOOD"}), seed=0)
    with pytest.raises(ValueError, match="environment"):
        episode_record(session, "variable_horizon")
    with pytest.raises(ValueError, match="env_params"):
        episode_record(
            session,
            "recovery",
            extra={"env_params": {"recover_success_probability": 0}},
        )


def test_episode_record_rejects_metadata_overwriting_environment():
    session = play_episode(RecoveryEnv(), scripted({0: "GOOD"}), seed=0)
    with pytest.raises(ValueError, match="cannot replace recorded fields"):
        episode_record(session, "recovery", extra={"env": "variable_horizon"})


def test_custom_environment_requires_explicit_parameters():
    from helpers import bandit_case

    mdp, _ = bandit_case()
    session = play_episode(mdp, scripted({0: "A"}), seed=0)
    with pytest.raises(ValueError, match="custom environments require explicit"):
        episode_record(session, "custom_bandit")
    record = episode_record(
        session, "custom_bandit", extra={"env_params": {"reward_a": 1.0}}
    )
    assert record["env_params"] == {"reward_a": 1.0}


@pytest.mark.parametrize("total", [None, -100.0, math.nan, math.inf])
def test_validated_record_rejects_inconsistent_total(total):
    mdp = RecoveryEnv()
    session = play_episode(mdp, scripted({0: "GOOD"}), seed=0)
    record = episode_record(session, "recovery")
    record["total_return"] = total
    with pytest.raises(ValueError, match="total_return"):
        trajectory_from_record(record, mdp=mdp)


def test_validated_record_rejects_zero_probability_outcomes():
    mdp = RecoveryEnv(recover_success_probability=0)
    session = play_episode(mdp, scripted({0: "BAD", 1: "RECOVER"}), seed=0)
    record = episode_record(session, "recovery")
    record["turns"][1].update(next_state="success", reward=1.0)
    record["total_return"] = 1.0
    with pytest.raises(ValueError, match="turn 1:.*positive-probability"):
        trajectory_from_record(record, mdp=mdp)


def test_validated_record_rejects_incomplete_and_disconnected_episodes():
    mdp = RecoveryEnv()
    session = play_episode(mdp, scripted({0: "BAD", 1: "RECOVER"}), seed=0)
    record = episode_record(session, "recovery")
    record["turns"][1]["state"] = "s0"
    with pytest.raises(ValueError, match="turn 1: state"):
        trajectory_from_record(record, mdp=mdp)
    record["turns"].pop()
    record["total_return"] = 0.0
    with pytest.raises(ValueError, match="turn 0: episode is incomplete"):
        trajectory_from_record(record, mdp=mdp)
    record["turns"] = []
    with pytest.raises(ValueError, match="no turns"):
        trajectory_from_record(record, mdp=mdp)


def test_validated_record_rejects_steps_after_termination():
    mdp = RecoveryEnv()
    session = play_episode(mdp, scripted({0: "GOOD"}), seed=0)
    record = episode_record(session, "recovery")
    record["turns"].append(dict(record["turns"][0], timestep=1))
    with pytest.raises(ValueError, match="turn 0:.*after termination"):
        trajectory_from_record(record, mdp=mdp)


def test_validated_record_allows_horizon_exhaustion_without_termination():
    mdp = TableMDP(
        1, "s0", {(0, "s0", "GO"): (Transition("s0", 2.0, 1.0, False),)}
    )
    session = play_episode(mdp, scripted({0: "GO"}), seed=0)
    record = episode_record(session, "horizon_only", extra={"env_params": {}})
    assert trajectory_from_record(record, mdp=mdp) == session.trajectory
    record["turns"].append(dict(record["turns"][0], timestep=1))
    with pytest.raises(ValueError, match="turn 1:.*horizon"):
        trajectory_from_record(record, mdp=mdp)


def test_validated_record_accepts_dense_rewards_and_reports_missing_turn_fields():
    mdp = VariableHorizonEnv(stop_rewards=(1.0, 3.0), continue_reward=0.5)
    session = play_episode(mdp, scripted({0: "CONTINUE", 1: "STOP"}), seed=0)
    record = episode_record(session, "variable_horizon")
    assert trajectory_from_record(record, mdp=mdp) == session.trajectory
    del record["turns"][1]["reward"]
    with pytest.raises(ValueError, match="turn 1: invalid step fields"):
        trajectory_from_record(record, mdp=mdp)


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
