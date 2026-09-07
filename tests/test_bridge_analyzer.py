"""Input-validation tests for the real-rollout bridge analyzer."""

import importlib.util
import json
from pathlib import Path

import pytest

from agent_credit_bench.envs import RecoveryEnv
from agent_credit_bench.estimators import BatchCenteredBroadcast, OracleAdvantage
from agent_credit_bench.integrations.bridge import episode_record, play_episode

ANALYZER_PATH = (
    Path(__file__).resolve().parent.parent
    / "recipes"
    / "verl_bridge"
    / "analyze_checkpoint.py"
)


def load_analyzer():
    spec = importlib.util.spec_from_file_location("bridge_analyzer", ANALYZER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_environment_spec_accepts_equal_parameter_objects() -> None:
    analyzer = load_analyzer()
    records = [
        {
            "env": "recovery",
            "env_params": {"recover_success_probability": 0.5, "horizon": 2},
        },
        {
            "env": "recovery",
            "env_params": {"horizon": 2, "recover_success_probability": 0.5},
        },
    ]
    assert analyzer.environment_spec(records) == (
        "recovery",
        {"recover_success_probability": 0.5, "horizon": 2},
    )


def test_environment_spec_rejects_mixed_parameters_for_same_env() -> None:
    analyzer = load_analyzer()
    records = [
        {"env": "recovery", "env_params": {"recover_success_probability": 1.0}},
        {"env": "recovery", "env_params": {"recover_success_probability": 0.5}},
    ]
    with pytest.raises(SystemExit, match="mix env_params for 'recovery'"):
        analyzer.environment_spec(records)


def test_environment_spec_rejects_boolean_horizon() -> None:
    analyzer = load_analyzer()
    records = [
        {"env": "recovery", "env_params": {"horizon": 2}},
        {"env": "recovery", "env_params": {"horizon": True}},
    ]
    with pytest.raises(SystemExit, match="invalid env_params.*integer"):
        analyzer.environment_spec(records)


def test_environment_spec_accepts_legacy_partial_parameters():
    records = [
        {"env": "recovery", "env_params": {}},
        {
            "env": "recovery",
            "env_params": {"horizon": 2, "recover_success_probability": 1.0},
        },
    ]
    assert load_analyzer().environment_spec(records) == (
        "recovery", {"horizon": 2, "recover_success_probability": 1.0}
    )


def test_analyzer_rejects_invalid_inputs() -> None:
    analyzer = load_analyzer()
    with pytest.raises(SystemExit, match="non-object env_params"):
        analyzer.environment_spec(
            [{"env": "recovery", "env_params": "not-a-parameter-object"}]
        )
    for value in ("0", "-1", "not-an-integer"):
        with pytest.raises(
            analyzer.argparse.ArgumentTypeError, match="positive integer"
        ):
            analyzer.positive_integer_argument(value)
    assert analyzer.positive_integer_argument("3") == 3


@pytest.mark.parametrize("fields", [{}, {"env_params": None}])
def test_environment_spec_rejects_unknown_parameters(fields):
    with pytest.raises(SystemExit, match="env_params"):
        load_analyzer().environment_spec([{"env": "recovery", **fields}])


def test_recorded_parameters_preserve_analyzer_oracle():
    from agent_credit_bench.integrations.bridge import EmpiricalTabularPolicy, make_env
    from agent_credit_bench.oracle import solve_exact_values

    mdp = RecoveryEnv(recover_success_probability=0)
    sessions = [
        play_episode(mdp, lambda _messages: "GOOD", seed=0),
        play_episode(mdp, lambda _messages: "BAD RECOVER", seed=1),
    ]
    records = [episode_record(session, "recovery") for session in sessions]
    name, params = load_analyzer().environment_spec(records)
    policy = EmpiricalTabularPolicy.from_trajectories(s.trajectory for s in sessions)
    values = solve_exact_values(make_env(name, **params), policy)
    assert values.advantages[(0, "s0", "BAD")] == -0.5


@pytest.mark.parametrize("window", [[], ["--last", "1"]])
def test_analyzer_rejects_single_episode_before_reporting(
    tmp_path, monkeypatch, capsys, window
):
    analyzer = load_analyzer()
    session = play_episode(RecoveryEnv(), lambda _messages: "GOOD", seed=0)
    record = json.dumps(episode_record(session, "recovery")) + "\n"
    source = tmp_path / "episodes.jsonl"
    source.write_text(record * (2 if window else 1))
    output = tmp_path / "results.csv"
    monkeypatch.setattr(
        "sys.argv", [str(ANALYZER_PATH), str(source), "--out", str(output), *window]
    )
    with pytest.raises(SystemExit) as error:
        analyzer.main()
    assert error.value.code == 2
    captured = capsys.readouterr()
    assert "at least two episodes" in captured.err
    assert captured.out == ""
    assert not output.exists()


def test_analyzer_exports_two_episode_batch(tmp_path, monkeypatch):
    analyzer = load_analyzer()
    session = play_episode(RecoveryEnv(), lambda _messages: "GOOD", seed=0)
    source = tmp_path / "episodes.jsonl"
    source.write_text((json.dumps(episode_record(session, "recovery")) + "\n") * 2)
    output = tmp_path / "results.csv"
    monkeypatch.setattr(
        "sys.argv", [str(ANALYZER_PATH), str(source), "--out", str(output)]
    )
    monkeypatch.setattr(
        analyzer,
        "build_estimators",
        lambda: [OracleAdvantage(), BatchCenteredBroadcast()],
    )
    analyzer.main()
    assert "batch_centered_broadcast" in output.read_text()


@pytest.mark.parametrize(
    "change",
    [
        {"reward": -100.0},
        {"state": "mistake"},
        {"timestep": 1},
        {"action": "UNKNOWN"},
        {"next_state": "failure"},
        {"terminated": False},
    ],
)
def test_analyzer_rejects_impossible_steps_before_scoring(
    tmp_path, monkeypatch, capsys, change
):
    analyzer = load_analyzer()
    session = play_episode(RecoveryEnv(), lambda _messages: "GOOD", seed=0)
    record = episode_record(session, "recovery")
    record["turns"][0].update(change)
    record["total_return"] = record["turns"][0]["reward"]
    source, output = tmp_path / "episodes.jsonl", tmp_path / "scores.csv"
    source.write_text((json.dumps(record) + "\n") * 2)
    monkeypatch.setattr(
        "sys.argv", [str(ANALYZER_PATH), str(source), "--out", str(output)]
    )
    with pytest.raises(SystemExit) as error:
        analyzer.main()
    assert error.value.code == 2
    captured = capsys.readouterr()
    assert "episode 0" in captured.err and "turn 0" in captured.err
    assert captured.out == ""
    assert not output.exists()
