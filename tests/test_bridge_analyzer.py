"""Input-validation tests for the real-rollout bridge analyzer."""

import importlib.util
from pathlib import Path

import pytest

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


def test_environment_spec_does_not_conflate_json_parameter_types() -> None:
    analyzer = load_analyzer()
    records = [
        {"env": "recovery", "env_params": {"horizon": 1}},
        {"env": "recovery", "env_params": {"horizon": True}},
    ]
    with pytest.raises(SystemExit, match="mix env_params for 'recovery'"):
        analyzer.environment_spec(records)


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
