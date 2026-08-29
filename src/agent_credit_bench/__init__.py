"""Conformance tests for turn-level credit estimators (see plan.md)."""

from agent_credit_bench.benchmark import BenchmarkResult, run_benchmark
from agent_credit_bench.mdp import FiniteHorizonMDP
from agent_credit_bench.oracle import solve_exact_values
from agent_credit_bench.policy import Policy, TabularPolicy, UniformPolicy
from agent_credit_bench.sampling import sample_trajectories
from agent_credit_bench.types import (
    Action,
    ExactValues,
    State,
    Step,
    Trajectory,
    Transition,
)

__version__ = "0.5.0"

__all__ = [
    "Action",
    "BenchmarkResult",
    "ExactValues",
    "FiniteHorizonMDP",
    "Policy",
    "State",
    "Step",
    "TabularPolicy",
    "Trajectory",
    "Transition",
    "UniformPolicy",
    "run_benchmark",
    "sample_trajectories",
    "solve_exact_values",
]
