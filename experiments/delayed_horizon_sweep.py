"""Delayed-effect horizon sweep — the first diagnostic figure (plan.md §11.1).

M2 exit condition: a reproducible plot of credit leaked onto zero-advantage
distractor actions, versus horizon.

Usage:
    pip install -e ".[experiments]"   # matplotlib, one-time
    python experiments/delayed_horizon_sweep.py --batch-size 1000 --seed 0
"""

import argparse
from collections.abc import Sequence
from pathlib import Path

from agent_credit_bench.envs import DelayedEffectEnv
from agent_credit_bench.estimators import (
    BatchCenteredBroadcast,
    EstimatorContext,
    GiGPOStyle,
    GRPOStyleNormalized,
    OracleAdvantage,
    OutcomeBroadcast,
)
from agent_credit_bench.policy import UniformPolicy
from agent_credit_bench.sampling import sample_trajectories
from agent_credit_bench.types import Trajectory

HORIZONS = (2, 4, 8, 16, 32)
ESTIMATORS = (
    OracleAdvantage(),
    OutcomeBroadcast(),
    BatchCenteredBroadcast(),
    GRPOStyleNormalized(),
    GiGPOStyle(),
)


def mean_distractor_credit(
    trajectories: Sequence[Trajectory],
    credits: Sequence[Sequence[float]],
) -> float:
    """Mean |credit| over steps at t >= 1, where exact advantage is zero.

    The inline precursor of the M3 leakage metric (plan.md §10.4).
    """
    magnitudes = [
        abs(credit)
        for trajectory, row in zip(trajectories, credits, strict=True)
        for step, credit in zip(trajectory.steps, row, strict=True)
        if step.timestep >= 1
    ]
    return sum(magnitudes) / len(magnitudes)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=Path("results/delayed_leakage.png"))
    args = parser.parse_args()

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise SystemExit(
            'matplotlib missing — run: pip install -e ".[experiments]"'
        ) from None

    leakage: dict[str, list[float]] = {est.name: [] for est in ESTIMATORS}
    for horizon in HORIZONS:
        env = DelayedEffectEnv(horizon=horizon)
        policy = UniformPolicy()
        trajectories = sample_trajectories(env, policy, args.batch_size, args.seed)
        context = EstimatorContext(mdp=env, policy=policy, trajectories=trajectories)
        for estimator in ESTIMATORS:
            credits = estimator.estimate(context)
            leakage[estimator.name].append(
                mean_distractor_credit(trajectories, credits)
            )
        print(f"horizon={horizon}: done")

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for name, values in leakage.items():
        ax.plot(HORIZONS, values, marker="o", label=name)
    ax.set_xlabel("horizon")
    ax.set_ylabel("mean |credit| on zero-advantage distractor steps")
    ax.set_xscale("log", base=2)
    ax.set_title(
        f"Credit leakage vs horizon (batch={args.batch_size}, seed={args.seed})"
    )
    ax.legend()
    fig.tight_layout()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
