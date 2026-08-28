"""Monte Carlo advantage convergence (plan.md §9.5, §16 item 2).

How many continuation rollouts does approximate ground truth need before it
agrees with the exact oracle? RMSE vs exact advantage as num_rollouts grows,
on the delayed-effect environment.

Usage:
    python experiments/monte_carlo_convergence.py --batch-size 200 --seed 0
"""

import argparse
import math
from pathlib import Path

from agent_credit_bench.envs import DelayedEffectEnv
from agent_credit_bench.estimators import EstimatorContext, MonteCarloAdvantage
from agent_credit_bench.oracle import solve_exact_values
from agent_credit_bench.policy import UniformPolicy
from agent_credit_bench.sampling import sample_trajectories

ROLLOUT_COUNTS = (1, 4, 16, 64, 256, 1024)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--horizon", type=int, default=8)
    parser.add_argument(
        "--out", type=Path, default=Path("results/monte_carlo_convergence.png")
    )
    args = parser.parse_args()

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise SystemExit(
            'matplotlib missing — run: pip install -e ".[experiments]"'
        ) from None

    env = DelayedEffectEnv(horizon=args.horizon)
    policy = UniformPolicy()
    values = solve_exact_values(env, policy)
    trajectories = sample_trajectories(env, policy, args.batch_size, args.seed)
    context = EstimatorContext(mdp=env, policy=policy, trajectories=trajectories)

    errors = []
    for num_rollouts in ROLLOUT_COUNTS:
        estimator = MonteCarloAdvantage(num_rollouts=num_rollouts, seed=args.seed)
        credits = estimator.estimate(context)
        squared = [
            (c - values.advantages[(s.timestep, s.state, s.action)]) ** 2
            for trajectory, row in zip(trajectories, credits, strict=True)
            for s, c in zip(trajectory.steps, row, strict=True)
        ]
        errors.append(math.sqrt(sum(squared) / len(squared)))
        print(f"num_rollouts={num_rollouts}: rmse={errors[-1]:.4f}")

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(ROLLOUT_COUNTS, errors, marker="o", label="MonteCarloAdvantage")
    reference = [errors[0] / math.sqrt(k / ROLLOUT_COUNTS[0]) for k in ROLLOUT_COUNTS]
    ax.plot(
        ROLLOUT_COUNTS, reference, linestyle="--", label="1/sqrt(K) reference"
    )
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("continuation rollouts per (t, s, a)")
    ax.set_ylabel("RMSE vs exact advantage")
    ax.set_title(
        f"MC convergence, DelayedEffectEnv(horizon={args.horizon}), "
        f"batch={args.batch_size}"
    )
    ax.legend()
    fig.tight_layout()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
