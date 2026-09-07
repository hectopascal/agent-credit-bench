"""Variable-horizon sweep -- the flagship diagnostic (plan.md §11.3).

Sweep the three variables named in the experiment plan: maximum horizon,
policy stop probability, and group/batch size.  The summary CSV has one row
per configuration and estimator; the turn-statistics CSV contains the
per-timestep measurements used by the credit-bias figure.

The old single-configuration interface remains available through
``--batch-size`` and ``--horizon``.  Use the plural options for a sweep::

    python experiments/variable_horizon_sweep.py \
        --batch-sizes 2,4,8,32,128,500 --horizons 3,5,8 --num-seeds 200
"""

import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path
from statistics import fmean

from agent_credit_bench.benchmark import BenchmarkResult, run_benchmark
from agent_credit_bench.envs.variable_horizon import (
    StopProbabilityPolicy,
    VariableHorizonEnv,
)
from agent_credit_bench.estimators import BatchCenteredBroadcast, OracleAdvantage
from agent_credit_bench.estimators.turn_loo import TurnLOO
from agent_credit_bench.gradients import exact_policy_gradient, norm
from agent_credit_bench.oracle import solve_exact_values

DEFAULT_STOP_PROBABILITIES = (0.2, 0.35, 0.5, 0.65, 0.8)
DEFAULT_BATCH_SIZES = (2, 4, 8, 32, 128, 500)
DEFAULT_HORIZONS = (3, 5, 8)
REWARD_PATTERN = (0.0, 2.0, 1.0, 3.0)
ESTIMATORS = (OracleAdvantage(), BatchCenteredBroadcast(), TurnLOO())


@dataclass(frozen=True)
class SweepResult:
    benchmark: BenchmarkResult
    normalized_gradient_mse: float


SUMMARY_FIELDS = (
    "horizon",
    "stop_rewards",
    "continue_reward",
    "batch_size",
    "stop_probability",
    "estimator",
    "num_seeds",
    "rmse_mean",
    "centered_rmse_mean",
    "spearman_mean",
    "sign_accuracy_mean",
    "gradient_cosine_mean",
    "mean_gradient_cosine",
    "relative_mean_gradient_error",
    "gradient_variance",
    "normalized_gradient_variance",
    "normalized_gradient_mse",
)
TURN_FIELDS = (
    "horizon",
    "stop_rewards",
    "continue_reward",
    "batch_size",
    "stop_probability",
    "estimator",
    "num_seeds",
    "timestep",
    "credit_bias",
    "credit_variance",
    "step_count",
)


def _comma_separated_ints(value: str) -> tuple[int, ...]:
    try:
        values = tuple(int(item.strip()) for item in value.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError("expected comma-separated integers") from error
    if not values or any(item < 1 for item in values):
        raise argparse.ArgumentTypeError("values must be positive integers")
    return values


def _comma_separated_probabilities(value: str) -> tuple[float, ...]:
    try:
        values = tuple(float(item.strip()) for item in value.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError("expected comma-separated numbers") from error
    if not values or any(not 0.0 < item < 1.0 for item in values):
        raise argparse.ArgumentTypeError(
            "probabilities must be strictly between 0 and 1"
        )
    return values


def _environment(horizon: int) -> VariableHorizonEnv:
    """Repeat the documented reward motif while varying maximum horizon."""
    if horizon < 2:
        raise ValueError("horizon must be at least 2")
    rewards = tuple(REWARD_PATTERN[t % len(REWARD_PATTERN)] for t in range(horizon))
    return VariableHorizonEnv(stop_rewards=rewards)


def _mean_optional(values: list[float | None]) -> float | str:
    present = [value for value in values if value is not None]
    return fmean(present) if present else ""


def _summary_row(
    *,
    horizon: int,
    stop_rewards: tuple[float, ...],
    continue_reward: float,
    batch_size: int,
    stop_probability: float,
    result: BenchmarkResult,
    exact_gradient_norm: float,
) -> dict[str, object]:
    metrics = result.seed_metrics
    if exact_gradient_norm == 0.0 or result.relative_mean_gradient_error is None:
        raise ValueError("the sweep requires a nonzero exact policy gradient")
    normalized_variance = result.gradient_variance / exact_gradient_norm**2
    # E[||g_hat - g*||^2] / ||g*||^2 = relative squared bias + relative
    # variance.  The mean-gradient term is empirical over ``num_seeds``.
    normalized_mse = result.relative_mean_gradient_error**2 + normalized_variance
    return {
        "horizon": horizon,
        "stop_rewards": "|".join(format(value, ".12g") for value in stop_rewards),
        "continue_reward": continue_reward,
        "batch_size": batch_size,
        "stop_probability": stop_probability,
        "estimator": result.estimator,
        "num_seeds": len(metrics),
        "rmse_mean": fmean(metric.rmse for metric in metrics),
        "centered_rmse_mean": fmean(metric.centered_rmse for metric in metrics),
        "spearman_mean": fmean(metric.spearman for metric in metrics),
        "sign_accuracy_mean": _mean_optional(
            [metric.sign_accuracy for metric in metrics]
        ),
        "gradient_cosine_mean": _mean_optional(
            [metric.gradient_cosine for metric in metrics]
        ),
        "mean_gradient_cosine": result.mean_gradient_cosine,
        "relative_mean_gradient_error": result.relative_mean_gradient_error,
        "gradient_variance": result.gradient_variance,
        "normalized_gradient_variance": normalized_variance,
        "normalized_gradient_mse": normalized_mse,
    }


def _stable_value(value: object) -> object:
    """Avoid platform-dependent last-bit noise in committed CSV artifacts."""
    if isinstance(value, float):
        return format(value, ".12g")
    return value


def _write_csv(
    path: Path,
    rows: list[dict[str, object]],
    fields: tuple[str, ...],
) -> None:
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(
            {key: _stable_value(row[key]) for key in fields} for row in rows
        )


def _closest(values, target):
    return min(values, key=lambda value: (abs(value - target), value))


def _plot_gradient_summary(
    plt,
    results,
    batch_sizes,
    horizons,
    stop_probabilities,
    out_dir,
):
    focus_horizon = _closest(horizons, 5)
    focus_probability = _closest(stop_probabilities, 0.5)
    focus_batch = max(batch_sizes)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8.5))
    ax_cosine, ax_variance, ax_magnitude, ax_mse = axes.flat

    for estimator in ESTIMATORS:
        probability_results = [
            results[(focus_horizon, focus_batch, p, estimator.name)]
            for p in stop_probabilities
        ]
        ax_cosine.plot(
            stop_probabilities,
            [
                result.benchmark.mean_gradient_cosine
                for result in probability_results
            ],
            marker="o",
            label=estimator.name,
        )
        ax_variance.plot(
            stop_probabilities,
            [result.benchmark.gradient_variance for result in probability_results],
            marker="o",
            label=estimator.name,
        )

        batch_rows = [
            results[(focus_horizon, batch_size, focus_probability, estimator.name)]
            for batch_size in batch_sizes
        ]
        ax_magnitude.plot(
            batch_sizes,
            [result.benchmark.relative_mean_gradient_error for result in batch_rows],
            marker="o",
            label=estimator.name,
        )
        ax_mse.plot(
            batch_sizes,
            [result.normalized_gradient_mse for result in batch_rows],
            marker="o",
            label=estimator.name,
        )

    ax_cosine.set_xlabel("policy stop probability")
    ax_cosine.set_ylabel("cosine(mean gradient, exact gradient)")
    ax_cosine.set_title(f"Direction, horizon={focus_horizon}, batch={focus_batch}")
    ax_cosine.ticklabel_format(axis="y", style="plain", useOffset=False)
    ax_cosine.legend()
    ax_variance.set_xlabel("policy stop probability")
    ax_variance.set_ylabel("E ||g - mean g||²")
    ax_variance.set_yscale("log")
    ax_variance.set_title(f"Variance, horizon={focus_horizon}, batch={focus_batch}")
    ax_magnitude.set_xlabel("group/batch size")
    ax_magnitude.set_ylabel("||mean g - exact g|| / ||exact g||")
    ax_magnitude.set_xscale("log", base=2)
    ax_magnitude.set_xticks(batch_sizes, labels=batch_sizes)
    ax_magnitude.set_yscale("log")
    ax_magnitude.set_title(
        f"Empirical mean error, horizon={focus_horizon}, p={focus_probability}"
    )
    ax_mse.set_xlabel("group/batch size")
    ax_mse.set_ylabel("normalized gradient MSE")
    ax_mse.set_xscale("log", base=2)
    ax_mse.set_xticks(batch_sizes, labels=batch_sizes)
    ax_mse.set_yscale("log")
    ax_mse.set_title(
        "Empirical mean error² + variance, "
        f"horizon={focus_horizon}, p={focus_probability}"
    )
    fig.suptitle("Variable-horizon gradient diagnostics")
    fig.tight_layout()
    path = out_dir / "variable_horizon_gradient.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"wrote {path}")


def _plot_turn_bias(
    plt,
    results,
    batch_sizes,
    horizons,
    stop_probabilities,
    out_dir,
):
    focus_horizon = _closest(horizons, 5)
    focus_probability = _closest(stop_probabilities, 0.5)
    focus_batch = max(batch_sizes)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for estimator in ESTIMATORS:
        per_turn = results[
            (focus_horizon, focus_batch, focus_probability, estimator.name)
        ].benchmark.per_turn
        timesteps = sorted(per_turn)
        ax.plot(
            timesteps,
            [per_turn[t].bias for t in timesteps],
            marker="o",
            label=estimator.name,
        )
    ax.axhline(0.0, linewidth=0.8, color="black")
    ax.set_xlabel("timestep")
    ax.set_ylabel("mean(credit - exact advantage)")
    ax.set_title(
        "Per-turn credit bias "
        f"(horizon={focus_horizon}, batch={focus_batch}, p={focus_probability})"
    )
    ax.legend()
    fig.tight_layout()
    path = out_dir / "variable_horizon_turn_bias.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"wrote {path}")


def _plot_crossover_frontier(
    plt,
    results,
    batch_sizes,
    horizons,
    stop_probabilities,
    out_dir,
):
    """Heatmaps of TurnLOO MSE divided by trajectory-centered MSE."""
    fig, axes = plt.subplots(
        1,
        len(horizons),
        figsize=(4.2 * len(horizons), 4.8),
        squeeze=False,
        sharey=True,
    )
    images = []
    for axis, horizon in zip(axes[0], horizons, strict=True):
        ratios = []
        for probability in stop_probabilities:
            row = []
            for batch_size in batch_sizes:
                turn = results[(horizon, batch_size, probability, "turn_loo")]
                trajectory = results[
                    (
                        horizon,
                        batch_size,
                        probability,
                        "batch_centered_broadcast",
                    )
                ]
                row.append(
                    math.log10(
                        turn.normalized_gradient_mse
                        / trajectory.normalized_gradient_mse
                    )
                )
            ratios.append(row)
        image = axis.imshow(
            ratios,
            aspect="auto",
            origin="lower",
            cmap="coolwarm",
            vmin=-0.5,
            vmax=0.5,
        )
        images.append(image)
        axis.set_xticks(range(len(batch_sizes)), labels=batch_sizes, rotation=45)
        axis.set_yticks(range(len(stop_probabilities)), labels=stop_probabilities)
        axis.set_xlabel("group/batch size")
        axis.set_title(f"horizon={horizon}")
    axes[0][0].set_ylabel("policy stop probability")
    fig.subplots_adjust(left=0.08, right=0.88, bottom=0.18, top=0.83, wspace=0.15)
    color_axis = fig.add_axes((0.91, 0.2, 0.015, 0.58))
    colorbar = fig.colorbar(images[-1], cax=color_axis)
    colorbar.set_label("log10(TurnLOO MSE / trajectory-centered MSE)")
    fig.suptitle("Empirical gradient-error crossover (< 0 favors TurnLOO)")
    path = out_dir / "variable_horizon_frontier.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"wrote {path}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--batch-sizes",
        type=_comma_separated_ints,
        help="comma-separated group sizes",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        help="run one group size (backward-compatible singular option)",
    )
    parser.add_argument(
        "--horizons",
        type=_comma_separated_ints,
        help="comma-separated maximum horizons",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        help="run one maximum horizon",
    )
    parser.add_argument(
        "--stop-probabilities",
        type=_comma_separated_probabilities,
        default=DEFAULT_STOP_PROBABILITIES,
    )
    parser.add_argument("--num-seeds", type=int, default=200)
    parser.add_argument("--out-dir", type=Path, default=Path("results"))
    args = parser.parse_args()
    if args.batch_size is not None and args.batch_sizes is not None:
        parser.error("use either --batch-size or --batch-sizes, not both")
    if args.batch_size is not None:
        if args.batch_size < 2:
            parser.error("--batch-size must be at least 2")
        args.batch_sizes = (args.batch_size,)
    elif args.batch_sizes is None:
        args.batch_sizes = DEFAULT_BATCH_SIZES
    if args.horizon is not None and args.horizons is not None:
        parser.error("use either --horizon or --horizons, not both")
    if args.horizon is not None:
        if args.horizon < 2:
            parser.error("--horizon must be at least 2")
        args.horizons = (args.horizon,)
    elif args.horizons is None:
        # Preserve the original ``--batch-size 500`` command, which swept
        # stop probabilities in the default five-step environment.
        args.horizons = (5,) if args.batch_size is not None else DEFAULT_HORIZONS
    if any(horizon < 2 for horizon in args.horizons):
        parser.error("all horizons must be at least 2")
    if any(batch_size < 2 for batch_size in args.batch_sizes):
        parser.error("all batch sizes must be at least 2")
    if args.num_seeds < 2:
        parser.error("--num-seeds must be at least 2 for variance estimates")
    args.batch_sizes = tuple(sorted(set(args.batch_sizes)))
    args.horizons = tuple(sorted(set(args.horizons)))
    args.stop_probabilities = tuple(sorted(set(args.stop_probabilities)))
    return args


def main() -> None:
    args = _parse_args()
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        raise SystemExit(
            'matplotlib missing -- run: pip install -e ".[experiments]"'
        ) from None

    summary_rows = []
    turn_rows = []
    results = {}
    total = len(args.horizons) * len(args.batch_sizes) * len(args.stop_probabilities)
    completed = 0
    for horizon in args.horizons:
        env = _environment(horizon)
        for batch_size in args.batch_sizes:
            for probability in args.stop_probabilities:
                policy = StopProbabilityPolicy(probability)
                values = solve_exact_values(env, policy)
                exact_gradient_norm = norm(exact_policy_gradient(env, policy, values))
                for estimator in ESTIMATORS:
                    result = run_benchmark(
                        mdp=env,
                        policy=policy,
                        estimator=estimator,
                        batch_size=batch_size,
                        seeds=range(args.num_seeds),
                    )
                    row = _summary_row(
                        horizon=horizon,
                        stop_rewards=env.stop_rewards,
                        continue_reward=env.continue_reward,
                        batch_size=batch_size,
                        stop_probability=probability,
                        result=result,
                        exact_gradient_norm=exact_gradient_norm,
                    )
                    results[(horizon, batch_size, probability, estimator.name)] = (
                        SweepResult(
                            benchmark=result,
                            normalized_gradient_mse=row["normalized_gradient_mse"],
                        )
                    )
                    summary_rows.append(row)
                    for timestep, stats in result.per_turn.items():
                        turn_rows.append(
                            {
                                "horizon": horizon,
                                "stop_rewards": "|".join(
                                    format(value, ".12g")
                                    for value in env.stop_rewards
                                ),
                                "continue_reward": env.continue_reward,
                                "batch_size": batch_size,
                                "stop_probability": probability,
                                "estimator": estimator.name,
                                "num_seeds": args.num_seeds,
                                "timestep": timestep,
                                "credit_bias": stats.bias,
                                "credit_variance": stats.variance,
                                "step_count": stats.count,
                            }
                        )
                completed += 1
                print(
                    f"[{completed}/{total}] horizon={horizon}, "
                    f"batch={batch_size}, stop_probability={probability}: done"
                )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.out_dir / "variable_horizon_sweep.csv"
    _write_csv(summary_path, summary_rows, SUMMARY_FIELDS)
    print(f"wrote {summary_path}")
    turns_path = args.out_dir / "variable_horizon_turn_stats.csv"
    _write_csv(turns_path, turn_rows, TURN_FIELDS)
    print(f"wrote {turns_path}")

    _plot_gradient_summary(
        plt,
        results,
        args.batch_sizes,
        args.horizons,
        args.stop_probabilities,
        args.out_dir,
    )
    _plot_turn_bias(
        plt,
        results,
        args.batch_sizes,
        args.horizons,
        args.stop_probabilities,
        args.out_dir,
    )
    _plot_crossover_frontier(
        plt,
        results,
        args.batch_sizes,
        args.horizons,
        args.stop_probabilities,
        args.out_dir,
    )


if __name__ == "__main__":
    main()
