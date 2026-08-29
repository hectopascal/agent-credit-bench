"""Score TRL's advantage computations against the exact oracle.

TRL (https://github.com/huggingface/trl) computes GRPO and RLOO advantages
inline in each trainer's ``_generate_and_score_completions`` — there is no
importable advantage function the way verl's ``core_algos`` exposes one. This
module therefore *transcribes* those lines exactly (same ops, same shapes,
same epsilons), calls TRL's real ``nanstd`` helper for every std, and pins
the transcription to the installed TRL source with fingerprint tests
(``tests/test_trl_integration.py``): if a TRL release changes the math, the
fingerprint fails and the transcription must be re-verified.

Suite batches are sampled from one start state, so the whole batch is packed
as a single group (``num_generations`` = batch size), matching the verl
integration's single-group semantics. Both trainers are outcome estimators:
one scalar advantage per completion, broadcast here to every turn.

Not captured: token-level loss shaping, importance ratios, and everything
downstream of the advantage. NaN reward handling (TRL's unscorable-completion
machinery) never triggers because every suite trajectory has a return.

Requires the optional extra::

    pip install "agent-credit-bench[trl]"

Transcribed from and tested against TRL 1.12.0 (torch CPU is sufficient).
"""

from dataclasses import dataclass
from typing import Any

from agent_credit_bench.estimators.base import EstimatorContext


def _trl_parts() -> tuple[Any, Any]:
    """Return (torch, trl.trainer.utils.nanstd), or raise clearly."""
    try:
        import torch
        from trl.trainer.utils import nanstd
    except ImportError as exc:
        raise ImportError(
            "trl is required for agent_credit_bench.integrations.trl — "
            'install it with: pip install "agent-credit-bench[trl]"'
        ) from exc
    return torch, nanstd


def _returns_tensor(context: EstimatorContext) -> Any:
    torch, _ = _trl_parts()
    if not context.trajectories:
        raise ValueError("estimate needs at least one trajectory")
    return torch.tensor(
        [trajectory.total_return for trajectory in context.trajectories],
        dtype=torch.float64,
    )


def _broadcast(
    advantages: Any, context: EstimatorContext
) -> tuple[tuple[float, ...], ...]:
    return tuple(
        tuple(float(advantages[i]) for _ in trajectory.steps)
        for i, trajectory in enumerate(context.trajectories)
    )


@dataclass(frozen=True)
class TrlGRPO:
    """TRL GRPOTrainer's advantage lines, single group, default config path.

    Transcribed (``sum_then_normalize`` aggregation, TRL's default):

        mean_grouped_rewards = torch.nanmean(rewards.view(-1, n), dim=1)
        std_rewards = nanstd(rewards.view(-1, n), dim=1)        # "group"
        std_rewards = nanstd(rewards).expand_as(rewards)        # "batch"
        advantages = rewards - mean_grouped_rewards
        advantages = advantages / (std_rewards + 1e-4)          # unless "none"

    ``scale_rewards="group"`` is TRL's default; ``"none"`` is mean-centering
    only (the Dr. GRPO recommendation); ``"batch"`` is identical to
    ``"group"`` here because the suite batch is a single group. TRL's
    ``nanstd`` is Bessel-corrected like verl's ``torch.std``, but the epsilon
    is 1e-4 where verl uses 1e-6.
    """

    scale_rewards: str = "group"

    def __post_init__(self) -> None:
        if self.scale_rewards not in ("group", "batch", "none"):
            raise ValueError(
                "scale_rewards must be 'group', 'batch', or 'none', "
                f"got {self.scale_rewards!r}"
            )

    @property
    def name(self) -> str:
        return {
            "group": "trl_grpo",
            "none": "trl_dr_grpo",
            "batch": "trl_grpo_batch_scale",
        }[self.scale_rewards]

    def estimate(
        self, context: EstimatorContext
    ) -> tuple[tuple[float, ...], ...]:
        torch, nanstd = _trl_parts()
        rewards = _returns_tensor(context)
        num_generations = len(rewards)
        mean_grouped_rewards = torch.nanmean(
            rewards.view(-1, num_generations), dim=1
        ).repeat_interleave(num_generations, dim=0)
        if self.scale_rewards in ("group", "none"):
            std_rewards = nanstd(
                rewards.view(-1, num_generations), dim=1
            ).repeat_interleave(num_generations, dim=0)
        else:  # "batch"
            std_rewards = nanstd(rewards).expand_as(rewards)
        advantages = rewards - mean_grouped_rewards
        if self.scale_rewards != "none":
            advantages = advantages / (std_rewards + 1e-4)
        return _broadcast(advantages, context)


@dataclass(frozen=True)
class TrlRLOO:
    """TRL RLOOTrainer's advantage lines, single group.

    Transcribed:

        baselines = (grouped_sum - grouped_rewards) / (scorable_counts - 1)
        advantages = rewards - baselines
        advantages = (advantages - torch.nanmean(advantages)) \\
            / (nanstd(advantages) + 1e-4)                  # if normalize

    The default (``normalize_advantages=False``, TRL's default) is plain
    leave-one-out centering — algebraically the suite's
    BatchCenteredBroadcast and verl's RLOO. A single-trajectory batch raises
    ValueError here; TRL zeroes that case, but it cannot occur in training
    and silently returning zeros would misgrade the estimator.
    """

    normalize_advantages: bool = False

    @property
    def name(self) -> str:
        return "trl_rloo_whitened" if self.normalize_advantages else "trl_rloo"

    def estimate(
        self, context: EstimatorContext
    ) -> tuple[tuple[float, ...], ...]:
        torch, nanstd = _trl_parts()
        rewards = _returns_tensor(context)
        num_generations = len(rewards)
        if num_generations < 2:
            raise ValueError("TrlRLOO needs at least two trajectories")
        grouped_rewards = rewards.view(-1, num_generations)
        scorable_counts = (~torch.isnan(grouped_rewards)).sum(dim=1, keepdim=True)
        grouped_sum = torch.nansum(grouped_rewards, dim=1, keepdim=True)
        baselines = (grouped_sum - grouped_rewards) / (scorable_counts - 1)
        advantages = rewards - baselines.view(-1)
        if self.normalize_advantages:
            advantages = (advantages - torch.nanmean(advantages)) / (
                nanstd(advantages) + 1e-4
            )
        return _broadcast(advantages, context)
