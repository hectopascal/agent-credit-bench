"""Score verl's *actual* advantage implementations against the exact oracle.

verl (https://github.com/volcengine/verl) computes advantages on padded
(batch, response_length) token tensors. This module packs suite trajectories
into that format using **one tensor cell per turn**: cell (i, t) holds the
reward of trajectory i's turn t, the response mask marks live turns, and
padding past a trajectory's end is masked out.

The packing is faithful because of how verl's estimators consume the tensors:

- Outcome estimators (GRPO, Dr. GRPO, RLOO) reduce each row to its reward
  *sum* and broadcast over masked cells — identical at any granularity.
- Recursive estimators (GAE, REINFORCE++) recurse cell by cell, skipping
  masked cells. With one cell per turn the recursion runs over exactly the
  turn-level MDP this suite defines.

What this does *not* capture: within-turn token structure, PPO clipping, and
everything else downstream of the advantage computation. It scores the
advantage function itself — verl's real code, not a reimplementation.

All trajectories are packed as a single group (``index`` constant): a suite
batch is sampled from one start state, which is the group-relative
estimators' whole-batch semantics.

Requires the optional extra::

    python -m pip install -e ".[verl]"  # from the source checkout

Tested against verl 0.9.0 (torch CPU is sufficient).
"""

from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from types import SimpleNamespace
from typing import Any

from agent_credit_bench.estimators.base import EstimatorContext
from agent_credit_bench.oracle import solve_exact_values
from agent_credit_bench.types import Trajectory

_SUPPORTED_VERL_VERSION = "0.9.0"


def _require_supported_verl_version() -> None:
    """Fail closed when verl's private advantage API may have changed."""
    try:
        installed = version("verl")
    except PackageNotFoundError as exc:
        raise ImportError(
            "verl is required for agent_credit_bench.integrations.verl — "
            'from the AgentCreditBench checkout, run: '
            'python -m pip install -e ".[verl]"'
        ) from exc
    if installed != _SUPPORTED_VERL_VERSION:
        raise RuntimeError(
            "agent_credit_bench.integrations.verl targets verl "
            f"{_SUPPORTED_VERL_VERSION}, but {installed} is installed; install "
            f"verl=={_SUPPORTED_VERL_VERSION} or re-validate the adapter"
        )


def _verl_core_algos() -> tuple[Any, Any, Any]:
    """Return (torch, numpy, verl.trainer.ppo.core_algos), or raise clearly."""
    _require_supported_verl_version()
    try:
        import numpy
        import torch
        from verl.trainer.ppo import core_algos
    except ImportError as exc:
        raise ImportError(
            "verl is required for agent_credit_bench.integrations.verl — "
            'from the AgentCreditBench checkout, run: '
            'python -m pip install -e ".[verl]"'
        ) from exc
    return torch, numpy, core_algos


def _algo_config(gamma: float) -> Any:
    """verl's AlgoConfig if available, else a duck-typed stand-in."""
    try:
        from verl.trainer.config import AlgoConfig
    except ImportError:
        return SimpleNamespace(gamma=gamma)
    return AlgoConfig(gamma=gamma)


@dataclass(frozen=True)
class PackedBatch:
    token_level_rewards: Any  # torch.Tensor, float64, (batch, max_turns)
    response_mask: Any  # torch.Tensor, float64, (batch, max_turns)
    index: Any  # numpy.ndarray, (batch,), constant single group


def pack_trajectories(trajectories: tuple[Trajectory, ...]) -> PackedBatch:
    """Pack trajectories into verl's tensor format, one cell per turn.

    float64 throughout so conformance comparisons against the suite's pure
    Python estimators are exact to rounding, not float32 noise.
    """
    torch, numpy, _ = _verl_core_algos()
    if not trajectories:
        raise ValueError("pack_trajectories needs at least one trajectory")
    width = max(len(trajectory.steps) for trajectory in trajectories)
    rewards = torch.zeros((len(trajectories), width), dtype=torch.float64)
    mask = torch.zeros((len(trajectories), width), dtype=torch.float64)
    for i, trajectory in enumerate(trajectories):
        for t, step in enumerate(trajectory.steps):
            rewards[i, t] = step.reward
            mask[i, t] = 1.0
    index = numpy.zeros(len(trajectories), dtype=numpy.int64)
    return PackedBatch(rewards, mask, index)


def unpack_credits(
    advantages: Any, trajectories: tuple[Trajectory, ...]
) -> tuple[tuple[float, ...], ...]:
    """Read the (batch, max_turns) advantage tensor back into suite credits."""
    return tuple(
        tuple(float(advantages[i, t]) for t in range(len(trajectory.steps)))
        for i, trajectory in enumerate(trajectories)
    )


@dataclass(frozen=True)
class VerlGRPO:
    """verl's ``compute_grpo_outcome_advantage`` on the whole batch as one group.

    With ``norm_adv_by_std=False`` this is verl's Dr. GRPO variant
    (mean-centering only). Note two deltas from the suite's
    GRPOStyleNormalized reimplementation: verl divides by the Bessel-corrected
    sample std (``torch.std``), not the population std, and uses epsilon 1e-6.
    Those grouped formulas apply for groups of at least two; verl 0.9.0's
    explicit singleton fallback broadcasts the sequence score (divided by
    ``1 + epsilon`` in normalized GRPO).
    """

    norm_adv_by_std: bool = True

    @property
    def name(self) -> str:
        return "verl_grpo" if self.norm_adv_by_std else "verl_dr_grpo"

    def estimate(
        self, context: EstimatorContext
    ) -> tuple[tuple[float, ...], ...]:
        _, _, core_algos = _verl_core_algos()
        batch = pack_trajectories(context.trajectories)
        advantages, _ = core_algos.compute_grpo_outcome_advantage(
            batch.token_level_rewards,
            batch.response_mask,
            batch.index,
            norm_adv_by_std_in_grpo=self.norm_adv_by_std,
        )
        return unpack_credits(advantages, context.trajectories)


@dataclass(frozen=True)
class VerlRLOO:
    """verl's ``compute_rloo_outcome_advantage`` (leave-one-out baseline).

    For groups of at least two, this is algebraically identical to the suite's
    BatchCenteredBroadcast:
    ``r*n/(n-1) - mean*n/(n-1) == r - mean(others)``. verl 0.9.0 handles a
    singleton group separately by broadcasting its raw score.
    """

    name: str = "verl_rloo"

    def estimate(
        self, context: EstimatorContext
    ) -> tuple[tuple[float, ...], ...]:
        _, _, core_algos = _verl_core_algos()
        batch = pack_trajectories(context.trajectories)
        advantages, _ = core_algos.compute_rloo_outcome_advantage(
            batch.token_level_rewards, batch.response_mask, batch.index
        )
        return unpack_credits(advantages, context.trajectories)


@dataclass(frozen=True)
class VerlReinforcePlusPlus:
    """verl's ``compute_reinforce_plus_plus_outcome_advantage``.

    Per-turn discounted return-to-go, then whitened (zero mean, unit
    variance) across all live cells in the batch.
    """

    gamma: float = 1.0
    name: str = "verl_reinforce_plus_plus"

    def estimate(
        self, context: EstimatorContext
    ) -> tuple[tuple[float, ...], ...]:
        _, _, core_algos = _verl_core_algos()
        batch = pack_trajectories(context.trajectories)
        advantages, _ = core_algos.compute_reinforce_plus_plus_outcome_advantage(
            batch.token_level_rewards,
            batch.response_mask,
            config=_algo_config(self.gamma),
        )
        return unpack_credits(advantages, context.trajectories)


@dataclass(frozen=True)
class VerlGAE:
    """verl's ``compute_gae_advantage_return`` with a controlled critic.

    ``critic="exact"`` feeds the oracle's undiscounted V^pi(t, s) as the
    value tensor and therefore requires ``gamma=1`` — GAE with a perfect
    critic, isolating the estimator from critic error. ``critic="zero"``
    feeds zeros and permits other gamma values. Padding cells get value 0,
    which verl's masked recursion never reads.

    At ``lam=1`` GAE is return-to-go minus baseline, so even a perfect critic
    can give selected successful repairs positive BAD credit. Only ``lam < 1``
    bootstraps on the critic and separates the two selected turns.

    verl whitens the advantages across the batch before returning them, so
    even the perfect-critic variant returns shifted/scaled credit values —
    exactly the distortion the suite's centered/gradient metrics are built
    to separate from real errors.
    """

    gamma: float = 1.0
    lam: float = 1.0
    critic: str = "exact"

    def __post_init__(self) -> None:
        if self.critic not in ("exact", "zero"):
            raise ValueError(f"critic must be 'exact' or 'zero', got {self.critic!r}")
        if self.critic == "exact" and self.gamma != 1.0:
            raise ValueError(
                "critic='exact' is only available with gamma=1.0 because "
                "the suite oracle is undiscounted"
            )

    @property
    def name(self) -> str:
        return f"verl_gae_{self.critic}_lam{self.lam:g}"

    def estimate(
        self, context: EstimatorContext
    ) -> tuple[tuple[float, ...], ...]:
        torch, _, core_algos = _verl_core_algos()
        batch = pack_trajectories(context.trajectories)
        values = torch.zeros_like(batch.token_level_rewards)
        if self.critic == "exact":
            exact = solve_exact_values(context.mdp, context.policy)
            for i, trajectory in enumerate(context.trajectories):
                for t, step in enumerate(trajectory.steps):
                    values[i, t] = exact.state_values[(step.timestep, step.state)]
        advantages, _ = core_algos.compute_gae_advantage_return(
            batch.token_level_rewards,
            values,
            batch.response_mask,
            self.gamma,
            self.lam,
        )
        return unpack_credits(advantages, context.trajectories)
