"""Score OpenRLHF's *actual* advantage pipeline against the exact oracle.

OpenRLHF (https://github.com/OpenRLHF/OpenRLHF) computes advantages in
``RemoteExperienceMaker.compute_advantages_and_returns`` — group-relative
baselines (RLOO, group_norm, Dr. GRPO, REINFORCE++-baseline), reward
placement, GAE / cumulative returns, and batch whitening, all in one method.
The method only touches ``self`` for config (``strategy.args``, ``kl_ctl``,
``advantage_estimator``) and its two pure helper methods, so this module
calls it **unbound with a stubbed self**: every number comes from OpenRLHF's
real released code, end to end, not a transcription.

Packing follows the pipeline's own semantics. OpenRLHF's estimator interface
consumes one scalar reward per sequence (``Experience.rewards`` is ``(B,)``;
``compute_reward`` scatters it onto the last unmasked cell), so each
trajectory contributes its total return, placed on its final turn — with one
tensor cell per turn, as in the verl integration. For every suite
environment this coincides with the per-turn reward structure, because suite
rewards are only ever emitted on the turn that ends the episode. KL is zero,
length penalties are disabled, and the whole batch is one group
(``n_samples_per_prompt`` = batch size).

One caveat the tests inherit: the batch-whitening block casts advantages to
float32 before computing the mean/rstd, so whitened estimators (gae,
reinforce, reinforce_baseline) carry ~1e-7 float32 noise even on float64
input. Unwhitened ones (rloo, group_norm, dr_grpo) are exact to rounding.

OpenRLHF ships Linux-only wheels; install the extra on Linux (CI runs it)::

    pip install "agent-credit-bench[openrlhf]"

Tested against openrlhf 0.11.0 (torch CPU is sufficient).
"""

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

from agent_credit_bench.estimators.base import EstimatorContext
from agent_credit_bench.oracle import solve_exact_values

_GROUP_ESTIMATORS = ("rloo", "group_norm", "dr_grpo", "reinforce_baseline")
_OUTCOME_ESTIMATORS = _GROUP_ESTIMATORS + ("reinforce",)


def _openrlhf_parts() -> tuple[Any, Any, Any]:
    """Return (torch, RemoteExperienceMaker, Experience), or raise clearly."""
    try:
        import torch
        from openrlhf.trainer.ppo_utils.experience import Experience
        from openrlhf.trainer.ppo_utils.experience_maker import (
            RemoteExperienceMaker,
        )
    except ImportError as exc:
        raise ImportError(
            "openrlhf is required for agent_credit_bench.integrations.openrlhf "
            "(Linux-only wheels) — install it with: "
            'pip install "agent-credit-bench[openrlhf]"'
        ) from exc
    return torch, RemoteExperienceMaker, Experience


def _stub_self(estimator: str, gamma: float, lam: float, no_std_norm: bool) -> Any:
    """The slice of RemoteExperienceMaker state the pipeline actually reads.

    ``clip_range=None`` disables reward clipping; the two ``None`` reward
    fields disable apply_length_penalties' DAPO/ProRL penalties; ``kl_ctl``
    at 0 zeroes the KL term in compute_reward.
    """
    return SimpleNamespace(
        strategy=SimpleNamespace(
            args=SimpleNamespace(
                rollout=SimpleNamespace(n_samples_per_prompt=0),  # set per batch
                algo=SimpleNamespace(
                    advantage=SimpleNamespace(
                        estimator=estimator,
                        gamma=gamma,
                        lambd=lam,
                        no_std_norm=no_std_norm,
                    )
                ),
                reward=SimpleNamespace(
                    clip_range=None,
                    overlong_buffer_len=None,
                    stop_properly_penalty_coef=None,
                ),
            )
        ),
        kl_ctl=SimpleNamespace(value=0.0),
        advantage_estimator=estimator,
    )


def _run_pipeline(
    context: EstimatorContext,
    estimator: str,
    gamma: float,
    lam: float,
    no_std_norm: bool,
    critic: str | None,
) -> tuple[tuple[float, ...], ...]:
    torch, maker, experience_cls = _openrlhf_parts()
    trajectories = context.trajectories
    if not trajectories:
        raise ValueError("estimate needs at least one trajectory")
    n = len(trajectories)
    width = max(len(trajectory.steps) for trajectory in trajectories)

    action_mask = torch.zeros((n, width), dtype=torch.bool)
    values = torch.zeros((n, width), dtype=torch.float64)
    exact = (
        solve_exact_values(context.mdp, context.policy) if critic == "exact" else None
    )
    for i, trajectory in enumerate(trajectories):
        for t, step in enumerate(trajectory.steps):
            action_mask[i, t] = True
            if exact is not None:
                values[i, t] = exact.state_values[(step.timestep, step.state)]

    experience = experience_cls(
        action_mask=action_mask,
        values=values,
        kl=torch.zeros((n, width), dtype=torch.float64),
        rewards=torch.tensor(
            [trajectory.total_return for trajectory in trajectories],
            dtype=torch.float64,
        ),
        index=list(range(n)),
        info={},
    )
    stub = _stub_self(estimator, gamma, lam, no_std_norm)
    stub.strategy.args.rollout.n_samples_per_prompt = n
    # The pipeline calls its two pure helpers through self; bind the real
    # methods onto the stub (they never read self).
    stub.get_advantages_and_returns = maker.get_advantages_and_returns.__get__(stub)
    stub.get_cumulative_returns = maker.get_cumulative_returns.__get__(stub)
    (experience,) = maker.compute_advantages_and_returns(stub, [experience])
    return tuple(
        tuple(float(experience.advantages[i, t]) for t in range(len(trajectory.steps)))
        for i, trajectory in enumerate(trajectories)
    )


@dataclass(frozen=True)
class OpenRLHFOutcome:
    """One of OpenRLHF's outcome estimators, through the real pipeline.

    ``estimator`` is OpenRLHF's own ``--algo.advantage.estimator`` name:
    ``rloo`` (leave-one-out centering), ``group_norm`` (their GRPO:
    ``(r - mean) / (torch.std + 1e-9)``), ``dr_grpo`` (mean-centering),
    ``reinforce_baseline`` (mean-centering, then batch-whitened — their
    REINFORCE++-baseline), and ``reinforce`` (raw return, batch-whitened).

    Only ``reinforce`` honours ``gamma``; the pipeline force-resets it to
    1.0 for the group-baseline estimators, so passing it here is rejected
    rather than silently ignored. Group-baseline estimators need >= 2
    trajectories (the leave-one-out and std terms are undefined otherwise).
    """

    estimator: str = "rloo"
    gamma: float = 1.0

    def __post_init__(self) -> None:
        if self.estimator not in _OUTCOME_ESTIMATORS:
            raise ValueError(
                f"estimator must be one of {_OUTCOME_ESTIMATORS}, "
                f"got {self.estimator!r}"
            )
        if self.gamma != 1.0 and self.estimator != "reinforce":
            raise ValueError(
                "OpenRLHF forces gamma=1.0 for group-baseline estimators; "
                "gamma is only meaningful for estimator='reinforce'"
            )

    @property
    def name(self) -> str:
        return f"openrlhf_{self.estimator}"

    def estimate(
        self, context: EstimatorContext
    ) -> tuple[tuple[float, ...], ...]:
        if self.estimator in _GROUP_ESTIMATORS and len(context.trajectories) < 2:
            raise ValueError(f"{self.name} needs at least two trajectories")
        return _run_pipeline(
            context,
            estimator=self.estimator,
            gamma=self.gamma,
            lam=1.0,
            no_std_norm=False,
            critic=None,
        )


@dataclass(frozen=True)
class OpenRLHFGAE:
    """OpenRLHF's GAE (``get_advantages_and_returns``) with a controlled critic.

    Same contract as the verl integration's VerlGAE: ``critic="exact"``
    feeds the oracle's V^pi, ``critic="zero"`` feeds zeros; at OpenRLHF's
    default ``lambd=1`` even a perfect critic only sets the baseline. The
    pipeline batch-whitens GAE advantages (in float32 — see module note);
    ``no_std_norm=True`` keeps their mean-centering but skips the rescale.
    """

    gamma: float = 1.0
    lam: float = 1.0
    critic: str = "exact"
    no_std_norm: bool = False

    def __post_init__(self) -> None:
        if self.critic not in ("exact", "zero"):
            raise ValueError(f"critic must be 'exact' or 'zero', got {self.critic!r}")

    @property
    def name(self) -> str:
        return f"openrlhf_gae_{self.critic}_lam{self.lam:g}"

    def estimate(
        self, context: EstimatorContext
    ) -> tuple[tuple[float, ...], ...]:
        return _run_pipeline(
            context,
            estimator="gae",
            gamma=self.gamma,
            lam=self.lam,
            no_std_norm=self.no_std_norm,
            critic=self.critic,
        )
