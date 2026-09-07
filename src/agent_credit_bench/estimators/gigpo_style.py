"""GiGPO-style hierarchical group baseline (plan.md §16, item 1).

Reimplements the *mechanism* of GiGPO (Feng et al., 2025) — hierarchical
grouping — without claiming the paper's exact constants, hence "-style":

  - episode level: z-normalized trajectory return over the batch
    (the GRPO-style group advantage);
  - step level: steps are anchored by identical environment state; within
    each anchor group the return-to-go from that step is z-normalized;
  - credit = episode_weight * episode_advantage
           + step_weight * step_advantage.

Anchor grouping is by state only (as in the paper: identical states met at
possibly different times form one group), with gamma = 1 returns-to-go. A
group with zero variance contributes zero step advantage.
"""

import statistics
from dataclasses import dataclass
from fractions import Fraction

from agent_credit_bench._numerics import centered_values, finite_float
from agent_credit_bench.estimators.base import EstimatorContext


def _z_scores(values: list[float], epsilon: float) -> list[float]:
    std = statistics.pstdev(values)
    return [v / (std + epsilon) for v in centered_values(values)]


@dataclass(frozen=True)
class GiGPOStyle:
    episode_weight: float = 1.0
    step_weight: float = 1.0
    epsilon: float = 1e-4
    name: str = "gigpo_style"

    def estimate(
        self, context: EstimatorContext
    ) -> tuple[tuple[float, ...], ...]:
        trajectories = context.trajectories
        episode = _z_scores(
            [t.total_return for t in trajectories], self.epsilon
        )

        # Returns-to-go, and anchor groups keyed by state.
        returns_to_go: list[list[float]] = []
        groups: dict[object, list[tuple[int, int]]] = {}
        for i, trajectory in enumerate(trajectories):
            suffix = Fraction(0)
            rtg = [0.0] * len(trajectory.steps)
            for t in range(len(trajectory.steps) - 1, -1, -1):
                suffix += Fraction(trajectory.steps[t].reward)
                rtg[t] = finite_float(suffix)
            returns_to_go.append(rtg)
            for t, step in enumerate(trajectory.steps):
                groups.setdefault(step.state, []).append((i, t))

        step_advantage = [
            [0.0] * len(trajectory.steps) for trajectory in trajectories
        ]
        for members in groups.values():
            scores = _z_scores(
                [returns_to_go[i][t] for i, t in members], self.epsilon
            )
            for (i, t), score in zip(members, scores, strict=True):
                step_advantage[i][t] = score

        return tuple(
            tuple(
                self.episode_weight * episode[i] + self.step_weight * adv
                for adv in step_advantage[i]
            )
            for i in range(len(trajectories))
        )
