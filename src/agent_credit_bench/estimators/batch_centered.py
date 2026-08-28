"""Group-relative trajectory baseline with leave-one-out centering (plan.md §9.3).

    credit_i_t = return_i - mean(return_j for j != i)

A simple stand-in for group-relative baselines; not labeled as a complete
GRPO implementation.

Single-trajectory batches: the leave-one-out mean is undefined.
TODO(yan): decide the behavior (raise ValueError? credit 0?), document it in
this docstring, then unskip and pin it in
tests/test_estimators.py::test_batch_centered_single_trajectory.
"""

from dataclasses import dataclass

from agent_credit_bench.estimators.base import EstimatorContext


@dataclass(frozen=True)
class BatchCenteredBroadcast:
    name: str = "batch_centered_broadcast"

    def estimate(
        self, context: EstimatorContext
    ) -> tuple[tuple[float, ...], ...]:
        # TODO(yan): leave-one-out centered return, broadcast to every step.
        raise NotImplementedError("M2: BatchCenteredBroadcast not implemented yet")
