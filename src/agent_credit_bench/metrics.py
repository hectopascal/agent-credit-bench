"""Estimator metrics — identification family (plan.md §10).

All functions take flat, aligned sequences: one entry per sampled step.
Gradient-family metrics live in gradients.py.
"""

import math
from collections.abc import Hashable, Sequence
from dataclasses import dataclass


def _check_aligned(estimated: Sequence[float], exact: Sequence[float]) -> None:
    if len(estimated) != len(exact):
        raise ValueError(
            f"length mismatch: {len(estimated)} estimated vs {len(exact)} exact"
        )
    if not estimated:
        raise ValueError("empty input")


def rmse(estimated: Sequence[float], exact: Sequence[float]) -> float:
    """Root mean squared error (plan.md §10.1)."""
    _check_aligned(estimated, exact)
    return math.sqrt(
        sum((e - a) ** 2 for e, a in zip(estimated, exact, strict=True))
        / len(estimated)
    )


def _average_ranks(values: Sequence[float]) -> list[float]:
    """Ranks starting at 1; ties receive the mean of their positions."""
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        mean_rank = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks[order[k]] = mean_rank
        i = j + 1
    return ranks


def spearman(estimated: Sequence[float], exact: Sequence[float]) -> float:
    """Spearman rank correlation (plan.md §10.2).

    Returns 0.0 when either input has zero rank variance (all values equal),
    where the correlation is undefined.
    """
    _check_aligned(estimated, exact)
    rx = _average_ranks(estimated)
    ry = _average_ranks(exact)
    n = len(rx)
    mx = sum(rx) / n
    my = sum(ry) / n
    cov = sum((x - mx) * (y - my) for x, y in zip(rx, ry, strict=True))
    vx = sum((x - mx) ** 2 for x in rx)
    vy = sum((y - my) ** 2 for y in ry)
    if vx == 0 or vy == 0:
        return 0.0
    return cov / math.sqrt(vx * vy)


def _sign(x: float) -> int:
    return (x > 0) - (x < 0)


@dataclass(frozen=True)
class SignAccuracy:
    accuracy: float | None  # None when no step had |exact| >= epsilon
    num_scored: int
    num_excluded: int


def sign_accuracy(
    estimated: Sequence[float],
    exact: Sequence[float],
    epsilon: float = 1e-8,
) -> SignAccuracy:
    """Sign agreement on actions with non-negligible exact advantage (§10.3)."""
    _check_aligned(estimated, exact)
    scored = [
        (e, a) for e, a in zip(estimated, exact, strict=True) if abs(a) >= epsilon
    ]
    excluded = len(estimated) - len(scored)
    if not scored:
        return SignAccuracy(accuracy=None, num_scored=0, num_excluded=excluded)
    matches = sum(_sign(e) == _sign(a) for e, a in scored)
    return SignAccuracy(
        accuracy=matches / len(scored),
        num_scored=len(scored),
        num_excluded=excluded,
    )


@dataclass(frozen=True)
class Leakage:
    ratio: float
    mean_abs_zero_credit: float | None  # None when no zero-advantage steps
    num_zero_advantage: int


def leakage(
    estimated: Sequence[float],
    exact: Sequence[float],
    epsilon: float = 1e-8,
) -> Leakage:
    """Credit assigned to actions whose exact advantage is zero (§10.4)."""
    _check_aligned(estimated, exact)
    zero_credit = [
        abs(e) for e, a in zip(estimated, exact, strict=True) if abs(a) < epsilon
    ]
    total = sum(abs(e) for e in estimated)
    ratio = sum(zero_credit) / (total + epsilon)
    mean_abs = sum(zero_credit) / len(zero_credit) if zero_credit else None
    return Leakage(
        ratio=ratio,
        mean_abs_zero_credit=mean_abs,
        num_zero_advantage=len(zero_credit),
    )


@dataclass(frozen=True)
class CenteredRmse:
    value: float
    multi_visit_fraction: float  # steps at (t, s) visited >= 2 times


def centered_rmse(
    estimated: Sequence[float],
    exact: Sequence[float],
    keys: Sequence[Hashable],
) -> CenteredRmse:
    """Baseline-shift-invariant error (plan.md §10.6).

    Distance to the nearest member of the equivalence class
    A + b(t, s): residuals are centered per key before the RMS. Steps at a
    key visited only once center to exactly zero, which flatters the
    estimator — hence the multi_visit_fraction coverage report.
    """
    _check_aligned(estimated, exact)
    if len(keys) != len(estimated):
        raise ValueError("keys must align with estimated/exact")

    residuals: dict[Hashable, list[float]] = {}
    for e, a, key in zip(estimated, exact, keys, strict=True):
        residuals.setdefault(key, []).append(e - a)

    total = 0.0
    multi_visit = 0
    for group in residuals.values():
        mean = sum(group) / len(group)
        total += sum((r - mean) ** 2 for r in group)
        if len(group) >= 2:
            multi_visit += len(group)
    return CenteredRmse(
        value=math.sqrt(total / len(estimated)),
        multi_visit_fraction=multi_visit / len(estimated),
    )


@dataclass(frozen=True)
class TurnStats:
    bias: float  # mean(estimated - exact)
    variance: float  # population variance of estimated
    count: int


def per_turn_stats(
    timesteps: Sequence[int],
    estimated: Sequence[float],
    exact: Sequence[float],
) -> dict[int, TurnStats]:
    """Per-timestep bias and variance (plan.md §10.5)."""
    _check_aligned(estimated, exact)
    if len(timesteps) != len(estimated):
        raise ValueError("timesteps must align with estimated/exact")

    grouped: dict[int, list[tuple[float, float]]] = {}
    for t, e, a in zip(timesteps, estimated, exact, strict=True):
        grouped.setdefault(t, []).append((e, a))

    stats = {}
    for t, pairs in sorted(grouped.items()):
        n = len(pairs)
        bias = sum(e - a for e, a in pairs) / n
        mean_e = sum(e for e, _ in pairs) / n
        variance = sum((e - mean_e) ** 2 for e, _ in pairs) / n
        stats[t] = TurnStats(bias=bias, variance=variance, count=n)
    return stats
