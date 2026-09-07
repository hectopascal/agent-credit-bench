"""Stable core arithmetic; public results must fit in a finite float."""

import math
from collections.abc import Sequence
from fractions import Fraction


def finite_float(value: Fraction | float) -> float:
    try:
        result = float(value)
    except OverflowError:
        raise ValueError("numeric result exceeds the finite float range") from None
    if not math.isfinite(result):
        raise ValueError("numeric result exceeds the finite float range")
    return result


def centered_values(values: Sequence[float]) -> list[float]:
    """Subtract a group mean without rounding away a large shared offset."""
    reference = min(values, key=abs)
    if min(values) <= 0 <= max(values):
        reference = 0.0
    offsets = [v - reference for v in values]
    try:
        mean = math.fsum(offsets) / len(offsets)
        return [finite_float(v - mean) for v in offsets]
    except (OverflowError, ValueError):
        # A sum or offset can overflow although centered results still fit.
        exact = [Fraction(v) for v in values]
        mean_exact = sum(exact) / len(exact)
        return [finite_float(v - mean_exact) for v in exact]
