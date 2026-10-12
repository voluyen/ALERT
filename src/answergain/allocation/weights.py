"""Step coefficients for answer-gain allocation.
Scores are standardized, clipped and exponentially tilted with token-mass normalization.
Allocation recovers spectral SFT at interpolation=0."""

import math
import random
import sys


def _validate(scores: list[float], lengths: list[int], temperature: float) -> None:
    if len(scores) != len(lengths):
        raise ValueError("scores and lengths must have the same number of steps")
    if not scores:
        return
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    if any(length <= 0 for length in lengths):
        raise ValueError("selected step lengths must be positive")
    if not all(math.isfinite(value) for value in scores):
        raise ValueError("scores must be finite")






def step_weights(
    scores: list[float],
    lengths: list[int],
    temperature: float = 1.0,
    interpolation: float = 1.0,
    clip: float = 2.0,
    epsilon: float = 1e-8,
) -> list[float]:
    """Paper Eqs. 3-6: clipped standardized score, normalized by token mass."""
    if not 0.0 <= interpolation <= 1.0:
        raise ValueError("interpolation must be in [0, 1]")
    _validate(scores, lengths, temperature)
    if clip < 0:
        raise ValueError("clip must be non-negative")
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")
    if not scores:
        return []
    mean = sum(scores) / len(scores)
    variance = sum((score - mean) ** 2 for score in scores) / len(scores)
    std = math.sqrt(variance)
    z_scores = [0.0 if std < epsilon else (score - mean) / (std + epsilon) for score in scores]
    scaled = [max(-clip, min(clip, z_score)) / temperature for z_score in z_scores]
    # Within-trace normalization cancels the common max shift.
    maximum = max(scaled)
    raw = [max(math.exp(value - maximum), sys.float_info.min) for value in scaled]
    token_mass = sum(lengths)
    weighted_mass = sum(length * weight for length, weight in zip(lengths, raw))
    normalized = [weight * token_mass / weighted_mass for weight in raw]
    return [(1.0 - interpolation) + interpolation * weight for weight in normalized]


def raw_step_weights(
    scores: list[float],
    lengths: list[int],
    temperature: float = 1.0,
    clip: float = 2.0,
    epsilon: float = 1e-8,
) -> list[float]:
    """Positive within-trace-standardized ratios before token-mass normalization.
    Callers choose global normalization or no mass correction."""
    _validate(scores, lengths, temperature)
    if clip < 0:
        raise ValueError("clip must be non-negative")
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")
    if not scores:
        return []

    mean = sum(scores) / len(scores)
    variance = sum((score - mean) ** 2 for score in scores) / len(scores)
    std = math.sqrt(variance)
    z_scores = [0.0 if std < epsilon else (score - mean) / (std + epsilon) for score in scores]
    scaled = [max(-clip, min(clip, z_score)) / temperature for z_score in z_scores]
    # Preserve literal r_ik: a per-trace max shift would change the global/no-norm ablations.
    if max(scaled) > math.log(sys.float_info.max):
        raise ValueError("temperature is too small to represent unnormalized allocation ratios")
    return [math.exp(value) for value in scaled]


def no_answer_upweight_step_weights(
    scores: list[float],
    lengths: list[int],
    answer_only: list[bool],
    temperature: float = 1.0,
    interpolation: float = 1.0,
    clip: float = 2.0,
    epsilon: float = 1e-8,
) -> list[float]:
    """Fix answer-only steps at one; normalize eligible steps independently to preserve trace mass."""
    _validate(scores, lengths, temperature)
    if len(answer_only) != len(scores):
        raise ValueError("answer_only and scores must have the same number of steps")
    eligible = [index for index, is_answer_only in enumerate(answer_only) if not is_answer_only]
    if not eligible:
        return [1.0] * len(scores)

    eligible_weights = step_weights(
        [scores[index] for index in eligible],
        [lengths[index] for index in eligible],
        temperature,
        interpolation,
        clip,
        epsilon,
    )
    weights = [1.0] * len(scores)
    for index, weight in zip(eligible, eligible_weights):
        weights[index] = weight
    return weights


def shuffled_step_weights(
    scores: list[float],
    lengths: list[int],
    temperature: float = 1.0,
    interpolation: float = 1.0,
    clip: float = 2.0,
    epsilon: float = 1e-8,
    seed: int = 0,
) -> list[float]:
    """Shuffle scores before Eq. 13-16 weighting to test score-to-step correspondence.
    Renormalizing the shuffled score-length pairing preserves token mass, not coefficient values."""
    shuffled = list(scores)
    random.Random(seed).shuffle(shuffled)
    return step_weights(shuffled, lengths, temperature, interpolation, clip, epsilon)


def reverse_step_weights(
    scores: list[float],
    lengths: list[int],
    temperature: float = 1.0,
    interpolation: float = 1.0,
    clip: float = 2.0,
    epsilon: float = 1e-8,
) -> list[float]:
    """Negate score before standardizing to test the opposite weighting direction.
    z(-x) = -z(x); token-mass normalization is preserved."""
    return step_weights(
        [-score for score in scores], lengths, temperature, interpolation, clip, epsilon
    )
