"""Dynamic truncation selection of reasoning steps (paper Eq. 8) and token-mask emission (Eq. 9)."""


def select_steps_by_energy(strengths: list[float], energy_threshold: float) -> list[int]:
    """Minimal step set with cumulative strength >= p * total (paper Eq. 8).
    Rank by descending strength, breaking ties by index; return indices in original order.
    Nonempty inputs retain at least one step; non-positive total strength retains all steps."""
    if not strengths:
        return []

    total = sum(strengths)
    order = sorted(range(len(strengths)), key=lambda i: (-strengths[i], i))

    if total <= 0:  # degenerate: no signal, keep everything rather than drop the sample
        return list(range(len(strengths)))

    selected, accumulated = [], 0.0
    for index in order:
        selected.append(index)
        accumulated += strengths[index]
        if accumulated >= energy_threshold * total:
            break

    return sorted(selected)


def build_loss_mask(
    sequence_length: int, step_spans: list[tuple[int, int]], selected_steps: list[int]
) -> list[int]:
    """Eq. 9 token mask: supervise selected absolute [start, end) step spans; zero elsewhere."""
    mask = [0] * sequence_length
    for index in selected_steps:
        start, end = step_spans[index]
        for position in range(start, min(end, sequence_length)):
            mask[position] = 1
    return mask


def selection_stats(
    step_spans: list[tuple[int, int]], selected_steps: list[int]
) -> dict[str, float]:
    """Report step and token retention separately because step lengths differ."""
    total_steps = len(step_spans)
    total_tokens = sum(end - start for start, end in step_spans)
    kept_tokens = sum(step_spans[i][1] - step_spans[i][0] for i in selected_steps)
    step_retention = len(selected_steps) / total_steps if total_steps else 0.0
    token_retention = kept_tokens / total_tokens if total_tokens else 0.0
    return {
        "n_steps": total_steps,
        "n_steps_kept": len(selected_steps),
        "n_steps_dropped": total_steps - len(selected_steps),
        "step_retention": step_retention,
        "step_drop": 1.0 - step_retention if total_steps else 0.0,
        "n_tokens": total_tokens,
        "n_tokens_kept": kept_tokens,
        "n_tokens_dropped": total_tokens - kept_tokens,
        "token_retention": token_retention,
        "token_drop": 1.0 - token_retention if total_tokens else 0.0,
        "keeps_final_step": float(total_steps - 1 in selected_steps) if total_steps else 0.0,
    }
