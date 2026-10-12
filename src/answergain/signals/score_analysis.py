"""Matched descriptive analyses for answer gain versus step predictability."""

import argparse
import csv
import json
import math
import statistics
from pathlib import Path


def ranks(values: list[float]) -> list[float]:
    """Average ranks for ties, equivalent to scipy.stats.rankdata(method='average')."""
    order = sorted(range(len(values)), key=values.__getitem__)
    output = [0.0] * len(values)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values[order[end]] == values[order[start]]:
            end += 1
        rank = (start + end - 1) / 2.0
        for index in order[start:end]:
            output[index] = rank
        start = end
    return output


def spearman(left: list[float], right: list[float]) -> float:
    if len(left) != len(right):
        raise ValueError("score lists must have equal length")
    if len(left) < 2:
        return 0.0
    x, y = ranks(left), ranks(right)
    mean_x, mean_y = statistics.mean(x), statistics.mean(y)
    numerator = sum((a - mean_x) * (b - mean_y) for a, b in zip(x, y))
    denominator = math.sqrt(
        sum((a - mean_x) ** 2 for a in x) * sum((b - mean_y) ** 2 for b in y)
    )
    return numerator / denominator if denominator else 0.0


def top_indices(values: list[float], fraction: float = 0.1, largest: bool = True) -> set[int]:
    count = max(1, math.ceil(fraction * len(values)))
    return set(sorted(range(len(values)), key=values.__getitem__, reverse=largest)[:count])


def analyze_trace(gain: list[float], predictability: list[float]) -> dict[str, float]:
    if len(gain) != len(predictability):
        raise ValueError("answer-gain and predictability step counts differ")
    gain_high = top_indices(gain)
    gain_low = top_indices(gain, largest=False)
    pred_high = top_indices(predictability)
    pred_low = top_indices(predictability, largest=False)
    return {
        "n_steps": len(gain),
        "spearman": spearman(gain, predictability),
        "top10_overlap": len(gain_high & pred_high) / max(len(gain_high), 1),
        "high_gain_low_predictability": len(gain_high & pred_low) / max(len(gain), 1),
        "low_gain_high_predictability": len(gain_low & pred_high) / max(len(gain), 1),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--answer-gain", required=True)
    parser.add_argument("--predictability", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    gains = json.loads(Path(args.answer_gain).read_text())
    predictability = json.loads(Path(args.predictability).read_text())
    if gains.keys() != predictability.keys():
        missing_gain = sorted(predictability.keys() - gains.keys())
        missing_pred = sorted(gains.keys() - predictability.keys())
        raise ValueError(f"score record ids differ: missing gain={missing_gain[:3]}, missing pred={missing_pred[:3]}")

    rows, all_gain, all_predictability = [], [], []
    for record_id in gains:
        row = {"id": record_id, **analyze_trace(gains[record_id], predictability[record_id])}
        rows.append(row)
        all_gain.extend(gains[record_id])
        all_predictability.extend(predictability[record_id])

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "score_analysis.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        key: {
            "median": statistics.median(row[key] for row in rows),
            "q1": statistics.quantiles([row[key] for row in rows], n=4)[0],
            "q3": statistics.quantiles([row[key] for row in rows], n=4)[2],
        }
        for key in ("spearman", "top10_overlap", "high_gain_low_predictability", "low_gain_high_predictability")
    }
    (output_dir / "score_analysis_summary.json").write_text(json.dumps(summary, indent=2))

    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(6.0, 4.8))
    histogram = axis.hist2d(all_predictability, all_gain, bins=80, cmap="viridis")
    figure.colorbar(histogram[3], ax=axis, label="step count")
    axis.set_xlabel("Step predictability (mean log probability)")
    axis.set_ylabel("Answer-likelihood gain")
    figure.tight_layout()
    figure.savefig(output_dir / "score_correlation.pdf")
    plt.close(figure)
    print(f"wrote score analysis for {len(rows)} traces -> {output_dir}")


if __name__ == "__main__":
    main()
