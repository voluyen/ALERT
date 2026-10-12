"""Build weighted datasets using the same spectral selection and token IDs.
Allocation adds loss_weights while preserving loss_mask."""

import argparse
import json
import math
import statistics
from pathlib import Path

import pandas as pd
import yaml

from answergain.allocation.weights import (
    no_answer_upweight_step_weights,
    raw_step_weights,
    reverse_step_weights,
    shuffled_step_weights,
    step_weights,
)
from answergain.segment.sentence import record_step_spans
from answergain.selection.energy import build_loss_mask, select_steps_by_energy, selection_stats

# Stable controls share token-mass normalization to isolate the score mapping.
_STABLE_VARIANTS = {
    "answer-gain": lambda scores, lengths, temperature, interpolation, clip, epsilon, seed:
        step_weights(scores, lengths, temperature, interpolation, clip, epsilon),
    "answer-gain-lambda0": lambda scores, lengths, temperature, interpolation, clip, epsilon, seed:
        step_weights(scores, lengths, temperature, 0.0, clip, epsilon),
    "answer-gain-shuffled": lambda scores, lengths, temperature, interpolation, clip, epsilon, seed:
        shuffled_step_weights(scores, lengths, temperature, interpolation, clip, epsilon, seed),
    "answer-gain-reverse": lambda scores, lengths, temperature, interpolation, clip, epsilon, seed:
        reverse_step_weights(scores, lengths, temperature, interpolation, clip, epsilon),
}

_NO_ANSWER_VARIANT = "answer-gain-no-answer-upweight"
_GLOBAL_NORM_VARIANT = "answer-gain-global"
_NO_NORM_VARIANT = "answer-gain-none"


def build_example(
    record: dict,
    strengths: list[float],
    scores: list[float],
    threshold: float,
    variant: str,
    temperature: float,
    interpolation: float,
    clip: float,
    epsilon: float,
    seed: int = 42,
    answer_only: list[bool] | None = None,
    global_scale: float | None = None,
) -> tuple[dict, dict]:
    """Build one weighted record and its selection/allocation diagnostics."""
    step_spans = record_step_spans(record)
    if len(strengths) != len(step_spans) or len(scores) != len(step_spans):
        raise ValueError(
            f"record {record['id']}: steps={len(step_spans)}, strengths={len(strengths)}, "
            f"scores={len(scores)}"
        )
    selected = select_steps_by_energy(strengths, threshold)
    selected_scores = [scores[index] for index in selected]
    selected_lengths = [step_spans[index][1] - step_spans[index][0] for index in selected]
    if variant == _NO_ANSWER_VARIANT:
        if answer_only is None or len(answer_only) != len(step_spans):
            raise ValueError(f"record {record['id']}: {variant} needs one answer-only label per step")
        selected_weights = no_answer_upweight_step_weights(
            selected_scores,
            selected_lengths,
            [answer_only[index] for index in selected],
            temperature,
            interpolation,
            clip,
            epsilon,
        )
    elif variant in _STABLE_VARIANTS:
        # Offset the seed to avoid repeating a positional permutation across samples.
        selected_weights = _STABLE_VARIANTS[variant](
            selected_scores, selected_lengths, temperature, interpolation, clip, epsilon,
            seed + record["id"],
        )
    elif variant in (_GLOBAL_NORM_VARIANT, _NO_NORM_VARIANT):
        raw = raw_step_weights(
            selected_scores, selected_lengths, temperature, clip, epsilon,
        )
        if variant == _GLOBAL_NORM_VARIANT:
            if global_scale is None:
                raise ValueError("answer-gain-global needs a corpus global_scale")
            raw = [weight * global_scale for weight in raw]
        selected_weights = [(1.0 - interpolation) + interpolation * weight for weight in raw]
    else:
        raise ValueError(f"unknown allocation variant: {variant}")

    loss_mask = build_loss_mask(len(record["input_ids"]), step_spans, selected)
    loss_weights = [0.0] * len(loss_mask)
    for step_index, weight in zip(selected, selected_weights):
        start, end = step_spans[step_index]
        loss_weights[start:end] = [weight] * (end - start)

    # Unit stop-token weight preserves spectral supervision at lambda=0.
    for position in range(record["response_token_span"][1], len(loss_mask)):
        loss_mask[position] = 1
        loss_weights[position] = 1.0

    selected_token_mass = sum(selected_lengths)
    weighted_token_mass = sum(
        length * weight for length, weight in zip(selected_lengths, selected_weights)
    )
    stats = selection_stats(step_spans, selected)
    stats.update(
        {
            "selected_token_mass": selected_token_mass,
            "weighted_token_mass": weighted_token_mass,
            "weight_mass_ratio": weighted_token_mass / max(selected_token_mass, 1),
            "weight_min": min(selected_weights, default=1.0),
            "weight_max": max(selected_weights, default=1.0),
            "selected_weights": selected_weights,
            "selected_lengths": selected_lengths,
            "answer_only_steps": sum(answer_only[index] for index in selected)
            if answer_only is not None else 0,
            "answer_only_tokens": sum(
                length for index, length in zip(selected, selected_lengths) if answer_only[index]
            ) if answer_only is not None else 0,
        }
    )
    return {
        "id": record["id"],
        "input_ids": record["input_ids"],
        "loss_mask": loss_mask,
        "loss_weights": loss_weights,
    }, stats


def emit_dataset(
    records: list[dict],
    signals: dict[int, tuple[list[float], list[float]]],
    output_path: Path,
    threshold: float,
    variant: str,
    temperature: float,
    interpolation: float,
    clip: float,
    epsilon: float,
    seed: int = 42,
    answer_only: dict[int, list[bool]] | None = None,
) -> list[dict]:
    """Write one allocation arm and return per-record diagnostics."""
    global_scale = None
    if variant == _GLOBAL_NORM_VARIANT:
        # Compute the corpus factor over the full training set before lambda shrinkage.
        total_tokens = total_raw_mass = 0.0
        for record in records:
            strengths, scores = signals[record["id"]]
            spans = record_step_spans(record)
            selected = select_steps_by_energy(strengths, threshold)
            lengths = [spans[index][1] - spans[index][0] for index in selected]
            raw = raw_step_weights(
                [scores[index] for index in selected], lengths,
                temperature, clip, epsilon,
            )
            total_tokens += sum(lengths)
            total_raw_mass += sum(length * weight for length, weight in zip(lengths, raw))
        if total_raw_mass <= 0:
            raise ValueError("answer-gain-global has non-positive corpus raw token mass")
        global_scale = total_tokens / total_raw_mass
    all_stats = []
    with output_path.open("w") as handle:
        for record in records:
            strengths, scores = signals[record["id"]]
            example, stats = build_example(
                record, strengths, scores, threshold, variant, temperature,
                interpolation, clip, epsilon, seed,
                answer_only[record["id"]] if answer_only is not None else None,
                global_scale,
            )
            handle.write(json.dumps(example) + "\n")
            all_stats.append(stats)
    return all_stats


def summarize(stats: list[dict]) -> dict:
    """Corpus diagnostics needed to check the answer-gain allocation invariance claim."""
    weighted_steps = [
        (weight, length)
        for item in stats
        for weight, length in zip(item["selected_weights"], item["selected_lengths"])
    ]
    weights = sorted(weight for weight, _ in weighted_steps)
    total_mass = sum(weight * length for weight, length in weighted_steps)
    total_tokens = sum(length for _, length in weighted_steps)
    trace_budgets = [item["weight_mass_ratio"] for item in stats]
    top_count = math.ceil(0.1 * len(weighted_steps))
    top_mass = sum(
        weight * length
        for weight, length in sorted(weighted_steps, key=lambda pair: pair[0], reverse=True)[:top_count]
    )

    def quantile(fraction: float) -> float:
        if not weights:
            return 1.0
        position = fraction * (len(weights) - 1)
        lower = math.floor(position)
        upper = math.ceil(position)
        if lower == upper:
            return weights[lower]
        return weights[lower] * (upper - position) + weights[upper] * (position - lower)

    return {
        "samples": len(stats),
        "step_drop_mean": statistics.mean(item["step_drop"] for item in stats),
        "token_drop_mean": statistics.mean(item["token_drop"] for item in stats),
        "weight_mass_ratio_mean": statistics.mean(item["weight_mass_ratio"] for item in stats),
        "weight_mass_ratio_min": min(item["weight_mass_ratio"] for item in stats),
        "weight_mass_ratio_max": max(item["weight_mass_ratio"] for item in stats),
        "dataset_weight_mass_ratio": total_mass / max(total_tokens, 1e-12),
        "trace_budget_std": statistics.pstdev(trace_budgets),
        "weight_min": min(item["weight_min"] for item in stats),
        "weight_max": max(item["weight_max"] for item in stats),
        "weight_p10": quantile(0.1),
        "weight_median": quantile(0.5),
        "weight_p90": quantile(0.9),
        "steps_above_one_ratio": sum(weight > 1.0 for weight in weights) / max(len(weights), 1),
        "top_10pct_step_mass_ratio": top_mass / max(total_mass, 1e-12),
        "max_abs_mass_error": max(
            abs(item["weighted_token_mass"] - item["selected_token_mass"]) for item in stats
        ),
        "answer_only_steps": sum(item["answer_only_steps"] for item in stats),
        "answer_only_tokens": sum(item["answer_only_tokens"] for item in stats),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", help="optional yaml base; CLI flags override it")
    parser.add_argument("--data-path", help="segmented JSONL from answergain.data.prepare")
    parser.add_argument("--strengths", help="spectral-strengths.parquet from answergain.signals.capture")
    parser.add_argument("--energy-threshold-p", type=float, help="the shared spectral gate threshold")
    parser.add_argument("--temperature", type=float, help="score temperature tau")
    parser.add_argument("--interpolation", type=float, help="answer-gain allocation lambda in [0, 1]")
    parser.add_argument("--clip", type=float, help="answer-gain allocation standardized-score clip bound")
    parser.add_argument("--epsilon", type=float, help="standardization stabilizer")
    parser.add_argument("--seed", type=int, help="base seed for the shuffled-score control")
    parser.add_argument(
        "--variants",
        help="comma-separated answer-gain variants: answer-gain, answer-gain-global, answer-gain-none, "
        "answer-gain-lambda0, answer-gain-shuffled, answer-gain-reverse, answer-gain-no-answer-upweight",
    )
    parser.add_argument(
        "--answer-only-labels",
        help=f"{_NO_ANSWER_VARIANT}: JSON mapping record id to one boolean per step",
    )
    parser.add_argument(
        "--check-mass", action="store_true",
        help="fail unless every answer-gain* variant keeps the selected token mass "
        "(mean weighted/selected mass prints as 1.000000)",
    )
    parser.add_argument(
        "--output-name",
        help="with a single variant: write train-<output-name>.jsonl and "
        "<output-name>-selection-stats.json instead of train-<variant>.jsonl and "
        "allocation-selection-stats.json, so several arms can share one data dir",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    config = yaml.safe_load(Path(args.config).read_text()) if args.config else {}
    overrides = {
        "data_path": args.data_path,
        "strengths": args.strengths,
        "energy_threshold_p": args.energy_threshold_p,
        "temperature": args.temperature,
        "interpolation": args.interpolation,
        "clip": args.clip,
        "epsilon": args.epsilon,
        "seed": args.seed,
        "variants": args.variants,
        "answer_only_labels": args.answer_only_labels,
    }
    config.update({key: value for key, value in overrides.items() if value is not None})
    config.setdefault("energy_threshold_p", 0.95)
    config.setdefault("temperature", 1.0)
    config.setdefault("interpolation", 1.0)
    config.setdefault("clip", 2.0)
    config.setdefault("epsilon", 1e-8)
    config.setdefault("seed", 42)
    config.setdefault("variants", "answer-gain")
    required = [key for key in ("data_path", "strengths") if key not in config]
    if required:
        parser.error(f"missing required settings: {required}")

    with open(config["data_path"]) as handle:
        records = [json.loads(line) for line in handle]
    frame = pd.read_parquet(config["strengths"])
    if "step_entropies" not in frame.columns:
        raise ValueError(
            f"{config['strengths']} has no step_entropies; rerun answergain.signals.capture after the allocation update"
        )
    signals = {
        int(row.id): (list(row.step_strengths), list(row.step_entropies))
        for row in frame.itertuples()
    }
    missing = [record["id"] for record in records if record["id"] not in signals]
    if missing:
        raise ValueError(f"spectral signals missing for {len(missing)} records (first: {missing[0]})")

    variants = config["variants"].split(",")
    if args.output_name and len(variants) != 1:
        parser.error("--output-name needs exactly one variant")
    answer_only = None
    if _NO_ANSWER_VARIANT in variants:
        if "answer_only_labels" not in config:
            parser.error(f"{_NO_ANSWER_VARIANT} needs --answer-only-labels")
        raw_labels = json.loads(Path(config["answer_only_labels"]).read_text())
        answer_only = {int(record_id): [bool(value) for value in labels]
                       for record_id, labels in raw_labels.items()}
        missing_labels = [record["id"] for record in records if record["id"] not in answer_only]
        if missing_labels:
            raise ValueError(
                f"answer-only labels missing for {len(missing_labels)} records (first: {missing_labels[0]})"
            )
    data_dir = Path(config["data_path"]).parent
    summaries = {}
    for variant in variants:
        output_path = data_dir / f"train-{args.output_name or variant}.jsonl"
        stats = emit_dataset(
            records, signals, output_path, config["energy_threshold_p"], variant,
            config["temperature"], config["interpolation"], config["clip"], config["epsilon"],
            config["seed"],
            answer_only,
        )
        summaries[variant] = summarize(stats)
        print(
            f"{variant}: {len(stats)} records -> {output_path}; "
            f"mean weighted/selected mass={summaries[variant]['weight_mass_ratio_mean']:.6f}"
        )
        if args.check_mass and variant in (_GLOBAL_NORM_VARIANT, _NO_NORM_VARIANT):
            # These variants do not constrain per-sample mass.
            if variant == _GLOBAL_NORM_VARIANT and abs(summaries[variant]["dataset_weight_mass_ratio"] - 1.0) > 1e-5:
                raise SystemExit(f"{variant}: corpus token mass not preserved")
        elif args.check_mass and (variant.startswith("answer-gain")):
            if summaries[variant]["max_abs_mass_error"] > 1e-5:
                error = summaries[variant]["max_abs_mass_error"]
                raise SystemExit(f"{variant}: per-trace token mass not preserved (max abs error={error:.3g})")

    summary_path = data_dir / (
        f"{args.output_name}-selection-stats.json" if args.output_name else "allocation-selection-stats.json"
    )
    summary_path.write_text(json.dumps({"config": config, "variants": summaries}, indent=2))
    print(f"diagnostics -> {summary_path}")


if __name__ == "__main__":
    main()
