"""Deterministic transforms of per-step score JSON artifacts for matched ablations."""

import argparse
import json
import math
import random
from pathlib import Path


def transform_scores(
    scores: dict[str, list[float]], mode: str, seed: int = 42
) -> tuple[dict[str, list[float]], dict[str, list[int]] | None]:
    transformed, permutations = {}, {} if mode == "shuffle" else None
    for record_id, values in scores.items():
        if not all(math.isfinite(value) for value in values):
            raise ValueError(f"record {record_id} contains a non-finite score")
        if mode == "negate":
            transformed[record_id] = [-value for value in values]
        elif mode == "shuffle":
            order = list(range(len(values)))
            random.Random(seed + int(record_id)).shuffle(order)
            transformed[record_id] = [values[index] for index in order]
            permutations[record_id] = order
        else:
            raise ValueError(f"unknown score transform: {mode}")
    return transformed, permutations


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--mode", required=True, choices=("negate", "shuffle"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--permutations-output", help="shuffle only: record the exact index permutations")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    scores = json.loads(Path(args.input).read_text())
    transformed, permutations = transform_scores(scores, args.mode, args.seed)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(transformed))
    if args.mode == "shuffle":
        if not args.permutations_output:
            raise ValueError("--permutations-output is required for shuffle")
        permutation_path = Path(args.permutations_output)
        permutation_path.parent.mkdir(parents=True, exist_ok=True)
        permutation_path.write_text(json.dumps(permutations))
        print(f"recorded shuffle permutations -> {permutation_path}")
    print(f"wrote {args.mode} scores for {len(transformed)} traces -> {output}")


if __name__ == "__main__":
    main()
