"""Summarize E0-E10 answer-gain ablations, benchmark results, and weight diagnostics."""

import argparse
import csv
import json
import statistics
from pathlib import Path

BENCHMARKS = ["aime24", "aime25", "amc12", "math500"]
METRICS = ["pass@1", "pass@3"]
ARMS = {
    "e0-uniform": "E0 Uniform SFT",
    "e1-alg": "E1 ALG",
    "e2-predictability-easy": "E2 Predictability-Easy",
    "e3-predictability-hard": "E3 Predictability-Hard",
    "e4-random-assignment": "E4 Random Assignment",
    "e5-alg-no-answer-upweight": "E5 ALG-NoAnswerUpweight",
    "e6-alg-lambda1": "E6 ALG lambda=1",
    "e7-alg-tau1": "E7 ALG tau=1",
    "e8-alg-tau4": "E8 ALG tau=4",
    "e9-alg-global-norm": "E9 ALG global norm",
    "e10-alg-no-norm": "E10 ALG no norm",
}
MAIN_ARMS = list(ARMS)[:6]
# (parameter, value, arm); E0 is lambda=0 and E1 is the default of both sweeps.
SENSITIVITY = [
    ("lambda", "0", "e0-uniform"), ("lambda", "0.5", "e1-alg"), ("lambda", "1", "e6-alg-lambda1"),
    ("tau", "1", "e7-alg-tau1"), ("tau", "2", "e1-alg"), ("tau", "4", "e8-alg-tau4"),
]
SEEDS = (42, 123, 456)


def run_tag(arm: str, track: str, seed: int) -> str:
    return f"{arm}-{track}" if seed == 42 else f"{arm}-{track}-s{seed}"


def load_run(results_dir: Path, tag: str) -> dict[str, dict] | None:
    path = results_dir / tag / "summary.json"
    if not path.exists():
        return None
    return {row["benchmark"]: row for row in json.loads(path.read_text())}


def macro(rows: dict[str, dict] | None, metric: str) -> float | None:
    """Mean over the four benchmarks; None unless all of them were evaluated."""
    if rows is None or any(name not in rows or metric not in rows[name] for name in BENCHMARKS):
        return None
    return sum(rows[name][metric] for name in BENCHMARKS) / len(BENCHMARKS)


def weight_stats(data_dir: Path, arm: str) -> dict | None:
    path = data_dir / f"{arm}-selection-stats.json"
    if not path.exists():
        return None
    variants = json.loads(path.read_text()).get("variants", {})
    return next(iter(variants.values()), None)


def pct(value: float | None) -> str:
    return "-" if value is None else f"{value:.2%}"


def number(value: float | None, digits: int = 3) -> str:
    return "-" if value is None else f"{value:.{digits}f}"


def mean_std(values: list[float]) -> str:
    if not values:
        return "-"
    spread = statistics.stdev(values) if len(values) > 1 else 0.0
    return f"{statistics.mean(values):.2%} ± {spread:.2%} (n={len(values)})"


def read_status(path: Path | None) -> list[list[str]]:
    if path is None or not path.exists():
        return []
    return [line.rstrip("\n").split("\t") for line in path.read_text().splitlines() if line.strip()]


def build_report(
    results_dir: Path, data_dir: Path, track: str, status: list[list[str]],
) -> tuple[str, list[dict], list[dict]]:
    runs = {(arm, seed): load_run(results_dir, run_tag(arm, track, seed)) for arm in ARMS for seed in SEEDS}
    lines = ["# Answer-gain ablation summary", "",
             f"Track `{track}`, results from `{results_dir}`. `-` means not run / not finished.", ""]

    lines += ["## Main ablation (training seed 42)", "",
              "| Method | AIME24 p@1 | AIME25 p@1 | AMC12 p@1 | MATH500 p@1 | Avg p@1 | Avg p@3 |",
              "|---|---:|---:|---:|---:|---:|---:|"]
    for arm in MAIN_ARMS:
        rows = runs[(arm, 42)]
        cells = [pct(rows[name]["pass@1"]) if rows and name in rows else "-" for name in BENCHMARKS]
        lines.append(f"| {ARMS[arm]} | " + " | ".join(cells)
                     + f" | {pct(macro(rows, 'pass@1'))} | {pct(macro(rows, 'pass@3'))} |")

    lines += ["", "## Hyperparameter sensitivity (seed 42)", "",
              "| Parameter | Value | Arm | Avg p@1 | Avg p@3 | w P10 | w median | w P90 | steps w>1 | top-10% mass |",
              "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|"]
    for parameter, value, arm in SENSITIVITY:
        rows, stats = runs[(arm, 42)], weight_stats(data_dir, arm) or {}
        lines.append(
            f"| {parameter} | {value} | {ARMS[arm]} | {pct(macro(rows, 'pass@1'))} | {pct(macro(rows, 'pass@3'))} | "
            f"{number(stats.get('weight_p10'))} | {number(stats.get('weight_median'))} | "
            f"{number(stats.get('weight_p90'))} | {pct(stats.get('steps_above_one_ratio'))} | "
            f"{pct(stats.get('top_10pct_step_mass_ratio'))} |"
        )

    lines += ["", "## Token-mass normalization (seed 42, tau=1, lambda=0.5)", "",
              "Each benchmark cell is `pass@1 / pass@3`.", "",
              "| Method | Normalization | AIME24 | AIME25 | AMC12 | MATH500 | Avg p@1 | Avg p@3 | "
              "Dataset mass | Trace-budget std | w P10 | w median | w P90 |",
              "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for arm, normalization in [("e7-alg-tau1", "per-trace"),
                               ("e9-alg-global-norm", "global corpus"),
                               ("e10-alg-no-norm", "none")]:
        rows, stats = runs[(arm, 42)], weight_stats(data_dir, arm) or {}
        benchmark_scores = [
            f"{rows[name]['pass@1']:.2%} / {rows[name]['pass@3']:.2%}" if rows and name in rows else "-"
            for name in BENCHMARKS
        ]
        lines.append(
            f"| {ARMS[arm]} | {normalization} | " + " | ".join(benchmark_scores)
            + f" | {pct(macro(rows, 'pass@1'))} | "
            f"{pct(macro(rows, 'pass@3'))} | {number(stats.get('dataset_weight_mass_ratio'))} | "
            f"{number(stats.get('trace_budget_std'))} | {number(stats.get('weight_p10'))} | "
            f"{number(stats.get('weight_median'))} | {number(stats.get('weight_p90'))} |"
        )

    seed_rows = []
    lines += ["", "## Training-seed robustness (mean ± std over finished seeds)", "",
              "| Method | Avg p@1 | Avg p@3 | Seeds finished |", "|---|---:|---:|---|"]
    for arm in ARMS:
        done = [seed for seed in SEEDS if macro(runs[(arm, seed)], "pass@1") is not None]
        if len(done) < 2:
            continue
        p1 = [macro(runs[(arm, seed)], "pass@1") for seed in done]
        p3 = [macro(runs[(arm, seed)], "pass@3") for seed in done if macro(runs[(arm, seed)], "pass@3") is not None]
        lines.append(f"| {ARMS[arm]} | {mean_std(p1)} | {mean_std(p3)} | {', '.join(map(str, done))} |")
        seed_rows.append({"arm": arm, "n_seeds": len(done), "seeds": " ".join(map(str, done)),
                          "avg_pass@1_mean": statistics.mean(p1),
                          "avg_pass@1_std": statistics.stdev(p1),
                          "avg_pass@3_mean": statistics.mean(p3) if p3 else "",
                          "avg_pass@3_std": statistics.stdev(p3) if len(p3) > 1 else ""})
    if not seed_rows:
        lines.append("| (needs at least two finished seeds) | - | - | - |")

    missing = [run_tag(arm, track, 42) for arm in ARMS if runs[(arm, 42)] is None]
    lines += ["", "## Missing seed-42 runs", "", ", ".join(f"`{tag}`" for tag in missing) if missing else "none"]

    if status:
        lines += ["", "## Run status (this launcher invocation)", "",
                  "| Arm | Stages | GPU | Status | Seconds | Log |", "|---|---|---:|---|---:|---|"]
        lines += ["| " + " | ".join(row[:6] + [""] * (6 - len(row))) + " |" for row in status]

    per_benchmark = []
    for (arm, seed), rows in runs.items():
        for name in BENCHMARKS:
            if rows and name in rows:
                row = rows[name]
                per_benchmark.append({
                    "run": run_tag(arm, track, seed), "arm": arm, "seed": seed, "benchmark": name,
                    "pass@1": row.get("pass@1"), "pass@3": row.get("pass@3"),
                    "length": row.get("length"), "truncation_rate": row.get("truncation_rate"),
                    "no_boxed_answer_rate": row.get("no_boxed_answer_rate"),
                })
    return "\n".join(lines) + "\n", per_benchmark, seed_rows


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--out-dir", default="experiments/answer_gain/results")
    parser.add_argument("--track", default="r1-qwen-1.5b-palign")
    parser.add_argument("--data-dir", help="default: data/<track>")
    parser.add_argument("--status-file", help="TSV written by project_commands_ablation_answer_gain.sh")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    report, per_benchmark, seed_rows = build_report(
        Path(args.results_dir), Path(args.data_dir or f"data/{args.track}"), args.track,
        read_status(Path(args.status_file) if args.status_file else None),
    )
    (out_dir / "ablation_summary.md").write_text(report)
    write_csv(out_dir / "per_benchmark.csv", per_benchmark,
              ["run", "arm", "seed", "benchmark", "pass@1", "pass@3", "length",
               "truncation_rate", "no_boxed_answer_rate"])
    write_csv(out_dir / "seed_summary.csv", seed_rows,
              ["arm", "n_seeds", "seeds", "avg_pass@1_mean", "avg_pass@1_std",
               "avg_pass@3_mean", "avg_pass@3_std"])
    data_dir = Path(args.data_dir or f"data/{args.track}")
    diagnostics = []
    for arm, normalization in [("e7-alg-tau1", "per-trace"),
                               ("e9-alg-global-norm", "global"),
                               ("e10-alg-no-norm", "none")]:
        rows = load_run(Path(args.results_dir), run_tag(arm, args.track, 42))
        stats = weight_stats(data_dir, arm) or {}
        diagnostics.append({
            "arm": arm, "normalization": normalization,
            "avg_pass@1": macro(rows, "pass@1"), "avg_pass@3": macro(rows, "pass@3"),
            "dataset_weight_mass_ratio": stats.get("dataset_weight_mass_ratio"),
            "trace_budget_std": stats.get("trace_budget_std"),
            "weight_p10": stats.get("weight_p10"), "weight_median": stats.get("weight_median"),
            "weight_p90": stats.get("weight_p90"),
            "steps_above_one_ratio": stats.get("steps_above_one_ratio"),
            "top_10pct_step_mass_ratio": stats.get("top_10pct_step_mass_ratio"),
        })
    write_csv(out_dir / "normalization_diagnostics.csv", diagnostics,
              ["arm", "normalization", "avg_pass@1", "avg_pass@3", "dataset_weight_mass_ratio",
               "trace_budget_std", "weight_p10", "weight_median", "weight_p90",
               "steps_above_one_ratio", "top_10pct_step_mass_ratio"])
    print(report)
    print(f"summary -> {out_dir / 'ablation_summary.md'} "
          "(+ per_benchmark.csv, seed_summary.csv, normalization_diagnostics.csv)")


if __name__ == "__main__":
    main()
