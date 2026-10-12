"""Run CPU pipeline stages on fixed assets and compare output hashes with golden.json.
Relative paths keep outputs stable; Parquet is hashed by content to ignore pyarrow metadata.
"""
import argparse
import hashlib
import json
import os
import random
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
ASSETS = HERE / "assets"
GOLDEN = HERE / "golden.json"
TOKENIZER = str(ASSETS / "tokenizer")
SSFT = ROOT / "references" / "SegmentSelectiveSFT"
if not SSFT.exists():
    SSFT = ROOT / "SegmentSelectiveSFT"

PY = sys.executable


def answergain(module: str) -> list[str]:
    """Command prefix for one answergain pipeline module."""
    return [PY, "-m", f"answergain.{module}"]


PREP_ARGS = [
    "--dataset-name", str(ASSETS / "palign_sample.jsonl"), "--tokenizer", TOKENIZER,
    "--max-tokens", "100000", "--seed", "42",
]

# (stage name, argv, stdout capture file or None). Run in order, cwd = work dir.
STAGES = [
    ("data_prep", answergain("data.prepare") + PREP_ARGS + [
        "--chat-template", "--no-enable-thinking", "--palign-prompt",
        "--output-path", "data/train-segmented.jsonl",
    ], None),
    ("data_prep_plain", answergain("data.prepare") + PREP_ARGS + [
        "--no-chat-template", "--output-path", "plain/train-segmented.jsonl",
    ], None),
    ("signals", None, None),  # synthetic spectral/entropy/gain signals, see write_signals()
    ("build_masks", answergain("selection.build_masks") + [
        "--data-path", "data/train-segmented.jsonl", "--strengths", "signals/spectral.parquet",
        "--energy-threshold-p", "0.8", "--vanilla",
    ], None),
    ("build_masks_sweep", answergain("selection.build_masks") + [
        "--data-path", "data/train-segmented.jsonl", "--strengths", "signals/spectral.parquet",
        "--sweep", "0.5,0.8,0.95",
    ], None),
    ("build_gain_signal", answergain("signals.gain_parquet") + [
        "--data-path", "gain/train-segmented.jsonl", "--gains", "signals/gains.json",
        "--output", "gain/signal-answer-gain.parquet",
    ], None),
    ("build_answer_gain", answergain("allocation.build") + [
        "--data-path", "gain/train-segmented.jsonl", "--strengths", "gain/signal-answer-gain.parquet",
        "--energy-threshold-p", "1.0", "--interpolation", "0.5", "--variants", "answer-gain",
    ], None),
    ("build_dft-sources", answergain("data.dft_sources") + [
        "--data-dir", "data", "--tokenizer", TOKENIZER,
    ], None),
    ("build_dft-sources_junction", answergain("data.dft_sources") + [
        "--data-dir", "data", "--tokenizer", TOKENIZER, "--junction-tokens", "8",
        "--output-name", "train-dft-sources-j8.jsonl",
    ], None),
    ("rescore", answergain("eval.evaluate") + [
        "--rescore", str(ASSETS / "raw_aime24.jsonl"), "--grader", "math_verify",
    ], "eval/rescore-math_verify.json"),
    ("rescore_builtin", answergain("eval.evaluate") + [
        "--rescore", str(ASSETS / "raw_aime24.jsonl"), "--grader", "builtin",
    ], "eval/rescore-builtin.json"),

    ("ssft_split_paragraph", [PY, str(SSFT / "Attribution" / "segment_split.py"),
        "--input_data_file", "ssft/train.jsonl", "--output_data_file", "ssft/segments-paragraph.jsonl",
        "--tokenizer", "none", "--segment_mode", "paragraph",
    ], None),
    ("ssft_split_cue", [PY, str(SSFT / "Attribution" / "segment_split.py"),
        "--input_data_file", "ssft/train.jsonl", "--output_data_file", "ssft/segments-cue.jsonl",
        "--tokenizer", "none", "--segment_mode", "cue",
    ], None),
    ("ssft_ig", None, None),  # synthetic compact IG per segment, see write_ssft_ig()
    ("ssft_select", [PY, str(SSFT / "Attribution" / "get_important_segments.py"),
        "--input_data_file", "ssft/segments-paragraph.jsonl", "--IG_score_data_file", "ssft/ig_compact.jsonl",
        "--output_data_file", "ssft/selected.jsonl",
    ], None),
    ("data_prep_paragraph", answergain("data.prepare") + PREP_ARGS + [
        "--chat-template", "--no-enable-thinking", "--palign-prompt", "--segmenter", "paragraph",
        "--output-path", "paragraph/train-segmented.jsonl",
    ], None),
    ("answergain_ssft_select", answergain("ssft.select") + [
        "--data-path", "ssft/segments-paragraph.jsonl", "--ig", "ssft/ig_compact.jsonl",
        "--output", "ssft/selected-answergain.jsonl",
    ], None),
    ("answergain_ssft_build", answergain("ssft.build") + [
        "--data-path", "ssft/selected-answergain.jsonl", "--tokenizer", "tokenizer-chatml",
        "--output", "ssft/train-ssft.jsonl",
        "--no-deepseek", "--think-prefix", "plain", "--prompt-style", "palign",
    ], None),
]

# Pairs of outputs that must be byte-identical: framework port vs the upstream script.
SAME_OUTPUT = [("ssft/selected.jsonl", "ssft/selected-answergain.jsonl")]

# Outputs that are figures or otherwise not byte-stable; everything else in the work dir is hashed.
UNHASHED_SUFFIXES = {".png"}


def write_inputs(work: Path) -> None:
    """Inputs that are copies of repo files rather than stage outputs."""
    # Segment-Selective SFT probes for a plain ChatML generation prompt ("<|im_start|>assistant\n");
    # the asset tokenizer opens <think> after it, so give that stage a plain-ChatML copy.
    chatml = work / "tokenizer-chatml"
    shutil.copytree(TOKENIZER, chatml)
    template = chatml / "chat_template.jinja"
    template.write_text(template.read_text().replace("assistant\n<think>\n", "assistant\n"))

    # SegmentSelectiveSFT reads question / solution / answer rows (its prepare_s1k.py format).
    (work / "ssft").mkdir()
    with (ASSETS / "palign_sample.jsonl").open() as handle, (work / "ssft" / "train.jsonl").open("w") as out:
        for line in handle:
            row = json.loads(line)
            solution = row["output"]
            answer = solution.rsplit("\\boxed{", 1)[-1].split("}", 1)[0]
            out.write(json.dumps({"question": row["input"], "solution": solution, "answer": answer}) + "\n")


def write_signals(work: Path) -> None:
    """Deterministic per-step spectral strengths, entropies and answer gains."""
    import pandas as pd

    records = [json.loads(line) for line in (work / "data" / "train-segmented.jsonl").open()]
    rows, gains = [], {}
    for record in records:
        rng = random.Random(1000 + record["id"])
        n_steps = len(record["steps"])
        rows.append({
            "id": record["id"],
            "k_star": rng.randint(1, 8),
            "n_response_tokens": record["response_token_span"][1] - record["response_token_span"][0],
            "n_steps": n_steps,
            "step_strengths": [round(rng.uniform(0.0, 1.0), 6) for _ in range(n_steps)],
            "step_entropies": [round(rng.uniform(0.0, 3.0), 6) for _ in range(n_steps)],
        })
        if record["id"] % 5 != 4:  # leave some records without a gain signal on purpose
            gains[str(record["id"])] = [round(rng.gauss(0.0, 1.0), 6) for _ in range(n_steps)]
    (work / "signals").mkdir()
    pd.DataFrame(rows).to_parquet(work / "signals" / "spectral.parquet")
    (work / "signals" / "gains.json").write_text(json.dumps(gains))
    (work / "gain").mkdir()
    shutil.copyfile(work / "data" / "train-segmented.jsonl", work / "gain" / "train-segmented.jsonl")


def write_ssft_ig(work: Path) -> None:
    """Deterministic compact IG rows [n_tok, sum|IG|, sum IG] per segment, with edge cases."""
    with (work / "ssft" / "segments-paragraph.jsonl").open() as handle, \
         (work / "ssft" / "ig_compact.jsonl").open("w") as out:
        for index, line in enumerate(handle):
            segments = json.loads(line)["segments"]
            rng = random.Random(2000 + index)
            compact = []
            for segment in segments:
                n_tok = max(1, len(segment) // 4)
                sum_abs = round(rng.uniform(0.0, 1.0), 6)
                compact.append([n_tok, sum_abs, round(sum_abs * rng.uniform(-1.0, 1.0), 6)])
            if index == 0:  # no attribution signal at all -> empty selection
                compact = [[n, 0.0, 0.0] for n, _, _ in compact]
            out.write(json.dumps({"segments": compact}) + "\n")


SYNTHETIC = {"signals": write_signals, "ssft_ig": write_ssft_ig}


def content_hash(path: Path) -> str:
    if path.suffix == ".parquet":
        import pandas as pd

        frame = pd.read_parquet(path)
        payload = frame.to_json(orient="records", double_precision=15).encode()
    else:
        payload = path.read_bytes()
    return hashlib.sha256(payload).hexdigest()


def run_pipeline(work: Path) -> dict[str, str]:
    env = dict(os.environ, PYTHONPATH=str(ROOT / "src"), TOKENIZERS_PARALLELISM="false",
               HF_HUB_OFFLINE="1", MPLBACKEND="Agg")
    write_inputs(work)
    for name, argv, capture in STAGES:
        if argv is None:
            SYNTHETIC[name](work)
            continue
        result = subprocess.run(argv, cwd=work, env=env, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"stage {name} failed:\n{result.stdout[-3000:]}\n{result.stderr[-3000:]}")
        if capture:
            target = work / capture
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(result.stdout)
    return {
        str(path.relative_to(work)): content_hash(path)
        for path in sorted(work.rglob("*"))
        if path.is_file() and path.suffix not in UNHASHED_SUFFIXES
        and path.name != "allocation-selection-stats.json"
    }


def diff(expected: dict[str, str], actual: dict[str, str]) -> list[str]:
    problems = [f"port differs from upstream: {a} vs {b}" for a, b in SAME_OUTPUT if actual.get(a) != actual.get(b)]
    problems += [f"missing output: {name}" for name in sorted(expected.keys() - actual.keys())]
    problems += [f"unexpected output: {name}" for name in sorted(actual.keys() - expected.keys())]
    problems += [
        f"changed: {name}" for name in sorted(expected.keys() & actual.keys())
        if expected[name] != actual[name]
    ]
    return problems


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--update", action="store_true", help="rewrite golden.json from this run")
    parser.add_argument("--keep", help="run in this directory and keep it, for debugging")
    args = parser.parse_args()

    if args.keep:
        work = Path(args.keep)
        shutil.rmtree(work, ignore_errors=True)
        work.mkdir(parents=True)
        hashes = run_pipeline(work)
    else:
        with tempfile.TemporaryDirectory() as tmp:
            hashes = run_pipeline(Path(tmp))

    if args.update:
        GOLDEN.write_text(json.dumps(hashes, indent=1, sort_keys=True) + "\n")
        print(f"wrote {len(hashes)} hashes -> {GOLDEN}")
        return
    problems = diff(json.loads(GOLDEN.read_text()), hashes)
    print("\n".join(problems) if problems else f"all {len(hashes)} outputs identical")
    sys.exit(1 if problems else 0)


if __name__ == "__main__":
    main()
