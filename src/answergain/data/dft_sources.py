"""Add prompt, teacher-prefix, student-continuation and junction token dft-sources.
Junction tokens allow separate supervision where low student probability suppresses DFT gradients."""

import argparse
import json
from pathlib import Path

from transformers import AutoTokenizer

END_MARKER = "<End_of_Prefix>"


def prefix_token_count(tokenizer, response: str, response_ids: list[int]) -> int | None:
    """Number of response tokens that belong to the teacher prefix (marker included).

    Tokens are assigned by character offset: a token is prefix iff it starts before the end of
    the marker. None when the response has no marker.
    """
    cut = response.find(END_MARKER)
    if cut < 0:
        return None
    boundary = cut + len(END_MARKER)
    encoded = tokenizer(response, add_special_tokens=False, return_offsets_mapping=True)
    if encoded["input_ids"] != response_ids:
        raise ValueError("re-tokenized response does not match the stored response tokens")
    return sum(1 for start, _ in encoded["offset_mapping"] if start < boundary)


def build(
    segmented_path: Path, vanilla_path: Path, output_path: Path, tokenizer_name: str, junction_tokens: int = 0,
) -> dict:
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    segmented = {}
    with segmented_path.open() as handle:
        for line in handle:
            record = json.loads(line)
            segmented[record["id"]] = record

    counts = {"samples": 0, "no_marker": 0, "prefix_tokens": 0, "cont_tokens": 0, "junction_tokens": 0}
    with vanilla_path.open() as source, output_path.open("w") as sink:
        for line in source:
            record = json.loads(line)
            seg = segmented[record["id"]]
            if record["input_ids"] != seg["input_ids"]:
                raise ValueError(f"record {record['id']}: vanilla and segmented input_ids differ")
            start, end = seg["response_token_span"]
            n_prefix = prefix_token_count(tokenizer, seg["response"], seg["input_ids"][start:end])
            if n_prefix is None:
                counts["no_marker"] += 1
                n_prefix = 0  # no teacher prefix: the whole response counts as continuation
            token_source = [0] * len(record["input_ids"])
            for position, keep in enumerate(record["loss_mask"]):
                if keep:
                    if position < start + n_prefix:
                        token_source[position] = 1
                    elif n_prefix and position < start + n_prefix + junction_tokens:
                        token_source[position] = 3
                    else:
                        token_source[position] = 2
            record["token_source"] = token_source
            counts["samples"] += 1
            counts["prefix_tokens"] += token_source.count(1)
            counts["cont_tokens"] += token_source.count(2)
            counts["junction_tokens"] += token_source.count(3)
            sink.write(json.dumps(record) + "\n")
    return counts


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True, help="dir holding train-segmented.jsonl and train-vanilla.jsonl")
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--output-name", default="train-dft-sources.jsonl")
    parser.add_argument(
        "--junction-tokens", type=int, default=0,
        help="label the first N continuation tokens as junction (token_source 3)",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    data_dir = Path(args.data_dir)
    output_path = data_dir / args.output_name
    counts = build(
        data_dir / "train-segmented.jsonl", data_dir / "train-vanilla.jsonl", output_path, args.tokenizer,
        args.junction_tokens,
    )
    total = counts["prefix_tokens"] + counts["cont_tokens"] + counts["junction_tokens"]
    print(
        f"{counts['samples']} records -> {output_path}; prefix tokens {counts['prefix_tokens']:,} "
        f"({counts['prefix_tokens'] / max(total, 1):.1%}), continuation tokens {counts['cont_tokens']:,}, "
        f"junction tokens {counts['junction_tokens']:,}; "
        f"records without {END_MARKER}: {counts['no_marker']}"
    )


if __name__ == "__main__":
    main()
