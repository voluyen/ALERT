"""Conservative answer-only labels for the ALG-NoAnswerUpweight ablation.
Only a boxed gold answer with a short final-answer wrapper qualifies; derivations do not."""

import argparse
import csv
import json
import re
from pathlib import Path

from transformers import AutoTokenizer

from answergain.eval.graders.answer_scoring import extract_boxed, normalize_math

_WRAPPER = re.compile(
    r"^(?:(?:therefore|thus|hence|so|finally|consequently|overall|in conclusion)[,:]?\s*)*"
    r"(?:(?:(?:the\s+)?(?:final\s+)?answer\s*(?:is|equals|:|=)\s*)|"
    r"(?:we\s+(?:get|obtain|have|find)\s*))?"
    r"<answer>"
    r"(?:\s*is\s+the\s+(?:final\s+)?answer)?$",
    re.IGNORECASE,
)


def boxed_spans(text: str) -> list[tuple[int, int, str]]:
    """Return complete ``\\boxed{...}`` spans, including nested braces."""
    spans = []
    for match in re.finditer(r"\\boxed\s*{", text):
        depth, index = 1, match.end()
        while index < len(text) and depth:
            if text[index] == "{":
                depth += 1
            elif text[index] == "}":
                depth -= 1
            index += 1
        if depth == 0:
            spans.append((match.start(), index, text[match.end(): index - 1].strip()))
    return spans


def is_answer_only_step(step_text: str, gold_answer: str) -> bool:
    """Whether a step only states the gold answer under the documented strict rule."""
    matching = [
        (start, end)
        for start, end, answer in boxed_spans(step_text)
        if normalize_math(answer) == normalize_math(gold_answer)
    ]
    if not matching:
        return False
    # Replace from the end so earlier character offsets remain valid. Multiple repetitions are
    # deliberately rejected by the wrapper regex, keeping the classifier conservative.
    reduced = step_text
    for start, end in reversed(matching):
        reduced = reduced[:start] + "<answer>" + reduced[end:]
    reduced = re.sub(r"</?(?:think|analysis|final)>\s*", " ", reduced, flags=re.IGNORECASE)
    reduced = re.sub(r"\\(?:\[|\]|\(|\))|\$+", " ", reduced)
    reduced = re.sub(r"[.!;\s]+$", "", reduced.strip())
    reduced = re.sub(r"\s+", " ", reduced)
    return bool(_WRAPPER.fullmatch(reduced))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--audit-output", required=True, help="CSV for manual label inspection")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    records = [json.loads(line) for line in open(args.data_path)]
    labels, audit_rows = {}, []
    labeled_tokens = 0
    for record in records:
        gold = extract_boxed(record["response"])
        record_labels = []
        for step_index, step in enumerate(record["steps"]):
            token_ids = record["input_ids"][step["token_start"]:step["token_end"]]
            text = tokenizer.decode(token_ids, skip_special_tokens=False)
            label = bool(gold) and is_answer_only_step(text, gold)
            record_labels.append(label)
            token_count = step["token_end"] - step["token_start"]
            labeled_tokens += token_count if label else 0
            audit_rows.append(
                {
                    "id": record["id"],
                    "step": step_index,
                    "answer_only": int(label),
                    "tokens": token_count,
                    "gold_answer": gold or "",
                    "text": text.replace("\x00", ""),
                }
            )
        labels[str(record["id"])] = record_labels

    output = Path(args.output)
    audit_output = Path(args.audit_output)
    output.parent.mkdir(parents=True, exist_ok=True)
    audit_output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(labels))
    with audit_output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=audit_rows[0].keys())
        writer.writeheader()
        writer.writerows(audit_rows)
    n_labeled = sum(sum(values) for values in labels.values())
    print(
        f"labeled {n_labeled}/{len(audit_rows)} answer-only steps ({labeled_tokens} tokens) "
        f"-> {output}; audit -> {audit_output}"
    )


if __name__ == "__main__":
    main()
