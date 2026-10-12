"""Frozen-student step predictability: d_k = mean_t log q0(y_t | x, y_<t).
Outputs a record-ID-to-step-scores JSON mapping; vocabulary projection is chunked."""

import argparse
import json
import math
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM

from answergain.segment.sentence import record_step_spans
from answergain.signals.gradients import shift_for_causal_lm


def token_log_probabilities(
    hidden_states: torch.Tensor,
    target_ids: torch.Tensor,
    unembedding: torch.Tensor,
    chunk_size: int = 1024,
) -> torch.Tensor:
    """Target-token log probabilities from already causally shifted hidden states."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    values = torch.empty(len(target_ids), dtype=torch.float32, device=hidden_states.device)
    unembed32 = unembedding.float()
    for start in range(0, len(target_ids), chunk_size):
        end = min(start + chunk_size, len(target_ids))
        logits = hidden_states[start:end].float() @ unembed32.T
        values[start:end] = torch.log_softmax(logits, dim=-1).gather(
            1, target_ids[start:end, None]
        ).squeeze(1)
        del logits
    return values


def step_mean_log_probabilities(
    token_logps: torch.Tensor,
    step_spans: list[tuple[int, int]],
    response_start: int,
) -> list[float]:
    """Average response-relative token log probabilities over absolute step spans."""
    values = []
    for start, end in step_spans:
        relative_start, relative_end = start - response_start, end - response_start
        value = float(token_logps[relative_start:relative_end].mean().item())
        if not math.isfinite(value):
            raise ValueError(f"non-finite step predictability at span {(start, end)}")
        values.append(value)
    return values


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", required=True, help="train-segmented.jsonl")
    parser.add_argument("--model", required=True, help="frozen q0 checkpoint")
    parser.add_argument("--output", required=True)
    parser.add_argument("--chunk-size", type=int, default=1024)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--attn-implementation", default="sdpa")
    parser.add_argument("--limit", type=int)
    return parser


@torch.no_grad()
def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    records = [json.loads(line) for line in open(args.data_path)]
    if args.limit is not None:
        records = records[: args.limit]

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=getattr(torch, args.dtype),
        attn_implementation=args.attn_implementation,
    ).to(args.device).eval()
    unembedding = model.get_output_embeddings().weight.float()

    scores = {}
    for index, record in enumerate(records, start=1):
        response_start, response_end = record["response_token_span"]
        input_ids = torch.tensor([record["input_ids"]], device=args.device)
        hidden = model.model(input_ids=input_ids).last_hidden_state[0]
        rows, targets = shift_for_causal_lm(hidden, input_ids[0], (response_start, response_end))
        token_logps = token_log_probabilities(rows, targets, unembedding, args.chunk_size)
        scores[str(record["id"])] = step_mean_log_probabilities(
            token_logps, record_step_spans(record), response_start
        )
        del input_ids, hidden, rows, targets, token_logps
        if index % 50 == 0:
            print(f"scored {index}/{len(records)} traces")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(scores))
    print(f"wrote predictability scores for {len(scores)} traces -> {output}")


if __name__ == "__main__":
    main()
