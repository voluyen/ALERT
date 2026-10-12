"""Sentence-level CoT segmentation with exact token-offset boundaries.
Character boundaries map through one tokenizer pass because BPE is context-dependent."""

import bisect
import re

_STEP_BOUNDARY = re.compile(r"([.?!\}\]])([\s\n]+)([A-Z])")


def split_into_step_texts(response: str) -> list[str]:
    """Split at sentence boundaries while preserving all characters for token-offset alignment."""
    if not response:
        return []

    pieces, cursor = [], 0
    for match in _STEP_BOUNDARY.finditer(response):
        boundary = match.start(3)  # cut before the capital letter, keep whitespace behind
        pieces.append(response[cursor:boundary])
        cursor = boundary
    if cursor < len(response):
        pieces.append(response[cursor:])
    return pieces


def record_step_spans(record: dict) -> list[tuple[int, int]]:
    """Absolute (start, end) token spans of a answergain.data.prepare record's steps."""
    return [(step["token_start"], step["token_end"]) for step in record["steps"]]


def solution_step_start(step_texts: list[str]) -> int | None:
    """Return the index of the step containing </think>, where the final solution begins.
    Return None for a whole-response fallback when no closing tag exists."""
    for index, text in enumerate(step_texts):
        if "</think>" in text:
            return index
    return None


def encode_with_offsets(tokenizer, text: str) -> tuple[list[int], list[int]]:
    """Tokenize once with a fast tokenizer; return IDs and token-start character offsets."""
    encoding = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    return encoding["input_ids"], [start for start, _ in encoding["offset_mapping"]]


def step_token_spans(
    response: str,
    token_starts: list[int],
    offset: int = 0,
    segmenter: str = "sentence",
) -> list[tuple[int, int]]:
    """Map lossless character boundaries to absolute token spans shifted by offset.
    Tokens belong to the step containing their first character; zero-token steps are dropped."""
    from answergain.segment import split

    cuts, position = [], 0
    for text in split(response, segmenter):
        cuts.append(position)
        position += len(text)

    bounds = [bisect.bisect_left(token_starts, cut) for cut in cuts] + [len(token_starts)]
    return [
        (offset + start, offset + end)
        for start, end in zip(bounds, bounds[1:])
        if end > start
    ]


def segment_response_token_spans(
    tokenizer,
    prompt_ids: list[int],
    response: str,
) -> tuple[list[int], list[tuple[int, int]]]:
    """Return response IDs and step spans as absolute indices in prompt_ids + response_ids."""
    response_ids, token_starts = encode_with_offsets(tokenizer, response)
    return response_ids, step_token_spans(response, token_starts, offset=len(prompt_ids))
