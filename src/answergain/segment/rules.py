"""Segment-Selective SFT and P-ALIGN segmentation rules.
SSFT retains delimiters and drops empty segments; P-ALIGN strips pieces and appends periods."""
import re

# The paper's rule: cut at backtracking cues.
CUE_PATTERN = (
    r"(\n\nWait|\n\nAlternatively|\n\nBut wait|\n\nBut alternatively|\n\nBut just to|\n\nHowever"
    r"|\n\nNot sure|\n\nGoing back|\n\nBacktrack|\n\nTrace back|\n\nAnother)"
)
PARAGRAPH_PATTERN = r"(\n\n)"

SEGMENT_PATTERNS = {"cue": CUE_PATTERN, "paragraph": PARAGRAPH_PATTERN}


def split_segments(text: str, mode: str = "paragraph") -> list[str]:
    if mode not in SEGMENT_PATTERNS:
        raise ValueError(f"mode must be one of {sorted(SEGMENT_PATTERNS)}, got {mode!r}")
    parts = re.split(SEGMENT_PATTERNS[mode], text)
    segments = [parts[0]]
    for index in range(1, len(parts), 2):
        segments.append(parts[index] + parts[index + 1])
    return _drop_empty(segments, text)


def _drop_empty(segments: list[str], text: str) -> list[str]:
    merged: list[str] = []
    for segment in segments:
        if segment == "":
            continue
        merged.append(segment)
    if not merged:
        return [text]
    assert "".join(merged) == text, "split_segments lost characters"
    return merged


def paragraph_segments(text: str) -> list[str]:
    return split_segments(text, "paragraph")


def cue_segments(text: str) -> list[str]:
    return split_segments(text, "cue")


def palign_sentences(text: str) -> list[str]:
    sentences = text.split(". ")
    return [
        sentence.strip() + "." if not sentence.endswith(".") else sentence.strip()
        for sentence in sentences
        if sentence.strip()
    ]
