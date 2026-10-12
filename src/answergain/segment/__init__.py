"""Lossless sentence, paragraph and cue segmentation with character-to-token span mapping.
P-ALIGN uses a separate lossy sentence splitter for prefix construction."""
from answergain.segment.rules import cue_segments, palign_sentences, paragraph_segments, split_segments
from answergain.segment.sentence import (
    encode_with_offsets,
    record_step_spans,
    split_into_step_texts,
    step_token_spans,
)

SEGMENTERS = {
    "sentence": split_into_step_texts,
    "paragraph": paragraph_segments,
    "cue": cue_segments,
}


def split(text: str, mode: str = "sentence") -> list[str]:
    """Split `text` into steps with the named lossless rule."""
    if mode not in SEGMENTERS:
        raise ValueError(f"unknown segmenter {mode!r}; choose from {sorted(SEGMENTERS)}")
    if not text:
        return []
    return SEGMENTERS[mode](text)


__all__ = [
    "SEGMENTERS",
    "encode_with_offsets",
    "palign_sentences",
    "record_step_spans",
    "split",
    "split_segments",
    "step_token_spans",
]
