"""Analytic loss gradients w.r.t. final hidden states (paper Eq. 1-2).
g = W_U^T (softmax(W_U h) - onehot(y)); chunking avoids a full T x V logits matrix."""

import torch


def analytic_hidden_gradients(
    hidden_states: torch.Tensor,
    target_ids: torch.Tensor,
    unembedding: torch.Tensor,
    chunk_size: int = 1024,
) -> torch.Tensor:
    """Return float32 (N, d) CE gradients (Eq. 2) from shifted (N, d) states.
    target_ids is (N,); unembedding is (V, d); chunk_size bounds vocabulary-projection memory."""
    n_targets = hidden_states.shape[0]
    unembed32 = unembedding.float()
    out = torch.empty(n_targets, unembed32.shape[1], dtype=torch.float32, device=hidden_states.device)

    for start in range(0, n_targets, chunk_size):
        end = min(start + chunk_size, n_targets)
        logits = hidden_states[start:end].float() @ unembed32.T  # (chunk, V)
        probs = torch.softmax(logits, dim=-1)
        # subtract the one-hot target: d(CE)/d(logits) = p - onehot(y)
        probs[torch.arange(end - start, device=probs.device), target_ids[start:end]] -= 1.0
        out[start:end] = probs @ unembed32  # (chunk, V) @ (V, d)
        del logits, probs

    return out


def analytic_hidden_gradients_and_entropies(
    hidden_states: torch.Tensor,
    target_ids: torch.Tensor,
    unembedding: torch.Tensor,
    chunk_size: int = 1024,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return Eq. 2 gradients and predictive entropies from the same chunked logits.
    H = logsumexp(logits) - sum(p * logits) avoids a second vocabulary-sized tensor."""
    n_targets = hidden_states.shape[0]
    unembed32 = unembedding.float()
    gradients = torch.empty(
        n_targets, unembed32.shape[1], dtype=torch.float32, device=hidden_states.device
    )
    entropies = torch.empty(n_targets, dtype=torch.float32, device=hidden_states.device)

    for start in range(0, n_targets, chunk_size):
        end = min(start + chunk_size, n_targets)
        logits = hidden_states[start:end].float() @ unembed32.T
        probs = torch.softmax(logits, dim=-1)
        entropies[start:end] = torch.logsumexp(logits, dim=-1) - (probs * logits).sum(dim=-1)
        probs[torch.arange(end - start, device=probs.device), target_ids[start:end]] -= 1.0
        gradients[start:end] = probs @ unembed32
        del logits, probs

    return gradients, entropies


def shift_for_causal_lm(
    last_hidden_state: torch.Tensor, input_ids: torch.Tensor, target_span: tuple[int, int]
) -> tuple[torch.Tensor, torch.Tensor]:
    """Align (T, d) states to (T,) target IDs in the absolute [start, end) span.
    State j-1 predicts token j; returns (N, d) states and (N,) targets."""
    start, end = target_span
    if start < 1:
        raise ValueError("target span must start at position >= 1 (needs a preceding token)")
    return last_hidden_state[start - 1 : end - 1], input_ids[start:end]


@torch.no_grad()
def capture_sequence_gradients(
    model,
    input_ids: torch.Tensor,
    target_span: tuple[int, int],
    chunk_size: int = 1024,
    unembedding: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return gradient matrix G over one sequence's supervised span.
    Use the backbone and a chunked lm_head projection to avoid logits for all positions."""
    if input_ids.dim() == 1:
        input_ids = input_ids.unsqueeze(0)
    if unembedding is None:
        unembedding = model.get_output_embeddings().weight

    hidden = model.model(input_ids=input_ids).last_hidden_state[0]  # (T, d), post-norm
    rows, targets = shift_for_causal_lm(hidden, input_ids[0], target_span)
    return analytic_hidden_gradients(rows, targets, unembedding, chunk_size=chunk_size)


@torch.no_grad()
def capture_sequence_gradients_and_entropies(
    model,
    input_ids: torch.Tensor,
    target_span: tuple[int, int],
    chunk_size: int = 1024,
    unembedding: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Capture Eq. 2 gradients plus predictive entropy for each response token."""
    if input_ids.dim() == 1:
        input_ids = input_ids.unsqueeze(0)
    if unembedding is None:
        unembedding = model.get_output_embeddings().weight

    hidden = model.model(input_ids=input_ids).last_hidden_state[0]
    rows, targets = shift_for_causal_lm(hidden, input_ids[0], target_span)
    return analytic_hidden_gradients_and_entropies(rows, targets, unembedding, chunk_size=chunk_size)
