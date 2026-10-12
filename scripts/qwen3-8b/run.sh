#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
answergain run qwen3-8b-s1k/zero-shot "$@"
answergain run qwen3-8b-s1k/sft "$@"
answergain run qwen3-8b-s1k/spectral "$@"
answergain run qwen3-8b-palign/sft-nll-lora "$@"
answergain run qwen3-8b-palign/answer-gain-lora "$@"
palign run qwen3-8b-lora "$@"
ssft run qwen3-8b-lora "$@"
