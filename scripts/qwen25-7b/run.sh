#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
answergain run qwen25-7b-s1k/zero-shot "$@"
answergain run qwen25-7b-s1k/sft "$@"
answergain run qwen25-7b-s1k/spectral "$@"
answergain run qwen25-7b-palign/sft-nll-lora "$@"
answergain run qwen25-7b-palign/answer-gain-lora "$@"
palign run qwen25-7b-lora "$@"
ssft run qwen25-7b-lora "$@"
