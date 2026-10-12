"""Download the released P-ALIGN traces and prepare a reference-trainer data directory."""

import argparse
import json
import shutil
from pathlib import Path

DATASET = "qizheyanger/P-ALIGN"
FILENAME = "P-ALIGN.json"


def released_path(*, local_files_only: bool = False) -> Path:
    from huggingface_hub import hf_hub_download

    return Path(hf_hub_download(
        repo_id=DATASET, repo_type="dataset", filename=FILENAME, local_files_only=local_files_only,
    ))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    source = released_path()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, args.output_dir / FILENAME)
    manifest = {"palign_hf": {
        "file_name": FILENAME,
        "columns": {"prompt": "instruction", "query": "input", "response": "output"},
    }}
    (args.output_dir / "dataset_info.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
