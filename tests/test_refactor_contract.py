"""Protect resolved experiments and HF dataset routing across the layout change."""

import json
from pathlib import Path

from answergain import cli
from answergain.config import load_config
from answergain.data import palign_download, prepare

AUDIT = Path(__file__).resolve().parents[1] / "docs" / "refactor"


def migrate(value):
    if isinstance(value, dict):
        return {key: migrate(item) for key, item in value.items()}
    if isinstance(value, list):
        return [migrate(item) for item in value]
    if isinstance(value, str):
        return (value.replace("configs/deepspeed/", "accelerate_configs/")
                .replace("answergain.transforms.dft_sources", "answergain.data.dft_sources")
                .replace("references/P-ALIGN/data/palign_sft_qwen2.5-7b.json.gz", "qizheyanger/P-ALIGN")
                .replace("/mnt/local/_data/aiskylimit_new_nothingnew_2/s1K-1.1", "baesad/s1K-1.1-deepseek-cot"))
    return value


def test_existing_resolved_configs_only_change_paths_and_dataset_sources():
    before = json.loads((AUDIT / "resolved-before.json").read_text())
    moves = json.loads((AUDIT / "move-map.json").read_text())
    assert len(before) == 27
    for old, config in before.items():
        expected = migrate(config)
        if "reference_data" in expected["stages"]:
            expected["stages"]["reference_data"] = {
                "module": "answergain.data.palign_download",
                "creates": "data/palign-reference/dataset_info.json",
                "args": {"output-dir": "data/palign-reference"},
            }
            positional = expected["stages"]["train_reference"]["positional"]
            positional[1:1] = ["dataset_dir=../../data/palign-reference", "dataset=palign_hf"]
        assert load_config(cli.REPO_DIR / moves[old]) == expected, old


def test_family_discovery_does_not_include_baselines_or_building_blocks():
    for family in cli.FAMILIES:
        for path in cli.family_configs(family):
            config = load_config(path)
            assert config["family"] == family
            assert config["pipeline"]
            name = str(path.relative_to(cli.family_root(family)).with_suffix(""))
            assert cli.resolve_config(name, family) == path


def test_palign_download_uses_the_released_hf_file(monkeypatch, tmp_path):
    import huggingface_hub

    seen = {}

    def download(**kwargs):
        seen.update(kwargs)
        return str(tmp_path / "P-ALIGN.json")

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", download)
    assert palign_download.released_path(local_files_only=True) == tmp_path / "P-ALIGN.json"
    assert seen == {"repo_id": "qizheyanger/P-ALIGN", "repo_type": "dataset",
                    "filename": "P-ALIGN.json", "local_files_only": True}


def test_palign_preparation_keeps_shuffle_and_alpaca_outputs(monkeypatch, tmp_path):
    seen = {}
    source = tmp_path / "P-ALIGN.json"
    response = "<Begin_of_Prefix>Reason.<End_of_Prefix>\nAnswer."

    class Dataset:
        def shuffle(self, **kwargs):
            seen["shuffle"] = kwargs
            return [{"instruction": "instruction", "input": "question", "output": response}]

    def load(*args, **kwargs):
        seen["load"] = (args, kwargs)
        return Dataset()

    monkeypatch.setattr(palign_download, "released_path", lambda: source)
    monkeypatch.setattr(prepare, "load_dataset", load)
    config = {"dataset_name": "qizheyanger/P-ALIGN", "dataset_split": "train", "seed": 42}
    assert list(prepare.iter_samples(config)) == [("question", response)]
    assert seen == {
        "load": (("json",), {"data_files": str(source), "split": "train", "streaming": True}),
        "shuffle": {"seed": 42, "buffer_size": prepare.SHUFFLE_BUFFER},
    }


def test_converted_paper_drivers_preserve_training_parameters():
    recipes = json.loads((AUDIT / "legacy-training-parameters.json").read_text())
    for config, recipe in recipes.items():
        args = load_config(cli.REPO_DIR / config)["stages"]["train"]["args"]
        for key, value in recipe["preserved_training_parameters"].items():
            assert float(args[key]) == value, (config, key)
