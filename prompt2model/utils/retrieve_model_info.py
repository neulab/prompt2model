"""Retrieve HuggingFace model's size, description, and downloads."""
import json
import os
import re
import subprocess
from pathlib import Path

from huggingface_hub import HfApi


def main(pretrained_model_name: str, cache_dir: str = None) -> None:
    """Downloads and caches a Hugging Face model's metadata.

    Args:
        pretrained_model_name: HuggingFace pretrained_model_name.
        cache_dir: A directory to cache the metadata.
    """
    if cache_dir is None:
        cache_dir = "model_info"
    cache_path = Path.cwd() / cache_dir
    cache_path.mkdir(parents=True, exist_ok=True)

    if len(pretrained_model_name.split("/")) == 2:
        _, model_name = pretrained_model_name.split("/")
    else:
        model_name = pretrained_model_name

    subprocess.run(
        ["git", "clone", f"https://huggingface.co/{pretrained_model_name}"],
        env=dict(os.environ, GIT_LFS_SKIP_SMUDGE="1"),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    model_bin_file = Path(f"{model_name}/pytorch_model.bin")

    try:
        if model_bin_file.exists():
            with open(model_bin_file, "r") as file:
                content = file.read()
                size = re.search(r"size (\d+)", content).group(1)  # type: ignore
                print(size)
        else:
            model_bin_file = Path(f"{model_name}/pytorch_model.bin.index.json")
            with open(model_bin_file, "r") as file:
                content = json.loads(file.read())  # type: ignore
                size = content["metadata"]["total_size"]  # type: ignore
                print(size)
    except (
        FileNotFoundError,
        PermissionError,
        IOError,
        json.decoder.JSONDecodeError,
    ) as e:
        raise Exception(f"Failed to read {model_name} in {model_bin_file}: {e}")

    with open(f"{model_name}/README.md", "r", encoding="utf-8") as f:
        readme_content = f.read()
        print(readme_content)

    api = HfApi()
    model_meta = api.model_info(pretrained_model_name)
    downloads = model_meta.downloads
    print(pretrained_model_name, downloads)

    model_info = {
        "pretrained_model_name": pretrained_model_name,
        "description": readme_content,
        "size_bytes": size,
        "downloads": downloads,
    }
    model_info_path = Path(f"{cache_dir}/{model_name}.json")
    model_info_path.touch()
    with open(model_info_path, "w") as file:
        file.write(json.dumps(model_info))

    # The model must exist because it can be found by
    # `model_meta = api.model_info(pretrained_model_name)`
    subprocess.run(["rm", "-rf", model_name])


if __name__ == "__main__":
    main("facebook/roscoe-512-roberta-base")
    main("gpt2")
