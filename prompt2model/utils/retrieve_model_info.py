"""Retrieve HuggingFace model's size, description, and downloads."""
import json
import os
import re
import subprocess
from pathlib import Path

from huggingface_hub import HfApi


def main(modelId: str, cache_dir: str = None) -> None:
    """Downloads and caches a Hugging Face model's metadata.

    Args:
        modelId: HuggingFace ModelId, like "gpt2" or "facebook/roscoe-512-roberta-base".
        cache_dir: A directory to cache the metadata.
    """
    if cache_dir is None:
        cache_dir = "model_info"
    cache_path = Path.cwd() / cache_dir
    cache_path.mkdir(parents=True, exist_ok=True)

    if len(modelId.split("/")) == 2:
        _, model_name = modelId.split("/")
    else:
        model_name = modelId

    subprocess.run(
        ["git", "clone", f"https://huggingface.co/{modelId}"],
        env=dict(os.environ, GIT_LFS_SKIP_SMUDGE="1"),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    model_bin_file = Path(f"{model_name}/pytorch_model.bin")

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

    with open(f"{model_name}/README.md", "r", encoding="utf-8") as f:
        readme_content = f.read()
        print(readme_content)

    api = HfApi()
    model_meta = api.model_info(modelId)
    downloads = model_meta.downloads
    print(modelId, downloads)

    model_info = {
        "modelId": modelId,
        "description": readme_content,
        "size": size,
        "downloads": downloads,
    }
    model_info_path = Path(f"{cache_dir}/{model_name}.json")
    model_info_path.touch()
    with open(model_info_path, "w") as file:
        file.write(json.dumps(model_info))

    subprocess.run(["rm", "-rf", model_name])


if __name__ == "__main__":
    main("facebook/roscoe-512-roberta-base")
    main("gpt2")
