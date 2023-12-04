"""Filtering out datasets before we do heavy processing on them."""
import json
from typing import Any

from huggingface_hub import list_datasets

# Constants
ALL_DATASETS_FILE = "all_datasets.json"
FILTERED_DATASETS_FILE = "filtered_datasets.json"
MIN_WORDS_IN_DESC = 4
MIN_DOWNLOADS = 10


def load_datasets(file_path: str, is_first_time=False) -> list[dict[str, Any]]:
    """Load datasets from a JSON file."""
    if is_first_time:
        all_datasets = list(list_datasets())
        with open(ALL_DATASETS_FILE, "w") as f:
            ds = json.dumps([ob.__dict__ for ob in all_datasets])
            f.write(ds)
        return all_datasets
    else:
        with open(file_path, "r") as file:
            return json.load(file)


def filter_datasets(datasets: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Filter datasets based on specific criteria."""
    filtered_datasets = []
    descr_none = descr_small = downloads_less = common_descr = 0
    unique_descriptions: set[str] = set()

    for dataset_info in datasets:
        description = dataset_info.get("description")

        if not description:
            descr_none += 1
            continue
        if len(description.split()) < MIN_WORDS_IN_DESC:
            descr_small += 1
            continue
        if dataset_info.get("downloads", 0) < MIN_DOWNLOADS:
            downloads_less += 1
            continue
        if description in unique_descriptions:
            common_descr += 1
            continue

        filtered_datasets.append(dataset_info)
        unique_descriptions.add(description)

    print(
        f"descr_none: {descr_none}, descr_small: {descr_small}, "
        f"downloads_less: {downloads_less}, common_descr: {common_descr}"
    )

    return filtered_datasets


def main():
    """Main function to load and filter datasets."""
    all_datasets = load_datasets(ALL_DATASETS_FILE)
    filtered_datasets = filter_datasets(all_datasets)
    with open(FILTERED_DATASETS_FILE, "w") as f:
        json.dump(filtered_datasets, f)


if __name__ == "__main__":
    main()
