"""Tools for retrieving dataset metadata from HuggingFace.

Before calling this script, set the HF_USER_ACCESS_TOKEN environment variable.
"""

from __future__ import annotations  # noqa FI58

import argparse
import json
from collections.abc import MutableMapping

import datasets
import requests
from huggingface_hub import list_datasets

parser = argparse.ArgumentParser()
parser.add_argument(
    "--dataset-index-file",
    type=str,
    default="huggingface_data/huggingface_datasets/dataset_index.json",
)


def get_dataset_validity(dataset_name):
    """Get the list of loadable datasets from HuggingFace."""
    API_URL = f"https://datasets-server.huggingface.co/is-valid?dataset={dataset_name}"
    response = requests.get(API_URL)
    if response.status_code != 200:
        return False
    response = response.json()
    if "preview" not in response or "viewer" not in response:
        return False
    return response["preview"] & response["viewer"]


def get_fully_supported_dataset_names():
    """Get the list of loadable datasets from HuggingFace."""
    API_URL = "https://datasets-server.huggingface.co/is-valid?dataset={dataset_name}"
    response = requests.get(API_URL)
    datasets_list = response.json()
    fully_supported_datasets = datasets_list["viewer"] + datasets_list["preview"]
    return fully_supported_datasets


def construct_search_documents(
    all_dataset_names,
    all_dataset_descriptions,
    fully_supported_dataset_names,
    minimum_description_length=4,
):
    """Select the datasets and corresponding descriptions to store in our index."""
    filtered_dataset_names = []
    nonempty_descriptions = []
    for dataset_name, description in zip(all_dataset_names, all_dataset_descriptions):
        if dataset_name not in fully_supported_dataset_names:
            continue
        if (
            description is not None
            and len(description.split()) > minimum_description_length
        ):
            filtered_dataset_names.append(dataset_name)
            nonempty_descriptions.append(description)
    return filtered_dataset_names, nonempty_descriptions


def flatten_dict(
    d: MutableMapping, parent_key: str = "", sep: str = "."
) -> MutableMapping:
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, MutableMapping):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def construct_search_document_v2(all_datasets, minimum_description_length=4):
    excluded_words = [
        "arxiv",
        "region",
        "license",
        "size_categories",
        "language_creators",
    ]

    dataset_index = {}
    counter = 1
    for z in range(15, len(all_datasets)):
        dataset_info = all_datasets[z]
        print("Currently on : ", dataset_info.id)
        is_gated = hasattr(dataset_info, "gated") and dataset_info.gated
        if hasattr(dataset_info, "disabled") and dataset_info.disabled:
            print("dataset is gated or disabled")
            continue
        dataset_name = dataset_info.id
        description = dataset_info.description
        if not get_dataset_validity(dataset_name):
            continue
        if (
            description is not None
            and len(description.split()) < minimum_description_length
        ):
            print("dataset is not valid")
            continue

        config_names = datasets.get_dataset_config_names(dataset_name)

        all_configs = []
        for config_name in config_names:

            if "train" not in datasets.get_dataset_split_names(
                dataset_name, config_name
            ):
                continue
            dataset = datasets.load_dataset(
                dataset_name, config_name, split="train", streaming=True
            )
            sample_rows = next(iter(dataset))
            sample_rows = flatten_dict(sample_rows)
            dataset_columns = sample_rows.keys()
            dataset_columns = ", ".join(dataset.column_names)
            all_configs.append(
                {
                    "config_name": config_name,
                    "columns": dataset_columns,
                    "sample_rows": json.dumps(sample_rows, indent=4),
                }
            )
            print("compelted config: ", config_name)
            del dataset

        filtered_tags = [
            tag
            for tag in dataset_info.tags
            if not any(excluded_word in tag for excluded_word in excluded_words)
        ]

        dataset_index[dataset_name] = {
            "name": dataset_name,
            "description": description,
            "downloads": dataset_info.downloads,
            "configs": all_configs,
            "tags": filtered_tags,
            "is_gated": is_gated,
        }
        del config_names, all_configs, filtered_tags
        counter += 1
        print(f"completed {counter} out of {len(all_datasets)}")
    return dataset_index


if __name__ == "__main__":
    args = parser.parse_args()

    fully_supported_dataset_names = []
    all_datasets = list(list_datasets())
    dataset_index = construct_search_document_v2(all_datasets)

    json.dump(dataset_index, open(args.dataset_index_file, "w"))
