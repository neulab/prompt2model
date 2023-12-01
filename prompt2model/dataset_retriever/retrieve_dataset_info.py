"""Tools for retrieving dataset metadata from HuggingFace.

Before calling this script, set the HF_USER_ACCESS_TOKEN environment variable.
"""

from __future__ import annotations  # noqa FI58

import argparse
import json

import requests
from huggingface_hub import list_datasets

parser = argparse.ArgumentParser()
parser.add_argument(
    "--dataset-index-file",
    type=str,
    default="huggingface_data/huggingface_datasets/dataset_index.json",
)


def get_fully_supported_dataset_names():
    """Get the list of loadable datasets from HuggingFace."""
    API_URL = "https://datasets-server.huggingface.co/valid"
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


if __name__ == "__main__":
    args = parser.parse_args()

    fully_supported_dataset_names = get_fully_supported_dataset_names()
    all_datasets = list(list_datasets())
    dataset_names = [dataset.id for dataset in all_datasets]
    dataset_descriptions = [dataset.description for dataset in all_datasets]

    filtered_dataset_names, filtered_descriptions = construct_search_documents(
        dataset_names, dataset_descriptions, fully_supported_dataset_names
    )
    dataset_index = {}
    for name, description in zip(filtered_dataset_names, filtered_descriptions):
        dataset_index[name] = {
            "name": name,
            "description": description,
        }

    json.dump(dataset_index, open(args.dataset_index_file, "w"))
