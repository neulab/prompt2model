"""Tools for retrieving dataset metadata from HuggingFace.

Before calling this script, set the HF_USER_ACCESS_TOKEN environment variable.
"""

import argparse
import importlib
import json
import os
import pickle

import requests
from huggingface_hub import list_datasets
from tqdm import tqdm

hf_eval_utils = importlib.import_module("model-evaluator.utils")

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


def get_eval_metadata(dataset):
    """Load the evaluation metadata from HuggingFace."""
    dataset_metadata = hf_eval_utils.get_metadata(
        dataset, os.environ["HF_USER_ACCESS_TOKEN"]
    )
    if dataset_metadata is None:
        return None
    else:
        return dataset_metadata


SUPPORTED_TASKS = [
    "text-classification",
    "text2text-generation",
    "question-answering",
    "summarization",
    "text-generation",
    "token-classification",
]


def load_dataset_metadata(
    dataset_names, dataset_metadata_cache_file="/tmp/dataset_metadata_cache.pkl"
):
    """Load the evaluation metadata for all datasets."""
    if os.path.exists(dataset_metadata_cache_file):
        dataset_metadata_cache = pickle.load(open(dataset_metadata_cache_file, "rb"))
    else:
        dataset_metadata_cache = {}
    all_dataset_metadata = {}
    for dataset in tqdm(dataset_names):
        if dataset in dataset_metadata_cache:
            dataset_metadata = dataset_metadata_cache[dataset]
        else:
            try:
                dataset_metadata = get_eval_metadata(dataset)
            except:  # noqa E722
                dataset_metadata = None
        if dataset_metadata is not None:
            filtered_task_metadata = []
            for task_metadata in dataset_metadata:
                if "task" in task_metadata and task_metadata["task"] in SUPPORTED_TASKS:
                    filtered_task_metadata.append(task_metadata)
            if len(filtered_task_metadata) > 0:
                all_dataset_metadata[dataset] = filtered_task_metadata
        dataset_metadata_cache[dataset] = dataset_metadata

    pickle.dump(dataset_metadata_cache, open(dataset_metadata_cache_file, "wb"))
    return all_dataset_metadata


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
    all_dataset_metadata = load_dataset_metadata(dataset_names)

    filtered_dataset_names, filtered_descriptions = construct_search_documents(
        dataset_names, dataset_descriptions, fully_supported_dataset_names
    )
    dataset_info = {}
    for name, description in zip(filtered_dataset_names, filtered_descriptions):
        if name in all_dataset_metadata:
            evaluation_metadata = all_dataset_metadata[name]
        else:
            evaluation_metadata = {}

        dataset_info[name] = {
            "name": name,
            "description": description,
            "evaluation_metadata": evaluation_metadata,
        }

    json.dump(dataset_info, open(args.dataset_index_file, "w"))
