from __future__ import annotations  # noqa FI58

import argparse
import json
import multiprocessing
import threading
from collections.abc import MutableMapping
from pathlib import Path

import datasets
import requests
from huggingface_hub import list_datasets

parser = argparse.ArgumentParser()
parser.add_argument(
    "--dataset-index-file",
    type=str,
    default="huggingface_data/huggingface_datasets/dataset_index.json",
)

import time


def get_dataset_validity(dataset_name, max_retries=5):
    """Get the list of loadable datasets from HuggingFace with backoff strategy."""
    API_URL = f"https://datasets-server.huggingface.co/is-valid?dataset={dataset_name}"
    retries = 0
    backoff = 10

    while retries < max_retries:
        response = requests.get(API_URL)

        if response.status_code == 200:
            response = response.json()
            return (
                "preview" in response
                and "viewer" in response
                and response["preview"] & response["viewer"]
            )

        elif response.status_code == 429:
            retry_after = response.headers.get("Retry-After")
            wait = int(retry_after) if retry_after else backoff
            time.sleep(wait)
            backoff *= 2  # Exponential increase
            retries += 1
        else:
            # Handle other HTTP errors
            break

    return False


def fetch_first_row_with_timeout(dataset, timeout=30):
    def fetch_sample_row(container):
        try:
            container.append(next(iter(dataset)))
        except Exception as e:
            container.append(e)

    result_container = []
    fetch_thread = threading.Thread(target=fetch_sample_row, args=(result_container,))
    fetch_thread.start()
    fetch_thread.join(timeout)

    if fetch_thread.is_alive():
        # Operation took too long
        return None

    return result_container[0]


def truncate_row(example_row: dict, max_length=50) -> str:
    """Truncate the row before displaying if it is too long."""
    truncated_row = {}
    for key in example_row.keys():
        curr_row = json.dumps(example_row[key])
        truncated_row[key] = (
            curr_row
            if len(curr_row) <= max_length - 3
            else curr_row[:max_length] + "..."
        )
    return json.dumps(truncated_row)


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


def construct_search_document_v2(chunk, minimum_description_length=4):
    excluded_words = [
        "arxiv",
        "region",
        "license",
        "size_categories",
        "language_creators",
    ]

    dataset_index = {}
    counter = 1
    max_attempts = 3
    for z in range(len(chunk)):
        attempt = 0
        delay = 10
        while attempt < max_attempts:
            try:
                dataset_info = chunk[z]
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
                if "alt" == dataset_name.lower():
                    # This dataset is just taking really long..
                    continue

                all_configs = []
                for config_name in config_names:

                    if "train" not in datasets.get_dataset_split_names(
                        dataset_name, config_name
                    ):
                        continue
                    dataset = datasets.load_dataset(
                        dataset_name, config_name, split="train", streaming=True
                    )
                    sample_rows = fetch_first_row_with_timeout(dataset, timeout=30)
                    sample_rows = flatten_dict(sample_rows)
                    dataset_columns = sample_rows.keys()
                    dataset_columns = ", ".join(dataset.column_names)
                    all_configs.append(
                        {
                            "config_name": config_name,
                            "columns": dataset_columns,
                            "sample_rows": truncate_row(sample_rows),
                        }
                    )
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
                print(f"completed {counter} out of {len(chunk)}")
            except Exception as e:  # Catching a broader exception
                if "429 Client Error" in str(e):
                    # Wait for the specified delay period
                    time.sleep(delay)
                    # Double the delay for next time
                    delay *= 2
                    attempt += 1
                else:
                    # If it's not a rate limiting issue, re-raise the error
                    print(f"Error processing {dataset_info.id}: {e}")
                    continue

    return dataset_index


def worker(chunk, index, temp_folder):
    result = construct_search_document_v2(chunk)
    temp_file = temp_folder / f"temp_{index}.json"
    with open(temp_file, "w") as f:
        json.dump(result, f)
    print("written to temp file..")


def chunkify(lst, n):
    """Divide the input list into n chunks."""
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


if __name__ == "__main__":
    args = parser.parse_args()
    all_datasets = list(list_datasets())
    all_datasets = all_datasets[:10000]

    # Split the dataset into 100 chunks
    chunk_size = len(all_datasets) // 100
    chunks = list(chunkify(all_datasets, chunk_size))
    temp_folder = Path("temp_data")
    temp_folder.mkdir(exist_ok=True)

    # Setup multiprocessing
    processes = []
    for i, chunk in enumerate(chunks):
        p = multiprocessing.Process(target=worker, args=(chunk, i, temp_folder))
        processes.append(p)
        p.start()

    for p in processes:
        p.join()

    # Combine results from temp files
    dataset_index = {}
    for temp_file in temp_folder.glob("temp_*.json"):
        with open(temp_file, "r") as f:
            dataset_index.update(json.load(f))

    # Write the final result
    with open(args.dataset_index_file, "w") as f:
        json.dump(dataset_index, f)

    # Optional: clean up temp files
    for temp_file in temp_folder.glob("temp_*.json"):
        temp_file.unlink()
    temp_folder.rmdir()
