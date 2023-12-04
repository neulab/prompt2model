from __future__ import annotations  # noqa FI58

import argparse
import gc
import json
import multiprocessing
import sys
import threading
import time
from collections.abc import MutableMapping
from pathlib import Path
from typing import Any

import datasets
import requests

EXCLUDED_TAGS = [
    "arxiv",
    "region",
    "license",
    "size_categories",
    "language_creators",
]

parser = argparse.ArgumentParser()
parser.add_argument("--index", type=int, default=100)

parser.add_argument("--num_processes", type=int, default=4)


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


def replace_duplicate_columns(original_dataset_columns):
    """Utility function to remove duplicate columns, after flattening dataset."""
    columns_mapping: dict[str, str] = {}
    new_columns = []
    counter: dict[str, int] = {}
    # convert flattened columns like answer.text -> answer_text
    for col in original_dataset_columns:
        new_col = col.replace(".", "_")
        if new_col in columns_mapping.values():
            counter[new_col] = counter.get(new_col, 0) + 1
            new_col = f"{new_col}_{counter[new_col]}"
        columns_mapping[col] = new_col
        new_columns.append(new_col)
    return new_columns, columns_mapping


def fetch_first_row_with_timeout(dataset, timeout=30):
    """Don't load dataset if it takes more than 30s."""

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


def flatten_dict(d: MutableMapping, parent_key: str = "", sep: str = ".") -> dict:
    """Utility function to flatten Streaming HuggingFace datasets."""
    items: list[tuple[str, Any]] = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, MutableMapping):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def get_dataset_configs(dataset_name):
    """Get all valid configs for a given dataset."""
    config_names = datasets.get_dataset_config_names(dataset_name)
    all_configs = {}
    for config_name in config_names:
        if "train" not in datasets.get_dataset_split_names(dataset_name, config_name):
            continue
        dataset = datasets.load_dataset(
            dataset_name,
            config_name,
            split="train",
            streaming=True,
            download_mode="force_redownload",
        )
        sample_rows = fetch_first_row_with_timeout(dataset, timeout=30)
        if not sample_rows:
            raise ValueError("no sample rows")
        sample_rows = flatten_dict(sample_rows)
        if any(
            "ImageFile" in sample_rows[key].__class__.__name__
            or "DateTime" in sample_rows[key].__class__.__name__
            for key in sample_rows
        ):
            raise ValueError("Image File")
        columns, columns_mapping = replace_duplicate_columns(sample_rows.keys())

        columns = ", ".join(columns)
        all_configs[config_name] = {
            "config_name": config_name,
            "sample_row": truncate_row(sample_rows),
            "columns": columns,
            "columns_mapping": columns_mapping,
            "dataset_description": dataset.info.description,
            "dataset_name": dataset_name,
        }

        return all_configs


def process_datasets(chunk, process_index):
    """Process through the chunk of datasets and get dataset info to store."""
    dataset_index = {}
    max_attempts = 3
    for z in range(len(chunk)):
        print(f"Process index: {process_index} : currently on {z} out of {len(chunk)}")

        attempt = 0
        delay = 10
        while attempt < max_attempts:
            try:
                dataset_info = chunk[z]
                dataset_name = dataset_info["id"]
                description = dataset_info["description"]

                is_gated = hasattr(dataset_info, "gated") and dataset_info["gated"]
                if hasattr(dataset_info, "disabled") and dataset_info["disabled"]:
                    raise ValueError("dataset is disabled")
                if not get_dataset_validity(dataset_name):
                    raise ValueError("dataset is not valid")

                all_configs = get_dataset_configs(dataset_name)

                filtered_tags = [
                    tag
                    for tag in dataset_info["tags"]
                    if not any(excluded_word in tag for excluded_word in EXCLUDED_TAGS)
                ]
                dataset_index[dataset_name] = {
                    "dataset_name": dataset_name,
                    "description": description,
                    "downloads": dataset_info["downloads"],
                    "configs": all_configs,
                    "tags": filtered_tags,
                    "is_gated": is_gated,
                }
                print(
                    f"""completed {z} out of {len(chunk)}, dataset is
                    {dataset_name}, and it has {len(all_configs)} configs in it"""
                )
                del all_configs, filtered_tags

                break
            except Exception as e:
                if "429 Client Error" in str(e):
                    time.sleep(delay)
                    delay *= 2
                    attempt += 1
                else:
                    print("Error processing +", dataset_info["id"], ": ", e)
                    break
            except SystemExit as e:
                print("Error processing +", dataset_info["id"], ": ", e)
                break
            gc.collect()

    return dataset_index


def worker(chunk, index, temp_folder):
    """Utility function for Multiprocessing."""
    try:
        result = process_datasets(chunk, index)
        temp_file = temp_folder / f"temp_{index}.json"
        with open(temp_file, "w") as f:
            json.dump(result, f)
    except:  # noqa: E722
        e = sys.exc_info()[0]

        print(f"Process {index} died because of {e}.")  # noqa: E501


def chunkify(lst, n):
    """Divide the input list into n chunks."""
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


if __name__ == "__main__":
    start_time = time.time()
    args = parser.parse_args()
    all_datasets_file = "processed_datasets.json"

    with open(all_datasets_file, "r") as f:
        all_datasets = json.load(f)

    # Split the dataset into num_processes chunks
    chunk_size = len(all_datasets) // args.num_processes
    chunks = list(chunkify(all_datasets, chunk_size))
    temp_folder = Path("temp_data_" + str(args.index))
    temp_folder.mkdir(exist_ok=True)
    final_folder = Path("final_folder")
    final_folder.mkdir(exist_ok=True)
    output_file = final_folder / f"final_{args.index}.json"

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
        with open(temp_file, "r", encoding="utf-8") as f:
            dataset_index.update(json.load(f))

    # Write the final result
    with open(output_file, "w+") as f:
        json.dump(dataset_index, f)

    # Optional: clean up temp files
    for temp_file in temp_folder.glob("temp_*.json"):
        temp_file.unlink()
    temp_folder.rmdir()

    end_time = time.time()
    print(
        f"Process took {end_time-start_time} seconds, {(end_time-start_time)/60} mins"
    )
