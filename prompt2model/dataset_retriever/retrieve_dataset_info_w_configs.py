from __future__ import annotations  # noqa FI58

import argparse
import json
import multiprocessing
import sys
import time
from pathlib import Path

import datasets
import requests
from utils.dataset_retriever_utils import (
    fetch_first_row_with_timeout,
    flatten_dict,
    get_dataset_validity,
    replace_duplicate_columns,
    truncate_row,
)

parser = argparse.ArgumentParser()
parser.add_argument(
    "--index",
    type=int,
)

parser.add_argument("--num_processes", type=int, default=16)


def get_fully_supported_dataset_names():
    """Get the list of loadable datasets from HuggingFace."""
    API_URL = "https://datasets-server.huggingface.co/is-valid?dataset={dataset_name}"
    response = requests.get(API_URL)
    datasets_list = response.json()
    fully_supported_datasets = datasets_list["viewer"] + datasets_list["preview"]
    return fully_supported_datasets


def process_datasets(chunk, minimum_description_length=4):
    """New: Process through the chunk of datasets and get dataset info to store."""
    excluded_words = [
        "arxiv",
        "region",
        "license",
        "size_categories",
        "language_creators",
    ]

    dataset_index = {}
    max_attempts = 3
    for z in range(len(chunk)):
        print(f"currently on {z} out of {len(chunk)}")

        attempt = 0
        delay = 10
        while attempt < max_attempts:
            try:
                dataset_info = chunk[z]
                is_gated = hasattr(dataset_info, "gated") and dataset_info["gated"]
                if hasattr(dataset_info, "disabled") and dataset_info["disabled"]:
                    raise ValueError("dataset is gated or disabled")
                dataset_name = dataset_info["id"]
                description = dataset_info["description"]
                if not get_dataset_validity(dataset_name):
                    raise ValueError("dataset is not valid")
                if (
                    description is not None
                    and len(description.split()) < minimum_description_length
                ):
                    raise ValueError("dataset len not enough")

                config_names = datasets.get_dataset_config_names(dataset_name)
                print(f"{z}: {dataset_name} has {len(config_names)} config names..")
                all_configs = []
                for config_name in config_names:
                    if "train" not in datasets.get_dataset_split_names(
                        dataset_name, config_name
                    ):
                        raise ValueError(f"train not in split for {config_name}")
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
                        raise ValueError(
                            "Image File"
                        )  # We dont want to handle image datasets.
                    columns, columns_mapping = replace_duplicate_columns(
                        sample_rows.keys()
                    )

                    columns = ", ".join(columns)
                    all_configs.append(
                        {
                            "config_name": config_name,
                            "sample_row": truncate_row(sample_rows),
                            "columns": columns,
                            "columns_mapping": columns_mapping,
                            "dataset_description": dataset.info.description,
                            "dataset_name": dataset_name,
                        }
                    )
                    del dataset

                filtered_tags = [
                    tag
                    for tag in dataset_info["tags"]
                    if not any(excluded_word in tag for excluded_word in excluded_words)
                ]

                dataset_index[dataset_name] = {
                    "name": dataset_name,
                    "description": description,
                    "downloads": dataset_info["downloads"],
                    "configs": all_configs,
                    "tags": filtered_tags,
                    "is_gated": is_gated,
                }
                len_ac = len(all_configs)
                del config_names, all_configs, filtered_tags
                print(
                    f"""completed {z} out of {len(chunk)}, dataset is
                    {dataset_name}, and it has {len_ac} configs in it"""
                )
                break
            except Exception as e:  # Catching a broader exception
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
    return dataset_index


def worker(chunk, index, temp_folder):
    """Utility function for Multiprocessing."""
    try:
        result = process_datasets(chunk)
        temp_file = temp_folder / f"temp_{index}.json"
        with open(temp_file, "w") as f:
            json.dump(result, f)
        print(f"written to temp_{index} file..")
    except:  # noqa: E722
        e = sys.exc_info()[0]

        print(
            f"Process {index} died because of {e}. It was supposed to do {chunk[0]['id']} onwards.."  # noqa: E501
        )


def chunkify(lst, n):
    """Divide the input list into n chunks."""
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


if __name__ == "__main__":
    start_time = time.time()
    args = parser.parse_args()
    starting_index = args.index * 10000
    ending_index = int((args.index + 1) * 10000)
    all_datasets_file = "all_datasets.json"
    # all_datasets = list(list_datasets())
    # with open(all_datasets_file, "w") as f:
    #     ds = json.dumps([ob.__dict__ for ob in all_datasets])
    #     f.write(ds)
    with open(all_datasets_file, "r") as f:
        all_datasets = json.load(f)

    all_datasets = all_datasets[starting_index:ending_index]

    # Split the dataset into 100 chunks
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
        if p.exitcode != 0:
            print(f"Finally Process {p.name} terminated with exit code {p.exitcode}")

    # Combine results from temp files
    dataset_index = {}
    for temp_file in temp_folder.glob("temp_*.json"):
        print("File name: ", temp_file)
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
