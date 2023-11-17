from __future__ import annotations  # noqa FI58

import argparse
import json
import multiprocessing
import os
import threading
from collections.abc import MutableMapping

import datasets
import requests
from huggingface_hub import list_datasets

# Other necessary imports...



parser = argparse.ArgumentParser()
parser.add_argument(
    "--dataset-index-file",
    type=str,
    default="huggingface_data/huggingface_datasets/dataset_index.json",
)


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


def process_dataset(dataset_infos, temp_file_path, minimum_description_length=4):
    print("Entered function")

    batch_result = {}
    for dataset_info in dataset_infos:
        print("in dataset info.....", dataset_info.idx)
        try:
            excluded_words = [
                "arxiv",
                "region",
                "license",
                "size_categories",
                "language_creators",
            ]
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
            print("got config names, there are ", len(config_names), "configs")
            # if "alt" == dataset_name.lower():
            #     #This dataset is just taking really long..
            #     return

            all_configs = []
            for config_name in config_names:

                if "train" not in datasets.get_dataset_split_names(
                    dataset_name, config_name
                ):
                    continue
                dataset = datasets.load_dataset(
                    dataset_name, config_name, split="train", streaming=True
                )
                sample_rows = fetch_first_row_with_timeout(dataset)
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

            processed_data = {
                "name": dataset_name,
                "description": description,
                "downloads": dataset_info.downloads,
                "configs": all_configs,
                "tags": filtered_tags,
                "is_gated": is_gated,
            }
            batch_result[dataset_info.id] = processed_data
            del config_names, all_configs, filtered_tags
        except Exception as e:
            print(f"Error processing {dataset_info.id}: {e}")
            continue
    with open(temp_file_path, "w") as f:
        print("helloooo, ", temp_file_path)

        json.dump(batch_result, f)


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


def construct_search_document_v2_parallel(
    all_datasets, output_file, temp_dir, process_count=1, batch_size=10
):
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)

    dataset_batches = list(chunks(all_datasets, batch_size))
    temp_files = [
        os.path.join(temp_dir, f"temp_batch_{i}.json")
        for i in range(len(dataset_batches))
    ]

    with multiprocessing.Pool(process_count) as pool:
        for batch, temp_file in zip(dataset_batches, temp_files):
            pool.apply_async(process_dataset, (batch, temp_file))

    pool.close()
    pool.join()

    # Combine temp files into final JSON
    final_data = {}
    for temp_file in temp_files:
        with open(temp_file, "r") as f:
            data = json.load(f)
            final_data.update(data)
            os.remove(temp_file)  # Clean up temp file

    # Remove temporary directory
    os.rmdir(temp_dir)

    with open(output_file, "w") as f:
        json.dump(final_data, f)


if __name__ == "__main__":
    args = parser.parse_args()
    temp_dir = "temp_directory"
    all_datasets = list(list_datasets())
    construct_search_document_v2_parallel(
        all_datasets, args.dataset_index_file, temp_dir
    )
