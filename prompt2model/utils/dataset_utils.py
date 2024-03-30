"""Util functions for datasets."""


import datasets
import requests

from prompt2model.utils.logging_utils import get_formatted_logger

logger = get_formatted_logger("dataset_utils")


def query(API_URL):
    """Returns a response json for a URL."""
    try:
        response = requests.get(API_URL)
        if response.status_code == 200:
            return response.json()
        else:
            logger.error(f"Error occurred in fetching size: {response.status_code}")
    except requests.exceptions.RequestException as e:
        logger.error("Error occurred in making the request: " + str(e))

    return {}


def get_dataset_size(dataset_name):
    """Fetches dataset size for a dataset in MB from hugging face API."""
    API_URL = f"https://datasets-server.huggingface.co/size?dataset={dataset_name}"
    data = query(API_URL)
    size_dict = data.get("size", {})
    return (
        "NA"
        if size_dict == {}
        else "{:.2f}".format(size_dict["dataset"]["num_bytes_memory"] / 1024 / 1024)
    )


def make_combined_datasets(
    dataset_list: list[datasets.Dataset], dataset_type: str = "input_output"
) -> datasets.Dataset:
    """Combine multiple datasets into one.

    Args:
        dataset_list: List of datasets to combine.
        dataset_type: Type of dataset to combine. Can be "text" or "input_output".
        "text" is for combining datasets with a single column "text".
        "input_output" is for combining datasets with 2 columns "input_col"
        and "output_col".

    Returns:
        A combined dataset.
        Singe column "text" if dataset_type is "text".
        Two columns "input_col" and "output_col" if dataset_type is "input_output".
        ValueError if dataset_type is not "text" or "input_output".
    """
    if dataset_type == "text":
        text_col = []
        for dataset in dataset_list:
            text_col.extend(dataset["text"])
        return datasets.Dataset.from_dict({"text": text_col})
    elif dataset_type == "input_output":
        input_col = []
        output_col = []
        for dataset in dataset_list:
            input_col.extend(dataset["input_col"])
            output_col.extend(dataset["output_col"])

        dataset = datasets.Dataset.from_dict(
            {"input_col": input_col, "output_col": output_col}
        )
        return dataset
    else:
        raise ValueError(
            f"dataset_type can be either 'text' or 'input_output' but got {dataset_type}"  # noqa E501
        )


def format_train_data(train_dataset: datasets.Dataset):
    """Formats the train dataset for training."""
    final_texts = []
    for row in train_dataset:
        final_texts.append(f"{row['input_col'].strip()} {row['output_col'].strip()}")
    return datasets.Dataset.from_dict({"text": final_texts})
