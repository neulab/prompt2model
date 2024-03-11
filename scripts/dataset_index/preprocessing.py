"""Filtering out datasets before we do heavy processing on them."""
import argparse
import json
import os
from typing import Any

from huggingface_hub import list_datasets


def parse_arguments():
    """Parse command line arguments for the script.

    Returns:
        argparse.Namespace: An object with the following attributes:
            - unprocessed_datasets_file (str): Filename for unprocessed datasets.
            - preprocessed_datasets_file (str): Filename for preprocessed datasets.
            - min_words_in_desc (int): Minimum words in a dataset description.
            - min_downloads (int): Minimum downloads for a dataset.

    """
    parser = argparse.ArgumentParser(description="Process dataset files.")
    parser.add_argument(
        "--unprocessed_datasets_file",
        type=str,
        default="unprocessed.json",
        help="File name for unprocessed datasets",
    )
    parser.add_argument(
        "--preprocessed_datasets_file",
        type=str,
        default="preprocessed_datasets.json",
        help="File name for preprocessed datasets",
    )
    parser.add_argument(
        "--min_words_in_desc",
        type=int,
        default=4,
        help="Minimum number of words in description",
    )
    parser.add_argument(
        "--min_downloads", type=int, default=10, help="Minimum number of downloads"
    )

    args = parser.parse_args()
    return args


def load_datasets(file_path: str) -> list[dict[str, Any]]:
    """Load all huggingface datasets from a JSON file.

    Check if the unfiltered dataset file exists, if not generate it.
    Generating it is helpful for multiple iterations of preprocessing.

    Args:
        file_path: File path of unfiltered datasets.

    Returns:
        List of datasets from HuggingFace. This doesn't have configs or rows.
    """
    if not os.path.exists(file_path):
        all_datasets = list(list_datasets())
        with open(file_path, "w") as f:
            ds = json.dumps([ob.__dict__ for ob in all_datasets])
            f.write(ds)
        return all_datasets
    else:
        with open(file_path, "r") as file:
            return json.load(file)


def filter_datasets(
    datasets: list[dict[str, Any]], min_downloads, min_words_in_desc
) -> list[dict[str, Any]]:
    """Filter datasets based on specific criteria.

    Filter if description is None, if number of words in description is < 4,
    if number of downloads is less than a threshold, or if it is a duplicated
    dataset (if the description is the same).

    Args:
        datasets: List of datasets from HuggingFace.

    Returns:
        Datasets filtered based on above criteria.

    """
    filtered_datasets = []
    descr_none = descr_small = downloads_less = common_descr = 0
    unique_descriptions: set[str] = set()

    for dataset_info in datasets:
        description = dataset_info.get("description")

        if not description:
            descr_none += 1
            continue
        if len(description.split()) < min_words_in_desc:
            descr_small += 1
            continue
        if dataset_info.get("downloads", 0) < min_downloads:
            downloads_less += 1
            continue
        if description in unique_descriptions:
            common_descr += 1
            continue

        filtered_datasets.append(dataset_info)
        unique_descriptions.add(description)

    print(f"{descr_none=}, {descr_small=}, {downloads_less=}, {common_descr=}")

    return filtered_datasets


def main(args):
    """Main function to load and filter datasets."""
    datasets = load_datasets(args.unprocessed_datasets_file)
    filtered_datasets = filter_datasets(
        datasets=datasets,
        min_downloads=args.min_downloads,
        min_words_in_desc=args.min_words_in_desc,
    )
    with open(args.preprocessed_datasets_file, "w") as f:
        json.dump(filtered_datasets, f)


if __name__ == "__main__":
    args = parse_arguments()
    main(args)
