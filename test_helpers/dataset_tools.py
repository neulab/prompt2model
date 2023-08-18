"""Tools for testing if two Datasets or DatasetDicts are identical."""


from __future__ import annotations  # noqa FI58

import datasets


def are_datasets_identical(
    dataset1: datasets.Dataset, dataset2: datasets.Dataset
) -> bool:
    """Check if two datasets are identical in terms of instance values and order."""
    if len(dataset1) != len(dataset2):
        return False

    return all(
        instance1 == instance2 for instance1, instance2 in zip(dataset1, dataset2)
    )


def are_dataset_dicts_identical(
    dataset_dict1: datasets.DatasetDict, dataset_dict2: datasets.DatasetDict
) -> bool:
    """Check if two DatasetDict objects are identical."""
    if set(dataset_dict1.keys()) != set(dataset_dict2.keys()):
        return False

    return all(
        are_datasets_identical(dataset_dict1[split_name], dataset_dict2[split_name])
        for split_name in dataset_dict1.keys()
    )
