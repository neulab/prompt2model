"""Tools for testing of two datasets or datasetDicts are identical."""


from __future__ import annotations  # noqa FI58

import datasets


def are_datasets_identical(
    dataset1: datasets.Dataset, dataset2: datasets.Dataset
) -> bool:
    """Check if two datasets are identical in terms of instance values and order."""
    if len(dataset1) != len(dataset2):
        return False

    assert isinstance(dataset1, datasets.Dataset)
    assert isinstance(dataset2, datasets.Dataset)

    for instance1, instance2 in zip(dataset1, dataset2):
        if instance1 != instance2:
            return False

    return True


def are_dataset_dicts_identical(
    dataset_dict1: datasets.DatasetDict, dataset_dict2: datasets.DatasetDict
) -> bool:
    """Check if two DatasetDict objects are identical."""
    if set(dataset_dict1.keys()) != set(dataset_dict2.keys()):
        return False

    assert isinstance(dataset_dict1, datasets.DatasetDict)
    assert isinstance(dataset_dict2, datasets.DatasetDict)

    for split_name in dataset_dict1.keys():
        dataset1 = dataset_dict1[split_name]
        dataset2 = dataset_dict2[split_name]

        if not are_datasets_identical(dataset1, dataset2):
            return False

    return True
