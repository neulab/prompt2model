"""An interface for dataset processer."""

from abc import ABC, abstractmethod
from functools import partial

import datasets


class BaseProcessor(ABC):
    """A base class for post-processing datasets."""

    @abstractmethod
    def process_dataset_dict(
        self, instruction: str, dataset_dicts: list[datasets.DatasetDict]
    ) -> list[datasets.DatasetDict]:
        """Post-process a list of DatasetDicts.

        Args:
            instruction: The instruction added to `input_col` to explain the task.
            dataset_dicts: A list of DatasetDicts (generated or retrieved).

        Returns:
            A list of DatasetDicts, all examples are converted into text2text fashion.
        """


def post_process_example(example: dict, instruction: str, task_id: int) -> dict:
    """Modifies the input column of a given example dictionary.

    Args:
        example: A dictionary representing an example.
        instruction: The instruction to convert task into a text2text generation.
        task_id: The dataset index in dataset_dicts, typical for multi-task training.

    Returns:
        A dictionary representing the modified example.
    """
    assert (
        "input_col" in example and "output_col" in example
    ), "Example dictionary must have 'input_col' and 'output_col' keys"
    example["input_col"] = f"<task {task_id}> {instruction} {example['input_col']}"
    return example


class DatasetProcessor(BaseProcessor):
    """A class for post-processing datasets."""

    def process_dataset_dict(
        self, instruction: str, dataset_dicts: list[datasets.DatasetDict]
    ) -> list[datasets.DatasetDict]:
        """Post-process a list of DatasetDicts.

        Args:
            instruction: The instruction to convert example into a text2text fashion.
            dataset_dicts: A list of DatasetDicts (generated or retrieved).

        Returns:
            A list of DatasetDicts, all examples are converted into text2text fashion.
        """
        modified_dataset_dicts = []
        for task_id, dataset_dict in enumerate(dataset_dicts):
            mapping_function = partial(
                post_process_example, instruction=instruction, task_id=task_id
            )
            modified_dataset_dict = dataset_dict.map(mapping_function)
            modified_dataset_dicts.append(modified_dataset_dict)
        return modified_dataset_dicts
