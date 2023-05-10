"""An interface for dataset processor."""

from abc import ABC, abstractmethod
from functools import partial

import datasets


class BaseProcessor(ABC):
    """A class for post-processing datasets."""

    @abstractmethod
    def process_dataset_dict(
        self, instruction: str, dataset_dicts: list[datasets.DatasetDict]
    ) -> list[datasets.DatasetDict]:
        """Post-process a list of DatasetDicts.

        Args:
            instruction: The instruction used as a prefix to explain the task.
            dataset_dicts: A list of DatasetDicts (generated or retrieved).

        Returns:
            A list of DatasetDicts, all examples are converted into text2text fashion.
        """


class T5Processor(BaseProcessor):
    """A post-processing class for datasets, converting task into text2text fashion."""

    @staticmethod
    def post_process_example(example: dict, instruction: str, task_id: int) -> dict:
        """Modifies the input column of a given example dictionary.

        Args:
            example: A dictionary representing an example.
            instruction: The instruction to convert task into a text2text generation.
            task_id: The dataset index in dataset_dicts, used for multi-task training.

        Returns:
            A dictionary representing the modified example.
        """
        assert (
            "input_col" in example and "output_col" in example
        ), "Example dictionary must have 'input_col' and 'output_col' keys"
        example["input_col"] = f"<task {task_id}> {instruction} {example['input_col']}"
        return example

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
                self.post_process_example, instruction=instruction, task_id=task_id
            )
            modified_dataset_dict = dataset_dict.map(mapping_function)
            modified_dataset_dicts.append(modified_dataset_dict)
        return modified_dataset_dicts
