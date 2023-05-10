"""A mock dataset processor for testing purposes."""

import datasets

from prompt2model.dataset_processor.base import BaseProcessor


class MockProcessor(BaseProcessor):
    """A class for retrieving datasets."""

    def process_dataset_dict(
        self, instruction: str, dataset_dicts: list[datasets.DatasetDict]
    ) -> list[datasets.DatasetDict]:
        """A mock function post-process a list of DatasetDicts.

        Args:
            instruction: The instruction used as a prefix to explain the task.
            dataset_dicts: A list of DatasetDicts (generated or retrieved).

        Returns:
            A list of DatasetDicts, all examples are converted into text2text fashion.
        """
        _ = instruction
        return dataset_dicts

    def post_process_example(example: dict, instruction: str, task_id: int) -> dict:
        """A mock function modifies the input column of a given example dictionary.

        Args:
            example: A dictionary representing an example.
            instruction: The instruction used as a prefix to explain the task.
            task_id: The dataset index in dataset_dicts, used for multi-task training.

        Returns:
            A dictionary representing the modified example.
        """
        _ = instruction, task_id
        return example