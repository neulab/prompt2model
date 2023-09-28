"""A mock dataset processor for testing purposes."""
from __future__ import annotations

import datasets

from prompt2model.dataset_processor.base import BaseProcessor


class MockProcessor(BaseProcessor):
    """A class for retrieving datasets."""

    def process_dataset_dict(
        self, instruction: str, dataset_dicts: list[datasets.DatasetDict]
    ) -> list[datasets.DatasetDict]:
        """A mock function to post-process a list of DatasetDicts.

        Args:
            instruction: The instruction used as a prefix to explain the task.
            dataset_dicts: A list of DatasetDicts (generated or retrieved).

        Returns:
            A list of DatasetDicts, all examples are converted into text2text fashion.
        """
        _ = instruction
        return dataset_dicts

    @staticmethod
    def _post_process_example(
        example: dict,
        instruction: str,
        task_id: int,
        has_encoder: bool,
        dataset_split: str,
        eos_token: str,
    ) -> dict:
        """A mock function that modifies a given example dictionary.

        Args:
            example: A dictionary representing an example.
            instruction: The instruction used as a prefix to explain the task.
            task_id: A tag marking which dataset (from dataset_dicts) this example
                comes from. Used for multi-task training.
            has_encoder: Whether the retrieved model has an encoder.
            dataset_split: The split of the example, i.e. train/val/test.
            eos_token: The end-of-sentence token of the tokenizer.

        Returns:
            A dictionary with `model_input` as the input to models
            and `model_output` as the expected output of models.
        """
        _ = instruction, task_id
        example["model_input"] = example["input_col"]
        example["model_output"] = example["output_col"]
        return example
