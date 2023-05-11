"""An interface for dataset processor."""

from abc import ABC
from functools import partial

import datasets


class DatasetProcessor(ABC):
    """A class for post-processing datasets."""

    def __init__(self, has_encoder: bool) -> None:
        """Initialize the `DatasetProcessor`.

        Args:
            has_encoder: Whether the retrieved model has an encoder.
                Encoder-decoder model like T5 has two model inputs.
                Decoder-only model like GPT only has one model input, thus
                `model_input` should be added with the `output_col`.
        """
        self.has_encoder = has_encoder

    @staticmethod
    def post_process_example(
        example: dict, instruction: str, task_id: int, has_encoder: bool
    ) -> dict:
        """Modifies the input column of a given example dictionary.

        Args:
            example: A dictionary representing an example.
            instruction: The instruction used as a prefix to explain the task.
            task_id: The dataset index in dataset_dicts, used for multi-task training.

        Returns:
            A dictionary with `model_input` as the input to models.
        """
        assert (
            "input_col" in example and "output_col" in example
        ), "Example dictionary must have 'input_col' and 'output_col' keys"
        if has_encoder:
            model_input = (
                f"<task {task_id}> {instruction} Example: {example['input_col']}"
            )
        else:
            model_input = (
                f"<task {task_id}> {instruction} Example: {example['input_col']}"
                + f" Label: {example['output_col']}"
            )
        example["model_input"] = model_input
        return example

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
        modified_dataset_dicts = []
        for task_id, dataset_dict in enumerate(dataset_dicts):
            mapping_function = partial(
                self.post_process_example,
                instruction=instruction,
                task_id=task_id,
                has_encoder=self.has_encoder,
            )
            modified_dataset_dict = dataset_dict.map(mapping_function)
            modified_dataset_dicts.append(modified_dataset_dict)
        return modified_dataset_dicts
