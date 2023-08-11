"""A base class for dataset processor."""

from abc import ABC, abstractmethod
from functools import partial

import datasets


class BaseProcessor(ABC):
    """A base class for post-processing datasets."""

    def __init__(self, has_encoder: bool, eos_token: str) -> None:
        """Initialize the `BaseProcessor`.

        Args:
            has_encoder: Whether the retrieved model has an encoder.
                Encoder-decoder model like T5 has two model inputs.
                Decoder-only model like GPT only has one model input, thus
                `model_input` should be added with the `output_col`.
            eos_token: The end-of-sentence token of the tokenizer.
        """
        self.has_encoder = has_encoder
        self.eos_token = eos_token

    @staticmethod
    @abstractmethod
    def post_process_example(
        example: dict,
        instruction: str,
        task_id: int,
        has_encoder: bool,
        dataset_split: str,
        eos_token: str,
    ) -> dict:
        """Modifies the input column of a given example dictionary.

        Args:
            example: A dictionary representing an example.
            instruction: The instruction used as a prefix to explain the task.
            task_id: A tag marking which dataset (from dataset_dicts) this
                example comes from. Used for multi-task training.
            has_encoder: Whether the retrieved model has an encoder.
            dataset_split: The split of the example, i.e. train/val/test.
            eos_token: The end-of-sentence token of the tokenizer.

        Returns:
            A dictionary with `model_input` as the input to models
            and `model_output` as the expected output of models.
        """

    def process_dataset_dict(
        self, instruction: str, dataset_dicts: list[datasets.DatasetDict]
    ) -> list[datasets.DatasetDict]:
        """Post-process a list of DatasetDicts.

        Args:
            instruction: The instruction used as a prefix to explain the task.
            dataset_dicts: A list of DatasetDicts (generated or retrieved).

        Returns:
            A list of DatasetDicts, all examples are converted into text2text fashion.
            Note that if any example contain empty `input_col` or `output_col`,
            it will be discarded.
        """
        modified_dataset_dicts = []

        def filter_empty_strings(example: dict) -> bool:
            """Filter to exclude examples with empty 'input_col' or 'output_col'.

            Args:
                example: A dictionary representing a single example in the dataset.

            Returns:
                bool: True if both 'input_col' and 'output_col' are non-empty strings,
                    False otherwise.

            Raises:
                AssertionError: If no 'input_col' or 'output_col' inside the example.
            """
            assert (
                "input_col" in example and "output_col" in example
            ), "Example dictionary must have 'input_col' and 'output_col' keys."
            # Check if 'input_col' and 'output_col' are both non-empty strings
            return bool(example["input_col"]) and bool(example["output_col"])

        for task_id, dataset_dict in enumerate(dataset_dicts):
            modified_dataset_dict = {}
            for dataset_split in list(dataset_dict.keys()):
                mapping_function = partial(
                    self.post_process_example,
                    instruction=instruction,
                    task_id=task_id,
                    has_encoder=self.has_encoder,
                    dataset_split=dataset_split,
                    eos_token=self.eos_token,
                )
                modified_dataset_dict[dataset_split] = (
                    dataset_dict[dataset_split]
                    .filter(filter_empty_strings)
                    .map(mapping_function)
                )
            modified_dataset_dict = datasets.DatasetDict(modified_dataset_dict)
            modified_dataset_dicts.append(modified_dataset_dict)
        return modified_dataset_dicts
