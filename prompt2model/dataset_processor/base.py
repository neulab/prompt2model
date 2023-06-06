"""A base class for dataset processor."""

import logging
from abc import ABC, abstractmethod
from functools import partial

import datasets


class BaseProcessor(ABC):
    """A base class for post-processing datasets."""

    def __init__(self, has_encoder: bool) -> None:
        """Initialize the `BaseProcessor`.

        Args:
            has_encoder: Whether the retrieved model has an encoder.
                Encoder-decoder model like T5 has two model inputs.
                Decoder-only model like GPT only has one model input, thus
                `model_input` should be added with the `output_col`.
        """
        self.has_encoder = has_encoder

    @staticmethod
    @abstractmethod
    def post_process_example(
        example: dict,
        instruction: str,
        task_id: int,
        has_encoder: bool,
        dataset_split: str,
    ) -> dict:
        """Modifies the input column of a given example dictionary.

        Args:
            example: A dictionary representing an example.
            instruction: The instruction used as a prefix to explain the task.
            task_id: A tag marking which dataset (from dataset_dicts) this example
                comes from. Used for multi-task training.
            has_encoder: Whether the retrieved model has an encoder.
            dataset_split: The split of the example, i.e. train/val/test.
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
        """
        modified_dataset_dicts = []
        for task_id, dataset_dict in enumerate(dataset_dicts):
            for dataset_split in list(dataset_dict.keys()):
                mapping_function = partial(
                    self.post_process_example,
                    instruction=instruction,
                    task_id=task_id,
                    has_encoder=self.has_encoder,
                    dataset_split=dataset_split,
                )
                if self.has_encoder is False and dataset_split == "val":
                    logging.warning(
                        "Decoder-only model doesn't support evaluation during training"
                    )
                dataset_dict[dataset_split] = dataset_dict[dataset_split].map(
                    mapping_function
                )
            modified_dataset_dicts.append(dataset_dict)
        return modified_dataset_dicts
