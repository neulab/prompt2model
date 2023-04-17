"""An interface for dataset generation."""

from __future__ import annotations  # noqa FI58

from abc import ABC, abstractmethod
from enum import Enum

import datasets

from prompt2model.prompt_parser import PromptSpec
from prompt2model.utils.rng import seed_generator


class DatasetSplit(Enum):
    """The split of a dataset."""

    TRAIN = "train"
    VAL = "val"
    TEST = "test"


class DatasetGenerator(ABC):
    """A class for generating datasets from a prompt specification."""


    @abstractmethod
    def generate_examples(
        self,
        prompt_spec: PromptSpec,
        num_examples: int,
        split: DatasetSplit,
    ) -> datasets.Dataset:
        """Generate data for a single named split of data.

        Args:
            prompt_spec: A prompt spec (containing a system description).
            num_examples: Number of examples in split.
            split: Name of dataset split to generate.)

        Returns:
            A single dataset split.

        """

    def generate_datasets(
        self,
        prompt_spec: PromptSpec,
        num_examples: dict[DatasetSplit, int],
        output_dir: str | None = None,
    ) -> datasets.DatasetDict:
        """Generate training/validation/testing datasets from a prompt.

        Args:
            prompt_spec: A prompt specification.
            num_examples: Number of examples per split (train/val/test/etc).

        Returns:
            A DatasetDict containing train, validation, and test splits.
        """
        dataset_dict = datasets.DatasetDict(
            {
                split: self.generate_examples(prompt_spec, num, split=split)
                for split, num in num_examples.items()
            }
        )

        if self.output_dir:
            dataset_dict.save_to_disk(self.output_dir)

        return dataset_dict
