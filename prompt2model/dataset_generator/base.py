"""An interface for dataset generation."""

from __future__ import annotations  # noqa FI58

from abc import ABC, abstractmethod
from enum import Enum
from pathlib import Path

import datasets

from prompt2model.prompt_parser import PromptSpec


class DatasetSplit(Enum):
    """The split of a dataset."""

    TRAIN = "train"
    VAL = "val"
    TEST = "test"


class DatasetGenerator(ABC):
    """A class for generating datasets from a prompt specification."""

    @abstractmethod
    def generate_dataset_split(
        self,
        prompt_spec: PromptSpec,
        num_examples: int,
        split: DatasetSplit,
    ) -> datasets.Dataset:
        """Generate data for a single named split of data.

        Args:
            prompt_spec: A prompt spec (containing a system description).
            num_examples: Number of examples in split.
            split: Name of dataset split to generate.

        Returns:
            A single dataset split.
        """

    def generate_dataset_dict(
        self,
        prompt_spec: PromptSpec,
        num_examples: dict[DatasetSplit, int],
        output_dir: str | None = None,
    ) -> datasets.DatasetDict:
        """Generate full dataset splits (e.g. train/dev/test) from a prompt.

        Args:
            prompt_spec: A prompt specification.
            num_examples: Number of examples per split (train/val/test/etc).

        Returns:
            A DatasetDict containing train, val, and test splits.
        """
        dataset_dict = datasets.DatasetDict(
            {
                split.value: self.generate_dataset(prompt_spec, num, split=split)
                for split, num in num_examples.items()
            }
        )

        if output_dir:
            save_dir = Path(output_dir)
            save_dir.mkdir(parents=True, exist_ok=True)
            dataset_dict.save_to_disk(str(save_dir))

        return dataset_dict
