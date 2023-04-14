"""An interface for dataset generation."""

from __future__ import annotations  # noqa FI58

from abc import ABC, abstractmethod
from enum import Enum

import datasets
import pandas as pd
from prompt_parser import PromptSpec
from utils.rng import seed_generator


class DatasetSplit(Enum):
    """The split of a dataset."""

    TRAIN = "train"
    VAL = "val"
    TEST = "test"


class DatasetGenerator(ABC):
    """A class for generating datasets from a prompt specification."""

    def __init__(
        self,
        model_config: dict | None = None,
        output_dir: str | None = None,
    ):
        """Construct a dataset generator."""
        self.model_config = model_config
        self.output_dir = output_dir
        self.random_seed = seed_generator.get_seed()

    @abstractmethod
    def generate_examples(
        self,
        prompt_spec: PromptSpec,
        num_examples: int | None,
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


class EmptyDatasetGenerator(DatasetGenerator):
    """A class for generating empty datasets (for testing purposes)."""

    def generate_examples(
        self,
        prompt_spec: PromptSpec,
        num_examples: int | None,
        split: DatasetSplit,
    ) -> datasets.Dataset:
        """Create empty versions of the datasets, for testing.

        Args:
            prompt_spec: A prompt specification.
            num_examples: Number of examples in split.
            split: Name of dataset split to generate.)

        Returns:
            A single dataset split.

        """
        _ = prompt_spec, split  # suppress unused variable warnings
        if num_examples is None:
            raise NotImplementedError
        else:
            col_values = ["" for i in range(num_examples)]
        # Construct empty-valued dataframe with length matching num_examples.
        df = pd.DataFrame.from_dict({"test_col": col_values})
        return datasets.Dataset.from_pandas(df)
