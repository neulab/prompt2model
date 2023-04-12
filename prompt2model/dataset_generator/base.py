"""An interface for dataset generation.

Input: A system description (optionally with few-shot examples)
Output:
   1) training dataset
   2) validation dataset
   3) testing dataset
"""

from __future__ import annotations  # noqa FI58

from abc import ABC, abstractmethod
from enum import Enum

import datasets
import pandas as pd
from prompt_parser import PromptSpec


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
        self.random_seed = 2023

    @abstractmethod
    def generate_examples(
        self,
        prompt_spec: PromptSpec,
        num_examples: int | None,
        split: DatasetSplit,
    ) -> datasets.Dataset:
        """Generate data for a single named split of data.

        Args:
            prompt_spec: A prompt specification.
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
        assert num_examples.keys() == {
            DatasetSplit.TRAIN,
            DatasetSplit.VAL,
            DatasetSplit.TEST,
        }

        train_examples = self.generate_examples(
            prompt_spec, num_examples[DatasetSplit.TRAIN], split=DatasetSplit.TRAIN
        )
        val_examples = self.generate_examples(
            prompt_spec, num_examples[DatasetSplit.VAL], split=DatasetSplit.VAL
        )
        test_examples = self.generate_examples(
            prompt_spec, num_examples[DatasetSplit.TEST], split=DatasetSplit.TEST
        )

        dataset_dict = datasets.DatasetDict(
            {
                DatasetSplit.TRAIN: train_examples,
                DatasetSplit.VAL: val_examples,
                DatasetSplit.TEST: test_examples,
            }
        )

        if self.output_dir:
            dataset_dict.save_to_disk(self.output_dir)

        return dataset_dict


class BaseGenerator(DatasetGenerator):
    """A class for generating datasets from a prompt specification."""

    def set_random_seed(self, seed: int):
        """Set the random seed for reproducibility."""
        self.random_seed = seed

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
