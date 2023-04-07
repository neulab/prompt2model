"""An interface for dataset generation.

Input: A system description (optionally with few-shot examples)
Output:
   1) training dataset
   2) validation dataset
   3) testing dataset
"""

from abc import ABC, abstractmethod
from enum import Enum
from typing import Optional

import datasets
import pandas as pd

from prompt_parser import PromptSpec


# pylint: disable=too-few-public-methods
class DatasetGenerator(ABC):
    """
    A class for generating datasets from a prompt specification.
    """

    @abstractmethod
    def generate_datasets(
        self,
        prompt_spec: PromptSpec,
        num_train_examples: Optional[int],
        num_val_examples: Optional[int],
        num_test_examples: Optional[int],
    ) -> datasets.DatasetDict:
        """Generate training/validation/testing datasets from a prompt (which
        may include a few demonstration examples). Use a Large Language Model
        to generate a large number of examples.
        Returns:
            datasets.DatasetDict: Includes train, validation, and test splits.
        """


class DatasetSplit(Enum):
    """The split of a dataset."""

    TRAIN = "train"
    VAL = "val"
    TEST = "test"


class BaseGenerator(DatasetGenerator):
    """
    A class for generating datasets from a prompt specification.
    """

    def __init__(
        self,
        model_config: Optional[dict] = None,
        output_dir: Optional[str] = None,
    ):
        """Construct a dataset generator."""
        self.model_config = model_config
        self.output_dir = output_dir
        self.random_seed = 2023

    def set_random_seed(self, seed: int):
        """Set the random seed for reproducibility."""
        self.random_seed = seed

    def generate_examples(
        self,
        prompt_spec: PromptSpec,
        num_examples: Optional[int],
        split: DatasetSplit,
    ) -> tuple[datasets.Dataset, datasets.Dataset, datasets.Dataset]:
        """Create empty versions of the datasets, for testing.
        Returns:
            datasets.Dataset: A single dataset split."""
        _ = prompt_spec, num_examples, split  # suppress unused variable warnings
        return datasets.Dataset.from_pandas(pd.DataFrame({}))

    def generate_datasets(
        self,
        prompt_spec: PromptSpec,
        num_train_examples: Optional[int] = 5000,
        num_val_examples: Optional[int] = 1500,
        num_test_examples: Optional[int] = 500,
    ) -> datasets.DatasetDict:
        """
        Generate training/validation/testing datasets from a prompt.
        Returns:
            datasets.DatasetDict: Includes train, validation, and test splits.
        """
        train_examples = self.generate_examples(
            prompt_spec, num_train_examples, split=DatasetSplit.TRAIN
        )
        val_examples = self.generate_examples(
            prompt_spec, num_val_examples, split=DatasetSplit.VAL
        )
        test_examples = self.generate_examples(
            prompt_spec, num_test_examples, split=DatasetSplit.TEST
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
