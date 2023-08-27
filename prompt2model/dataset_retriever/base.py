"""An interface for dataset retrieval."""

from __future__ import annotations  # noqa FI58

import dataclasses
from abc import ABC, abstractmethod

import datasets

from prompt2model.prompt_parser import PromptSpec


@dataclasses.dataclass
class DatasetInfo:
    """Store the dataset name, description, and query-dataset score for each dataset.

    Args:
        name: The name of the dataset.
        description: The description of the dataset.
        score: The retrieval score of the dataset.
    """

    name: str
    description: str
    score: float


# pylint: disable=too-few-public-methods
class DatasetRetriever(ABC):
    """A class for retrieving datasets."""

    @abstractmethod
    def retrieve_dataset_dict(
        self, prompt_spec: PromptSpec
    ) -> datasets.DatasetDict | None:
        """Retrieve full dataset splits (e.g. train/dev/test) from a prompt.

        Args:
            prompt_spec: A prompt spec (containing a system description).

        Returns:
            A retrieved DatasetDict containing train/val/test splits.
        """
