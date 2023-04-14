"""An interface for dataset retrieval."""

from abc import ABC, abstractmethod

import datasets
from prompt_parser import PromptSpec


# pylint: disable=too-few-public-methods
class DatasetRetriever(ABC):
    """A class for retrieving datasets."""

    @abstractmethod
    def retrieve_datasets(self, prompt_spec: PromptSpec) -> list[datasets.Dataset]:
        """Retrieve datasets from a prompt specification.

        Args:
            prompt_spec: A prompt spec (containing a system description).

        Returns:
            A list of retrieved datasets.

        """
