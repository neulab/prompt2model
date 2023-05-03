"""An interface for dataset retrieval."""

from abc import ABC, abstractmethod

import datasets

from prompt2model.prompt_parser import PromptSpec


# pylint: disable=too-few-public-methods
class DatasetRetriever(ABC):
    """A class for retrieving datasets."""

    @abstractmethod
    def retrieve_dataset_dict(
        self, prompt_spec: PromptSpec
    ) -> list[datasets.DatasetDict]:
        """Retrieve DatasetDicts from a prompt specification.

        Args:
            prompt_spec: A prompt spec (containing a system description).

        Returns:
            A list of retrieved DatasetDict containing train/val/test splits.
        """
