"""An interface for dataset retrieval.

Input: A system description (ideally with few shot-examples removed)
Output: A list of datasets.Dataset objects
"""

from abc import ABC, abstractmethod

import datasets
import pandas as pd

from prompt_parser import PromptSpec


# pylint: disable=too-few-public-methods
class DatasetRetriever(ABC):
    """A class for retrieving datasets.

    TO IMPLEMENT IN SUBCLASSES:
    def __init__(self):
    '''Construct a search index from HuggingFace Datasets.'''
    """

    @abstractmethod
    def retrieve_datasets(self, prompt_spec: PromptSpec) -> list[datasets.Dataset]:
        """Retrieve datasets from a prompt specification.
        Returns:
            list[datasets.DatasetDict]: A list of retrieved datasets."""


class BaseRetriever(DatasetRetriever):
    """A class for retrieving datasets."""

    def __init__(self):
        """Construct a mock dataset retriever."""

    def retrieve_datasets(self, prompt_spec: PromptSpec) -> list[datasets.Dataset]:
        """Return a single empty dataset for testing purposes."""
        _ = prompt_spec  # suppress unused variable warning
        return [datasets.Dataset.from_pandas(pd.DataFrame({}))]
