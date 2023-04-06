"""An interface for dataset retrieval.

Input: A system description (ideally with few shot-examples removed)
Output: A list of datasets.Dataset objects
"""

import datasets

from prompt_parser import PromptSpec


class DatasetRetriever:
    """A class for retrieving datasets."""

    def __init__(self):
        """Construct a search index from HuggingFace Datasets."""

    def retrieve_datasets(self, prompt_spec: PromptSpec) -> list[datasets.Dataset]:
        """Retrieve datasets from a prompt specification."""
        return []
