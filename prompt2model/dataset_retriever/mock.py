"""A mock dataset retriever for testing purposes."""

from __future__ import annotations  # noqa FI58

import datasets

from prompt2model.dataset_retriever.base import DatasetRetriever
from prompt2model.prompt_parser import PromptSpec


class MockRetriever(DatasetRetriever):
    """A class for retrieving datasets."""

    def __init__(self):
        """Construct a mock dataset retriever."""

    def retrieve_dataset_dict(
        self, prompt_spec: PromptSpec
    ) -> datasets.DatasetDict | None:
        """Return a single empty DatasetDict for testing purposes."""
        _ = prompt_spec  # suppress unused vaiable warning
        mock_dataset = datasets.Dataset.from_dict(
            {"input_col": [""], "output_col": [""]}
        )
        return [
            datasets.DatasetDict(
                {"train": mock_dataset, "val": mock_dataset, "test": mock_dataset}
            )
        ]
