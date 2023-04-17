"""A mock dataset retriever for testing purposes."""

import datasets
import pandas as pd

from prompt2model.dataset_retriever.base import DatasetRetriever
from prompt2model.prompt_parser import PromptSpec


class MockRetriever(DatasetRetriever):
    """A class for retrieving datasets."""

    def __init__(self):
        """Construct a mock dataset retriever."""

    def retrieve_datasets(self, prompt_spec: PromptSpec) -> list[datasets.Dataset]:
        """Return a single empty dataset for testing purposes."""
        _ = prompt_spec  # suppress unused vaiable warning
        test_df = pd.DataFrame.from_dict({"input_col": [""], "output_col": [""]})
        return [datasets.Dataset.from_pandas(test_df)]
