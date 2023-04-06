"""An interface for dataset generation.

Input: A system description (optionally with few-shot examples)
Output:
   1) training dataset
   2) validation dataset
   3) testing dataset
"""

import datasets
import pandas as pd

from prompt_parser import PromptSpec


class DatasetGenerator:
    """A class for generating datasets from a prompt specification."""

    def __init__(self):
        """Construct a dataset generator."""
        pass

    def generate_datasets(
        prompt_spec: PromptSpec,
    ) -> datasets.DatasetDict:
        """Generate training/validation/testing datasets from a prompt (which may
        include a few demonstration examples). The typical implementation of this
        will use a Large Language Model to generate a large number of examples."""
        training = datasets.Dataset.from_pandas(pd.DataFrame({}))
        validation = datasets.Dataset.from_pandas(pd.DataFrame({}))
        testing = datasets.Dataset.from_pandas(pd.DataFrame({}))
        return (training, validation, testing)
