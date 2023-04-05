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


def generate_datasets(
    prompt_spec: PromptSpec,
) -> tuple[datasets.Dataset, datasets.Dataset, datasets.Dataset]:
    # raise NotImplementedError
    training = datasets.Dataset.from_pandas(pd.DataFrame({}))
    validation = datasets.Dataset.from_pandas(pd.DataFrame({}))
    testing = datasets.Dataset.from_pandas(pd.DataFrame({}))
    return (training, validation, testing)
