import datasets
import pandas as pd
from typing import Tuple

from prompt_parser import PromptSpec

# Input: A system description (optionally with few-shot examples)
# Output:
#    1) training dataset (datasets.Dataset)
#    2) validation dataset (datasets.Dataset)
#    3) testing dataset (datasets.Dataset)


def generate_datasets(
    prompt_spec: PromptSpec,
) -> Tuple[datasets.Dataset, datasets.Dataset, datasets.Dataset]:
    # raise NotImplementedError
    training = datasets.Dataset.from_pandas(pd.DataFrame({}))
    validation = datasets.Dataset.from_pandas(pd.DataFrame({}))
    testing = datasets.Dataset.from_pandas(pd.DataFrame({}))
    return (training, validation, testing)
