"""An interface for dataset retrieval.
"""

from typing import Iterable

import datasets

from prompt_parser import PromptSpec

# Input: A system description (ideally with few shot-examples removed)
# Output: A list of datasets.Dataset objects


def retrieve_datasets(prompt_spec: PromptSpec) -> Iterable[datasets.Dataset]:
    # raise NotImplementedError
    return []
