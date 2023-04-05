"""An interface for dataset retrieval.

Input: A system description (ideally with few shot-examples removed)
Output: A list of datasets.Dataset objects
"""

import datasets

from prompt_parser import PromptSpec


def retrieve_datasets(prompt_spec: PromptSpec) -> list[datasets.Dataset]:
    # raise NotImplementedError
    return []
