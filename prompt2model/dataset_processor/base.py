"""An interface for dataset processor."""

from abc import ABC, abstractmethod

import datasets


# pylint: disable=too-few-public-methods
class BaseProcesser(ABC):
    """A class for post-processing datasets."""

    @abstractmethod
    def process_dataset_dict(
        self, instruction: str, dataset_dicts: list[datasets.DatasetDict]
    ) -> list[datasets.DatasetDict]:
        """Post-process a list of DatasetDicts.

        Args:
            instruction: The instruction added to `input_col` to explain the task.
            dataset_dicts: A list of DatasetDicts (generated or retrieved).

        Returns:
            A list of DatasetDicts, all examples are converted into text2text fashion.
        """
