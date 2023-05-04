"""A mock dataset processor for testing purposes."""

import datasets

from prompt2model.dataset_processor.base import BaseProcessor


class MockProcessor(BaseProcessor):
    """A class for retrieving datasets."""

    def process_dataset_dict(
        self, instruction: str, dataset_dicts: list[datasets.DatasetDict]
    ) -> list[datasets.DatasetDict]:
        """Post-process a list of DatasetDicts.

        Args:
            instruction: The instruction to convert example into a text2text fashion.
            dataset_dicts: A list of DatasetDicts (generated or retrieved).

        Returns:
            A list of DatasetDicts, all examples are converted into text2text fashion.
        """
        _ = instruction
        return dataset_dicts
