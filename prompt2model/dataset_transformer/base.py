"""An interface for dataset transformation."""

from __future__ import annotations  # noqa FI58

from abc import ABC, abstractmethod

import datasets

from prompt2model.prompt_parser import PromptSpec


class DatasetTransformer(ABC):
    """A class for transforming given dataset to required format."""

    @abstractmethod
    def transform_data(
        self, prompt_spec: PromptSpec, dataset: datasets.Dataset, num_transform: int
    ) -> datasets.Dataset:
        """Transform a split of data.

        Args:
            prompt_spec: A prompt spec (containing a system description).
            dataset: A dataset split.
            num_transform: number of data points you wish to transform.

        Returns:
            A single dataset split.
        """
