"""A base class for dataset processor."""

from __future__ import annotations  # noqa FI58

import sys
from abc import ABC, abstractmethod
from functools import partial

import datasets


class BaseProcessor(ABC):
    """A base class for post-processing datasets."""

    def __init__(self, has_encoder: bool, eos_token: str | None = None) -> None:
        """Initialize the `BaseProcessor`.

        Args:
            has_encoder: Whether the retrieved model has an encoder.
                Encoder-decoder model like T5 has two model inputs.
                Decoder-only model like GPT only has one model input, thus
                `model_input` should be added with the `output_col`.
            eos_token: The end-of-sentence token of the tokenizer.
        """
        self.has_encoder = has_encoder
        self.eos_token = eos_token

    @staticmethod
    @abstractmethod
    def _post_process_example(
        example: dict,
        instruction: str,
        task_id: int,
        has_encoder: bool,
        dataset_split: str,
        eos_token: str,
    ) -> dict:
        """Modifies the input column of a given example dictionary.

        Args:
            example: A dictionary representing an example.
            instruction: The instruction used as a prefix to explain the task.
            task_id: A tag marking which dataset (from dataset_dicts) this
                example comes from. Used for multi-task training.
            has_encoder: Whether the retrieved model has an encoder.
            dataset_split: The split of the example, i.e. train/val/test.
            eos_token: The end-of-sentence token of the tokenizer.

        Returns:
            A dictionary with `model_input` as the input to models
            and `model_output` as the expected output of models.
        """

    def process_dataset_dict(
        self, instruction: str, dataset_dicts: list[datasets.DatasetDict]
    ) -> list[datasets.DatasetDict]:
        """Post-process a list of DatasetDicts.

        Args:
            instruction: The instruction used as a prefix to explain the task.
            dataset_dicts: A list of DatasetDicts (generated or retrieved).

        Returns:
            A list of DatasetDicts, all examples are converted into text2text fashion.
            Note that if any example contain empty `input_col` or `output_col`,
            it will be discarded.
        """
        modified_dataset_dicts = []

        def filter_empty_strings(example: dict) -> bool:
            """Filter to exclude examples with empty 'input_col' or 'output_col'.

            Args:
                example: A dictionary representing a single example in the dataset.

            Returns:
                bool: True if both 'input_col' and 'output_col' are non-empty strings,
                    False otherwise.

            Raises:
                ValueError: If no 'input_col' or 'output_col' inside the example.
            """
            if not ("input_col" in example and "output_col" in example):
                raise ValueError(
                    "Example dictionary must have 'input_col' and 'output_col' keys."
                )
            # Check if 'input_col' and 'output_col' are both non-empty strings
            return bool(str(example["input_col"])) and bool(str(example["output_col"]))

        for task_id, dataset_dict in enumerate(dataset_dicts):
            modified_dataset_dict = {}
            for dataset_split in list(dataset_dict.keys()):
                mapping_function = partial(
                    self._post_process_example,
                    instruction=instruction,
                    task_id=task_id,
                    has_encoder=self.has_encoder,
                    dataset_split=dataset_split,
                    eos_token=self.eos_token,
                )
                modified_dataset_dict[dataset_split] = (
                    dataset_dict[dataset_split]
                    .filter(filter_empty_strings)
                    .map(mapping_function, remove_columns=["input_col", "output_col"])
                )
            modified_dataset_dict = datasets.DatasetDict(modified_dataset_dict)
            modified_dataset_dicts.append(modified_dataset_dict)
        return modified_dataset_dicts

    @staticmethod
    def _split_dataset_into_dataset_dict(
        dataset,
        train_proportion: float = 0.8,
        val_proportion: float = 0.1,
        maximum_example_num: dict[str, int] | None = None,
    ) -> datasets.DatasetDict:
        """Split a given dataset into `train`, `val`, and `test` splits.

        This function takes a dataset and splits it based on specified
        proportions for train, val and test. It respects a maximum
        number of examples to be included in each set, if specified.

        Args:
            dataset: The original dataset to be split.
            train_proportion: Proportion of examples for the `train` set.
            val_proportion: Proportion of examples for the `val` set.
            maximum_example_num: Maximum number of examples
                to include in each set.

        Returns:
            datasets.DatasetDict: A dictionary containing the `train`,
                `val`, and `test` datasets.
        """
        num_of_examples = len(dataset)
        train_num = int(train_proportion * num_of_examples)
        val_num = int(val_proportion * num_of_examples)
        test_num = num_of_examples - train_num - val_num

        if maximum_example_num is not None:
            train_num = min(train_num, maximum_example_num.get("train", sys.maxsize))
            val_num = min(val_num, maximum_example_num.get("val", sys.maxsize))
            test_num = min(test_num, maximum_example_num.get("test", sys.maxsize))

        train_dataset = datasets.Dataset.from_dict(dataset[:train_num])
        val_dataset = datasets.Dataset.from_dict(
            dataset[train_num : train_num + val_num]
        )
        test_dataset = datasets.Dataset.from_dict(
            dataset[train_num + val_num : train_num + val_num + test_num]
        )

        dataset_dict = datasets.DatasetDict(
            {"train": train_dataset, "val": val_dataset, "test": test_dataset}
        )
        return dataset_dict

    @staticmethod
    def wrap_single_input(instruction: str, input: str):
        """Wrap an input string into text2text fashion to be the input of model.

        Args:
            instruction: The instruction used as a prefix to explain the task.
            input: An input string to be wrapped.

        Return:
                A wrapped input string.
        """
        return f"<task 0>{instruction}\nExample:\n{input}\nLabel:\n"

    def process_dataset_lists(
        self,
        instruction: str,
        dataset_list: list[datasets.Dataset],
        train_proportion: float = 0.8,
        val_proportion: float = 0.1,
        maximum_example_num: dict[str, int] | None = None,
    ) -> list[datasets.DatasetDict]:
        """Post-processes both the generated and retrieved datasets.

        This function takes in datasets generated by `DatasetGenerator`
        and retrieved by `DatasetRetriever`. It modifies these datasets
        based on a given instruction, converting all examples into a
        text-to-text format.

        Args:
            instruction: The instruction used as a prefix to explain the task.
            dataset_list: A list of datasets. It can be either generated by
                the DatasetGenerator or retrieved by the DatasetRetriever.
            train_proportion: The proportion of examples used for `train`.
            val_proportion: The proportion of examples used for `val`.
            maxium_example_num: The maximum number of examples to
                be used for `train`, `val` and `test`.

        Returns:
            list[datasets.DatasetDict]: A list of DatasetDicts, all examples
                are converted into text2text fashion.

        Note:
            The DatasetRetriever returns a DatasetDict with multiple splits.
                Any of these splits can be passed into this function.
            The remaining proportion after allocating to `train` and
                `val` will be used for the `test` set.
        """
        if train_proportion + val_proportion >= 1:
            raise ValueError(
                f"train_proportion {train_proportion} + val_proportion {val_proportion} must be less than 1."  # noqa E501
            )

        dataset_dicts = [
            self._split_dataset_into_dataset_dict(
                each, train_proportion, val_proportion, maximum_example_num
            )
            for each in dataset_list
        ]
        return self.process_dataset_dict(instruction, dataset_dicts)
