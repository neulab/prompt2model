"""A dataset processor to convert datasets into Text2Text fashion."""

from __future__ import annotations  # noqa FI58

import logging

from prompt2model.dataset_processor.base import BaseProcessor


class TextualizeProcessor(BaseProcessor):
    """A class for post-processing datasets, convert them into Text2Text fashion."""

    def __init__(self, has_encoder: bool, eos_token: str | None = None) -> None:
        """Initialize the `TextualizeProcessor`.

        Args:
            has_encoder: Whether the retrieved model has an encoder.
                Encoder-decoder model like T5 has two model inputs.
                Decoder-only model like GPT only has one model input, thus
                `model_input` should be added with the `output_col`.
            eos_token: The end-of-sentence token of the tokenizer.
                The T5 tokenizer automatically adds eos token in the end of
                sequence. So only TextualizeProcessor for GPT model
                requires eos_token.
        """
        super().__init__(has_encoder, eos_token)
        if has_encoder and eos_token is not None:
            logging.info(
                (
                    "The T5 tokenizer automatically adds eos token in the end of sequence in when tokenizing."  # noqa E501
                    " So the eos_token of encoder-decoder model tokenizer is unnecessary."  # noqa E501
                )
            )
        elif not has_encoder and eos_token is None:
            logging.warning(
                (
                    "The autoregressive model tokenizer does not automatically add eos token in the end of the sequence."  # noqa E501
                    " So the `eos_token` of the autoregressive model is required."  # noqa E501
                )
            )

    @staticmethod
    def post_process_example(
        example: dict,
        instruction: str,
        task_id: int,
        has_encoder: bool,
        dataset_split: str,
        eos_token: str | None = None,
    ) -> dict:
        """Modifies the input column of a given example dictionary.

        Args:
            example: A dictionary representing an example.
            instruction: The instruction used as a prefix to explain the task.
            task_id: A tag marking which dataset (from dataset_dicts) this example
                comes from. Used for multi-task training.
            has_encoder: Whether the retrieved model has an encoder.
            dataset_split: The split of the example, i.e. train/val/test.
            eos_token: The end-of-sentence token of the tokenizer.

        Returns:
            A dictionary with `model_input` as the input to models.
        """
        assert (
            "input_col" in example and "output_col" in example
        ), "Example dictionary must have 'input_col' and 'output_col' keys"
        assert dataset_split in (
            "train",
            "val",
            "test",
        ), "Split must be one of train/val/test"

        if has_encoder:
            model_input = (
                f"<task {task_id}>{instruction}\nExample:\n{example['input_col']}\n"
                + "Label:\n"
            )
        else:
            # The T5 tokenizer automatically adds eos token in `add eos if not present`.
            # So the output_col for T5 model should not have eos token in the end.
            # On the contrary, output_col for GPT model should add eos token in the end.
            example["output_col"] += eos_token
            if dataset_split == "train":
                model_input = (
                    f"<task {task_id}>{instruction}\nExample:\n{example['input_col']}\n"
                    + f"Label:\n{example['output_col']}"
                )
            else:
                model_input = (
                    f"<task {task_id}>{instruction}\nExample:\n{example['input_col']}\n"
                    + "Label:\n"
                )
        example["model_input"] = model_input
        return example
