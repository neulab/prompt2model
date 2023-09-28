"""A dataset processor to convert datasets into Text2Text fashion."""

from __future__ import annotations  # noqa FI58

from prompt2model.dataset_processor.base import BaseProcessor
from prompt2model.utils.logging_utils import get_formatted_logger

logger = get_formatted_logger("DatasetProcessor")


class TextualizeProcessor(BaseProcessor):
    """A class for pre-processing datasets before training."""

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
            logger.info(
                (
                    "The T5 tokenizer automatically adds eos token in the end of sequence when tokenizing."  # noqa E501
                    " So the eos_token of encoder-decoder model tokenizer is unnecessary."  # noqa E501
                )
            )
        elif not has_encoder and eos_token is None:
            logger.warning(
                (
                    "The autoregressive model tokenizer does not automatically add eos token in the end of the sequence."  # noqa E501
                    " So the `eos_token` of the autoregressive model is required."  # noqa E501
                )
            )

    @staticmethod
    def _post_process_example(
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
            A dictionary with `model_input` as the input to models
            and `model_output` as the expected output of models.
        """
        if dataset_split not in (
            "train",
            "val",
            "test",
        ):
            raise ValueError("Datset split must be in train/val/test.")
        example["output_col"] = str(example["output_col"])
        if has_encoder:
            model_input = f"<task {task_id}>{instruction}\nExample:\n{example['input_col']}\nLabel:\n"  # noqa E501
            model_output = example["output_col"]
        else:
            # The T5 tokenizer automatically adds eos token in `add_eos_if_not_present`.
            # On the contrary, model_output of GPT model need eos token in the end.
            if dataset_split == "train":
                model_output = example["output_col"] + eos_token
                model_input = f"<task {task_id}>{instruction}\nExample:\n{example['input_col']}\nLabel:\n{model_output}"  # noqa E501
            else:
                # The val/test split is only used for evaluation. Since our decode
                # method in the ModelExecutor set `skip_special_tokens=True`,
                # we do not need to add eos token in the end.
                model_output = example["output_col"]
                model_input = f"<task {task_id}>{instruction}\nExample:\n{example['input_col']}\nLabel:\n"  # noqa E501
        example["model_input"] = model_input
        example["model_output"] = model_output
        return example
