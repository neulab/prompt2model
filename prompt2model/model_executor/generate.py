"""Model executor for generative models, including T5-type and GPT-type."""
from __future__ import annotations  # noqa FI58

import logging
from typing import Any

import datasets
import torch

from prompt2model.model_executor import ModelExecutor, ModelOutput


class GenerationModelExecutor(ModelExecutor):
    """Model executor for T5-type and GPT-type models."""

    def generate(
        self,
        input_ids: list[torch.Tensor],
        attention_mask: list[torch.Tensor],
        hyperparameter_choices: dict[str, Any],
    ) -> list[torch.Tensor]:
        """Generates sequences of token IDs using the model.

        Args:
            input_ids: A list of token ID sequences.
            attention_mask: A list of binary masks indicating attended tokens.
            hyperparameter_choices: A dictionary of hyperparameters for inference.

        Returns:
            A list of model output tensors, one for each element in input_ids.
        """
        generate_strategy = hyperparameter_choices.get("generate_strategy", "greedy")
        assert generate_strategy in [
            "beam",  # beam search.
            "top_k",  # top_k sampling.
            "top_p",  # top_p sampling.
            "greedy",  # greedy search.
            "intersect",  # If both top_k and top_p are set, the model will
            # sample from the intersection of the top-k tokens and the top-p tokens.
        ], f"Only support top_k/top_p/intersect sampling and beam/greedy search for inference. But the passed in generate_strategy is {generate_strategy}"  # noqa 501
        if generate_strategy == "greedy":
            output = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=self.sequence_max_length,
                eos_token_id=self.model.config.eos_token_id,
                early_stopping=True,
                repetition_penalty=hyperparameter_choices.get(
                    "repetition_penalty", 2.0
                ),
            )
        elif generate_strategy == "beam":
            output = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=self.sequence_max_length,
                eos_token_id=self.model.config.eos_token_id,
                early_stopping=True,
                do_sample=False,
                repetition_penalty=hyperparameter_choices.get(
                    "repetition_penalty", 2.0
                ),
                num_beams=hyperparameter_choices.get("num_beams", 3),
            )
        elif generate_strategy == "top_k":
            output = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=self.sequence_max_length,
                eos_token_id=self.model.config.eos_token_id,
                early_stopping=True,
                do_sample=True,
                repetition_penalty=hyperparameter_choices.get(
                    "repetition_penalty", 2.0
                ),
                top_k=hyperparameter_choices.get("top_k", 20),
            )
        elif generate_strategy == "top_p":
            output = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=self.sequence_max_length,
                eos_token_id=self.model.config.eos_token_id,
                early_stopping=True,
                do_sample=True,
                repetition_penalty=hyperparameter_choices.get(
                    "repetition_penalty", 2.0
                ),
                top_p=hyperparameter_choices.get("top_p", 0.95),
            )
        else:
            # For intersect sampling.
            output = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=self.sequence_max_length,
                eos_token_id=self.model.config.eos_token_id,
                early_stopping=True,
                do_sample=True,
                repetition_penalty=hyperparameter_choices.get(
                    "repetition_penalty", 2.0
                ),
                top_k=hyperparameter_choices.get("top_k", 20),
                top_p=hyperparameter_choices.get("top_p", 0.95),
            )
        return output

    def make_prediction(
        self,
        single_model_input: str = None,
        hyperparameter_choices: dict[str, Any] | None = None,
    ) -> list[ModelOutput]:
        """Evaluate a T5-type or GPT-type model on a test set.

        Args:
            single_model_input: An optional parameter. If `single_model_input` is None,
                the model executor will make prediction on self.test_set, else it will
                make prediction on the single_model_input.
            hyperparameter_choices: A dictionary of hyperparameter for inference.

        Returns:
            A list of model outputs, one for each element in the test set.
        """
        model_outputs = []
        if not single_model_input:
            inference_column = self.input_column
            num_examples = len(self.test_set)
            inference_dataset = self.test_set
        else:
            logging.info("Making single prediction for DemoCreator.")
            num_examples = 1
            inference_dataset = datasets.Dataset.from_dict(
                {"model_input": [single_model_input]}
            )
            inference_column = "model_input"
            assert len(inference_dataset) == num_examples
        longest_input = max(inference_dataset[inference_column], key=len)
        if (
            self.tokenizer_max_length is not None
            and len(self.tokenizer.tokenize(longest_input)) > self.tokenizer_max_length
        ):
            logging.warning(
                (
                    "Truncation happened when tokenizing dataset / input string."
                    " You should consider increasing the tokenizer_max_length."
                    " Otherwise the truncation may lead to unexpected results."
                )
            )
            inference_column = "model_input"
            assert len(inference_dataset) == num_examples

        for start_idx in range(0, num_examples, self.batch_size):
            end_idx = min(start_idx + self.batch_size, num_examples)
            batch = datasets.Dataset.from_dict(inference_dataset[start_idx:end_idx])

            input_texts = batch[inference_column]
            encoded_inputs = self.tokenizer.batch_encode_plus(
                input_texts,
                truncation=True,
                max_length=self.tokenizer_max_length,
                padding=True,
                return_tensors="pt",
            )
            device = self.model.device
            input_ids = encoded_inputs["input_ids"].to(device)
            attention_mask = encoded_inputs["attention_mask"].to(device)
            hyperparameter_choices = (
                hyperparameter_choices
                if hyperparameter_choices is not None
                else {"generate_strategy": "greedy"}
            )
            output = self.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                hyperparameter_choices=hyperparameter_choices,
            )

            for idx, input_text in enumerate(input_texts):
                logits = output[idx]
                decoded_output = self.tokenizer.decode(logits, skip_special_tokens=True)
                model_output = ModelOutput(
                    prediction=decoded_output,
                    auxiliary_info={
                        "input_text": input_text,
                        "logits": logits,
                    },
                )
                model_outputs.append(model_output)

        return model_outputs

    def make_single_prediction(
        self, model_input: str, hyperparameter_choices: dict[str, Any] | None = None
    ) -> ModelOutput:
        """Mock evaluation on one example.

        Args:
            model_input: The input string to the model.
            hyperparameter_choices: A dictionary of hyperparameter for inference.

        Returns:
            A single model output, useful for exposing a model to a user interface.
        """
        model_output = self.make_prediction(
            single_model_input=model_input,
            hyperparameter_choices=hyperparameter_choices,
        )[0]
        return model_output
