"""Model executor for generative models, including T5-type and GPT-type."""

import logging

import datasets
import torch

from prompt2model.model_executor import ModelExecutor, ModelOutput


class GenerationModelExecutor(ModelExecutor):
    """Model executor for T5-type and GPT-type models."""

    def make_prediction(self, single_model_input: str = None) -> list[ModelOutput]:
        """Evaluate a T5-type or GPT-type model on a test set.

        Args:
            single_model_input: An optional parameter. If `single_model_input` is None,
                the model executor will make prediction on self.test_set, else it will
                make prediction on the single_model_input.

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

            input_ids = encoded_inputs["input_ids"]
            attention_mask = encoded_inputs["attention_mask"]
            device = self.model.device
            output = self.model.generate(
                input_ids=input_ids.to(device),
                attention_mask=attention_mask.to(device),
                max_length=self.sequence_max_length,
                eos_token_id=self.model.config.eos_token_id,
                early_stopping=True,
                num_beams=3,
                no_repeat_ngram_size=3,
            )

            for i, example in enumerate(batch):
                decoded_output = self.tokenizer.decode(
                    output[i], skip_special_tokens=True
                )
                logits = output[i].float()
                probs = torch.softmax(logits, dim=-1)
                confidence = probs.mean().item()
                model_output = ModelOutput(
                    prediction=decoded_output,
                    confidence=confidence,
                    auxiliary_info={
                        "example": example,
                        "input_text": input_texts[i],
                        "logits": logits,
                        "probs": probs,
                    },
                )
                model_outputs.append(model_output)

        return model_outputs

    def make_single_prediction(self, model_input: str) -> ModelOutput:
        """Mock evaluation on one example.

        Args:
            model_input: The input string to the model.

        Returns:
            A single model output, useful for exposing a model to a user interface.
        """
        model_output = self.make_prediction(single_model_input=model_input)[0]
        return model_output
