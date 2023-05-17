"""Model executor for generative models, including T5-type and GPT-type."""

import datasets
import torch
import transformers

from prompt2model.model_executor import ModelExecutor, ModelOutput


class GenerationModelExecutor(ModelExecutor):
    """Model executor for T5-type and GPT-type models."""

    def make_predictions(self) -> list[ModelOutput]:
        """Evaluate a T5-type or GPT-type model on a test set.

        Returns:
            A list of model outputs, one for each element in the test set.
        """
        assert self.input_column == "model_input"

        model_outputs = []
        num_examples = len(self.test_set)

        for start_idx in range(0, num_examples, self.batch_size):
            end_idx = min(start_idx + self.batch_size, num_examples)
            batch = datasets.Dataset.from_dict(self.test_set[start_idx:end_idx])

            input_texts = batch[self.input_column]
            encoded_inputs = self.tokenizer.batch_encode_plus(
                input_texts,
                truncation=True,
                padding=True,
                return_tensors="pt",
            )

            input_ids = encoded_inputs["input_ids"]
            attention_mask = encoded_inputs["attention_mask"]

            if issubclass(
                self.model.__class__, transformers.T5ForConditionalGeneration
            ):
                output = self.model.generate(
                    input_ids=input_ids, attention_mask=attention_mask
                )
            else:
                output = self.model.generate(
                    input_ids=input_ids, attention_mask=attention_mask
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
