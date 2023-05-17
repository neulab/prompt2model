"""Model executor for generative models, including T5-type and GPT-type."""

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
        for example in self.test_set:
            input_text = example[self.input_column]
            encoded_input = self.tokenizer(
                input_text, truncation=True, padding=True, return_tensors="pt"
            )
            if issubclass(
                self.model.__class__, transformers.T5ForConditionalGeneration
            ):
                output = self.model.generate(**encoded_input)
            else:
                output = self.model.generate(input_ids=encoded_input["input_ids"])
            decoded_output = self.tokenizer.decode(output[0], skip_special_tokens=True)
            logits = output[0].float()
            probs = torch.softmax(logits, dim=-1)
            confidence = probs.mean().item()
            model_output = ModelOutput(
                prediction=decoded_output,
                confidence=confidence,
                auxiliary_info={
                    "example": example,
                    "input_text": input_text,
                    "logits": logits,
                    "probs": probs,
                },
            )
            model_outputs.append(model_output)

        return model_outputs
