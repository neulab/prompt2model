"""A simple dataset generator that uses OpenAI's GPT-3.5 API."""

import json

import openai

from prompt2model.dataset_generator.simple import OpenAIDatasetGenerator


class ClassifyTaskGenerator(OpenAIDatasetGenerator):
    """A dataset generator for examples and pesudo-label for classification tasks."""

    def generate_prompt(
        self, natrual_instruction: str, few_shot_examples: list[str] = None
    ) -> str:
        """Generates a prompt string.

        Args:
            natrual_instruction: The natural language instruction for the prompt.
            few_shot_examples: A list of few-shot examples. Defaults to None.

        Returns:
            The generated prompt string.
        """
        natrual_instruction = (
            "please give me a movie comment. If it's If it's positive, "
            "please give a label 1. Otherwise, give a label 0."
        )  # Get it from prompt_spec
        few_shot_examples = [
            "example: 'This movie is great!' label: 1",
            "example: 'This movie is bad!' label: 0",
        ]  # Get it from prompt_spec
        example_string = " ".join(few_shot_examples) if few_shot_examples else "NA"
        prompt = (
            f"Requirement: {natrual_instruction} \n"
            f"Few-Shot Examples: {example_string} \n"
            "example: \n"
            "label: \n"
            "Please answer me in JSON format, with `example` and `label` keys."
        )
        return prompt

    def response_mining(self, response: openai.Completion) -> tuple[str, str]:
        """Extracts the generated example and pseudo label from an OpenAI API response.

        Args:
            response (openai.Completion): The response object returned by OpenAI API.

        Returns:
            A tuple of (generated_example, pseudo_label), where:
            - generated_example is the generated example string extracted from the
            response, or "" if not found.
            - pseudo_label is the pseudo label int extracted from the response,
            or "" if not found.
        """
        try:
            response_dict = json.loads(response.choices[0]["message"]["content"])
            keys = response_dict.keys()
            generated_example = pseudo_label = None
            for key in keys:
                if "example" in key.lower():
                    generated_example = response_dict[key]
                elif "label" in key.lower():
                    pseudo_label = response_dict[key]
            if generated_example and pseudo_label:
                return generated_example, pseudo_label
            else:
                print("No example or label found")
                raise ValueError("No example or label found")
        except (
            json.JSONDecodeError,
            IndexError,
            TypeError,
            ValueError,
            AttributeError,
        ):
            return "", ""
