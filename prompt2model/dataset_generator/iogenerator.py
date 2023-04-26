"""A simple dataset generator that uses OpenAI's GPT-3.5 API."""

import json

import openai

from prompt2model.dataset_generator.openai import OpenAIDatasetGenerator


class InputOutputGenerator(OpenAIDatasetGenerator):
    """A dataset generator for input and output for NLG / NLI tasks."""

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
        # Get natrual_instruction and few_shot_examples from prompt_spec
        example_string = " ".join(few_shot_examples) if few_shot_examples else "NA"
        prompt = (
            f"Requirement: {natrual_instruction} \n"
            f"Few-Shot Examples: {example_string} \n"
            "input: \n"
            "output: \n"
            "Please answer me in JSON format, with `input` and `output` keys."
        )
        return prompt

    def response_mining(self, response: openai.Completion) -> tuple[str, str]:
        """Extracts the generated input and output from an OpenAI API response.

        Args:
            response (openai.Completion): The response object returned by OpenAI API.

        Returns:
            A tuple of (input, output), where:
            - input is the generated input string extracted from the
            response, or "" if not found.
            - output is the generated output string int extracted from
            the response, or "" if not found.
        """
        try:
            response_dict = json.loads(response.choices[0]["message"]["content"])
            keys = response_dict.keys()
            input = output = None
            for key in keys:
                if "input" in key.lower():
                    input = response_dict[key]
                elif "output" in key.lower():
                    output = response_dict[key]
            if input and output:
                return input, output
            else:
                print("No input or output found")
                raise ValueError("No input or output found")
        except (
            json.JSONDecodeError,
            IndexError,
            TypeError,
            ValueError,
            AttributeError,
        ):
            return "", ""
