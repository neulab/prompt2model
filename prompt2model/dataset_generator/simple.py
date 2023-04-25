"""A simple dataset generator that uses OpenAI's GPT-3.5 API."""

from abc import ABC, abstractmethod

import openai
from datasets import Dataset
from tqdm import tqdm

from prompt2model.dataset_generator.base import DatasetGenerator, DatasetSplit
from prompt2model.prompt_parser import PromptSpec


class OpenAIDatasetGenerator(DatasetGenerator, ABC):
    """A abstract class for dataset generator using OpenAI's GPT-3.5 API."""

    def __init__(self, api_key: str, max_api_call: int = None):
        """Initialize an OpenAI client with an API key and max API call allowed.

        Args:
            api_key: A valid OpenAI API key.
            max_api_call: The maximum number of API calls allowed. Defaults to 3000.
        """
        openai.api_key = api_key
        if max_api_call:
            self.max_api_call = max_api_call
        else:
            self.max_api_call = 3000
        self.current_api_call = 0

    @abstractmethod
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

    @abstractmethod
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

    def generate_example(self, prompt: str) -> openai.Completion:
        """Generate an exmaple and its pseudo_label using OpenAI's GPT-3 API.

        Args:
            prompt: A prompt asking for an example and its pseudo_label.

        Returns:
            A response object containing a generated example and its pseudo_label.
        """
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "user", "content": f"{prompt}"},
            ],
        )
        return response

    def generate_examples(
        self,
        prompt_spec: PromptSpec,
        num_examples: int,
        split: DatasetSplit,
    ) -> Dataset:
        """Generate examples using GPT-3.5.

        Args:
            prompt_spec: A prompt specification.
            num_examples: Number of examples in split.
            split: Name of dataset split to generate.

        Returns:
            A single dataset split with exmaples and pseudo_labels.
        """
        _ = prompt_spec, split  # suppress unused variable warnings
        natrual_instruction = (
            ""  # Get it from prompt_spec, current hard-coded in generate_prompt
        )
        few_shot_examples = [
            ""
        ]  # Get it from prompt_spec, current hard-coded in generate_prompt
        prompt = self.generate_prompt(natrual_instruction, few_shot_examples)

        input_cols = []  # type: list[str]
        output_cols = []  # type: list[str]
        for example_index in tqdm(range(num_examples), desc="Generating examples"):
            while True:
                if self.current_api_call >= self.max_api_call:
                    print("Maximum number of API calls reached.")
                    return Dataset.from_dict(
                        {"input_col": input_cols, "output_col": output_cols}
                    )
                else:
                    self.current_api_call += 1
                response = self.generate_example(prompt)
                input_col, output_col = self.response_mining(response)
                if (input != "") and (output_col != ""):
                    input_cols.append(input_col)
                    output_cols.append(output_col)
                    break
                else:
                    print(
                        "No input or output found",
                        f"for {example_index + 1} th example",
                    )

        return Dataset.from_dict({"input_col": input_cols, "output_col": output_cols})
