"""A simple dataset generator that uses OpenAI's GPT-3.5 API."""

import logging
from abc import ABC, abstractmethod

import openai
from datasets import Dataset
from tqdm import tqdm

from prompt2model.dataset_generator.base import DatasetGenerator, DatasetSplit
from prompt2model.prompt_parser import PromptSpec


class OpenAIDatasetGenerator(DatasetGenerator, ABC):
    """A abstract class for dataset generator using OpenAI's GPT-3.5 API."""

    def __init__(self, api_key: str, max_api_call: int = 3000):
        """Initialize an OpenAI client with an API key and max API call allowed.

        Args:
            api_key: A valid OpenAI API key.
            max_api_call: The maximum number of API calls allowed. Defaults to 3000.
        """
        openai.api_key = api_key
        self.max_api_call = max_api_call
        self.api_call_counter = 0

    @abstractmethod
    def generate_prompt(
        self, natural_instruction: str, few_shot_examples: list[str] = None
    ) -> str:
        """Generates a prompt string.

        Args:
            natural_instruction: The natural language instruction for the prompt.
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
            A tuple of (str, str), the first string is the input of generation task
            or exmaple of classification task. The second string is the output of
            generation task or label of classification task.
        """

    def generate_example(self, prompt: str) -> openai.Completion:
        """Generate a response using OpenAI's GPT-3 API.

        Args:
            prompt: A prompt asking for a response.

        Returns:
            A response object.
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
            A single dataset.
        """
        _ = split  # suppress unused variable warnings

        # expect to parse natural_instruction and few_shot_examples from prompt_spec
        # natural_instruction, few_shot_examples = prompt_spec.parse_from_prompt
        # currently hard-coded
        natural_instruction = (
            "Give me some translation from Chinese to English."
            " Input Chinese and output English."
        )
        # Get it from prompt_spec, current hard-coded in generate_prompt
        few_shot_examples = [
            "input: '人生苦短，我用 Python', output: 'Life is short, I use Python.'",
            "input: '明天是周末', output: 'Tomorrow is weekend.'",
        ]  # Get it from prompt_spec, current hard-coded in generate_prompt
        prompt = self.generate_prompt(natural_instruction, few_shot_examples)

        input_cols = []  # type: list[str]
        output_cols = []  # type: list[str]
        for example_index in tqdm(range(num_examples), desc="Generating examples"):
            while True:
                if self.api_call_counter >= self.max_api_call:
                    logging.warning("Maximum number of API calls reached.")
                    return Dataset.from_dict(
                        {"input_col": input_cols, "output_col": output_cols}
                    )
                else:
                    self.api_call_counter += 1
                response = self.generate_example(prompt)
                input_col, output_col = self.response_mining(response)
                if input_col != "" and output_col != "":
                    input_cols.append(input_col)
                    output_cols.append(output_col)
                    break
                else:
                    logging.warning(
                        "No input_col or output_col found",
                        f"for {example_index + 1} th example",
                    )

        return Dataset.from_dict({"input_col": input_cols, "output_col": output_cols})
