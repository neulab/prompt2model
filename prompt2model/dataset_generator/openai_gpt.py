"""A simple dataset generator that uses OpenAI's GPT-3.5 API."""

from __future__ import annotations  # noqa FI58

import json
import logging
import os

import openai
from datasets import Dataset
from tqdm import tqdm

from prompt2model.dataset_generator.base import DatasetGenerator, DatasetSplit
from prompt2model.prompt_parser import PromptSpec
from prompt2model.utils import OPENAI_ERRORS, ChatGPTAgent, handle_openai_error

PROMPT_TEMPLATE = (
    "You are DatasetGenerator. Here is the generation requirement for you. \n\n------\n\n"  # noqa: E501
    "Instruction: \n\n------\n\n{instruction} \n\n------\n\n"
    "Few-Shot Examples: \n\n------\n\n {examples} \n\n------\n\n"
    "Following the Instruction and, guided by the Few-Shot Examples,"
    " generate one more `input` and its `output`."
    " Please return a JSON dictionary containing `input` and `output` fields with the appropriate values."  # noqa: E501
)
# A string template for the prompt. Can be modified by the users.
# Prompt_template must contains `instruction` and `examples` fields.


class OpenAIDatasetGenerator(DatasetGenerator):
    """A abstract class for NLP dataset generation using OpenAI's GPT-3.5 API."""

    def __init__(self, api_key: str | None = None, max_api_calls: int = None):
        """Initialize an OpenAI DatasetGenerator with an API key and max API call.

        Args:
            api_key: A valid OpenAI API key. Alternatively, set as None and set
                the environment variable with `export OPENAI_API_KEY=<your key>`.
            max_api_calls: The maximum number of API calls allowed,
                or None for unlimited.
        """
        self.api_key: str | None = api_key if api_key else os.environ["OPENAI_API_KEY"]
        assert self.api_key is not None and self.api_key != "", (
            "API key must be provided"
            + " or set the environment variable with `export OPENAI_API_KEY=<your key>`"
        )
        if max_api_calls:
            assert max_api_calls > 0, "max_api_calls must be > 0"
        self.max_api_calls = max_api_calls
        self.api_call_counter = 0

    def generate_prompt(
        self,
        instruction: str,
        examples: list[str] = None,
    ) -> str:
        """Generates a prompt string.

        Args:
            instruction: The natural language instruction for the prompt.
            examples: A list of few-shot examples. Defaults to None.

        Returns:
            The generated prompt string.
        """
        # Replace placeholders in prompt template with actual values
        example_string = examples if examples else "NA"
        prompt = PROMPT_TEMPLATE.format(
            instruction=instruction, examples=example_string
        )
        return prompt

    def extract_response(self, response: openai.Completion) -> tuple[str, str]:
        """Extracts the generated sample and annotation from an OpenAI API response.

        Args:
            response (openai.Completion): The response object returned by OpenAI API.

        Returns:
            A tuple of (sample, annotation), where:
            - sample is the generated example string extracted from the response.
            - annotation is the generated label string extracted from the response.
        """
        try:
            response_json = json.loads(response.choices[0]["message"]["content"])
        except json.decoder.JSONDecodeError as e:
            logging.warning("API response was not a valid JSON")
            raise e
        required_keys = ["input", "output"]
        missing_keys = [key for key in required_keys if key not in response_json]
        assert (
            len(missing_keys) == 0
        ), f'API response must contain {", ".join(required_keys)} keys'
        input = str(response_json["input"]).strip()
        output = str(response_json["output"]).strip()
        return input, output

    def generate_dataset_split(
        self,
        prompt_spec: PromptSpec,
        num_examples: int,
        split: DatasetSplit,
    ) -> Dataset:
        """Generate a single dataset using GPT-3.5.

        Args:
            prompt_spec: A prompt specification.
            num_examples: Number of examples in split.
            split: Name of dataset split to generate.

        Returns:
            A single dataset.
        """
        _ = split  # suppress unused variable warnings
        prompt = self.generate_prompt(
            instruction=prompt_spec.get_instruction,
            examples=prompt_spec.get_examples,
        )
        logging.info(f"OpenAI Prompt: {prompt}")
        chat_api = ChatGPTAgent(self.api_key)
        input_cols = []  # type: list[str]
        output_cols = []  # type: list[str]

        for _ in tqdm(range(num_examples), desc="Generating examples"):
            while True:
                try:
                    if (
                        self.max_api_calls
                        and self.api_call_counter >= self.max_api_calls
                    ):
                        logging.warning("Maximum number of API calls reached.")
                        return Dataset.from_dict(
                            {"input_col": input_cols, "output_col": output_cols}
                        )
                    else:
                        self.api_call_counter += 1
                    response = chat_api.generate_openai_chat_completion(prompt)
                    input_col, output_col = self.extract_response(response)
                    logging.info(f"Input: {input_col}")
                    logging.info(f"Output: {output_col}")
                    input_cols.append(input_col)
                    output_cols.append(output_col)
                    break
                except OPENAI_ERRORS as e:
                    self.api_call_counter = handle_openai_error(
                        e, self.api_call_counter
                    )

        return Dataset.from_dict({"input_col": input_cols, "output_col": output_cols})
