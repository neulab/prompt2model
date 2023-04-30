"""A simple dataset generator that uses OpenAI's GPT-3.5 API."""

import json
import logging

import openai
from datasets import Dataset
from tqdm import tqdm

from prompt2model.dataset_generator.base import DatasetGenerator, DatasetSplit
from prompt2model.prompt_parser import PromptSpec
from prompt2model.utils.openai_tools import generate_openai_chat_completion


class OpenAIDatasetGenerator(DatasetGenerator):
    """A abstract class for NLP dataset generation using OpenAI's GPT-3.5 API."""

    def __init__(self, api_key: str, max_api_calls: int = None):
        """Initialize an OpenAI client with an API key and max API call allowed.

        Args:
            api_key: A valid OpenAI API key.
            max_api_calls: The maximum number of API calls allowed,
                or None for unlimited.
        """
        self.api_key = api_key
        openai.api_key = self.api_key
        self.max_api_calls = max_api_calls
        self.api_call_counter = 0

    def generate_prompt(
        self,
        instruction: str,
        examples: list[str] = None,
        prompt_template: str = None,
    ) -> str:
        """Generates a prompt string.

        Args:
            instruction: The natural language instruction for the prompt.
            examples: A list of few-shot examples. Defaults to None.
            prompt_template: A string template for the prompt. Defaults to None.
                Prompt_template must contains `instruction` and `examples` fields.

        Returns:
            The generated prompt string.
        """
        # Set default prompt template if not provided
        if not prompt_template:
            prompt_template = (
                "Requirement: {instruction} \n"
                "Few-Shot Examples: {examples} \n"
                "sample: \n"
                "annotation: \n"
                "Please answer me in JSON format, with `sample` and `annotation` keys."
            )

        # Replace placeholders in prompt template with actual values
        example_string = " ".join(examples) if examples else "NA"
        prompt = prompt_template.format(
            instruction=instruction, examples=example_string
        )
        return prompt

    def extract_response(self, response: openai.Completion) -> tuple[str, str]:
        """Extracts the generated sample and annotation from an OpenAI API response.

        Args:
            response (openai.Completion): The response object returned by OpenAI API.

        Returns:
            A tuple of (sample, annotation), where:
            - sample is the generated example string extracted from the
            response, or "" if not found.
            - annotation is the generated label/annotation string int extracted from
            the response, or "" if not found.
        """
        try:
            response_dict = json.loads(response.choices[0]["message"]["content"])
            keys = response_dict.keys()
            sample = annotation = None
            for key in keys:
                if "sample" in key.lower():
                    sample = response_dict[key]
                elif "annotation" in key.lower():
                    annotation = response_dict[key]
            if sample and annotation:
                return sample, annotation
            else:
                logging.error("No sample or annotation found")
                raise ValueError("No sample or annotation found")
        except (
            json.JSONDecodeError,
            IndexError,
            TypeError,
            ValueError,
            AttributeError,
        ):
            return "", ""

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
        prompt = self.generate_prompt(
            instruction=prompt_spec.instruction,
            examples=prompt_spec.examples,
            prompt_template=prompt_spec.prompt_template,
        )

        input_cols = []  # type: list[str]
        output_cols = []  # type: list[str]
        for example_index in tqdm(range(num_examples), desc="Generating examples"):
            while True:
                if self.max_api_calls and self.api_call_counter >= self.max_api_calls:
                    logging.warning("Maximum number of API calls reached.")
                    return Dataset.from_dict(
                        {"input_col": input_cols, "output_col": output_cols}
                    )
                else:
                    self.api_call_counter += 1
                response = generate_openai_chat_completion(self.api_key, prompt)
                input_col, output_col = self.extract_response(response)
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
