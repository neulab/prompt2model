"""A simple dataset generator that uses OpenAI's GPT-3.5 API."""

from __future__ import annotations  # noqa FI58

import asyncio
import json
import logging
import math
import os
import random
from collections import Counter, defaultdict, namedtuple
from pathlib import Path

import openai
from datasets import Dataset
from tqdm import tqdm

from prompt2model.dataset_generator.base import DatasetGenerator, DatasetSplit
from prompt2model.dataset_generator.openai_gpt_template import construct_meta_prompt
from prompt2model.prompt_parser import PromptSpec
from prompt2model.utils import (
    OPENAI_ERRORS,
    ChatGPTAgent,
    count_tokens_from_string,
    handle_openai_error,
)

Example = namedtuple("Example", ["input_col", "output_col"])


class OpenAIDatasetGenerator(DatasetGenerator):
    """A abstract class for NLP dataset generation using OpenAI's GPT-3.5 API."""

    def __init__(
        self,
        api_key: str | None = None,
        max_api_calls: int = None,
        temperature: float = 2.0,
        presence_penalty: float = 0,
        frequency_penalty: float = 0,
        batch_size: int = 5,
        responses_per_request: int = 5,
        requests_per_minute: int = 80,
        filter_duplicated_examples: bool = True,
        cache_root: str = "cached_genrated_dataset",
    ):
        """Initializes an instance of the OpenAI DatasetGenerator.

        Args:
            api_key: A valid OpenAI API key. If not provided, the environment
                variable OPENAI_API_KEY is used.
            max_api_calls: The maximum number of API calls allowed,
                or None for unlimited.
            temperature: The sampling temperature to use, ranging from 0 to 2.
                Higher values yield more random outputs, while lower values produce
                more deterministic outputs.
            presence_penalty: Value between -2.0 and 2.0 to penalize new tokens
                based on their presence in the text so far. Positive values increase
                the model's likelihood to discuss new topics in generated examples.
            frequency_penalty: Value between -2.0 and 2.0 to penalize new tokens
                based on their frequency in the text. Positive values discourage
                the model from repeating the same line verbatim in generated examples.
            batch_size: The number of requests to make in each batch.
            responses_per_request: The number of responses for each request.
            requests_per_minute: The maximum number of requests per minute.
            filter_duplicated_examples: If True, filters duplicated examples based
                on multi-votes.
            cache_root: The root directory for caching generated examples.

        Raises:
            AssertionError: If an API key is not provided and set as an environment
            variable, or if the 'max_api_calls' value is not greater than 0.
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
        self.temperature = temperature
        self.presence_penalty = presence_penalty
        self.frequency_penalty = frequency_penalty
        self.batch_size = batch_size
        self.responses_per_request = responses_per_request
        self.requests_per_minute = requests_per_minute
        self.filter_duplicated_examples = filter_duplicated_examples
        self.cache_root = Path(cache_root)
        self.cache_root.mkdir(exist_ok=True, parents=True)
        # This list stores all generated examples. These will later be
        # converted into `generated_dataset` and `input_output_map`
        # if `filter_duplicated_examples` is True.
        self.generated_examples: list[Example] = []

        # `generated_examples` will be transformed into `generated_dataset`.
        # If `filter_duplicated_examples` is True, `generated_examples` will
        # be filtered based on multi-votes before being used to construct
        # `generated_dataset`. If it's False, `generated_examples` will be
        # used directly to construct `generated_dataset`.
        self.generated_dataset: Dataset = Dataset.from_dict({})

        # If `filter_duplicated_examples` is True, `self.generated_examples`
        # will first be converted into `input_output_map`, and then into
        # `generated_dataset`. If it's False, `input_output_map` will remain
        # empty.
        self.input_output_map: dict[str, Counter] = defaultdict(Counter)

        # `generating_split` refers to the DatasetSplit currently being
        # generated. After each loop, `generated_examples` will be
        # stored as a Dataset at the path `{cache_root}/{generating_split}`.
        self.generating_split: DatasetSplit | None = None

    def generate_prompt(
        self,
        instruction: str,
        few_shot_example_string: str = None,
    ) -> tuple[str, str]:
        """Generates a prompt string.

        Args:
            instruction: The natural language instruction for the prompt.
            few_shot_example_string: A string representing the few-shot examples
                parsed from the user's prompt, which equality is higher than the
                genrated examples.

        Returns:
            The generated prompt string.
        """
        # The random_example_string is a string, which contains several random
        # few-shot examples as demonstrations for the DatasetGenerator. If
        # self.generated_examples is empty, then the random_example_string
        # is the few-shot examples parsed from the user's prompt.
        while True:
            if len(self.generated_examples) == 0:
                low_equality_example_string = "N/A\n"
                # Create default low_equality_example_string if self.generated_examples
                # is empty. few_shot_example_string is the high-equality few-shot
                # examples parsed from the user's prompt. But if user does not
                # provideany examples in the input prompt, the few_shot_example_string
                # will be "N/A"/""/None.
                random_selected_generated_example_num = 0
                # random_selected_generated_example_num is the number of selected
                # random examples from self.generated_examples that will be used to
                # create the low_equality_example_string. If generated_examples
                # is empty, then random_selected_generated_example_num is 0.
            else:
                # If self.generated_examples is not empty, low_equality_example_string
                # is sveral random generated examples from self.generated_examples.

                low_equality_example_string = ""
                random_selected_generated_example_num = random.randint(
                    1, min(len(self.generated_examples), 10)
                )
                # random_selected_generated_example_num is the number of selected
                # random examples from self.generated_examples that will
                # be concatenated to create low_equality_example_string.
                random_examples = random.sample(
                    self.generated_examples, random_selected_generated_example_num
                )
                # If generated_examples is not empty, then choose several
                # random examples from self.generated_examples to construct
                # new low_equality_example_string.
                for example in random_examples:
                    low_equality_example_string += (
                        f'input="{example.input_col}"\noutput="{example.output_col}"\n'
                    )
            # To increase the diversity of the prompt to DatasetGenerator, create three
            # prompt templates, COMPLEX, MIDDLE, and SIMPLE. The COMPLEX template
            # contains 4 meta examples, the MIDDLE template contains 3 meta examples,
            # and the SIMPLE template contains 2 meta examples.
            template_type_dict = {1: "COMPLEX", 2: "MIDDLE", 0: "SIMPLE"}
            template_type = template_type_dict[
                random_selected_generated_example_num % 3
            ]
            prompt = construct_meta_prompt(
                instruction=instruction,
                low_equality_example_string=low_equality_example_string,
                high_equality_example_string=few_shot_example_string,
                template_type=template_type,
            )
            # The max content length of gpt-3.5-turbo is 4097, so if the
            # generated prompt is longer than 3500, then the prompt
            # should be regenerated.
            if count_tokens_from_string(prompt) < 3500:
                return prompt
            else:
                continue

    def extract_responses(self, completions: list[openai.Completion]) -> None:
        """Extracts the generated sample and annotation from an OpenAI API response.

        Args:
            completions: The generated completion objects returned by OpenAI API.

        Returns:
                Each API call will return `responses_per_request` completion objects.
                If the response is a valid JSON object, create a namedtuple called
                `example` and append it to self.generated_examples. `example` consists
                of `input_col` and`output_col`, where:
                - input_col is the generated example string extracted from the response.
                - output_col is the generated label string extracted from the response.
                If the response is not a valid JSON object, discard it.
            There is 5 * len(completions) responses at a time.
        """
        for completion in completions:
            try:
                for choice in completion.choices:
                    try:
                        response_json = json.loads(choice["message"]["content"])
                    except Exception:
                        logging.warning(f"Error happened parsing API choice: {choice}")
                        continue
                        # If the response is not a valid JSON object, discard it.
                    required_keys = ["input", "output"]
                    missing_keys = [
                        key for key in required_keys if key not in response_json
                    ]
                    if len(missing_keys) != 0:
                        logging.warning(
                            f'API response must contain {", ".join(required_keys)} keys'
                        )
                        continue
                    input = str(response_json["input"]).strip()
                    output = str(response_json["output"]).strip()
                    self.generated_examples.append(Example(input, output))
                    logging.info(f"input: \n\n{input}\n\n")  # noqa: E501
                    logging.info(f"output: \n\n{output}\n\n")  # noqa: E501
            except Exception:
                logging.warning(
                    f"Error happened when parsing API completion: {completion}"
                )
                continue

    def generate_dataset_split(
        self,
        prompt_spec: PromptSpec,
        expected_num_examples: int,
        split: DatasetSplit,
    ) -> Dataset:
        """Generate a single dataset using GPT-3.5.

        Args:
            prompt_spec: A prompt specification.
            expected_num_examples: Number of examples in split.
                Each API call will return `responses_per_request` completion
                objects. The upper bound of the length of generated dataset
                is expected_num_examples + responses_per_request.
            split: Name of dataset split to generate.

        Returns:
            A single dataset.
        """
        _ = split  # suppress unused variable warnings
        chat_api = ChatGPTAgent(self.api_key)
        self.generated_examples = []
        pbar = tqdm(total=expected_num_examples, desc="Generating examples")
        while len(self.generated_examples) < expected_num_examples:
            try:
                if self.max_api_calls and self.api_call_counter >= self.max_api_calls:
                    logging.warning("Maximum number of API calls reached.")
                    return Dataset.from_dict(
                        {
                            "input_col": [
                                example.input_col for example in self.generated_examples
                            ],
                            "output_col": [
                                example.output_col
                                for example in self.generated_examples
                            ],
                        }
                    )
                else:
                    batch_size = (
                        min(
                            self.batch_size,
                            math.ceil(
                                (
                                    (
                                        expected_num_examples
                                        - len(self.generated_examples)
                                    )
                                    / self.responses_per_request
                                )
                            ),
                        )
                        if self.max_api_calls is None
                        else min(
                            self.batch_size,
                            math.ceil(
                                (
                                    (
                                        expected_num_examples
                                        - len(self.generated_examples)
                                    )
                                    / self.responses_per_request
                                )
                            ),
                            self.max_api_calls - self.api_call_counter,
                        )
                    )
                    assert batch_size > 0
                    self.api_call_counter += batch_size
                    prompts = [
                        self.generate_prompt(
                            instruction=prompt_spec.instruction,
                            few_shot_example_string=prompt_spec.examples,
                        )
                        for _ in range(batch_size)
                    ]

                    async def generate_responses():
                        responses = (
                            await chat_api.generate_batch_openai_chat_completion(
                                prompts,
                                temperature=self.temperature,
                                responses_per_request=self.responses_per_request,
                                requests_per_minute=self.requests_per_minute,
                            )
                        )
                        return responses

                    loop = asyncio.get_event_loop()
                    responses = loop.run_until_complete(generate_responses())
                    self.extract_responses(responses)
                    pbar.update(len(self.generated_examples))
            except OPENAI_ERRORS as e:
                self.api_call_counter = handle_openai_error(e, self.api_call_counter)
        # Each API call will return `responses_per_request` completion
        # objects. The upper bound of the length of generated dataset
        # is expected_num_examples + responses_per_request.
        assert (
            len(self.generated_examples)
            < expected_num_examples + self.responses_per_request
        )
        return Dataset.from_dict(
            {
                "input_col": [example.input_col for example in self.generated_examples],
                "output_col": [
                    example.output_col for example in self.generated_examples
                ],
            }
        )
