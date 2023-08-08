"""A simple dataset generator that uses OpenAI's GPT-3.5 API."""

from __future__ import annotations  # noqa FI58

import asyncio
import json
import logging
import math
import os
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
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


@dataclass(frozen=True)
class Example:
    """An example from a dataset, containing input and output columns."""

    input_col: str
    output_col: str


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

        # `generating_split` refers to the DatasetSplit currently being
        # generated. After each loop, `generated_examples` will be
        # stored as a Dataset at the path `{cache_root}/{generating_split}`.
        self.generating_split: DatasetSplit | None = None

    def generate_prompt(
        self,
        instruction: str,
        few_shot_example_string: str,
        generated_examples: list[Example],
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
        # generated_examples is empty, then the random_example_string
        # is the few-shot examples parsed from the user's prompt.
        while True:
            if len(generated_examples) == 0:
                low_equality_example_string = "N/A\n"
                # Create default low_equality_example_string if generated_examples
                # is empty. few_shot_example_string is the high-equality few-shot
                # examples parsed from the user's prompt. But if user does not
                # provideany examples in the input prompt, the few_shot_example_string
                # will be "N/A"/""/None.
                random_selected_generated_example_num = 0
                # random_selected_generated_example_num is the number of selected
                # random examples from generated_examples that will be used to
                # create the low_equality_example_string. If generated_examples
                # is empty, then random_selected_generated_example_num is 0.
            else:
                # If generated_examples is not empty, low_equality_example_string
                # is sveral random generated examples from generated_examples.

                low_equality_example_string = ""
                random_selected_generated_example_num = random.randint(
                    1, min(len(generated_examples), 10)
                )
                # random_selected_generated_example_num is the number of selected
                # random examples from generated_examples that will
                # be concatenated to create low_equality_example_string.
                random_examples = random.sample(
                    generated_examples, random_selected_generated_example_num
                )
                # If generated_examples is not empty, then choose several
                # random examples from generated_examples to construct
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

    def construct_input_output_map(self, generated_examples) -> dict[str, Counter]:
        """Constructs a dictionary mapping inputs to `Counter` objects of outputs.

        Ideally, each input should have a unique output (one-to-one mapping).
        However, language models may occasionally generate different outputs
        for identical inputs. For instance, given the input “What is the biggest
        city in China?”, it might produce different but correct outputs such as
        “Shanghai” and “The biggest city in China is Shanghai”. At other times,
        it may produce incorrect variations. For the input “What is the Chemical
        symbol of gold?”, the outputs might be “Au”, “Au”, and “AU”, where the
        last one is wrong due to capital letters.

        To address this, OpenAIDataSetGenerator uses a two-step multi-vote
        filtering mechanism. This function represents the first step, creating a
        dictionary to map inputs to a `Counter` of their outputs.

        The function iterates over all the examples, building a dictionary where
        inputs serve as keys and `Counter` objects as values. The `Counter`
        tracks the frequency of each output for a specific input.

        For example:
        input: ["apple", "banana", "apple", "orange", "apple"]
        output: ["A", "B", "A", "O", "D"]

        Then input_output_map value is:
        {
            "apple": Counter({"A": 2, "D": 1}),
            "banana": Counter({"B": 1}),
            "orange": Counter({"O": 1})
        }
        """
        # Whenever using the multi-vote filtering mechanism, refresh
        # input_output_map to avoid duplicately countering.
        input_output_map: dict[str, Counter] = defaultdict(Counter)

        # Iterate through the examples and construct the mapping.
        for example in generated_examples:
            input_str = example.input_col
            output_str = example.output_col

            # Increment the count of the output for the specific input.
            input_output_map[input_str][output_str] += 1

        # Ensure that the generated_examples list is not empty
        # and the map is constructed correctly.
        if len(generated_examples) != 0:
            assert input_output_map

        return input_output_map

    def apply_multi_vote_to_construct_generated_dataset(
        self, input_output_map: dict[str, Counter]
    ) -> Dataset:
        """Multi-vote to construct generated_dataset from input_output_map.

        This method uses multi-vote filtering to create a unique mapping from inputs
        to outputs. The input_col of generated_dataset contains unique inputs,
        while the output_col holds the shortest, most frequent output for the
        corresponding input.

        The function asserts that self.filter_duplicated_examples is True and that
        input_output_map is not None when generated_examples is not
        empty. It then iterates over input_output_map, finding the most frequent
        output for each input. If there are multiple outputs with the highest frequency,
        it selects the shortest one. If there are multiple shortest outputs with the
        highest frequency, it selects the one that comes first in lexicographical
        (alphabetical) order.

        Example:
        Suppose input_output_map is:
        {
            "apple": Counter({"A": 2, "D": 2}),
            "banana": Counter({"B": 2, "C": 1}),
            "orange": Counter({"O": 1})
        }

        The function will produce generated_dataset:
        {
            "input_col": ["apple", "banana", "orange"],
            "output_col": ["A", "B", "O"]
        }

        Note: When generated_examples is empty, both input_output_map
        and generated_dataset will be empty.

        Returns:
            Current generated dataset with multi-vote filtering applied.
        """
        # Ensure that multi-vote filtering is enabled.
        assert self.filter_duplicated_examples

        filtered_inputs = []
        filtered_outputs = []

        for input_str, output_counter in input_output_map.items():
            # Find the most frequent output count.
            most_common_count = output_counter.most_common(1)[0][1]

            # Get all the outputs that have the most common count.
            most_frequent_outputs = [
                output
                for output, count in output_counter.items()
                if count == most_common_count
            ]

            # Sort the outputs based on their lengths and select
            # the shortest ones. When several outputs have the
            # same length with the highest frequency, they will
            # be sorted in their lexicographical (alphabetical) order.
            most_frequent_outputs.sort(key=len)
            final_output = most_frequent_outputs[0]

            filtered_inputs.append(input_str)
            filtered_outputs.append(final_output)

        # Note that when `generated_examples` is empty,
        # `input_output_map` is None, and `generated_dataset`
        # will also be empty.
        generated_dataset = Dataset.from_dict(
            {"input_col": filtered_inputs, "output_col": filtered_outputs}
        )

        # `generated_examples` will be transformed into `generated_dataset`.
        # If `filter_duplicated_examples` is True, `generated_examples` will
        # be filtered based on multi-votes before being used to construct
        # `generated_dataset`. If it's False, `generated_examples` will be
        # used directly to construct `generated_dataset`.
        return generated_dataset

    def extract_responses(
        self, completions: list[openai.Completion], generated_examples: list[Example]
    ) -> None:
        """Extracts the generated sample and annotation from an OpenAI API response.

        Args:
            completions: The generated completion objects returned by OpenAI API.
            generated_examples: Current generated examples of DatasetGenerator.

        Returns:
                Each API call will return `responses_per_request` completion objects.
                If the response is a valid JSON object, create a namedtuple called
                `example` and append it to generated_examples. `example` consists
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
                    generated_examples.append(Example(input, output))
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
        generated_examples: list[Example] = []
        pbar = tqdm(total=expected_num_examples, desc="Generating examples")
        while len(generated_examples) < expected_num_examples:
            try:
                if self.max_api_calls and self.api_call_counter >= self.max_api_calls:
                    logging.warning("Maximum number of API calls reached.")
                    return Dataset.from_dict(
                        {
                            "input_col": [
                                example.input_col for example in generated_examples
                            ],
                            "output_col": [
                                example.output_col for example in generated_examples
                            ],
                        }
                    )
                else:
                    batch_size = (
                        min(
                            self.batch_size,
                            math.ceil(
                                (
                                    (expected_num_examples - len(generated_examples))
                                    / self.responses_per_request
                                )
                            ),
                        )
                        if self.max_api_calls is None
                        else min(
                            self.batch_size,
                            math.ceil(
                                (
                                    (expected_num_examples - len(generated_examples))
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
                            generated_examples=generated_examples,
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
                    self.extract_responses(
                        responses, generated_examples=generated_examples
                    )
                    pbar.update(len(generated_examples))
            except OPENAI_ERRORS as e:
                self.api_call_counter = handle_openai_error(e, self.api_call_counter)
        # Each API call will return `responses_per_request` completion
        # objects. The upper bound of the length of generated dataset
        # is expected_num_examples + responses_per_request.
        assert (
            len(generated_examples) < expected_num_examples + self.responses_per_request
        )
        return Dataset.from_dict(
            {
                "input_col": [example.input_col for example in generated_examples],
                "output_col": [example.output_col for example in generated_examples],
            }
        )
