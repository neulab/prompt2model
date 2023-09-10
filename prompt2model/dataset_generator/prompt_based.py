"""A simple dataset generator that uses APIs."""

from __future__ import annotations

import asyncio
import json
import math
import random
from collections import Counter, defaultdict
from dataclasses import dataclass

import nest_asyncio
import openai
from datasets import Dataset
from tqdm import tqdm

from prompt2model.dataset_generator.base import DatasetGenerator, DatasetSplit
from prompt2model.dataset_generator.prompt_template import construct_meta_prompt
from prompt2model.prompt_parser import PromptSpec
from prompt2model.utils import (
    API_ERRORS,
    APIAgent,
    api_tools,
    count_tokens_from_string,
    get_formatted_logger,
    handle_api_error,
)

nest_asyncio.apply()
logger = get_formatted_logger("DatasetGenerator")


@dataclass(frozen=True)
class Example:
    """An example from a dataset, containing input and output columns."""

    input_col: str
    output_col: str

    def __eq__(self, other) -> bool:
        """Example equality."""
        return self.input_col == other.input_col and self.output_col == other.output_col

    def __lt__(self, other) -> bool:
        """Example less than."""
        return self.input_col < other.input_col or self.output_col < other.output_col


class PromptBasedDatasetGenerator(DatasetGenerator):
    """A abstract class for NLP dataset generation using a prompted API."""

    def __init__(
        self,
        max_api_calls: int = None,
        initial_temperature: float = 0.5,
        max_temperature: float = 1.7,
        presence_penalty: float = 0,
        frequency_penalty: float = 0,
        max_batch_size: int = 5,
        responses_per_request: int = 5,
        requests_per_minute: int = 80,
        filter_duplicated_examples: bool = True,
        cache_root: str = "cached_generated_dataset",
    ):
        """Initializes an instance of the PromptBasedDatasetGenerator.

        Args:
            max_api_calls: The maximum number of API calls allowed,
                or None for unlimited.
            initial_temperature: The sampling temperature to use when initializing
                the generation.
            max_temperature: The sampling temperature to use when the generation
                is about to terminate.
            presence_penalty: Value between -2.0 and 2.0 to penalize new tokens
                based on their presence in the text so far. Positive values increase
                the model's likelihood to discuss new topics in generated examples.
            frequency_penalty: Value between -2.0 and 2.0 to penalize new tokens
                based on their frequency in the text. Positive values discourage
                the model from repeating the same line verbatim in generated examples.
            max_batch_size: The maximum number of requests to make in each batch.
            responses_per_request: The number of responses for each request.
            requests_per_minute: The maximum number of requests per minute.
            filter_duplicated_examples: If True, filters duplicated examples,
                using the most-frequent output for each input.

        Raises:
            ValueError: If the 'max_api_calls' value is not greater than 0.

        Note:
            Temperature ranges from 0 to 2. Higher
            values yield more random/diverse outputs with lower quality, while
            lower values produce more deterministic outputs with higher quality.
            We use a strategy to dynamically adjust the temperature from
            initial_temperature to max_temperature during generation.

            We incorporate random few-shot generated examples into the prompt.
            The initial temperature is set lower to obtain
            high-quality, low-diversity examples. As the number of generated examples
            increases, we gradually have more high-quality examples for in-context
            learning during generation. This allows us to achieve high-quality,
            high-diversity examples later on by using a higher temperature.
        """
        if max_api_calls and max_api_calls <= 0:
            raise ValueError("max_api_calls must be > 0")
        self.max_api_calls = max_api_calls
        self.api_call_counter = 0
        self.initial_temperature = initial_temperature
        self.max_temperature = max_temperature
        if self.initial_temperature < 0:
            raise ValueError(
                f"initial_temperature must be >= 0, but {self.initial_temperature=}"
            )
        if self.max_temperature > 2.0:
            raise ValueError(
                "max_temperature must be <= 2.0 but {self.max_temperature=}"
            )
        if self.initial_temperature > self.max_temperature:
            raise ValueError(
                f"{self.initial_temperature=} must be <= {self.max_temperature=}"
            )
        self.presence_penalty = presence_penalty
        self.frequency_penalty = frequency_penalty
        self.max_batch_size = max_batch_size
        self.responses_per_request = responses_per_request
        self.requests_per_minute = requests_per_minute
        self.filter_duplicated_examples = filter_duplicated_examples

    def construct_prompt(
        self,
        instruction: str,
        few_shot_example_string: str,
        generated_examples: list[Example],
        context_cutoff: int = 3500,
    ) -> str:
        """Generates a prompt string.

        The function generates a prompt string using the provided instruction and
        few_shot_example_string. It also selects random examples from the generated
        dataset to provide additional context for the prompt. At the start of dataset
        generation, it defaults to using the few_shot_example_string.

        The function uses different prompt templates based on the number of selected
        examples from the generated dataset.

        Args:
            instruction: The natural language instruction for the prompt.
            few_shot_example_string: A string representing the few-shot examples
                parsed from the user's prompt, which quality is higher than the
                generated examples.
            generated_examples: A list of currently generated examples.
            context_cutoff: If the total length of the prompt in tokens exceeds this
                value, repeat the prompt generation process to generate a shorter one.

        Returns:
            The generated prompt string.
        """
        while True:
            # Choose a few examples to add to the prompt if examples exist.
            if len(generated_examples) == 0:
                low_quality_example_string = "N/A\n"
                random_selected_generated_example_num = 0
            else:
                low_quality_example_string = ""
                random_selected_generated_example_num = random.randint(
                    1, min(len(generated_examples), 10)
                )
                random_examples = random.sample(
                    generated_examples, random_selected_generated_example_num
                )
                for example in random_examples:
                    low_quality_example_string += (
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
                low_quality_example_string=low_quality_example_string,
                high_quality_example_string=few_shot_example_string,
                template_type=template_type,
            )
            if count_tokens_from_string(prompt) < context_cutoff:
                return prompt
            else:
                orginal_input_string = (
                    instruction + few_shot_example_string
                    if few_shot_example_string
                    else instruction
                )
                if count_tokens_from_string(orginal_input_string) > context_cutoff:
                    logger.warning(
                        "The original input prompt is too long. "
                        "Consider writing a shorter prompt."
                    )
                continue

    def apply_multi_vote_filtering(
        self,
        generated_examples: list[Example],
    ) -> list[Example]:
        """Multi-vote to construct generated_dataset from input_output_map.

        Args:
            generated_examples: A list of currently generated examples.

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

        Returns:
            Currently generated dataset with multi-vote filtering applied.
        """
        # Ensure that multi-vote filtering is enabled.
        if not self.filter_duplicated_examples:
            raise ValueError("Multi-vote filtering is not enabled.")
        filtered_examples = []

        input_output_map: dict[str, Counter] = defaultdict(Counter)

        for ex in generated_examples:
            input_output_map[ex.input_col][ex.output_col] += 1

        for input_str, output_counter in input_output_map.items():
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

            filtered_examples.append(Example(input_str, final_output))
        return filtered_examples

    def compute_batch_size(self, num_examples: int, generated_dataset_size: int) -> int:
        """Computes the batch size for API calls in a batch.

        The batch size is determined based on the remaining number of examples to be
        generated and the number of responses per request. The function also respects
        the maximum limit of API calls if it is set.

        Args:
            num_examples: The total number of examples expected to be
                generated for the current dataset split.
            generated_dataset: Currently generated dataset.

        Returns:
            The batch size for the next batch of API calls with zeno-build.
        """
        # If max_api_calls is not set, make it equivalent to the batch size
        max_api_calls = (
            self.max_batch_size
            if self.max_api_calls is None
            else self.max_api_calls - self.api_call_counter
        )
        batch_size = min(
            self.max_batch_size,
            math.ceil(
                ((num_examples - generated_dataset_size) / self.responses_per_request)
            ),
            max_api_calls,
        )
        # Ensure that the batch_size is a positive value.
        if batch_size <= 0:
            raise ValueError("Batch size must be greater than 0.")
        return batch_size

    def extract_and_append_responses(
        self, completions: list[openai.Completion], generated_examples: list[Example]
    ) -> None:
        """Extracts the generated sample and annotation from an API response.

        Args:
            completions: A list of Completion objects returned by the API.
                Each API call returns a number of completion objects equivalent to
                `responses_per_request`. The default `responses_per_request` = 5.
            generated_examples: Currently generated examples of DatasetGenerator.

        This function iterates through the provided completions, attempting to
        extract and parse the content of each completion as a JSON object. It
        then checks for the presence of `input` and `output` keys in the JSON
        object. If either is missing, the completion is discarded.

        For valid completions, the function instantiates a class
        with `input_col` and `output_col` fields, representing the generated
        example and label strings respectively. The `example` is then added
        to generated_examples.
        """
        for completion in completions:
            try:
                for choice in completion.choices:
                    try:
                        response_json = json.loads(choice["message"]["content"])
                    except Exception:
                        logger.warning(f"Error happened parsing API choice: {choice}")
                        continue
                        # If the response is not a valid JSON object, discard it.
                    required_keys = ["input", "output"]
                    missing_keys = [
                        key for key in required_keys if key not in response_json
                    ]
                    if len(missing_keys) != 0:
                        logger.warning(
                            f'API response must contain {", ".join(required_keys)} keys'
                        )
                        continue
                    input = str(response_json["input"]).strip()
                    output = str(response_json["output"]).strip()
                    if input != "" and output != "":
                        generated_examples.append(Example(input, output))
                    else:
                        logger.info(
                            "Empty input or output ditected. Discard this example."
                        )
                        continue
                    logger.info(f"input: \n\n{input}\n\n")
                    logger.info(f"output: \n\n{output}\n\n")
            except Exception:
                logger.warning(
                    f"Error happened when parsing API completion: {completion}"
                )
                continue

    async def generate_responses(
        self,
        chat_api: APIAgent,
        generated_dataset_size: int,
        expected_num_examples: int,
        prompts: list[str],
    ) -> list[openai.Completion]:
        """Asynchronously generates responses using the GPT-3.5 API.

        Args:
            chat_api: APIAgent to generate responses.
            generated_dataset: Currently generated dataset.
            expected_num_examples: The number of examples expected
                to be generated.
            prompts: A list of prompts to generate responses.

        The temperature for generating responses dynamically adjusts
        based on the size of the generated dataset. As the dataset
        grows, the dynamic temperature gradually increases,
        approaching the max_temperature.

        To prevent potential round-off errors in Python, the dynamic
        temperature is rounded within the range [0, 2.0].

        Returns:
            A list of openai.Completion.
        """
        # Calculate the dynamic temperature based
        # on the size of the generated dataset
        dynamic_temperature = (
            (self.max_temperature - self.initial_temperature)
            * generated_dataset_size
            / expected_num_examples
            + self.initial_temperature
        )

        # Ensure the dynamic temperature is within the range [0, 2.0]
        clipped_temperature = max(0.0, min(2.0, dynamic_temperature))
        responses = await chat_api.generate_batch_completion(
            prompts,
            temperature=clipped_temperature,
            responses_per_request=self.responses_per_request,
            requests_per_minute=self.requests_per_minute,
        )
        return responses

    def generate_dataset_split(
        self,
        prompt_spec: PromptSpec,
        num_examples: int,
        split: DatasetSplit = DatasetSplit.TRAIN,
    ) -> Dataset:
        """Generates a dataset split using API-based LMs.

        This method iteratively makes API calls to generate a dataset split.
        Each API call yields a batch of responses. From these responses, new examples
        are extracted and added to 'generated_examples'. The process continues
        until the desired number of examples is reached, or the maximum limit on API
        calls is reached. If an error occurs during an API call, the error is handled
        appropriately, and the API call counter is adjusted.

        Args:
            prompt_spec: PromptParser to be used for generating examples.
            num_examples: The number of examples to be generated.

        Returns:
            The generated dataset split.
        """
        all_generated_examples: list[Example] = []
        generated_examples: list[Example] = []

        pbar = tqdm(total=num_examples, desc="Generating examples")
        chat_api = api_tools.default_api_agent

        while len(generated_examples) < num_examples:
            if self.max_api_calls and self.api_call_counter >= self.max_api_calls:
                logger.warning("Maximum number of API calls reached.")
                break

            batch_size = self.compute_batch_size(num_examples, len(generated_examples))
            self.api_call_counter += batch_size

            # Generate prompts for the batch call.
            prompts = [
                self.construct_prompt(
                    instruction=prompt_spec.instruction,
                    few_shot_example_string=prompt_spec.examples,
                    generated_examples=generated_examples,
                )
                for _ in range(batch_size)
            ]

            try:
                loop = asyncio.get_event_loop()
                responses = loop.run_until_complete(
                    self.generate_responses(
                        chat_api=chat_api,
                        generated_dataset_size=len(generated_examples),
                        expected_num_examples=num_examples,
                        prompts=prompts,
                    )
                )
            except API_ERRORS as e:
                handle_api_error(e)

            # Extract the responses and add new examples to the dataset.
            prev_length = len(generated_examples)
            self.extract_and_append_responses(responses, all_generated_examples)
            generated_examples = (
                self.apply_multi_vote_filtering(all_generated_examples)
                if self.filter_duplicated_examples
                else all_generated_examples
            )

            pbar.update(len(generated_examples) - prev_length)

        if len(generated_examples) >= num_examples:
            generated_examples = generated_examples[:num_examples]

        return Dataset.from_dict(
            {
                "input_col": [ex.input_col for ex in generated_examples],
                "output_col": [ex.output_col for ex in generated_examples],
            }
        )
