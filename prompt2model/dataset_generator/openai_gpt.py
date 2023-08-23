"""A simple dataset generator that uses OpenAI's GPT-3.5 API."""

from __future__ import annotations  # noqa FI58

import asyncio
import json
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
    get_formatted_logger,
    handle_openai_error,
)

logger = get_formatted_logger("DatasetGenerator")


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
        """Initializes an instance of the OpenAI DatasetGenerator.

        Args:
            api_key: A valid OpenAI API key. If not provided, the environment
                variable OPENAI_API_KEY is used.
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
            cache_root: The root directory for caching generated examples.

        Raises:
            AssertionError: If an API key is not provided and set as an environment
            variable, or if the 'max_api_calls' value is not greater than 0.

        Note:
            For the OpenAI GPT-3.5 API, Temperature ranges from 0 to 2. Higher
            values yield more random/diverse outputs with lower quality, while
            lower values produce more deterministic outputs with higher quality.
            We use a strategy to dynamically adjust the temperature from
            initial_temperature to max_temperature during generation.

            We incorporate random few-shot generated examples into the prompt
            to the OpenAI GPT-3.5 API. The initial temperature is set lower to obtain
            high-quality, low-diversity examples. As the number of generated examples
            increases, we gradually have more high-quality examples for in-context
            learning during generation. This allows us to achieve high-quality,
            high-diversity examples later on by using a higher temperature.
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
        self.cache_root = Path(cache_root)

    def construct_prompt(
        self,
        instruction: str,
        few_shot_example_string: str,
        generated_examples: list[Example],
    ) -> str:
        """Generates a prompt string.

        The function generates a prompt string using the provided instruction and
        few_shot_example_string. It also selects random examples from the generated
        dataset to provide additional context for the prompt. At the start of dataset
        generation, it defaults to using the few_shot_example_string.

        The function uses different prompt templates based on the number of selected
        examples from the generated dataset. If the total length of the prompt exceeds
        3500 tokens, repeat the prompt generation process to generate a shorter one.

        Args:
            instruction: The natural language instruction for the prompt.
            few_shot_example_string: A string representing the few-shot examples
                parsed from the user's prompt, which quality is higher than the
                genrated examples.
            generated_examples: A list of currently generated examples.

        Returns:
            The generated prompt string.
        """
        # The random_example_string is a string, which contains several random
        # few-shot examples as demonstrations for the DatasetGenerator. If
        # generated_examples is empty, then the random_example_string
        # is the few-shot examples parsed from the user's prompt.
        while True:
            if len(generated_examples) == 0:
                low_quality_example_string = "N/A\n"
                # Create default low_quality_example_string if generated_examples
                # is empty. few_shot_example_string is the high-quality few-shot
                # examples parsed from the user's prompt. But if user does not
                # provideany examples in the input prompt, the few_shot_example_string
                # will be "N/A"/""/None.
                random_selected_generated_example_num = 0
                # random_selected_generated_example_num is the number of selected
                # random examples from generated_examples that will be used to
                # create the low_quality_example_string. If generated_examples
                # is empty, then random_selected_generated_example_num is 0.
            else:
                # If generated_examples is not empty, low_quality_example_string
                # is sveral random generated examples from generated_examples.

                low_quality_example_string = ""
                random_selected_generated_example_num = random.randint(
                    1, min(len(generated_examples), 10)
                )
                # random_selected_generated_example_num is the number of selected
                # random examples from generated_examples that will
                # be concatenated to create low_quality_example_string.
                random_examples = random.sample(
                    generated_examples, random_selected_generated_example_num
                )
                # If generated_examples is not empty, then choose several
                # random examples from generated_examples to construct
                # new low_quality_example_string.
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
            # The max content length of gpt-3.5-turbo is 4097, so if the
            # generated prompt is longer than 3500, then the prompt
            # should be regenerated.
            if count_tokens_from_string(prompt) < 3500:
                return prompt
            else:
                orginal_input_string = (
                    instruction + few_shot_example_string
                    if few_shot_example_string
                    else instruction
                )
                if count_tokens_from_string(orginal_input_string) > 3500:
                    logger.warning(
                        "The original input prompt is too long. Consider writing a shorter prompt."  # noqa E501
                    )
                continue

    def construct_input_output_map(
        self,
        generated_examples: list[Example],
    ) -> dict[str, Counter]:
        """Constructs a dictionary mapping inputs to `Counter` objects of outputs.

        Args:
            generated_examples: A list of currently generated examples.

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
            Currently generated dataset with multi-vote filtering applied.
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
        return generated_dataset

    def create_all_examples_dataset_and_generated_dataset(
        self, generated_examples: list[Example]
    ) -> Dataset:
        """Converts generated_examples into generated_dataset.

        Args:
            generated_examples: A list of currently generated examples.

        Depending on the value of self.filter_duplicated_examples, the function either
        constructs a mapping for input-output pairs followed by multi-vote filtering
        to create a Dataset or directly converts the generated examples into a Dataset.

        The function also verifies the presence of data in the input-output map
        and the generated dataset if there are any generated examples and
        self.filter_duplicated_examples is True.

        Lastly, the function stores all generated examples, irrespective of the value
        of self.filter_duplicated_examples, into a Dataset on the disk.

        Returns:
            A dataset of all the generated exmaples and the currently generated
            dataset. If filter_duplicated_examples is True, multi-vote filtering is
            performed. Else, the generated examples are directly converted into
            a Dataset.
        """
        # Convert all generated examples into a Dataset.
        all_generated_examples_dataset = Dataset.from_dict(
            {
                "input_col": [example.input_col for example in generated_examples],
                "output_col": [example.output_col for example in generated_examples],
            }
        )

        if self.filter_duplicated_examples:
            # When filtering duplicated examples is
            # enabled, perform multi-vote filtering.
            input_output_map = self.construct_input_output_map(generated_examples)
            generated_dataset = self.apply_multi_vote_to_construct_generated_dataset(
                input_output_map
            )

            # Ensure that the input-output map and
            # the generated dataset are not empty.
            if len(generated_examples) != 0:
                assert input_output_map is not None
                assert generated_dataset is not None
        else:
            # When filtering duplicated examples is not enabled,
            # use all_generated_examples_dataset directly.
            generated_dataset = all_generated_examples_dataset

        # If there are generated examples, the
        # generated_dataset should not be empty.
        if len(generated_examples) != 0:
            assert len(generated_dataset) > 0

        return all_generated_examples_dataset, generated_dataset

    def compute_batch_size(
        self, expected_num_examples: int, generated_dataset: Dataset
    ) -> int:
        """Computes the batch size for OpenAI API calls in a batch.

        The batch size is determined based on the remaining number of examples to be
        generated and the number of responses per request. The function also respects
        the maximum limit of API calls if it is set.

        Args:
            expected_num_examples: The total number of examples expected to be
            generated for the current dataset split. Note that if max_api_calls is not
            set, the actual number of generated examples can be slightly higher due
            to each API call returning `responses_per_request` examples.
            generated_dataset: Currently generated dataset.

        Returns:
            The batch size for the next batch of OpenAI API calls with zeno-build.
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
                (
                    (expected_num_examples - len(generated_dataset))
                    / self.responses_per_request
                )
            ),
            max_api_calls,
        )
        # Ensure that the batch_size is a positive value.
        assert batch_size > 0
        return batch_size

    def extract_responses(
        self, completions: list[openai.Completion], generated_examples: list[Example]
    ) -> list[Example]:
        """Extracts the generated sample and annotation from an OpenAI API response.

        Args:
            completions: A list of Completion objects returned by the OpenAI API.
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

        Note: The function process `batch_size * responses_per_request`
        responses at a time.

        Example:
            Given a list of two completion objects: [completion_1, completion_2],
            where:
            completion_1.choices = [
                {"message": {"content": '{"input": "1", "output": "a"}'}},
                {"message": {"content": '{"input": "1", "output": "b"}'}},
                {"message": {"content": '{"input": "1", "output": "a"}'}},
            ]
            completion_2.choices = [
                {"message": {"content": '{"input": "1", "output": "c"}'}},
                {"message": {"content": '{"input": "2", "output": "a"}'}},
                {"message": {"content": '{"input": "2", "output": "b"}'}},
            ]

            The function will create 'example' namedtuples:
            Example(input_col="1", output_col="a")
            Example(input_col="1", output_col="b")
            Example(input_col="1", output_col="a")
            Example(input_col="1", output_col="c")
            Example(input_col="2", output_col="a")
            Example(input_col="2", output_col="b")

            It will then append them to generated_examples.

        Returns:
            A list of `Example` objects.
                Each API call will return `responses_per_request` completion objects.
                If the response is a valid JSON object, create a namedtuple called
                `example` and append it to generated_examples. `example` consists
                of `input_col` and`output_col`, where:
                - input_col is the generated example string extracted from the response.
                - output_col is the generated label string extracted from the response.
                If the response is not a valid JSON object, discard it.
                There is responses_per_request * len(completions) responses at a time.
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
        return generated_examples

    async def generate_responses(
        self,
        chat_api: ChatGPTAgent,
        generated_dataset: Dataset,
        expected_num_examples: int,
        prompts: list[str],
    ) -> list[openai.Completion]:
        """Asynchronously generates responses using the GPT-3.5 API.

        Args:
            chat_api: ChatGPTAgent to generate responses.
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
        dynamic_temperature = (self.max_temperature - self.initial_temperature) * len(
            generated_dataset
        ) / expected_num_examples + self.initial_temperature

        # Ensure the dynamic temperature is within the range [0, 2.0]
        clipped_temperature = max(0.0, min(2.0, dynamic_temperature))
        responses = await chat_api.generate_batch_openai_chat_completion(
            prompts,
            temperature=clipped_temperature,
            responses_per_request=self.responses_per_request,
            requests_per_minute=self.requests_per_minute,
        )
        return responses

    def generate_dataset_split(
        self,
        prompt_spec: PromptSpec,
        expected_num_examples: int,
        split: DatasetSplit,
    ) -> Dataset:
        """Generates a dataset split using GPT-3.5.

        This method iteratively makes API calls to GPT-3.5 to generate a dataset split.
        Each API call yields a batch of responses. From these responses, new examples
        are extracted and added to 'generated_examples'. The process continues
        until the desired number of examples is reached, or the maximum limit on API
        calls is reached. If an error occurs during an API call, the error is handled
        appropriately, and the API call counter is adjusted.

        Args:
            prompt_spec: PromptParser to be used for generating examples.
            expected_num_examples: The number of examples expected to be
                generated. If the maximum limit on API calls is not set, the actual
                number of generated examples can be slightly higher due to each
                API call returning `responses_per_request` examples.
            split: The dataset split (e.g., train, validation, test) for which the
                examples are being generated.

        Returns:
            The generated dataset split.
        """
        # Refresh the relevant data structures for the new split.
        self.cache_root.mkdir(parents=True, exist_ok=True)
        examples_cache_path = Path(
            self.cache_root / f"generated_examples_{split.value}"
        )
        dataset_cache_path = Path(self.cache_root / f"generated_dataset_{split.value}")

        if examples_cache_path.exists():
            # If cache exists, load generated examples from disk.
            logger.info(f"Loading cache from {str(examples_cache_path)}.")
            all_generated_examples_dataset = Dataset.load_from_disk(examples_cache_path)
            generated_examples = [
                Example(input_col=ex["input_col"], output_col=ex["output_col"])
                for ex in all_generated_examples_dataset
            ]
        else:
            # Initialize data structures for a new split.
            generated_examples = []

        pbar = tqdm(total=expected_num_examples, desc="Generating examples")
        chat_api = ChatGPTAgent(self.api_key)

        while True:
            # Each API call will return `responses_per_request` completion
            # objects. The upper bound of the length of the generated dataset
            # is expected_num_examples + responses_per_request.
            try:
                # Convert the generated examples into a
                # Dataset and update the progress bar.
                (
                    all_generated_examples_dataset,
                    generated_dataset,
                ) = self.create_all_examples_dataset_and_generated_dataset(
                    generated_examples
                )
                all_generated_examples_dataset.save_to_disk(examples_cache_path)
                generated_dataset.save_to_disk(dataset_cache_path)
                pbar.update(len(generated_dataset))

                if self.max_api_calls and self.api_call_counter >= self.max_api_calls:
                    logger.warning("Maximum number of API calls reached.")
                    break
                elif len(generated_dataset) >= expected_num_examples:
                    break
                else:
                    # Compute the batch size for the next API call.
                    batch_size = self.compute_batch_size(
                        expected_num_examples, generated_dataset
                    )
                    self.api_call_counter += batch_size

                    # Generate prompts for the batch call.
                    prompts = [
                        self.construct_prompt(
                            instruction=prompt_spec.instruction,
                            few_shot_example_string=prompt_spec.examples,
                            generated_examples=generated_examples
                            if not self.filter_duplicated_examples
                            else [
                                Example(each["input_col"], each["output_col"])
                                for each in generated_dataset
                            ],
                        )
                        for _ in range(batch_size)
                    ]

                    loop = asyncio.get_event_loop()
                    responses = loop.run_until_complete(
                        self.generate_responses(
                            chat_api=chat_api,
                            generated_dataset=generated_dataset,
                            expected_num_examples=expected_num_examples,
                            prompts=prompts,
                        )
                    )

                    # Extract the responses and add new examples to the dataset.
                    generated_examples = self.extract_responses(
                        responses, generated_examples
                    )
            except OPENAI_ERRORS as e:
                # Handle OpenAI API errors and adjust the API call counter.
                self.api_call_counter = handle_openai_error(e, self.api_call_counter)
        return generated_dataset
