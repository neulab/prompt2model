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
from test_helpers import are_datasets_identical

example = namedtuple("example", ["input_col", "output_col"])


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
        """Initialize an OpenAI DatasetGenerator with an API key and max API call.

        Args:
            api_key: A valid OpenAI API key. Alternatively, set as None and set
                the environment variable with `export OPENAI_API_KEY=<your key>`.
            max_api_calls: The maximum number of API calls allowed,
                or None for unlimited.
            temperature: What sampling temperature to use, between 0 and 2. Higher
                values like 0.8 will make the output more random, while lower values
                like 0.2 will make it more focused and deterministic.
            presence_penalty: Float between -2.0 and 2.0. Positive values penalize
                new tokens based on whether they appear in the text so far, increasing
                the model's likelihood to talk about new topics in generated examples.
            frequency_penalty: Float between -2.0 and 2.0. Positive values penalize
                new tokens based on their existing frequency in text, descouraging
                the model to repeat the same line verbatim in generated examples.
            batch_size: The number of requests to make in each batch.
            responses_per_request: Number of responses for each request.
                i.e. the parameter n of OpenAI API call.
            requests_per_minute: Number of requests per minute to allow.
            filter_duplicated_examples: Whether to filter duplicated examples.
                If it's True, the generated examples are filtered based on multi-votes.
            cache_root: If it's not None, the generated examples will be cached.
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
        self.generated_examples = []  # type: list[example]
        # Store all the generated examples in a list, which will be converted into
        # self.generated_dataset later and self.input_output_map if
        # filter_duplicated_examples is True.
        self.generated_dataset: Dataset = Dataset.from_dict({})
        # self.generated_examples will be converted into self.generated_dataset.
        # If filter_duplicated_examples is True, self.generated_examples will be
        # filtered based on multi-votes first to construct self.generated_dataset.
        # If filter_duplicated_examples is False, self.generated_examples will
        # not be filtered and directly used to construct self.generated_dataset.
        self.input_output_map: dict[str, Counter] = defaultdict(Counter)
        # If filter_duplicated_examples is True, self.generated_examples will
        # firstly be converted into self.input_output_map then into
        # self.generated_dataset. Else, self.input_output_map will always be {}.
        self.generating_split: DatasetSplit | None = None
        # generating_split is the DatasetSplit which is being generated.
        # After each while loop, self.generated_examples will be stored as
        # a Datast in f"{cache_root}/{generating_split}".

    def construct_prompt(
        self,
        instruction: str,
        few_shot_example_string: str = None,
    ) -> str:
        """Generates a prompt string.

        Args:
            instruction: The natural language instruction for the prompt.
            few_shot_example_string: A string representing the few-shot examples
                parsed from the user's prompt, which quality is higher than the
                genrated examples from self.generated_dataset.

        Returns:
            The generated prompt string.
        """
        # The random_example_string is a string, which contains several random
        # few-shot examples as demonstrations for the DatasetGenerator. If
        # self.generated_dataset is empty, then the random_example_string
        # is the few-shot examples parsed from the user's prompt.
        if len(self.generated_examples) != 0:
            assert len(self.generated_dataset) != 0
        while True:
            if len(self.generated_dataset) == 0:
                low_equality_example_string = "N/A\n"
                # Create default low_equality_example_string if self.generated_dataset
                # is empty. few_shot_example_string is the high-quality few-shot
                # examples parsed from the user's prompt. But if user does not
                # provideany examples in the input prompt (which is extremely
                # discouraged), the few_shot_example_string will be "N/A"/""/None.
                random_selected_generated_example_num = 0
                # random_selected_generated_example_num is the number of selected
                # random examples from self.generated_dataset that will be used to
                # create the low_equality_example_string. If self.generated_dataset
                # is empty, then random_selected_generated_example_num is 0.
            else:
                # self.generated_dataset is not empty, then low_equality_example_string
                # is the string representation of sveral random generated examples
                # from self.generated_dataset.

                low_equality_example_string = ""
                random_selected_generated_example_num = random.randint(
                    1, min(len(self.generated_dataset), 10)
                )
                # random_selected_generated_example_num is the number of
                # selected random examples from self.generated_examples that
                # will be concatenated to create low_equality_example_string.
                random_examples = Dataset.from_dict(
                    self.generated_dataset.shuffle()[
                        :random_selected_generated_example_num
                    ]
                )
                # self.generated_dataset is not empty. Choose several
                # random examples from self.generated_dataset to
                # construct low_equality_example_string.
                for example in random_examples:
                    low_equality_example_string += f"""input="{example["input_col"]}"\noutput="{example["output_col"]}"\n"""  # noqa 501
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

    def extract_responses(self, completions: list[openai.Completion]):
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
                    self.generated_examples.append(example(input, output))
                    logging.info(f"input: \n\n{input}\n\n")  # noqa: E501
                    logging.info(f"output: \n\n{output}\n\n")  # noqa: E501
            except Exception:
                logging.warning(
                    f"Error happened when parsing API completion: {completion}"
                )
                continue

    def construct_input_output_map(self):
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

        Then self.input_output_map value is:
        {
            "apple": Counter({"A": 2, "D": 1}),
            "banana": Counter({"B": 1}),
            "orange": Counter({"O": 1})
        }
        """
        # When ever using the multi-vote filtering mechanism, refresh
        # self.input_output_map to avoid duplicately countering.
        self.input_output_map = defaultdict(Counter)

        # Iterate through the examples and construct the mapping
        for example in self.generated_examples:
            input_str = example.input_col
            output_str = example.output_col

            self.input_output_map[input_str][output_str] += 1

        if len(self.generated_examples) != 0:
            assert self.input_output_map

    def use_multi_vote_to_construct_generated_dataset(self):
        """Multi-vote outputs self.input_output_map to construct self.generated_dataset.

        After multi-vote filtering, the input_col of self.generated_dataset is filtered
        unique inputs and the output_col is the shortest but most frequent output
        for the corresponding input.
        """
        # Only use multi-vote filtering if self.filter_duplicated_examples is True.
        # And self.input_output_map is not None when self.generated_examples
        # is not empty.
        assert self.filter_duplicated_examples
        if not (len(self.generated_examples) == 0):
            assert self.input_output_map is not None
        filtered_inputs = []
        filtered_outputs = []

        for input_str, output_counter in self.input_output_map.items():
            # Find the most frequent output count.
            most_common_count = output_counter.most_common(1)[0][1]

            # Get all the outputs that have the most common count.
            most_frequent_outputs = [
                output
                for output, count in output_counter.items()
                if count == most_common_count
            ]

            # Sort the outputs based on their lengths and select the
            # shortest ones. When several outputs have  the same
            # length with the highest frequency, they will be sorted
            # in their lexicographical (alphabetical) order when
            # using most_frequent_outputs.sort(key=len).

            most_frequent_outputs.sort(key=len)
            final_output = most_frequent_outputs[0]

            filtered_inputs.append(input_str)
            filtered_outputs.append(final_output)

        # Note that when self.generated_examples is empty,
        # self.input_output_map is None and self.generated_dataset
        # is empty.
        self.generated_dataset = Dataset.from_dict(
            {"input_col": filtered_inputs, "output_col": filtered_outputs}
        )

    def convert_generated_examples_to_generated_dataset(self):
        """Convert all generated examples to a Dataset.

        If self.filter_duplicated_examples is True, construct the
        input_output_map. Then use multi-vote filtering to convert
        the mapping to a Dataset. If self.filter_duplicated_examples
        is False, directly construct a Dataset.
        """
        all_generated_examples_dataset = Dataset.from_dict(
            {
                "input_col": [example.input_col for example in self.generated_examples],
                "output_col": [
                    example.output_col for example in self.generated_examples
                ],
            }
        )
        if self.filter_duplicated_examples:
            self.construct_input_output_map()
            self.use_multi_vote_to_construct_generated_dataset()
            if len(self.generated_examples) != 0:
                assert self.input_output_map is not None
                assert self.generated_dataset is not None
        else:
            self.generated_dataset = all_generated_examples_dataset
        # If self.generated_examples is not empty,
        # self.generated_examples is not empty.
        if len(self.generated_examples) != 0:
            assert not are_datasets_identical(
                self.generated_dataset, Dataset.from_dict({})
            )
        dataset_cache_path = Path(self.cache_root / f"{self.generating_split.value}")
        # No matter self.filter_duplicated_examples is True or False,
        # all the generated examples will be saved to disk as a Dataset.
        all_generated_examples_dataset.save_to_disk(dataset_cache_path)

    def compute_batch_size(self, expected_num_examples: int) -> int:
        """Compute the batch size to use zeno-bulid to call OpenAI API.

        Args:
            expected_num_examples: Number of expected examples in split.
        """
        if self.max_api_calls is None:
            # If there is no limit on the number of API calls, the batch_size should
            # be the min of self.batch_size and minimum calls to get more than
            # expected_num_examples examples.
            batch_size = min(
                self.batch_size,
                math.ceil(
                    (
                        (expected_num_examples - len(self.generated_dataset))
                        / self.responses_per_request
                    )
                ),
            )
        else:
            # If there is a limit on the number of API calls, the batch_size should
            # also take remaining API calls into account.
            batch_size = min(
                self.batch_size,
                math.ceil(
                    (
                        (expected_num_examples - len(self.generated_dataset))
                        / self.responses_per_request
                    )
                ),
                self.max_api_calls - self.api_call_counter,
            )
        assert batch_size > 0
        return batch_size

    def generate_dataset_split(
        self,
        prompt_spec: PromptSpec,
        expected_num_examples: int,
        split: DatasetSplit,
    ) -> Dataset:
        """Generate a single dataset using GPT-3.5.

        Args:
            prompt_spec: A prompt specification.
            expected_num_examples: Number of expected examples in split.
                Each API call will return `responses_per_request` completion
                objects. The upper bound of the length of generated dataset
                is expected_num_examples + responses_per_request.
            split: Name of dataset split to generate.

        Returns:
            A single dataset.
        """
        # Refresh the generated_dataset, generated_examples,
        # input_output_map, and generating_split for different split.
        self.generating_split = split
        dataset_cache_path = Path(self.cache_root / f"{self.generating_split.value}")
        if dataset_cache_path.exists():
            all_generated_examples_dataset = Dataset.load_from_disk(dataset_cache_path)
            self.generated_examples = [
                example(input_col=ex["input_col"], output_col=ex["output_col"])
                for ex in all_generated_examples_dataset
            ]
            # self.generated_examples will be load from disk. self.generated_dataset
            # will be initialized in the first loop. If self.filter_duplicated_examples
            # is True, it will also be constructed in the first loop.
        else:
            self.generated_dataset = Dataset.from_dict({})
            self.input_output_map = defaultdict(Counter)
            self.generated_examples = []

        chat_api = ChatGPTAgent(self.api_key)
        pbar = tqdm(total=expected_num_examples, desc="Generating examples")
        while True:
            # Each API call will return `responses_per_request` completion
            # objects. The upper bound of the length of generated dataset
            # is expected_num_examples + responses_per_request.
            try:
                self.convert_generated_examples_to_generated_dataset()
                pbar.update(len(self.generated_dataset))
                if self.max_api_calls and self.api_call_counter >= self.max_api_calls:
                    logging.warning("Maximum number of API calls reached.")
                    return self.generated_dataset
                elif len(self.generated_dataset) >= expected_num_examples:
                    return self.generated_dataset
                else:
                    batch_size = self.compute_batch_size(
                        expected_num_examples=expected_num_examples
                    )
                    self.api_call_counter += batch_size
                    prompts = [
                        self.construct_prompt(
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
            except OPENAI_ERRORS as e:
                self.api_call_counter = handle_openai_error(e, self.api_call_counter)
