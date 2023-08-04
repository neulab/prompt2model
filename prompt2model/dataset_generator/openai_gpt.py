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
logger = logging.getLogger("DatasetGenerator")
ch = logging.StreamHandler()
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
ch.setFormatter(formatter)
logger.addHandler(ch)


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
            batch_size: The number of requests to make in each batch.
            responses_per_request: The number of responses for each request.
            requests_per_minute: The maximum number of requests per minute.
            filter_duplicated_examples: If True, filters duplicated examples based
                on multi-votes.
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
            logger.warning(
                "The lowest temperature for GPT-3.5 API is 0. So the initial_temperature is set to 0."  # noqa E501
            )
            self.initial_temperature = 0
        if self.max_temperature > 2.0:
            logger.warning(
                "The highest temperature for GPT-3.5 API is 2. So the max_temperature is set to 2."  # noqa E501
            )
            self.max_temperature = 2
        if self.initial_temperature >= self.max_temperature:
            logger.warning(
                "The generator gradually increases the temperature from a lower value to a higher value. So the initial_temperature and the max_temperature are switched."  # noqa E501
            )
            self.initial_temperature, self.max_temperature = (
                self.max_temperature,
                self.initial_temperature,
            )
        self.presence_penalty = presence_penalty
        self.frequency_penalty = frequency_penalty
        self.batch_size = batch_size
        self.responses_per_request = responses_per_request
        self.requests_per_minute = requests_per_minute
        self.filter_duplicated_examples = filter_duplicated_examples
        self.cache_root = Path(cache_root)
        self.cache_root.mkdir(exist_ok=True, parents=True)
        self.generated_examples = []  # type: list[Example]
        # This list stores all generated examples. These will later be
        # converted into `generated_dataset` and `input_output_map`
        # if `filter_duplicated_examples` is True.

        self.generated_dataset: Dataset = Dataset.from_dict({})
        # `generated_examples` will be transformed into `generated_dataset`.
        # If `filter_duplicated_examples` is True, `generated_examples` will
        # be filtered based on multi-votes before being used to construct
        # `generated_dataset`. If it's False, `generated_examples` will be
        # used directly to construct `generated_dataset`.

        self.input_output_map: dict[str, Counter] = defaultdict(Counter)
        # If `filter_duplicated_examples` is True, `self.generated_examples`
        # will first be converted into `input_output_map`, and then into
        # `generated_dataset`. If it's False, `input_output_map` will remain
        # empty.

        self.generating_split: DatasetSplit | None = None
        # `generating_split` refers to the DatasetSplit currently being
        # generated. After each loop, `generated_examples` will be
        # stored as a Dataset at the path `{cache_root}/{generating_split}`.

        self.dataset_cache_path = Path()
        self.examples_cache_path = Path()
        # `genrated_examples` and `generated_dataset` are stored at
        # path `examples_cache_path` and `dataset_cache_path` respectively.

    def construct_prompt(
        self,
        instruction: str,
        few_shot_example_string: str = None,
    ) -> str:
        """Generates a prompt string.

        The function generates a prompt string using the provided instruction and
        few_shot_example_string. It also selects random examples from the generated
        dataset to provide additional context for the prompt. If the generated_dataset
        is empty, it defaults to using the few_shot_example_string.

        The function uses different prompt templates based on the number of selected
        examples from the generated dataset. If the total length of the prompt exceeds
        3500 tokens, repeat the prompt generation process to generate a shorter one.

        Args:
            instruction: Instruction for the prompt.
            few_shot_example_string: High-quality few-shot examples.

        Returns:
            A prompt string for the OpenAI API.
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
                # If `generated_dataset` is empty, we create a default
                # `low_equality_example_string`. `few_shot_example_string`
                # contains high-quality few-shot examples parsed from the
                # user's prompt.

                # If no examples are provided in the input prompt (which is
                # extremely discouraged), `few_shot_example_string` will
                # be set to "N/A" or an empty string ("") or None.
                random_selected_generated_example_num = 0
                # `random_selected_generated_example_num` is the count
                # of randomly selected examples from `generated_dataset`
                # used to create `low_equality_example_string`. If
                # `generated_dataset` is empty, then
                # `random_selected_generated_example_num` is 0.
            else:
                # If `generated_dataset` is not empty, `low_equality_example_string`
                # becomes the string representation of several randomly generated
                # examples from `generated_dataset`.

                low_equality_example_string = ""
                random_selected_generated_example_num = random.randint(
                    1, min(len(self.generated_dataset), 10)
                )
                # `random_selected_generated_example_num` is the count of
                # randomly selected examples from `generated_examples`
                # that will be joined to create `low_equality_example_string`.

                random_examples = Dataset.from_dict(
                    self.generated_dataset.shuffle()[
                        :random_selected_generated_example_num
                    ]
                )
                # When `generated_dataset` is not empty, select several
                # random examples from `generated_dataset` to construct
                # `low_equality_example_string`.

                random_example_strings = [
                    f'input="{example["input_col"]}"\noutput="{example["output_col"]}"\n'  # noqa E501
                    for example in random_examples
                ]
                random.shuffle(random_example_strings)
                low_equality_example_string = "\n".join(random_example_strings)

                # To diversify the prompt for DatasetGenerator, three prompt
                # templates (COMPLEX, MIDDLE, SIMPLE) and 24 meta examples
                # are created in `openai_gpt_template`. The COMPLEX template
                # includes 4 meta examples, MIDDLE has 3, and SIMPLE contains 2.

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

            # The maximum content length for gpt-3.5-turbo is 4097.
            # Therefore, if the generated prompt is too long (exceeds
            # 3500 tokens), the prompt should be regenerated.

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
                        "The original input prompt is too long. Consider writing a shorter prompt."  # noqa 501
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
        # Whenever using the multi-vote filtering mechanism, refresh
        # self.input_output_map to avoid duplicately countering.
        self.input_output_map = defaultdict(Counter)

        # Iterate through the examples and construct the mapping.
        for example in self.generated_examples:
            input_str = example.input_col
            output_str = example.output_col

            # Increment the count of the output for the specific input.
            self.input_output_map[input_str][output_str] += 1

        # Ensure that the generated_examples list is not empty
        # and the map is constructed correctly.
        if len(self.generated_examples) != 0:
            assert self.input_output_map

    def apply_multi_vote_to_construct_generated_dataset(self):
        """Multi-vote to construct self.generated_dataset from self.input_output_map.

        This method uses multi-vote filtering to create a unique mapping from inputs
        to outputs. The input_col of self.generated_dataset contains unique inputs,
        while the output_col holds the shortest, most frequent output for the
        corresponding input.

        The function asserts that self.filter_duplicated_examples is True and that
        self.input_output_map is not None when self.generated_examples is not
        empty. It then iterates over self.input_output_map, finding the most frequent
        output for each input. If there are multiple outputs with the highest frequency,
        it selects the shortest one. If there are multiple shortest outputs with the
        highest frequency, it selects the one that comes first in lexicographical
        (alphabetical) order.

        Example:
        Suppose self.input_output_map is:
        {
            "apple": Counter({"A": 2, "D": 2}),
            "banana": Counter({"B": 2, "C": 1}),
            "orange": Counter({"O": 1})
        }

        The function will produce self.generated_dataset:
        {
            "input_col": ["apple", "banana", "orange"],
            "output_col": ["A", "B", "O"]
        }

        Note: When self.generated_examples is empty, both self.input_output_map
        and self.generated_dataset will be empty.
        """
        # Ensure that multi-vote filtering is enabled.
        assert self.filter_duplicated_examples

        # Ensure `input_output_map` is not None when
        # `generated_examples` is not empty.
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
        self.generated_dataset = Dataset.from_dict(
            {"input_col": filtered_inputs, "output_col": filtered_outputs}
        )

    def convert_generated_examples_to_generated_dataset(self):
        """Converts self.generated_examples into self.generated_dataset.

        Depending on the value of self.filter_duplicated_examples, the function either
        constructs a mapping for input-output pairs followed by multi-vote filtering
        to create a Dataset or directly converts the generated examples into a Dataset.

        The function also verifies the presence of data in the input-output map
        and the generated dataset if there are any generated examples and
        self.filter_duplicated_examples is True.

        Lastly, the function stores all generated examples, irrespective of the value
        of self.filter_duplicated_examples, into a Dataset on the disk.
        """
        # Convert all generated examples into a Dataset.
        all_generated_examples_dataset = Dataset.from_dict(
            {
                "input_col": [example.input_col for example in self.generated_examples],
                "output_col": [
                    example.output_col for example in self.generated_examples
                ],
            }
        )

        if self.filter_duplicated_examples:
            # When filtering duplicated examples is
            # enabled, perform multi-vote filtering.
            self.construct_input_output_map()
            self.apply_multi_vote_to_construct_generated_dataset()

            # Ensure that the input-output map and
            # the generated dataset are not empty.
            if len(self.generated_examples) != 0:
                assert self.input_output_map is not None
                assert self.generated_dataset is not None
        else:
            # When filtering duplicated examples is not enabled,
            # use all_generated_examples_dataset directly.
            self.generated_dataset = all_generated_examples_dataset

        # If there are generated examples, the
        # generated_dataset should not be empty.
        if len(self.generated_examples) != 0:
            assert len(self.generated_dataset) > 0

        # Save all the generated examples to disk as
        # a Dataset, regardless of the filtering option.

        assert self.examples_cache_path != Path()
        assert self.dataset_cache_path != Path()
        assert self.dataset_cache_path != self.examples_cache_path
        all_generated_examples_dataset.save_to_disk(self.examples_cache_path)
        self.generated_dataset.save_to_disk(self.dataset_cache_path)

    def compute_batch_size(self, expected_num_examples: int) -> int:
        """Computes the batch size for OpenAI API calls in a batch.

        The batch size is determined based on the remaining number of examples to be
        generated and the number of responses per request. The function also respects
        the maximum limit of API calls if it is set.

        Args:
            expected_num_examples: The total number of examples expected to be
            generated for the current dataset split. Note that if max_api_calls is not
            set, the actual number of generated examples can be slightly higher due
            to each API call returning `responses_per_request` examples.

        Returns:
            The batch size for the next batch of OpenAI API calls with zeno-build.
        """
        if self.max_api_calls is None:
            # If there is no limit on the number of API calls, the batch_size should
            # be the minimum of self.batch_size and the minimum number of calls
            # to get more than expected_num_examples examples.
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
            # If there is a limit on the number of API calls, the batch_size
            # should also take remaining API calls into account.
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

        # Ensure that the batch_size is a positive value.
        assert batch_size > 0
        return batch_size

    def extract_responses(self, completions: list[openai.Completion]):
        """Extracts and stores generated input and output from OpenAI API responses.

        Args:
            completions: A list of Completion objects returned by the OpenAI API.
            Each API call returns a number of completion objects equivalent to
            `responses_per_request`. The default `responses_per_request` = 5.

        This function iterates through the provided completions, attempting to
        extract and parse the content of each completion as a JSON object. It
        then checks for the presence of `input` and `output` keys in the JSON
        object. If either is missing, the completion is discarded.

        For valid completions, the function creates a namedtuple `example`
        with `input_col` and `output_col` fields, representing the generated
        example and label strings respectively. The `example` is then added
        to self.generated_examples.

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

        And append them to self.generated_examples.
        """
        for completion in completions:
            try:
                for choice in completion.choices:
                    try:
                        response_json = json.loads(choice["message"]["content"])
                    except Exception:
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
                        # If the response doesn't contain required keys, discard it.
                    input = str(response_json["input"]).strip()
                    output = str(response_json["output"]).strip()
                    if input != "" and output != "":
                        self.generated_examples.append(Example(input, output))
                    else:
                        logger.info(
                            "Empty input or output ditected. Discard this example."
                        )
                        continue
                    # Add the validated example to the generated examples list.
                    logger.info(
                        f"input\n-------------------------------------------------\n{input}\n-------------------------------------------------"  # noqa: E501
                    )
                    logger.info(
                        f"output\n-------------------------------------------------\n{output}\n-------------------------------------------------"  # noqa: E501
                    )
            except Exception:
                continue
                # If an error occurs during processing a
                # completion, skip it and move to the next.

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
        self.generating_split = split
        self.examples_cache_path = Path(
            self.cache_root / f"generated_examples_{self.generating_split.value}"
        )
        self.dataset_cache_path = Path(
            self.cache_root / f"generated_dataset_{self.generating_split.value}"
        )

        if self.examples_cache_path.exists():
            # If cache exists, load generated examples from disk.
            logger.info(f"Loading cache from {str(self.examples_cache_path)}.")
            all_generated_examples_dataset = Dataset.load_from_disk(
                self.examples_cache_path
            )
            self.generated_examples = [
                Example(input_col=ex["input_col"], output_col=ex["output_col"])
                for ex in all_generated_examples_dataset
            ]
            # `generated_examples` will be loaded from disk. `generated_dataset`
            # will be initialized in the first loop. If `filter_duplicated_examples` is
            # True, `input_output_map` will also be constructed in the first loop.

        else:
            # Initialize data structures for a new split.
            self.generated_dataset = Dataset.from_dict({})
            self.input_output_map = defaultdict(Counter)
            self.generated_examples = []

        pbar = tqdm(total=expected_num_examples, desc="Generating examples")
        chat_api = ChatGPTAgent(self.api_key)

        while True:
            # Each API call will return `responses_per_request` completion
            # objects. The upper bound of the length of the generated dataset
            # is expected_num_examples + responses_per_request.
            try:
                # Convert the generated examples into a
                # Dataset and update the progress bar.
                self.convert_generated_examples_to_generated_dataset()
                pbar.update(len(self.generated_dataset) - pbar.n)

                if self.max_api_calls and self.api_call_counter >= self.max_api_calls:
                    logger.warning("Maximum number of API calls reached.")
                    return self.generated_dataset
                elif len(self.generated_dataset) >= expected_num_examples:
                    return self.generated_dataset
                else:
                    # Compute the batch size for the next API call.
                    batch_size = self.compute_batch_size(
                        expected_num_examples=expected_num_examples
                    )
                    self.api_call_counter += batch_size

                    # Generate prompts for the batch call.
                    prompts = [
                        self.construct_prompt(
                            instruction=prompt_spec.instruction,
                            few_shot_example_string=prompt_spec.examples,
                        )
                        for _ in range(batch_size)
                    ]

                    async def generate_responses():
                        """Asynchronously generates responses using the GPT-3.5 API.

                        The temperature for generating responses dynamically adjusts
                        based on the size of the generated dataset. As the dataset
                        grows, the dynamic temperature gradually increases,
                        approaching the max_temperature.

                        To prevent potential round-off errors in Python, the dynamic
                        temperature is rounded within the range [0, 2.0].
                        """
                        # Calculate the dynamic temperature based
                        # on the size of the generated dataset
                        dynamic_temperature = (
                            self.max_temperature - self.initial_temperature
                        ) * len(
                            self.generated_dataset
                        ) / expected_num_examples + self.initial_temperature

                        # Ensure the dynamic temperature is within the range [0, 2.0]
                        clipped_temperature = max(0.0, min(2.0, dynamic_temperature))
                        responses = (
                            await chat_api.generate_batch_openai_chat_completion(
                                prompts,
                                temperature=clipped_temperature,
                                responses_per_request=self.responses_per_request,
                                requests_per_minute=self.requests_per_minute,
                            )
                        )
                        return responses

                    loop = asyncio.get_event_loop()
                    responses = loop.run_until_complete(generate_responses())

                    # Extract the responses and add new examples to the dataset.
                    self.extract_responses(responses)
            except OPENAI_ERRORS as e:
                # Handle OpenAI API errors and adjust the API call counter.
                self.api_call_counter = handle_openai_error(e, self.api_call_counter)
