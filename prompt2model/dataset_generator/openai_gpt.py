"""A simple dataset generator that uses OpenAI's GPT-3.5 API."""

from __future__ import annotations  # noqa FI58

import asyncio
import json
import logging
import math
import os
import random
from collections import namedtuple

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
            batch_size: The number of examples to generate in each batch.
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
        self.generated_examples = []  # type: list[example]
        # Randomly selected several examples as addtional few-shot examples
        # from the generated examples to generate new examples.

    def generate_prompt(
        self,
        instruction: str,
        few_shot_example_string: str = None,
    ) -> tuple[str, str]:
        """Generates a prompt string.

        Args:
            instruction: The natural language instruction for the prompt.
            few_shot_example_string: A string representing the few-shot examples
                parsed from the user's prompt.

        Returns:
            The generated prompt string and the few-shot examples string.
        """
        # The random_example_string is a string, which contains several random
        # few-shot examples as demonstrations for the DatasetGenerator. If
        # self.generated_examples is empty, then the random_example_string
        # is the few-shot examples parsed from the user's prompt.
        while True:
            if len(self.generated_examples) == 0:
                random_example_string = (
                    (few_shot_example_string + "\n")
                    if (
                        few_shot_example_string is not None
                        and few_shot_example_string != "N/A"
                        and few_shot_example_string != ""
                    )
                    else "N/A\n"
                )
                # Create default random_example_string if self.generated_examples
                # is empty. few_shot_example_string is the few-shot examples
                # parsed from the user's prompt. But if user does not provide
                # any examples in the input prompt, the few_shot_example_string
                # will be "N/A"/""/None.
                random_selected_generated_example_num = 0
                # random_selected_generated_example_num is the number of selected
                # random examples from self.generated_examples that will
                # be added to random_example_string. If self.generated_examples
                # is empty, then random_selected_generated_example_num is 0.
            else:
                # If self.generated_examples is not empty, then random_example_string
                # is the few-shot examples parsed from the user's input prompt by the
                # PromptParser, together with sveral random generated examples from
                # self.generated_examples.

                # To increase the diversity of the random_example_string, first select
                # several random examples from self.generated_examples.
                # And then the few-shot examples parsed from the user's input prompt
                # will be inserted into these random examples in a random index.
                random_example_string = ""
                random_selected_generated_example_num = random.randint(
                    1, min(len(self.generated_examples), 10)
                )
                # random_selected_generated_example_num is the number of selected
                # random examples from self.generated_examples that will
                # be added to random_example_string.
                random_examples = random.sample(
                    self.generated_examples, random_selected_generated_example_num
                )
                # If generated_examples is not empty, then choose several
                # random examples from self.generated_examples to construct
                # new random_example_string, else use the default random_example_string.
                user_examples_insert_index = random.randint(0, len(random_examples) - 1)
                for index, example in enumerate(random_examples):
                    random_example_string += (
                        f'input="{example.input_col}"\noutput="{example.output_col}"\n'
                    )
                    if (
                        # If the index equals to user_examples_insert_index and the
                        # few_shot_example_string is valid, then add the few-shot
                        # into the random_example_string at the index.
                        index == user_examples_insert_index
                        and few_shot_example_string is not None
                        and few_shot_example_string != "N/A"
                        and few_shot_example_string != ""
                    ):
                        random_example_string += few_shot_example_string + "\n"
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
                few_shot_example_string=random_example_string,
                template_type=template_type,
            )
            # The max content length of gpt-3.5-turbo is 4097, so if the
            # generated prompt is longer than 3500, then the prompt
            # should be regenerated.
            if count_tokens_from_string(prompt) < 3500:
                return prompt, random_example_string
            else:
                continue

    def extract_responses(self, completions: list[openai.Completion]) -> None:
        """Extracts the generated sample and annotation from an OpenAI API response.

        Args:
            completions: The generated completion objects returned by OpenAI API.

        Returns:
            Each api call would return a completion object with 5 responses.
                If the response is a valid JSON object, create a namedtuple called
                `example` and append it to self.generated_examples. `example` consists
                of `input_col` and`output_col`, where:
                - input_col is the generated example string extracted from the response.
                - output_col is the generated label string extracted from the response.
                If the response is not a valid JSON object, discard it.
            There is 5 * len(completions) responses at a time.
        """
        # from IPython import embed
        # embed()
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
                Since each API call returns 5 responses, the total number
                of examples less than expected_num_examples + 5.
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
                                    / 5
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
                                    / 5
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
                        )[0]
                        for _ in range(batch_size)
                    ]

                    async def generate_responses():
                        responses = (
                            await chat_api.generate_batch_openai_chat_completion(
                                prompts,
                                temperature=self.temperature,
                            )
                        )
                        return responses

                    loop = asyncio.get_event_loop()
                    responses = loop.run_until_complete(generate_responses())
                    self.extract_responses(responses)
                    pbar.update(len(self.generated_examples))
            except OPENAI_ERRORS as e:
                self.api_call_counter = handle_openai_error(e, self.api_call_counter)
        # Since each API call would return five completion objects, the upper bound
        # of the length of generated dataset is expected_num_examples + 5.
        assert len(self.generated_examples) < expected_num_examples + 5
        return Dataset.from_dict(
            {
                "input_col": [example.input_col for example in self.generated_examples],
                "output_col": [
                    example.output_col for example in self.generated_examples
                ],
            }
        )
