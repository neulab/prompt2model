"""A simple dataset generator that uses OpenAI's GPT-3.5 API."""

from abc import ABC, abstractmethod

import openai

from prompt2model.dataset_generator.base import DatasetGenerator


class OpenAIDatasetGenerator(DatasetGenerator, ABC):
    """A abstract class for dataset generator using OpenAI's GPT-3.5 API."""

    def __init__(self, api_key: str, max_api_call: int = None):
        """Initialize an OpenAI client with an API key and max API call allowed.

        Args:
            api_key: A valid OpenAI API key.
            max_api_call: The maximum number of API calls allowed. Defaults to 3000.
        """
        openai.api_key = api_key
        if max_api_call:
            self.max_api_call = max_api_call
        else:
            self.max_api_call = 3000
        self.current_api_call = 0

    @abstractmethod
    def generate_prompt(
        self, natrual_instruction: str, few_shot_examples: list[str] = None
    ) -> str:
        """Generates a prompt string.

        Args:
            natrual_instruction: The natural language instruction for the prompt.
            few_shot_examples: A list of few-shot examples. Defaults to None.

        Returns:
            The generated prompt string.
        """

    def generate_example(self, prompt: str) -> openai.Completion:
        """Generate an exmaple and its pseudo_label using OpenAI's GPT-3 API.

        Args:
            prompt: A prompt asking for expected fileds.

        Returns:
            A openai.Completion object.
        """
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "user", "content": f"{prompt}"},
            ],
        )
        return response

    @abstractmethod
    def response_mining(self, response: openai.Completion) -> tuple[str, int]:
        """Extracts expected fileds from an OpenAI API response.

        Args:
            response (openai.Completion): The response object returned by OpenAI API.

        Returns:
            A tuple of strings (generated_example, pseudo_label), where:
            - generated_example is the generated example string extracted from the
            response, or "" if not found.
            - pseudo_label is the pseudo label int extracted from the response,
            or -1 if not found.
        """
