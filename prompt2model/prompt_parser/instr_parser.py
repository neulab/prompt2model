"""An interface for prompt parsing."""

from __future__ import annotations  # noqa FI58

import json
import os

import openai

from prompt2model.prompt_parser.base import PromptSpec, TaskType

from prompt2model.prompt_parser.instr_parser_prompt import (  # isort: split
    construct_prompt_for_instruction_parsing,
)

from prompt2model.utils import api_tools, get_formatted_logger
from prompt2model.utils.api_tools import API_ERRORS, handle_api_error

logger = get_formatted_logger("PromptParser")

os.environ["TOKENIZERS_PARALLELISM"] = "false"


class PromptBasedInstructionParser(PromptSpec):
    """Parse the prompt to separate instructions from task demonstrations."""

    def __init__(self, task_type: TaskType, max_api_calls: int = None):
        """Initialize the prompt spec with empty parsed fields.

        We initialize the "instruction" and "examples" fields with None.
        These fields can be populated with the parse_from_prompt method.

        Args:
            task_type: Set a constant task type to use for all prompts.
            max_api_calls: The maximum number of API calls allowed,
                or None for unlimited.
        """
        self.task_type = task_type
        self._instruction: str | None = None
        self._examples: str | None = None
        if max_api_calls and max_api_calls <= 0:
            raise ValueError("max_api_calls must be > 0.")
        self.max_api_calls = max_api_calls
        self.api_call_counter = 0

    def extract_response(self, response: openai.Completion) -> tuple[str, str] | None:
        """Parse stuctured fields from the API response.

        Args:
            response: API response.

        Returns:
            If the API response is a valid JSON object and contains the required_keys,
                then returns a tuple consisting of:
                1) Instruction: The instruction parsed from the API response.
                2) Demonstrations: (Optional) demonstrations parsed from the
                API response.
            Else returns None.
        """
        response_text = response.choices[0]["message"]["content"]
        try:
            response_json = json.loads(response_text, strict=False)
        except json.decoder.JSONDecodeError:
            logger.warning(f"API response was not a valid JSON: {response_text}")
            return None

        required_keys = ["Instruction", "Demonstrations"]
        missing_keys = [key for key in required_keys if key not in response_json]
        if len(missing_keys) != 0:
            logger.warning(f'API response must contain {", ".join(required_keys)} keys')
            return None
        instruction_string = response_json["Instruction"].strip()
        demonstration_string = response_json["Demonstrations"].strip()
        return instruction_string, demonstration_string

    def parse_from_prompt(self, prompt: str) -> None:
        """Parse prompt into specific fields, stored as class member variables.

        This function directly stores the parsed fields into the class's member
        variables `instruction` and `examples`. So it has no return value.

        Args:
            prompt: User prompt to parse into two specific fields:
                    "instruction" and "demonstrations".
        """
        parsing_prompt_for_chatgpt = construct_prompt_for_instruction_parsing(prompt)

        chat_api = api_tools.default_api_agent
        last_error = None
        while True:
            self.api_call_counter += 1
            try:
                response: openai.ChatCompletion | Exception = (
                    chat_api.generate_one_completion(
                        parsing_prompt_for_chatgpt,
                        temperature=0,
                        presence_penalty=0,
                        frequency_penalty=0,
                    )
                )
                extraction = self.extract_response(response)
                if extraction is not None:
                    self._instruction, self._examples = extraction
                    return
            except API_ERRORS as e:
                last_error = e
                handle_api_error(e)

            if self.max_api_calls and self.api_call_counter >= self.max_api_calls:
                # In case we reach maximum number of API calls, we raise an error.
                logger.error("Maximum number of API calls reached.")
                raise RuntimeError(
                    "Maximum number of API calls reached."
                ) from last_error
