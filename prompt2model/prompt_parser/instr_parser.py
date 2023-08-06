"""An interface for prompt parsing."""

from __future__ import annotations  # noqa FI58

import json
import logging
import os

import deepl
import openai

from prompt2model.prompt_parser.base import PromptSpec, TaskType

from prompt2model.prompt_parser.instr_parser_prompt import (  # isort: split
    construct_prompt_for_instruction_parsing,
)
from prompt2model.utils import OPENAI_ERRORS, ChatGPTAgent, handle_openai_error


class OpenAIInstructionParser(PromptSpec):
    """Parse the prompt to separate instructions from task demonstrations."""

    def __init__(
        self, task_type: TaskType, api_key: str | None = None, max_api_calls: int = None, translate_instruction_to_English=True,
    ):
        """Initialize the prompt spec with empty parsed fields.

        We initialize the "instruction" and "examples" fields with None.
        These fields can be populated with the parse_from_prompt method.

        Args:
            task_type: Set a constant task type to use for all prompts.
            api_key: A valid OpenAI API key. Alternatively, set as None and set
                the environment variable with `export OPENAI_API_KEY=<your key>`.
            max_api_calls: The maximum number of API calls allowed,
                or None for unlimited.
        """
        self.task_type = task_type
        self._instruction: str | None = None
        self._examples: str | None = None
        self.api_key: str | None = api_key if api_key else os.environ["OPENAI_API_KEY"]
        assert self.api_key is not None and self.api_key != "", (
            "API key must be provided"
            + " or set the environment variable with `export OPENAI_API_KEY=<your key>`"
        )
        if max_api_calls:
            assert max_api_calls > 0, "max_api_calls must be > 0"
        self.max_api_calls = max_api_calls
        self.api_call_counter = 0
        self.translate_instruction_to_English = translate_instruction_to_English

    def extract_response(self, response: openai.Completion) -> tuple[str, str]:
        """Parse stuctured fields from the OpenAI API response.

        Args:
            response: OpenAI API response.

        Returns:
            tuple[str, str]: Tuple consisting of:
                1) Instruction: The instruction parsed from the API response.
                2) Demonstrations: (Optional) demonstrations parsed from the
                   API response.
        """
        response_text = response.choices[0]["message"]["content"]
        try:
            response_json = json.loads(response_text, strict=False)
        except json.decoder.JSONDecodeError as e:
            logging.warning("API response was not a valid JSON")
            raise e

        required_keys = ["Instruction", "Demonstrations"]
        missing_keys = [key for key in required_keys if key not in response_json]
        assert (
            len(missing_keys) == 0
        ), f'API response must contain {", ".join(required_keys)} keys'
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

        chat_api = ChatGPTAgent(self.api_key)
        while True:
            try:
                self.api_call_counter += 1
                response = chat_api.generate_openai_chat_completion(
                    parsing_prompt_for_chatgpt
                )
                self._instruction, self._examples = self.extract_response(response)
                break
            except OPENAI_ERRORS as e:
                self.api_call_counter = handle_openai_error(e, self.api_call_counter)
                if self.max_api_calls and self.api_call_counter >= self.max_api_calls:
                    logging.error("Maximum number of API calls reached.")
                    raise ValueError("Maximum number of API calls reached.") from e

        if self.translate_instruction_to_English:
            pass
        else:
            assert "DEEPL_KEY" in os.environ, "A DeepL API key must be set as the environment variable DEEPL_KEY"
            translator = deepl.Translator(os.environ["DEEPL_KEY"])
            result = translator.translate_text(self._instruction, target_lang="EN-US")
            if not result.detected_source_lang.startswith("EN"):
                self._instruction_english = result.text

        return None
