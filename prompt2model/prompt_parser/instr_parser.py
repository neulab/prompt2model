"""An interface for prompt parsing."""

from __future__ import annotations  # noqa FI58

import json
import re

import openai

from prompt2model.prompt_parser.base import PromptSpec, TaskType
from prompt2model.utils import ChatGPTAgent

from prompt2model.prompt_parser.instr_parser_prompt import (  # isort: split
    construct_full_parsing_prompt,
    construct_single_demonstration,
)


# pylint: disable=too-few-public-methods
class OpenAIInstructionParser(PromptSpec):
    """Parse the prompt to separate instructions from task demonstrations."""

    def __init__(self, task_type: TaskType, api_key: str | None = None):
        """By default, assume that every task is a text generation task."""
        self.task_type = task_type
        self.instruction: str | None = None
        self.demonstration: str | None = None
        self.api_key: str | None = api_key

    def get_prompt_for_instruction_parsing(self, user_prompt: str) -> str:
        """A (GPT-3) prompt for separating instructions from demonstrations.

        Args:
            user_prompt: A user-generated prompt asking for a response.

        Returns:
            A prompt to instruct GPT-3 to parse the user's provided prompt.
        """
        filled_template = construct_single_demonstration(
            user_prompt, None, None, input_only=True
        )
        final_prompt = construct_full_parsing_prompt() + filled_template
        return final_prompt

    def extract_response(self, response: openai.Completion) -> tuple[str, str | None]:
        """Parse stuctured fields from the OpenAI API response.

        Args:
            response (openai.Completion): OpenAI API response.

        Returns:
            tuple[str, str | None]: Tuple consisting of:
                1) Instruction: The instruction parsed from the API response.
                2) Demonstrations: (Optional) demonstrations parsed from the
                   API response.
        """
        response_text = json.loads(response.choices[0]["message"]["content"])
        # This regex pattern matches any text that's either between
        # "1) Instruction:" and "2) Demonstrations:" or after
        # "2) Demonstrations:". An arbitrary amount of uncaptured whitespace is
        # allowed immediately after "1)" or "2)" or after the colon following
        # each section header (e.g. "Instruction:").
        pattern = r"1\)\s*Instruction[:]\s*(.+)\s*2\)\s*Demonstrations[:]\s*(.+)"
        matches = re.findall(pattern, response_text, re.DOTALL)
        assert len(matches) == 1
        assert len(matches[0]) == 2
        instruction_string, demonstration_string = matches[0]
        instruction_string = instruction_string.strip()
        demonstration_string = demonstration_string.strip()
        if demonstration_string == "NO DEMONSTRATION.":
            # This special output sequence means "demonstration is None".
            demonstration_string = None
        return instruction_string, demonstration_string

    def parse_from_prompt(self, prompt: str) -> None:
        """Parse the prompt into an instruction and demonstrations."""
        parsing_prompt_for_chatgpt = self.get_prompt_for_instruction_parsing(prompt)

        chat_api = ChatGPTAgent(self.api_key)
        response = chat_api.generate_openai_chat_completion(parsing_prompt_for_chatgpt)
        self.instruction, self.demonstration = self.extract_response(response)
