"""An interface for prompt parsing."""

from __future__ import annotations  # noqa FI58

import json
import os

import openai

from prompt2model.prompt_parser.base import PromptSpec, TaskType

from prompt2model.prompt_parser.instr_parser_prompt import (  # isort: split
    construct_prompt_for_instruction_parsing,
)

from prompt2model.utils import api_tools, get_formatted_logger, parse_json_responses
from prompt2model.utils.api_tools import API_ERRORS, handle_api_error

logger = get_formatted_logger("PromptParser")

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# TODO: what to do about max_api_calls
# TODO: should parse_from_prompt be separate from json_parsing
# TODO: how to do the try except things with it abstracted to a separate module--prolly done
# TODO: What to name the file and class?
# TODO: temparature and all has been abstracted out, mention that in documentation
# TODO: one of the parse_from_propmt na
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
        self.max_api_calls = max_api_calls
        

    def parse_from_prompt(self, prompt: str) -> None:
        """Parse prompt into specific fields, stored as class member variables.

        This function directly stores the parsed fields into the class's member
        variables `instruction` and `examples`. So it has no return value.

        Args:
            prompt: User prompt to parse into two specific fields:
                    "instruction" and "demonstrations".
        """
        parsing_prompt_for_chatgpt = construct_prompt_for_instruction_parsing(prompt)
        json_parser = parse_json_responses.JsonParsingFromLLMResponse(self.max_api_calls)
        required_keys = ["Instruction", "Demonstrations"] 
        #FIXME: The error handling is already done previously isn't it?
        extraction = json_parser.get_fields_from_llm(parsing_prompt_for_chatgpt, required_keys)
        self._instruction = extraction['Instruction']
        self._examples = extraction['Demonstrations']
        


