"""Testing integration of components locally."""

from functools import partial
from unittest.mock import patch

import openai
import pytest

from prompt2model.prompt_parser import OpenAIInstructionParser, TaskType
from test_helpers import mock_openai_response

GPT3_RESPONSE_WITH_DEMONSTRATIONS = (
    '{"Instruction": "Convert each date from an informal description into a'
    ' MM/DD/YYYY format.", "Demonstrations": "Fifth of November 2024 ->'
    ' 11/05/2024\nJan. 9 2023 -> 01/09/2023\nChristmas 2016 -> 12/25/2016"}'
)
GPT3_RESPONSE_WITHOUT_DEMONSTRATIONS = (
    '{"Instruction": "Turn the given fact into a question by a simple rearrangement'
    " of words. This typically involves replacing some part of the given fact with a"
    " WH word. For example, replacing the subject of the provided fact with the word"
    ' \\"what\\" can form a valid question. Don\'t be creative! You just need to'
    " rearrange the words to turn the fact into a question - easy! Don't just"
    " randomly remove a word from the given fact to form a question. Remember that"
    " your question must evaluate scientific understanding. Pick a word or a phrase"
    " in the given fact to be the correct answer, then make the rest of the question."
    " You can also form a question without any WH words. For example, 'A radio"
    ' converts electricity into?\'", "Demonstrations": "N/A"}'
)
GPT3_RESPONSE_WITH_INVALID_JSON = (
    '{"Instruction": "A", "Demonstrations": "B}'  # Missing final quotation mark
)


class UNKNOWN_GPT3_EXCEPTION(Exception):
    """This is a newly-defined exception for testing purposes."""

    pass


mock_prompt_parsing_example_with_demonstrations = partial(
    mock_openai_response, content=GPT3_RESPONSE_WITH_DEMONSTRATIONS
)
mock_prompt_parsing_example_without_demonstrations = partial(
    mock_openai_response, content=GPT3_RESPONSE_WITHOUT_DEMONSTRATIONS
)
mock_prompt_parsing_example_with_invalid_json = partial(
    mock_openai_response, content=GPT3_RESPONSE_WITH_INVALID_JSON
)


@patch(
    "prompt2model.utils.ChatGPTAgent.generate_openai_chat_completion",
    side_effect=mock_prompt_parsing_example_with_demonstrations,
)
def test_instruction_parser_with_demonstration(mocked_parsing_method):
    """Test a prompt-based instruction (with the LLM call mocked).

    Args:
        mocked_parsing_method: Mocked function for parsing a prompt using GPT.
    """
    prompt = """Convert each date from an informal description into a MM/DD/YYYY format.
Fifth of November 2024 -> 11/05/2024
Jan. 9 2023 -> 01/09/2023
Christmas 2016 -> 12/25/2016"""
    prompt_spec = OpenAIInstructionParser(task_type=TaskType.TEXT_GENERATION)
    prompt_spec.parse_from_prompt(prompt)

    assert prompt_spec.task_type == TaskType.TEXT_GENERATION
    assert (
        prompt_spec.instruction
        == "Convert each date from an informal description into a MM/DD/YYYY format."
    )
    assert (
        prompt_spec.demonstration
        == """Fifth of November 2024 -> 11/05/2024
Jan. 9 2023 -> 01/09/2023
Christmas 2016 -> 12/25/2016"""
    )
    assert mocked_parsing_method.call_count == 1


@patch(
    "prompt2model.utils.ChatGPTAgent.generate_openai_chat_completion",
    side_effect=mock_prompt_parsing_example_without_demonstrations,
)
def test_instruction_parser_without_demonstration(mocked_parsing_method):
    """Test a prompt-based instruction (with the LLM call mocked).

    Args:
        mocked_parsing_method: Mocked function for parsing a prompt using GPT.
    """
    prompt = """Turn the given fact into a question by a simple rearrangement of words. This typically involves replacing some part of the given fact with a WH word. For example, replacing the subject of the provided fact with the word \"what\" can form a valid question. Don't be creative! You just need to rearrange the words to turn the fact into a question - easy! Don't just randomly remove a word from the given fact to form a question. Remember that your question must evaluate scientific understanding. Pick a word or a phrase in the given fact to be the correct answer, then make the rest of the question. You can also form a question without any WH words. For example, 'A radio converts electricity into?'"""  # noqa: E501
    prompt_spec = OpenAIInstructionParser(task_type=TaskType.TEXT_GENERATION)
    prompt_spec.parse_from_prompt(prompt)

    assert prompt_spec.task_type == TaskType.TEXT_GENERATION
    assert (
        prompt_spec.instruction
        == "Turn the given fact into a question by a simple rearrangement of words. This typically involves replacing some part of the given fact with a WH word. For example, replacing the subject of the provided fact with the word \"what\" can form a valid question. Don't be creative! You just need to rearrange the words to turn the fact into a question - easy! Don't just randomly remove a word from the given fact to form a question. Remember that your question must evaluate scientific understanding. Pick a word or a phrase in the given fact to be the correct answer, then make the rest of the question. You can also form a question without any WH words. For example, 'A radio converts electricity into?'"  # noqa: E501
    )
    assert prompt_spec.demonstration == "N/A"
    assert mocked_parsing_method.call_count == 1


@patch(
    "prompt2model.utils.ChatGPTAgent.generate_openai_chat_completion",
    side_effect=mock_prompt_parsing_example_with_invalid_json,
)
def test_instruction_parser_with_invalid_json(mocked_parsing_method):
    """Verify that we handle when the API returns a invalid JSON response.

    Args:
        mocked_parsing_method: Mocked function for parsing a prompt using GPT.
    """
    prompt = """This prompt will be ignored by the parser in this test."""
    with pytest.raises(ValueError):
        prompt_spec = OpenAIInstructionParser(
            task_type=TaskType.TEXT_GENERATION, max_api_calls=3
        )
        prompt_spec.parse_from_prompt(prompt)

    assert mocked_parsing_method.call_count == 3


@patch("time.sleep")
@patch(
    "prompt2model.utils.ChatGPTAgent.generate_openai_chat_completion",
    side_effect=openai.error.Timeout("timeout"),
)
def test_instruction_parser_with_timeout(mocked_parsing_method, mocked_sleep_method):
    """Verify that we wait and retry (a set number of times) if the API times out.

    Args:
        mocked_parsing_method: Mocked function for parsing a prompt using the OpenAI
                               API. The mocked API call raises a `openai.error.Timeout`
                               error when we request a chat completion.
        mocked_sleep_method: When `time.sleep` is called, we mock it to do nothing.
                             We simply use this mock to verify that the function waits
                             some time after each API timeout.
    """
    prompt = """This prompt will be ignored by the parser in this test."""
    with pytest.raises(ValueError):
        prompt_spec = OpenAIInstructionParser(
            task_type=TaskType.TEXT_GENERATION, max_api_calls=3
        )
        prompt_spec.parse_from_prompt(prompt)

    # If we allow 3 API calls, we should have 2 sleep calls (1 after each
    # timeout).
    assert mocked_sleep_method.call_count == 3
    assert mocked_parsing_method.call_count == 3


@patch(
    "prompt2model.utils.ChatGPTAgent.generate_openai_chat_completion",
    side_effect=UNKNOWN_GPT3_EXCEPTION(),
)
def test_instruction_parser_with_unexpected_error(mocked_parsing_method):
    """Verify we don't retry the API call if an unexpected exception appears.

    Args:
        mocked_parsing_method: Mocked function for parsing a prompt using GPT.
    """
    prompt = """This prompt will be ignored by the parser in this test."""
    with pytest.raises(UNKNOWN_GPT3_EXCEPTION):
        prompt_spec = OpenAIInstructionParser(
            task_type=TaskType.TEXT_GENERATION, max_api_calls=3
        )
        prompt_spec.parse_from_prompt(prompt)

    # Check that we only tried calling the API once.
    assert mocked_parsing_method.call_count == 1
