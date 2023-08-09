"""Testing integration of components locally."""

import gc
import os
from functools import partial
from unittest.mock import patch

import openai
import pytest

from prompt2model.prompt_parser import OpenAIInstructionParser, TaskType
from test_helpers import UnknownGpt3Exception, mock_one_openai_response

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


mock_prompt_parsing_example_with_demonstrations = partial(
    mock_one_openai_response, content=GPT3_RESPONSE_WITH_DEMONSTRATIONS
)
mock_prompt_parsing_example_without_demonstrations = partial(
    mock_one_openai_response, content=GPT3_RESPONSE_WITHOUT_DEMONSTRATIONS
)
mock_prompt_parsing_example_with_invalid_json = partial(
    mock_one_openai_response, content=GPT3_RESPONSE_WITH_INVALID_JSON
)


@patch(
    "prompt2model.utils.ChatGPTAgent.generate_one_openai_chat_completion",
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
    os.environ["OPENAI_API_KEY"] = "fake_api_key"
    prompt_spec = OpenAIInstructionParser(task_type=TaskType.TEXT_GENERATION)
    prompt_spec.parse_from_prompt(prompt)

    assert prompt_spec.task_type == TaskType.TEXT_GENERATION
    correct_instruction = (
        "Convert each date from an informal description into a MM/DD/YYYY format."
    )
    assert prompt_spec.instruction == correct_instruction
    assert prompt_spec.instruction == correct_instruction
    assert (
        prompt_spec.examples
        == """Fifth of November 2024 -> 11/05/2024
Jan. 9 2023 -> 01/09/2023
Christmas 2016 -> 12/25/2016"""
    )
    assert (
        prompt_spec.examples
        == """Fifth of November 2024 -> 11/05/2024
Jan. 9 2023 -> 01/09/2023
Christmas 2016 -> 12/25/2016"""
    )
    assert mocked_parsing_method.call_count == 1


@patch(
    "prompt2model.utils.ChatGPTAgent.generate_one_openai_chat_completion",
    side_effect=mock_prompt_parsing_example_without_demonstrations,
)
def test_instruction_parser_without_demonstration(mocked_parsing_method):
    """Test a prompt-based instruction (with the LLM call mocked).

    Args:
        mocked_parsing_method: Mocked function for parsing a prompt using GPT.
    """
    api_key = "fake_api_key"
    prompt = """Turn the given fact into a question by a simple rearrangement of words. This typically involves replacing some part of the given fact with a WH word. For example, replacing the subject of the provided fact with the word \"what\" can form a valid question. Don't be creative! You just need to rearrange the words to turn the fact into a question - easy! Don't just randomly remove a word from the given fact to form a question. Remember that your question must evaluate scientific understanding. Pick a word or a phrase in the given fact to be the correct answer, then make the rest of the question. You can also form a question without any WH words. For example, 'A radio converts electricity into?'"""  # noqa: E501
    prompt_spec = OpenAIInstructionParser(
        task_type=TaskType.TEXT_GENERATION, api_key=api_key
    )
    prompt_spec.parse_from_prompt(prompt)

    assert prompt_spec.task_type == TaskType.TEXT_GENERATION
    correct_instruction = "Turn the given fact into a question by a simple rearrangement of words. This typically involves replacing some part of the given fact with a WH word. For example, replacing the subject of the provided fact with the word \"what\" can form a valid question. Don't be creative! You just need to rearrange the words to turn the fact into a question - easy! Don't just randomly remove a word from the given fact to form a question. Remember that your question must evaluate scientific understanding. Pick a word or a phrase in the given fact to be the correct answer, then make the rest of the question. You can also form a question without any WH words. For example, 'A radio converts electricity into?'"  # noqa: E501
    assert prompt_spec.instruction == correct_instruction
    assert prompt_spec.instruction == correct_instruction
    assert prompt_spec.examples == "N/A"
    assert prompt_spec.examples == "N/A"
    assert mocked_parsing_method.call_count == 1
    gc.collect()


@patch(
    "prompt2model.utils.ChatGPTAgent.generate_one_openai_chat_completion",
    side_effect=mock_prompt_parsing_example_with_invalid_json,
)
def test_instruction_parser_with_invalid_json(mocked_parsing_method):
    """Verify that we handle when the API returns a invalid JSON response.

    Args:
        mocked_parsing_method: Mocked function for parsing a prompt using GPT.
    """
    os.environ["OPENAI_API_KEY"] = "fake_api_key"
    prompt = """This prompt will be ignored by the parser in this test."""
    prompt_spec = OpenAIInstructionParser(
        task_type=TaskType.TEXT_GENERATION, max_api_calls=3
    )
    with patch("logging.info") as mock_info, patch("logging.warning") as mock_warning:
        prompt_spec.parse_from_prompt(prompt)
        mock_info.assert_not_called()
        warning_list = [each.args[0] for each in mock_warning.call_args_list]
        assert warning_list == ["API response was not a valid JSON"] * 3 + [
            "Maximum number of API calls reached for PromptParser."
        ]
    assert mocked_parsing_method.call_count == 3
    assert prompt_spec._instruction is None
    assert prompt_spec._examples is None
    gc.collect()


@patch("time.sleep")
@patch(
    "prompt2model.utils.ChatGPTAgent.generate_one_openai_chat_completion",
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
    api_key = "fake_api_key"
    prompt = """This prompt will be ignored by the parser in this test."""
    with pytest.raises(ValueError) as exc_info:
        prompt_spec = OpenAIInstructionParser(
            task_type=TaskType.TEXT_GENERATION, api_key=api_key, max_api_calls=3
        )
        prompt_spec.parse_from_prompt(prompt)

    # If we allow 3 API calls, we should have 2 sleep calls (1 after each
    # timeout).
    assert mocked_sleep_method.call_count == 3
    assert mocked_parsing_method.call_count == 3

    # Check if the ValueError was raised
    assert isinstance(exc_info.value, ValueError)
    # Check if the original exception (e) is present as the cause
    original_exception = exc_info.value.__cause__
    assert isinstance(original_exception, openai.error.Timeout)
    gc.collect()


@patch(
    "prompt2model.utils.ChatGPTAgent.generate_one_openai_chat_completion",
    side_effect=UnknownGpt3Exception(),
)
def test_instruction_parser_with_unexpected_error(mocked_parsing_method):
    """Verify we don't retry the API call if an unexpected exception appears.

    Args:
        mocked_parsing_method: Mocked function for parsing a prompt using GPT.
    """
    os.environ["OPENAI_API_KEY"] = "fake_api_key"
    prompt = """This prompt will be ignored by the parser in this test."""
    with pytest.raises(UnknownGpt3Exception):
        prompt_spec = OpenAIInstructionParser(
            task_type=TaskType.TEXT_GENERATION, max_api_calls=3
        )
        prompt_spec.parse_from_prompt(prompt)

    # Check that we only tried calling the API once.
    assert mocked_parsing_method.call_count == 1
    gc.collect()


def test_openai_key_init():
    """Test openai key initialization."""
    api_key = None
    os.environ["OPENAI_API_KEY"] = ""
    with pytest.raises(AssertionError) as exc_info:
        _ = OpenAIInstructionParser(task_type=TaskType.TEXT_GENERATION)
        assert str(exc_info.value) == (
            "API key must be provided or set the environment variable"
            + " with `export OPENAI_API_KEY=<your key>`"
        )
    os.environ["OPENAI_API_KEY"] = "fake_api_key"
    environment_key_parser = OpenAIInstructionParser(task_type=TaskType.TEXT_GENERATION)
    assert (
        environment_key_parser.api_key == os.environ["OPENAI_API_KEY"]
        and os.environ["OPENAI_API_KEY"] is not None
    )
    os.environ["OPENAI_API_KEY"] = ""
    api_key = "qwertwetyriutytwreytuyrgtwetrueytttr"
    explicit_api_key_paser = OpenAIInstructionParser(
        task_type=TaskType.TEXT_GENERATION, api_key=api_key
    )
    assert explicit_api_key_paser.api_key == api_key and api_key is not None
    gc.collect()
