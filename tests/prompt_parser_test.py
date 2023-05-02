"""Testing integration of components locally."""

from functools import partial
from unittest.mock import patch

from prompt2model.prompt_parser import OpenAIInstructionParser, TaskType
from test_helpers import mock_openai_response

gpt3_response_with_demonstrations = """
1) Instruction:
Convert each date from an informal description into a MM/DD/YYYY format.

2) Demonstrations:
- Fifth of November 2024 -> 11/05/2024
- Jan. 9 2023 -> 01/09/2023
- Christmas 2016 -> 12/25/2016
"""

mock_prompt_parsing_example_with_demonstrations = partial(
    mock_openai_response, content=gpt3_response_with_demonstrations
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
    prompt = """
    Convert each date from an informal description into a MM/DD/YYYY format.
    Fifth of November 2024 -> 11/05/2024
    Jan. 9 2023 -> 01/09/2023
    Christmas 2016 -> 12/25/2016
    """
    prompt_spec = OpenAIInstructionParser(task_type=TaskType.TEXT_GENERATION)
    prompt_spec.parse_from_prompt(prompt)

    assert prompt_spec.task_type == TaskType.TEXT_GENERATION
    assert (
        prompt_spec.instruction
        == "Convert each date from an informal description into a MM/DD/YYYY format."
    )
    assert (
        prompt_spec.demonstration
        == """- Fifth of November 2024 -> 11/05/2024
- Jan. 9 2023 -> 01/09/2023
- Christmas 2016 -> 12/25/2016"""
    ), breakpoint()
    assert mocked_parsing_method.call_count == 1


gpt3_response_without_demonstrations = """1) Instruction:
Turn the given fact into a question by a simple rearrangement of words. This typically involves replacing some part of the given fact with a WH word. For example, replacing the subject of the provided fact with the word "what" can form a valid question. Don't be creative! You just need to rearrange the words to turn the fact into a question - easy! Don't just randomly remove a word from the given fact to form a question. Remember that your question must evaluate scientific understanding. Pick a word or a phrase in the given fact to be the correct answer, then make the rest of the question. You can also form a question without any WH words. For example, 'A radio converts electricity into?'

2) Demonstrations:
NO DEMONSTRATION."""  # noqa: E501
mock_prompt_parsing_example_without_demonstrations = partial(
    mock_openai_response, content=gpt3_response_without_demonstrations
)


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
    assert prompt_spec.demonstration is None
    assert mocked_parsing_method.call_count == 1
