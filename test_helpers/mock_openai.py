"""Tools for mocking OpenAI API responses (for testing purposes)."""

from __future__ import annotations  # noqa FI58

import typing


class MockCompletion:
    """Mock openai completion object."""

    def __init__(self, content: str | None = None, responses_per_request: int = 1):
        """Initialize a new instance of `MockCompletion` class.

        Args:
            content: The mocked content to be returned, i.e.,
                `json.dumps({"comment": "This is a great movie!",
                "label": 1})`.
            responses_per_request: Number of responses
                for each request.
        """
        # We generate 5 identical responses for each API call by default.
        if content is not None:
            # Mock an OpenAI ChatCompletion with identical responses.
            self.choices = [{"message": {"content": content}}] * responses_per_request
        else:
            # Mock an OpenAI ChatCompletion with different responses.
            # Only used in mock_batch_openai_response_with_different_completion.
            # The choice will be replaced later in the function.
            self.choices = []

    def __repr__(self):
        """Return a string representation.

        Returns:
            _string: A string representation of the object, including its choices.
        """
        _string = f"<MockObject choices={self.choices}>"
        return _string


def mock_one_openai_response(
    prompt: str,
    temperature: float,
    presence_penalty: float,
    frequency_penalty: float,
    content: str,
) -> MockCompletion:
    """Generate a mock completion object containing a choice with example content.

    This function creates a `MockCompletion`
    object with a `content` attribute set to an LLM completion string.

    Args:
        prompt: A mocked prompt that won't be used.
        temperature: A mocked temperature.
        presence_penalty: A mocked presence penalty.
        frequency_penalty: A mocked frequency penalty.
        content: The example string to be returned.

    Returns:
        A mock completion object simulating an OpenAI ChatCompletion API response.
    """
    _ = prompt, temperature, presence_penalty, frequency_penalty
    mock_completion = MockCompletion(content=content)
    return mock_completion


def mock_batch_openai_response_with_identical_completions(
    prompts: list[str],
    content: str,
    temperature: float,
    presence_penalty: float = 0,
    frequency_penalty: float = 0,
    responses_per_request: int = 5,
    requests_per_minute: int = 80,
) -> list[MockCompletion]:
    """Generate a batch of mock completion objects.

        This function creates a batch of `MockCompletion`
        object with a `content` attribute set to an LLM completion string.

    Args:
        prompts: A batch of mocked prompts that won't be used.
        content: The example string to be returned.
        temperature: A mocked temperature.
        presence_penalty: A mocked presence penalty.
        frequency_penalty: A mocked frequency penalty.
        responses_per_request: Number of responses for each request.
        requests_per_minute: Number of requests per minute to allow.

    Returns:
        A mock completion object simulating an OpenAI ChatCompletion API response.
    """
    _ = prompts, temperature, presence_penalty, frequency_penalty, requests_per_minute
    mock_completions = [
        MockCompletion(content=content, responses_per_request=responses_per_request)
        for _ in prompts
    ]
    return mock_completions


@typing.no_type_check
def mock_batch_openai_response_with_different_completions(
    prompts: list[str] = None,
    content: str = None,
    temperature: float = None,
    presence_penalty: float = 0,
    frequency_penalty: float = 0,
    responses_per_request: int = 5,
    requests_per_minute: int = 80,
):
    """Returns a batch of diffenrent `MockCompletion` objects at each call.

    This function is carefully designed to similuate the response of ChatGPTAgent
    and tests the generation process of the OpenAIDataSetGenerator with
    filter_duplicated_examples = True in `dataset_generator_with_filter_test`.

    This function works together with an OpenAIDataSetGenerator with
    batch_size = 2, responses_per_request = 3, expected_num_examples
    = 5, and filter_duplicated_examples = True.

    In the first API call, the generator produce 2 * 3 = 6 responses. After filtering
    duplicates, the generated_dataset will be:
        Dataset.from_dict(
        {
            "input_col": ["1", "2"],
            "output_col": ["a", "a"],
        })

    batch_size = (expected_num_examples - len(generated_dataset))
    / responses_per_request = (5 - 2) / 3 = 1.

    The second API call reduces batch_size to 1 and generates 3 more responses.


    After filtering duplicates, the generated_dataset will be:
        Dataset.from_dict(
        {
            "input_col": ["1", "2", "3"],
            "output_col": ["a", "a", "a"],
        })

    The third API call again uses batch_size = 1 and generates another 3 responses.
    After filtering duplicates, the generated_dataset will be:
        Dataset.from_dict(
        {
            "input_col": ["1", "2", "3"],
            "output_col": ["b", "a", "a"],
        })

    The fourth and final API call also uses batch_size = 1 and generates a final 3
    responses. After filtering duplicates, the generated_dataset will be:
        Dataset.from_dict(
        {
            "input_col": ["1", "2", "3", "4", "5"],
            "output_col": ["b", "a", "a", "c", "a"],
        })

    The generator will then be exhausted, and the generation process will end.
    """
    _ = (
        prompts,
        content,
        temperature,
        presence_penalty,
        frequency_penalty,
        requests_per_minute,
        responses_per_request,
    )
    # Add explicit types to the function attributes.
    if not hasattr(
        mock_batch_openai_response_with_different_completions, "mock_completions"
    ):
        # Initialize mock_completions if it doesn't exist yet.
        mock_batch_openai_response_with_different_completions.mock_completions = [
            [MockCompletion(), MockCompletion()],
            [MockCompletion()],
            [MockCompletion()],
            [MockCompletion()],
        ]
        mock_batch_openai_response_with_different_completions.current_index = 0

        # Populate mock_completions with desired choices as before.
        mock_completion_1 = MockCompletion()
        mock_completion_1.choices = [
            {"message": {"content": '{"input": "1", "output": "a"}'}},
            {"message": {"content": '{"input": "1", "output": "b"}'}},
            {"message": {"content": '{"input": "1", "output": "a"}'}},
        ]
        mock_completion_2 = MockCompletion()
        mock_completion_2.choices = [
            {"message": {"content": '{"input": "1", "output": "c"}'}},
            {"message": {"content": '{"input": "2", "output": "a"}'}},
            {"message": {"content": '{"input": "2", "output": "b"}'}},
        ]
        mock_batch_openai_response_with_different_completions.mock_completions[0] = [
            mock_completion_1,
            mock_completion_2,
        ]
        mock_completion_3 = MockCompletion()
        mock_completion_3.choices = [
            {"message": {"content": '{"input": "3", "output": "a"}'}},
            {"message": {"content": '{"input": "3", "output": "a"}'}},
            {"message": {"content": '{"input": "3", "output": "b"}'}},
        ]
        mock_batch_openai_response_with_different_completions.mock_completions[1] = [
            mock_completion_3
        ]

        mock_completion_4 = MockCompletion()
        mock_completion_4.choices = [
            {"message": {"content": '{"input": "1", "output": "b"}'}},
            {"message": {"content": '{"input": "1", "output": "b"}'}},
            {"message": {"content": '{"input": "1", "output": "b"}'}},
        ]
        mock_batch_openai_response_with_different_completions.mock_completions[2] = [
            mock_completion_4
        ]
        mock_completion_5 = MockCompletion()
        mock_completion_5.choices = [
            {"message": {"content": '{"input": "4", "output": "c"}'}},
            {"message": {"content": '{"input": "4", "output": "c"}'}},
            {"message": {"content": '{"input": "5", "output": "a"}'}},
        ]
        mock_batch_openai_response_with_different_completions.mock_completions[3] = [
            mock_completion_5
        ]

    # Get the current index and increment it for the next call.
    current_index = mock_batch_openai_response_with_different_completions.current_index
    mock_batch_openai_response_with_different_completions.current_index += 1

    mock_completions = (
        mock_batch_openai_response_with_different_completions.mock_completions[
            current_index % 4
        ]
    )
    assert len(mock_completions) == len(prompts)
    # Return the corresponding MockCompletion object for this call.
    return mock_completions


def reset_mock_batch_openai_response_with_different_completions():
    """Resets the state of the `mock_batch_openai_response_with_different_completions`.

    Reset the state of the `mock_batch_openai_response_with_different_completions`
    function by deleting its `mock_completions` and `current_index` attributes, if they
    exist. This allows the `mock_batch_openai_response_with_different_completions`
    function to be reused in a fresh state in subsequent tests.

    The `mock_batch_openai_response_with_different_completions` function simulates
    the behavior of the OpenAI API during the generation of OpenAIDataSetGenerator.
    It returns a batch of different `MockCompletion` objects on each call, which are
    used to simulate different API responses. The `mock_completions` attribute stores
    these `MockCompletion` objects, while the `current_index` attribute keeps track
    of the current position in the `mock_completions` list.
    """
    if hasattr(
        mock_batch_openai_response_with_different_completions, "mock_completions"
    ):
        del mock_batch_openai_response_with_different_completions.mock_completions
    if hasattr(mock_batch_openai_response_with_different_completions, "current_index"):
        del mock_batch_openai_response_with_different_completions.current_index
