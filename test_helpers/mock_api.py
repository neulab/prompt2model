"""Tools for mocking API responses (for testing purposes)."""

from __future__ import annotations

import openai

from prompt2model.utils.api_tools import APIAgent


class MockCompletion:
    """Mock completion object."""

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
            # Mock a ChatCompletion with identical responses.
            self.choices = [{"message": {"content": content}}] * responses_per_request
        else:
            # Mock a ChatCompletion with different responses.
            # Only used in mock_batch_api_response_with_different_completion.
            # The choice will be replaced later in the function.
            self.choices = []

    def __repr__(self):
        """Return a string representation.

        Returns:
            _string: A string representation of the object, including its choices.
        """
        _string = f"<MockObject choices={self.choices}>"
        return _string


class MockBatchDifferentCompletions:
    """Mock batch completion object."""

    def __init__(self, length: int = 4) -> None:
        """Init a new instance of `MockBatchDifferentCompletions`.

        Args:
            length: Length of the batch completions.

        This class is designed to simulate the response of APIAgent and test the
        generation process of the PromptBasedDatasetGenerator with
        `filter_duplicated_examples` set to True in
        `dataset_generator_with_filter_test`.

        The class works in conjunction with PromptBasedDatasetGenerator with
        batch_size = 2, responses_per_request = 3, expected_num_examples
        = 5, and filter_duplicated_examples = True.

        Explanation of the generation process:

        In the first API call, the generator produces 2 * 3 = 6 responses.
        After filtering duplicates, the generated_dataset will be:

        Dataset.from_dict({
            "input_col": ["1", "2"],
            "output_col": ["a", "a"],
        })

        The second API call reduces batch_size to 1 and generates 3 more
        responses. After filtering duplicates, the generated_dataset will be:

        Dataset.from_dict({
            "input_col": ["1", "2", "3"],
            "output_col": ["a", "a", "a"],
        })

        The third API call again uses batch_size = 1 and generates another
        3 responses. After filtering duplicates, the generated_dataset will be:

        Dataset.from_dict({
            "input_col": ["1", "2", "3"],
            "output_col": ["b", "a", "a"],
        })

        The fourth and API call also uses batch_size = 1 and generates the final
        3 responses. After filtering duplicates, the generated_dataset will be:

        Dataset.from_dict({
            "input_col": ["1", "2", "3", "4", "5"],
            "output_col": ["b", "a", "a", "c", "a"],
        })

        The fivth and API call is specifically designed for
        testing generate dataset_dict.
        """
        assert length == 4 or length == 5
        self.mock_completions: list[list[MockCompletion]] = []
        self.current_index = 0
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
        self.mock_completions.append(
            [
                mock_completion_1,
                mock_completion_2,
            ]
        )
        mock_completion_3 = MockCompletion()
        mock_completion_3.choices = [
            {"message": {"content": '{"input": "3", "output": "a"}'}},
            {"message": {"content": '{"input": "3", "output": "a"}'}},
            {"message": {"content": '{"input": "3", "output": "b"}'}},
        ]
        self.mock_completions.append([mock_completion_3])

        mock_completion_4 = MockCompletion()
        mock_completion_4.choices = [
            {"message": {"content": '{"input": "1", "output": "b"}'}},
            {"message": {"content": '{"input": "1", "output": "b"}'}},
            {"message": {"content": '{"input": "1", "output": "b"}'}},
        ]
        self.mock_completions.append([mock_completion_4])
        mock_completion_5 = MockCompletion()
        mock_completion_5.choices = [
            {"message": {"content": '{"input": "4", "output": "c"}'}},
            {"message": {"content": '{"input": "4", "output": "c"}'}},
            {"message": {"content": '{"input": "5", "output": "a"}'}},
        ]
        self.mock_completions.append([mock_completion_5])
        if length == 5:
            self.mock_completions.append(
                [
                    mock_completion_1,
                    mock_completion_2,
                ]
            )


def mock_batch_api_response_identical_completions(
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
        A mock completion object simulating an ChatCompletion API response.
    """
    _ = prompts, temperature, presence_penalty, frequency_penalty, requests_per_minute
    mock_completions = [
        MockCompletion(content=content, responses_per_request=responses_per_request)
        for _ in prompts
    ]
    return mock_completions


class MockAPIAgent(APIAgent):
    """A mock API agent that always returns the same content."""

    def __init__(self, default_content):
        """Initialize the API agent."""
        self.generate_one_call_counter = 0
        self.generate_batch_call_counter = 0
        self.default_content = default_content

    def generate_one_completion(
        self,
        prompt: str,
        temperature: float = 0,
        presence_penalty: float = 0,
        frequency_penalty: float = 0,
    ) -> openai.Completion:
        """Return a mocked object and increment the counter."""
        self.generate_one_call_counter += 1
        return MockCompletion(content=self.default_content)

    async def generate_batch_completion(
        self,
        prompts: list[str],
        temperature: float = 1,
        responses_per_request: int = 5,
        requests_per_minute: int = 80,
    ) -> list[openai.Completion]:
        """Return a mocked object and increment the counter."""
        self.generate_batch_call_counter += 1
        return [MockCompletion(content=self.default_content) for _ in prompts]


class UnknownGpt3Exception(Exception):
    """This is a newly-defined exception for testing purposes."""

    pass
