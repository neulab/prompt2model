"""Tools for mocking OpenAI API responses (for testing purposes)."""


class MockCompletion:
    """Mock openai completion object."""

    def __init__(self, content: str):
        """Initialize a new instance of `MockCompletion` class.

        Args:
            content: The mocked content to be returned, i.e.,
            `json.dumps({"comment": "This is a great movie!",
            "label": 1})`.
        """
        self.choices = [{"message": {"content": content}}]

    def __repr__(self):
        """Return a string representation of the `MockCompletion` object.

        Returns:
            _string: A string representation of the object, including its choices.
        """
        _string = f"<MockObject choices={self.choices}>"
        return _string


def mock_openai_response(
    prompt: str,
    temperature: float,
    presence_penalty: float,
    frequency_penalty: float,
    content: str,
) -> MockCompletion:
    """Generate a mock completion object containing a choice with example content.

    This function creates a `MockCompletion` object with a `content` attribute set to
    an LLM completion string.

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
