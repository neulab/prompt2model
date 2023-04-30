"""Tools for accessing OpenAI's API."""

import openai


class ChatGPTAgent:
    """A class for accessing OpenAI's ChatCompletion API."""

    def __init__(self, api_key):
        """Initialize ChatGPTAgent with an API key."""
        openai.api_key = api_key

    def generate_openai_chat_completion(self, prompt: str) -> openai.Completion:
        """Generate a chat completion using OpenAI's gpt-3.5-turbo.

        Args:
            prompt: A prompt asking for a response.

        Returns:
            A response object.
        """
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "user", "content": f"{prompt}"},
            ],
        )
        return response
