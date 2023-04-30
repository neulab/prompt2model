"""Tools for accessing OpenAI's API."""

import openai


def generate_openai_chat_completion(api_key: str, prompt: str) -> openai.Completion:
    """Generate a chat completion using OpenAI's gpt-3.5-turbo.

    Args:
        prompt: A prompt asking for a response.

    Returns:
        A response object.
    """
    openai.api_key = api_key
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "user", "content": f"{prompt}"},
        ],
    )
    return response
