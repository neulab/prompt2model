"""Tools for accessing OpenAI's API."""

from __future__ import annotations  # noqa FI58

import json
import time
import logging
import openai

OPENAI_ERRORS = (
    openai.error.APIError,
    openai.error.Timeout,
    openai.error.RateLimitError,
    openai.error.ServiceUnavailableError,
    json.decoder.JSONDecodeError,
    AssertionError,
)


class ChatGPTAgent:
    """A class for accessing OpenAI's ChatCompletion API."""

    def __init__(self, api_key: str | None):
        """Initialize ChatGPTAgent with an API key.

        Args:
            api_key: A valid OpenAI API key. Alternatively, set as None and set
                     the environment variable with `export OPENAI_API_KEY=<your key>`.
        """
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


def handle_openai_error(e, api_call_counter):
    """Handle OpenAI errors or related errors that the OpenAI API may raise.

    Args:
        e: The error to handle. This could be an OpenAI error or a related
           non-fatal error, such as JSONDecodeError or AssertionError.
        api_call_counter: The number of API calls made so far.

    Returns:
        The api_call_counter (if no error was raised), else raise the error.
    """
    logging.error(e)
    if isinstance(
        e,
        (openai.error.APIError, openai.error.Timeout, openai.error.RateLimitError),
    ):
        # For these errors, OpenAI recommends waiting before retrying.
        time.sleep(1)

    if isinstance(e, OPENAI_ERRORS):
        # For these errors, we can increment a counter and retry the API call.
        return api_call_counter
    else:
        # For all other errors, immediately throw an exception.
        raise e
