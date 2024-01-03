"""Utility file for parsing OpenAI json responses."""
from __future__ import annotations

import json

import openai

from prompt2model.utils import api_tools, get_formatted_logger
from prompt2model.utils.api_tools import API_ERRORS, handle_api_error

logger = get_formatted_logger("ParseJsonResponses")


def extract_response(
    response: openai.Completion, required_keys: list, optional_keys: list
) -> dict | None:
    """Parse stuctured fields from the API response.

    Args:
        response: API response.
        required_keys: Required keys from the response
        optional_keys: Optional keys from the response

    Returns:
        If the API response is a valid JSON object and contains the
        required and optional keys then returns the
        final response as a Dictionary
        Else returns None.
    """
    response_text = response.choices[0]["message"]["content"]
    try:
        response_json = json.loads(response_text, strict=False)
    except json.decoder.JSONDecodeError:
        logger.warning(f"API response was not a valid JSON: {response_text}")
        return None

    missing_keys = [key for key in required_keys if key not in response_json]
    if len(missing_keys) != 0:
        logger.warning(f'API response must contain {", ".join(required_keys)} keys')
        return None

    final_response = {}
    for key in required_keys + optional_keys:
        if key not in response_json:
            # This is an optional key, so exclude it from the final response.
            continue
        if type(response_json[key]) == str:
            final_response[key] = response_json[key].strip()
        else:
            final_response[key] = response_json[key]
    return final_response


def parse_prompt_to_fields(
    prompt: str,
    required_keys: list,
    optional_keys: list = [],
    max_api_calls: int = 5,
) -> dict:
    """Parse prompt into specific fields, and return to the calling function.

    This function calls the required api, has the logic for the retrying,
    passes the response to the parsing function, and return the
    response back or throws an error

    Args:
        prompt: User prompt into specific fields
        required_keys: Fields that need to be present in the response
        optional_keys: Field that may/may not be present in the response
        max_api_calls: Max number of retries, defaults to 5 to avoid
                        being stuck in an infinite loop

    Returns:
        Parsed Response or throws error
    """
    chat_api = api_tools.default_api_agent
    if max_api_calls <= 0:
        raise ValueError("max_api_calls must be > 0.")

    api_call_counter = 0
    last_error = None
    while True:
        api_call_counter += 1
        try:
            response: openai.ChatCompletion | Exception = (
                chat_api.generate_one_completion(
                    prompt,
                    temperature=0.01,
                    presence_penalty=0,
                    frequency_penalty=0,
                )
            )
            extraction = extract_response(response, required_keys, optional_keys)
            if extraction is not None:
                return extraction
        except API_ERRORS as e:
            last_error = e
            handle_api_error(e)

        if api_call_counter >= max_api_calls:
            # In case we reach maximum number of API calls, we raise an error.
            logger.error("Maximum number of API calls reached.")
            raise RuntimeError("Maximum number of API calls reached.") from last_error
