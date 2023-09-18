import json

import openai
from prompt2model.utils.api_tools import API_ERRORS, handle_api_error
from prompt2model.utils import api_tools, get_formatted_logger

logger = get_formatted_logger("parse_json_responses")

class JsonParsingFromLLMResponse:
    def __init__(self, max_api_calls: int = None):
        if max_api_calls and max_api_calls <= 0:
            raise ValueError("max_api_calls must be > 0.")
        self.max_api_calls = max_api_calls
        self.api_call_counter = 0

    def extract_response(self, response: openai.Completion, required_keys: list, optional_keys: list) -> dict:
        """Parse stuctured fields from the API response.

        Args:
            response: API response.

        Returns:
            If the API response is a valid JSON object and contains the required_keys,
                then returns a tuple consisting of:
                1) Instruction: The instruction parsed from the API response.
                2) Demonstrations: (Optional) demonstrations parsed from the
                API response.
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
        
        final_response = {key:response_json[key].strip() for key in required_keys }
        final_response |= {key:response_json[key].strip() for key in optional_keys if key in response_json}
        return final_response
    
    def get_fields_from_llm(self, prompt: str, required_keys: list, optional_keys: list =None, temperature: float =0.01, presence_penalty:float =0, frequency_penalty:float =0) -> dict:
        """Parse prompt into specific fields, stored as class member variables.

        This function directly stores the parsed fields into the class's member
        variables `instruction` and `examples`. So it has no return value.

        Args:
            prompt: User prompt to parse into two specific fields:
                    "instruction" and "demonstrations".
        """
        chat_api = api_tools.default_api_agent
        last_error = None
        while True:
            self.api_call_counter += 1
            try:
                response: openai.ChatCompletion | Exception = (
                    chat_api.generate_one_completion(
                        prompt,
                        temperature=temperature,
                        presence_penalty=presence_penalty,
                        frequency_penalty=frequency_penalty,
                    )
                )
                extraction = self.extract_response(response, required_keys, optional_keys)
                if extraction is not None:
                    return extraction
            except API_ERRORS as e:
                last_error = e
                handle_api_error(e)

            if self.max_api_calls and self.api_call_counter >= self.max_api_calls:
                # In case we reach maximum number of API calls, we raise an error.
                logger.error("Maximum number of API calls reached.")
                raise RuntimeError(
                    "Maximum number of API calls reached."
                ) from last_error

