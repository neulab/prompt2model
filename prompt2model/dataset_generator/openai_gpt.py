"""A simple dataset generator that uses OpenAI's GPT-3.5 API."""

from __future__ import annotations  # noqa FI58

import json
import logging
import os

import openai
from datasets import Dataset
from tqdm import tqdm

from prompt2model.dataset_generator.base import DatasetGenerator, DatasetSplit
from prompt2model.prompt_parser import PromptSpec
from prompt2model.utils import OPENAI_ERRORS, ChatGPTAgent, handle_openai_error

PROMPT_TEMPLATE = """
As a DatasetGenerator, your task is to generate a new example (input and output) based on the [new instruction] and [few-shot examples]. Please provide a JSON dictionary response that includes the new input and its corresponding output. Use the `input` and `output` keys in the dictionary. The 'input' field should be marked as 'N/A' if the instruction doesn't require additional input. It is important that the input and output you generate are distinct from the examples provided. Please ensure that your response is diverse, detailed, precise, comprehensive, and of high-quality.

--------------------------------------------------------------------------------------------
Here are some exmaples you can refer to:

- Example 1

instruction: Which exercises are best for reducing belly fat at home?
input: N/A
output:
- Lying Leg Raises
- Leg In And Out
- Plank
- Side Plank
- Sit-ups

- Example 2

instruction: Extract all the country names in the paragraph, and list them separated by commas.
input: Dr. No is the sixth novel by the English author Ian Fleming to feature his British Secret Service agent James Bond. Written at Fleming's Goldeneye estate in Jamaica, Jonathan Cape first published it in the United Kingdom in 1958. In the novel Bond looks into the disappearance in Jamaica of two fellow MI6 operatives who had been investigating Doctor No. Bond travels to No's Caribbean island and meets Honeychile Rider, who is there to collect shells. They are captured and taken to a luxurious facility carved into a mountain. The character of Doctor No, the son of a German missionary and a Chinese woman, was influenced by Sax Rohmer's Fu Manchu stories. Dr. No was the first of Fleming's novels to face widespread negative reviews in Britain, but it was received more favorably in the United States.
output: English, British, Jamaica, the United Kingdom, German, Chinese, Britain, the United States.

- Example 3

instruction: Converting 85 F to Celsius.
input: N/A
output: 85°F = 29.44°C

- Example 4
instruction: Sort the given list ascendingly.
input: [10, 92, 2, 5, -4, 92, 5, 101]
output: [-4, 2, 5, 5, 10, 92, 92, 101]

- Example 5
instruction: Suggest a better and more professional rephrasing of the following sentence.
input: This house is surprisingly not constructed very well, and you probably need more money to fix it after you buy it. If you ask me, I would suggest you consider other candidates.
output: This house does not seem to be constructed well, so you may need to spend more money to fix it after you purchase it. I would suggest that you look at other properties.

- Example 6
instruction: Read the following paragraph and answer a math question about the paragraph. You need to write out the calculation to get the final answer.
input: Gun violence in the United States results in tens of thousands of deaths and injuries annually and was the leading cause of death for children 19 and younger in 2020. In 2018, the most recent year for which data are available as of 2021, the Centers for Disease Control and Prevention's (CDC) National Center for Health Statistics reports 38,390 deaths by firearm, of which 24,432 were by suicide. The rate of firearm deaths per 100,000 people rose from 10.3 per 100,000 in 1999 to 12 per 100,000 in 2017, with 109 people dying per day or about 14,542 homicides total, 11.9 per 100,000 in 2018. In 2010, there were 19,392 firearm-related suicides and 11,078 firearm-related homicides in the U.S. In 2010, 358 murders were reported involving a rifle, while 6,009 were reported involving a handgun; another 1,939 were reported with an unspecified type of firearm. In 2011, a total of 478,400 fatal and nonfatal violent crimes were committed with a firearm. How many more firearm-related deaths were there in 2018 compared to 2010?
output:
38390 - (19392 + 11078) = 38390 - 30470 = 7920. So, in 2018, there were 7920 more deaths by firearm than in 2010.

- Example 7
instruction: Write Python code to solve this leet code problem.
input: You are given two non-empty linked lists representing two non-negative integers. The digits are stored in reverse order, and each of their nodes contains a single digit. Add the two numbers and return the sum as a linked list. You may assume the two numbers do not contain any leading zero except the number 0 itself.
output:
class Solution(object):
    def addTwoNumbers(self, l1, l2):
        carry = 0
        root = n = ListNode(0)
        while l1 or l2 or carry:
            v1 = v2 = 0
            if l1:
                v1 = l1.val
                l1 = l1.next
            if l2:
                v2 = l2.val
                l2 = l2.next
            carry, val = divmod(v1+v2+carry, 10)
            n.next = ListNode(val)
            n = n.next
        return root.next

- Example 8
instruction: Solve the equation and find the value of X. Show your steps.
input: 10X + 5 = 10
output: 10X = 5,  X = 0.5

- Example 9
instruction: Write a program to compute the sum of integers from k to n.
input: N/A
output:
def sum(k, n):
    sum = 0
    for i in range(k, n+1):
        sum += i
    return sum

- Example 10
instruction: Select the oldest person from the given list.
input: George Washington, Confucius, Michael Jordan, Michelangelo
output: Confucious

- Example 11
instruction: Turn down a job offer by sending an email to a recruiter explaining the reason.
input: N/A
output: Hi  Recruiter,
Thank you so much for the generous offer to join your team. As we discussed, I've admired the company for a number of years, and am a proud endorser of its products. However, after further consideration of where I currently am in my career, I've decided to accept an offer at another company.
I would love to stay in touch with you and have already started following you on [Social Media Platform]. Again, thank you so much for your time and consideration.
Thanks again,
Your Name
--------------------------------------------------------------------------------------------

[new instruction]:
{instruction}

[few-shot examples]:
{examples}

[new example (in JSON)]:"""  # noqa: E501
# A string template for the prompt. Can be modified by the users.
# Prompt_template must contains `instruction` and `examples` fields.


class OpenAIDatasetGenerator(DatasetGenerator):
    """A abstract class for NLP dataset generation using OpenAI's GPT-3.5 API."""

    def __init__(self, api_key: str | None = None, max_api_calls: int = None):
        """Initialize an OpenAI DatasetGenerator with an API key and max API call.

        Args:
            api_key: A valid OpenAI API key. Alternatively, set as None and set
                the environment variable with `export OPENAI_API_KEY=<your key>`.
            max_api_calls: The maximum number of API calls allowed,
                or None for unlimited.
        """
        self.api_key: str | None = api_key if api_key else os.environ["OPENAI_API_KEY"]
        assert self.api_key is not None and self.api_key != "", (
            "API key must be provided"
            + " or set the environment variable with `export OPENAI_API_KEY=<your key>`"
        )
        if max_api_calls:
            assert max_api_calls > 0, "max_api_calls must be > 0"
        self.max_api_calls = max_api_calls
        self.api_call_counter = 0

    def generate_prompt(
        self,
        instruction: str,
        examples: list[str] = None,
    ) -> str:
        """Generates a prompt string.

        Args:
            instruction: The natural language instruction for the prompt.
            examples: A list of few-shot examples. Defaults to None.

        Returns:
            The generated prompt string.
        """
        # Replace placeholders in prompt template with actual values
        example_string = examples if examples else "NA"
        prompt = PROMPT_TEMPLATE.format(
            instruction=instruction, examples=example_string
        )
        return prompt

    def extract_response(self, response: openai.Completion) -> tuple[str, str]:
        """Extracts the generated sample and annotation from an OpenAI API response.

        Args:
            response (openai.Completion): The response object returned by OpenAI API.

        Returns:
            A tuple of (sample, annotation), where:
            - sample is the generated example string extracted from the response.
            - annotation is the generated label string extracted from the response.
        """
        try:
            response_json = json.loads(response.choices[0]["message"]["content"])
        except json.decoder.JSONDecodeError as e:
            logging.warning("API response was not a valid JSON")
            raise e
        required_keys = ["input", "output"]
        missing_keys = [key for key in required_keys if key not in response_json]
        assert (
            len(missing_keys) == 0
        ), f'API response must contain {", ".join(required_keys)} keys'
        input = str(response_json["input"]).strip()
        output = str(response_json["output"]).strip()
        return input, output

    def generate_dataset_split(
        self,
        prompt_spec: PromptSpec,
        num_examples: int,
        split: DatasetSplit,
    ) -> Dataset:
        """Generate a single dataset using GPT-3.5.

        Args:
            prompt_spec: A prompt specification.
            num_examples: Number of examples in split.
            split: Name of dataset split to generate.

        Returns:
            A single dataset.
        """
        _ = split  # suppress unused variable warnings
        prompt = self.generate_prompt(
            instruction=prompt_spec.get_instruction,
            examples=prompt_spec.get_examples,
        )
        logging.info(
            f"LLM Prompt: \n\n {prompt}\n\n============================\n\n"  # noqa: E501
        )
        chat_api = ChatGPTAgent(self.api_key)
        input_cols = []  # type: list[str]
        output_cols = []  # type: list[str]

        for _ in tqdm(range(num_examples), desc="Generating examples"):
            while True:
                try:
                    if (
                        self.max_api_calls
                        and self.api_call_counter >= self.max_api_calls
                    ):
                        logging.warning("Maximum number of API calls reached.")
                        return Dataset.from_dict(
                            {"input_col": input_cols, "output_col": output_cols}
                        )
                    else:
                        self.api_call_counter += 1
                    response = chat_api.generate_openai_chat_completion(prompt)
                    input_col, output_col = self.extract_response(response)
                    logging.info(
                        f" Input: \n\n {input_col}\n\n======================\n\n"  # noqa: E501
                    )
                    logging.info(
                        f" Output: \n\n {output_col}\n\nn======================\n\n"  # noqa: E501
                    )
                    input_cols.append(input_col)
                    output_cols.append(output_col)
                    break
                except OPENAI_ERRORS as e:
                    self.api_call_counter = handle_openai_error(
                        e, self.api_call_counter
                    )

        return Dataset.from_dict({"input_col": input_cols, "output_col": output_cols})
