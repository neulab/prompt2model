"""A simple dataset generator that uses OpenAI's GPT-3.5 API."""

import json

import openai
from datasets import Dataset
from tqdm import tqdm

from prompt2model.dataset_generator.base import DatasetSplit
from prompt2model.dataset_generator.simple import OpenAIDatasetGenerator
from prompt2model.prompt_parser import PromptSpec


class ClassifyDatasetGenerator(OpenAIDatasetGenerator):
    """A dataset generator for examples and pesudo-label for classification tasks."""

    def generate_prompt(
        self, natrual_instruction: str, few_shot_examples: list[str] = None
    ) -> str:
        """Generates a prompt string.

        Args:
            natrual_instruction: The natural language instruction for the prompt.
            few_shot_examples: A list of few-shot examples. Defaults to None.

        Returns:
            The generated prompt string.
        """
        example_string = " ".join(few_shot_examples) if few_shot_examples else "NA"
        prompt = (
            f"Requirement: {natrual_instruction} \n"
            f"Few-Shot Examples: {example_string} \n"
            "New Example: \n"
            "Label: \n"
            "Please answer me in JSON format."
        )
        return prompt

    def response_mining(self, response: openai.Completion) -> tuple[str, int]:
        """Extracts the generated example and pseudo label from an OpenAI API response.

        Args:
            response (openai.Completion): The response object returned by OpenAI API.

        Returns:
            A tuple of strings (generated_example, pseudo_label), where:
            - generated_example is the generated example string extracted from the
            response, or "" if not found.
            - pseudo_label is the pseudo label int extracted from the response,
            or -1 if not found.
        """
        try:
            response_dict = json.loads(response.choices[0]["message"]["content"])
            keys = response_dict.keys()
            generated_example = None
            pseudo_label = -1
            for key in keys:
                if "comment" in key.lower():
                    generated_example = response_dict[key]
                elif "label" in key.lower():
                    pseudo_label = int(response_dict[key])
            if generated_example and pseudo_label:
                return generated_example, pseudo_label
            else:
                raise ValueError("No examples or pseudo_labels found")
        except (
            json.JSONDecodeError,
            IndexError,
            TypeError,
            ValueError,
            AttributeError,
        ):
            # Catch specific exceptions that you expect to occur
            return "", -1

    def generate_examples(
        self,
        prompt_spec: PromptSpec,
        num_examples: int,
        split: DatasetSplit,
    ) -> Dataset:
        """Generate examples using GPT-3.5.

        Args:
            prompt_spec: A prompt specification.
            num_examples: Number of examples in split.
            split: Name of dataset split to generate.

        Returns:
            A single dataset split with exmaples and pseudo_labels.
        """
        _ = prompt_spec, split  # suppress unused variable warnings
        natrual_instruction = (
            "please give me a movie comment. If  it's If it's positive, "
            "please give a label '1'. Otherwise, give a label '0'."
        )  # Get it from prompt_spec
        few_shot_examples = [
            "This movie is great!",
            "This movie is bad!",
        ]  # Get it from prompt_spec

        prompt = self.generate_prompt(natrual_instruction, few_shot_examples)

        examples = []  # type: list[str]
        pseudo_labels = []  # type: list[int]
        for example_index in tqdm(range(num_examples), desc="Generating examples"):
            while True:
                if self.current_api_call >= self.max_api_call:
                    print("Maximum number of API calls reached.")
                    return Dataset.from_dict(
                        {"input_col": examples, "output_col": pseudo_labels}
                    )
                else:
                    self.current_api_call += 1
                response = self.generate_example(prompt)
                generated_example, pseudo_label = self.response_mining(response)
                if (generated_example != "") and (pseudo_label != -1):
                    examples.append(generated_example)
                    pseudo_labels.append(pseudo_label)
                    break
                else:
                    print(
                        "No examples or pseudo_labels found",
                        f"for {example_index + 1} th example",
                    )

        return Dataset.from_dict({"input_col": examples, "output_col": pseudo_labels})
