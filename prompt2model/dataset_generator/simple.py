"""A simple dataset generator that uses OpenAI's GPT-3.5 API."""

import openai
from datasets import ClassLabel, Dataset, DatasetInfo
from tqdm import tqdm

from prompt2model.dataset_generator.base import DatasetGenerator, DatasetSplit
from prompt2model.prompt_parser import PromptSpec


class SimpleDatasetGenerator(DatasetGenerator):
    """A simple dataset generator that uses OpenAI's GPT-3.5 API."""

    def __init__(self, api_key: str):
        """Initialize an OpenAI API client with the provided API key.

        Args:
            api_key: A valid OpenAI API key.
        """
        openai.api_key = api_key

    def generate_example(self, prompt: str) -> openai.Completion:
        """Generate an exmaple and its pseudo_label using OpenAI's GPT-3 API.

        Args:
            prompt: A prompt asking for an example and its pseudo_label.

        Returns:
            A response object containing a generated example and its pseudo_label.
        """
        response = openai.Completion.create(
            engine="text-davinci-002",
            prompt=prompt,
            max_tokens=1024,
            n=1,
            stop=None,
            temperature=0.5,
        )
        return response

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
        prompt = "please give me some movie comments"

        examples = []  # type: list[str]
        pseudo_labels = []  # type: list[int]
        for _ in tqdm(range(num_examples)):
            response = self.generate_example(prompt)
            comment = response.choices[0].text.strip()
            pseudo_label = int(response.choices[1].text.strip())
            examples.append(comment)
            pseudo_labels.append(pseudo_label)

        dataset_info = DatasetInfo(
            features={
                "input_col": {
                    "description": "Generated movie comments",
                    "type": "string",
                },
                "label": {
                    "description": "Label of the movie comment\
                        (0: negative, 1: positive)",
                    "type": ClassLabel(names=["negative", "positive"]),
                },
            },
            split={
                split: {
                    "num_examples": num_examples,
                    "description": f"{split} split generated using OpenAI's GPT-3 API",
                },
            },
        )

        return Dataset.from_dict(
            {"input_col": examples, "output_col": pseudo_labels},
            dataset_info=dataset_info,
        )
