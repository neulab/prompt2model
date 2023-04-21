"""A simple dataset generator that uses OpenAI's GPT-3.5 API."""

import datasets
import openai
import pandas as pd
from tqdm import tqdm

from prompt2model.dataset_generator.base import DatasetGenerator, DatasetSplit
from prompt2model.prompt_parser import PromptSpec


class SimpleDatasetGenerator(DatasetGenerator):
    """A simple dataset generator that uses OpenAI's GPT-3.5 API."""

    def __init__(self, api_key: str):
        """Initialize an OpenAI API client with the provided API key.

        Args:
            api_key (str): A valid OpenAI API key.

        Returns:
            None
        """
        openai.api_key = api_key

    def generate_examples(
        self,
        prompt_spec: PromptSpec,
        num_examples: int,
        split: DatasetSplit,
    ) -> datasets.Dataset:
        """Generate movie comments using GPT-3.5.

        Args:
            prompt_spec: A prompt specification.
            num_examples: Number of examples in split.
            split: Name of dataset split to generate.

        Returns:
            A single dataset split.
        """
        _ = prompt_spec, split  # suppress unused variable warnings
        prompt = "please give me some movie comments"

        examples = []  # type: list[str]
        for _ in tqdm(range(num_examples)):
            response = openai.Completion.create(
                engine="davinci",
                prompt=prompt,
                max_tokens=1024,
                n=1,
                stop=None,
                temperature=0.5,
            )

            message = response.choices[0].text.strip()
            if message:
                examples.append(message)

        df = pd.DataFrame.from_dict(
            {"input_col": [prompt] * num_examples, "output_col": examples}
        )

        return datasets.Dataset.from_pandas(df)
