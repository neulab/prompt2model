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
            api_key: A valid OpenAI API key.
        """
        openai.api_key = api_key


    def generate_examples(
        self,
        prompt_spec: PromptSpec,
        num_examples: int,
        split: DatasetSplit,
    ) -> datasets.Dataset:
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
        pseudo_labels = []  # type: list[str]
        for _ in tqdm(range(num_examples)):
            message_response = openai.Completion.create(
                engine="text-davinci-003",
                prompt=prompt,
                max_tokens=1024,
                n=1,
                stop=None,
                temperature=0.5,
            )
            comment = message_response.choices[0].text.strip()
            examples.append(comment)
            label_response = openai.Completion.create(
                engine="text-davinci-003",
                prompt=f"Here is a comment: {comment} If\
                    it's postive, please give me '1'. If\
                    it's negtive, please give me '0'.",
                max_tokens=1024,
                n=1,
                stop=None,
                temperature=0.5,
            )
            pseudo_label = label_response.choices[0].text.strip()
            pseudo_labels.append(pseudo_label)

        df = pd.DataFrame.from_dict(
            {"input_col": examples, "output_col": pseudo_labels}
        )

        return datasets.Dataset.from_pandas(df)
