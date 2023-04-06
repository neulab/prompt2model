"""
An interface for dataset generators.
Users input a system description (optionally with few-shot examples), a model configuration (OpenAI API or other Models API),
and how many examples to generate for each split (train, val, test), and a random seed for reproducibility, and their output directory.
We will use the OpenAI API to generate the dataset.

Input:
   1) system_description: A system description (optionally with few-shot examples)
   2) model_config: Model configuration (OpenAI API or other Models), currently I will only use OpenAI API as an example.
   3) few_shot_examples: Some few-shot examples (e.g. "example 1." and "example 2.")
   4) output_dir: if user pass in this parameter, then the output DatsetDict will be store here.
   5) num_train_examples, num_val_examples, num_test_examples: How many examples to generate for each split (train, val, test)
   6) random_seed: A random seed for API usage

Output:
   1) A DatasetDict including training, validation, and testing datasets.
"""

import datasets
from typing import Optional, List
import openai


class DatasetGenerator:
    """
    A class for generating datasets from a prompt specification.
    """

    def __init__(
        self,
        system_description: str,
        model_config: dict,
        few_shot_examples: Optional[List[str]] = None,
        output_dir: Optional[str] = None,
        num_train_examples: Optional[int] = 5000,
        num_val_examples: Optional[int] = 1500,
        num_test_examples: Optional[int] = 500,
        random_seed: Optional[int] = 42,
    ):
        """Construct a dataset generator."""
        self.system_description = system_description
        self.model_config = model_config
        self.few_shot_examples = few_shot_examples
        self.output_dir = output_dir
        self.num_train_examples = num_train_examples
        self.num_val_examples = num_val_examples
        self.num_test_examples = num_test_examples
        self.random_seed = random_seed

    @staticmethod
    def post_process(response: str):
        """
        Post-process the response from the API
        """
        start_quote = response.find('"')
        end_quote = response.find('"', start_quote + 1)
        content = response[start_quote + 1 : end_quote]
        return content

    def generate_examples(self, split: str = "train") -> datasets.Dataset:
        """
        Generate examples for a given dataset split (train, validation, or test).

        Args:
            split (str): The dataset split to generate examples for. Must be one of "train", "val", or "test". Default is "train".

        Returns:
            datasets.Dataset: The generated dataset.
        """
        assert split in ["train", "val", "test"], f"Invalid dataset split: {split}"

        num_examples = {
            "train": self.num_train_examples,
            "val": self.num_val_examples,
            "test": self.num_test_examples,
        }[split]

        openai.api_key = "YOUR_API_KEY"

        prompt = self.system_description
        if self.few_shot_examples:
            prompt += "\nFew-Shot Examples:" + "\n".join(self.few_shot_examples)

        examples = []
        for _ in range(num_examples):
            response = (
                openai.Completion.create(
                    engine=self.model_config["engine"],
                    prompt=prompt,
                    max_tokens=self.model_config["max_tokens"],
                    n=self.model_config["n"],
                    stop=self.model_config["stop"],
                    temperature=0.5,
                    random_seed=self.random_seed
                    + {"train": 0, "val": 1, "test": 2}[split],
                )
                .choices[0]
                .text.strip()
            )
        examples.append(response)

        examples = [self.post_process(ex) for ex in examples]

        dataset = datasets.Dataset.from_dict({"text": examples})
        return dataset

    def generate_datasets(self) -> datasets.DatasetDict:
        """
        Generate training/validation/testing datasets from a prompt (which may
        include a few demonstration examples). Use a Large Language Model to generate
        a large number of examples.

        Returns:
            datasets.DatasetDict: A DatasetDict including training, validation, and testing datasets.
        """
        train_examples = self.generate_examples(split="train")
        val_examples = self.generate_examples(split="val")
        test_examples = self.generate_examples(split="test")

        dataset_dict = datasets.DatasetDict(
            {
                "train": train_examples,
                "validation": val_examples,
                "test": test_examples,
            }
        )

        if self.output_dir:
            dataset_dict.save_to_disk(self.output_dir)

        return dataset_dict
