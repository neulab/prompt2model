"""An dual-encoder dataset retriever using HuggingFace dataset descriptions."""

from __future__ import annotations  # noqa FI58

import json
import os
import urllib.request

import datasets
import numpy as np
import torch

from prompt2model.dataset_retriever.base import DatasetRetriever
from prompt2model.prompt_parser import PromptSpec
from prompt2model.utils import encode_text, retrieve_objects

datasets.utils.logging.disable_progress_bar()


class DatasetInfo:
    """Store the dataset name, description, and query-dataset score for each dataset."""

    def __init__(
        self,
        name: str,
        description: str,
        score: float,
    ):
        """Initialize a DatasetInfo object.

        Args:
            name: The name of the dataset.
            description: The description of the dataset.
            score: The similarity of the dataset to a given prompt from a user.
        """
        self.name = name
        self.description = description
        self.score = score


def input_string():
    """Read a string from stdin."""
    description = str(input())
    return description


def input_y_n() -> bool:
    """Get a yes/no answer from the user via stdin."""
    y_n = str(input())
    return not (y_n.strip() == "" or y_n.strip().lower() == "n")


class DescriptionDatasetRetriever(DatasetRetriever):
    """Retrieve a dataset from HuggingFace, based on similarity to the prompt."""

    def __init__(
        self,
        search_index_path: str = "huggingface_data/huggingface_datasets/"
        + "huggingface_datasets_datafinder_index",
        first_stage_search_depth: int = 1000,
        max_search_depth: int = 25,
        encoder_model_name: str = "viswavi/datafinder-huggingface-prompt-queries",
        dataset_info_file: str = "huggingface_data/huggingface_datasets/"
        + "dataset_index.json",
        device: torch.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        ),
    ):
        """Initialize a dual-encoder retriever against a search index.

        Args:
            search_index_path: Where to store the search index (e.g. encoded vectors).
            first_stage_search_depth: The # of datasets to retrieve before filtering.
            max_search_depth: The number of most-relevant datasets to retrieve.
            encoder_model_name: The name of the model to use for the dual-encoder.
            dataset_info_file: The file containing dataset names and descriptions.
            device: The device to use for encoding text for our dual-encoder model.
        """
        self.search_index_path = search_index_path
        self.first_stage_search_depth = first_stage_search_depth
        self.max_search_depth = max_search_depth
        self.encoder_model_name = encoder_model_name
        self.dataset_infos: list[DatasetInfo] = []
        if not os.path.exists(dataset_info_file):
            # Download the dataset search index if one is not on disk already.
            urllib.request.urlretrieve(
                "http://phontron.com/data/prompt2model/dataset_index.json",
                dataset_info_file,
            )
        self.full_dataset_metadata = json.load(open(dataset_info_file, "r"))
        for dataset_name in sorted(self.full_dataset_metadata.keys()):
            self.dataset_infos.append(
                DatasetInfo(
                    name=dataset_name,
                    description=self.full_dataset_metadata[dataset_name]["description"],
                    score=0.0,
                )
            )
        self.device = device

        assert not os.path.isdir(
            search_index_path
        ), f"Search index must either be a valid file or not exist yet. But {search_index_path} is provided."  # noqa E501

    def encode_dataset_descriptions(
        self, dataset_infos, search_index_path
    ) -> np.ndarray:
        """Encode dataset descriptions into a vector for indexing."""
        dataset_descriptions = [
            dataset_info.description for dataset_info in dataset_infos
        ]
        dataset_vectors = encode_text(
            self.encoder_model_name,
            text_to_encode=dataset_descriptions,
            encoding_file=search_index_path,
            device=self.device,
        )
        return dataset_vectors

    @staticmethod
    def print_divider():
        """Utility function to assist with the retriever's command line interface."""
        print("\n-------------------------------------------------\n")

    def choose_dataset(self, top_datasets: list[DatasetInfo]) -> str | None:
        """Have the user choose an appropriate dataset from a list of top datasets."""
        self.print_divider()
        print("Here are the datasets I've retrieved for you:")
        print("#\tName\tDescription")
        for i, d in enumerate(top_datasets):
            description_no_spaces = d.description.replace("\n", " ")
            print(f"{i+1}):\t{d.name}\t{description_no_spaces}")

        self.print_divider()
        print(
            "If none of these are relevant to your prompt, we'll only use "
            + "generated data. Are any of these datasets relevant? (y/N)"
        )
        any_datasets_relevant = input_y_n()
        if any_datasets_relevant:
            print(
                "Which dataset would you like to use? Give the number between "
                + f"1 and {len(top_datasets)}."
            )
            dataset_idx = int(input())
            chosen_dataset_name = top_datasets[dataset_idx - 1].name
        else:
            chosen_dataset_name = None
        self.print_divider()  # noqa E501
        return chosen_dataset_name

    @staticmethod
    def canonicalize_dataset_using_columns_for_split(
        dataset_split: datasets.Dataset,
        input_columns: list[str],
        output_column: str,
    ) -> datasets.DatasetDict:
        """Canonicalize a single dataset split into a suitable text-to-text format."""
        input_col = []
        output_col = []
        for i in range(len(dataset_split)):
            input_string = ""
            for col in input_columns:
                input_string += f"{col}: {dataset_split[i][col]}\n"
            input_string = input_string.strip()
            input_col.append(input_string)
            output_col.append(dataset_split[i][output_column])
        return datasets.Dataset.from_dict(
            {"input_col": input_col, "output_col": output_col}
        )

    def canonicalize_dataset_using_columns(
        self,
        dataset: datasets.DatasetDict,
        input_columns: list[str],
        output_columns: str,
    ) -> datasets.DatasetDict:
        """Canonicalize a dataset into a suitable text-to-text format."""
        dataset_dict = {}
        for split in dataset:
            dataset_dict[split] = self.canonicalize_dataset_using_columns_for_split(
                dataset[split], input_columns, output_columns
            )
        return datasets.DatasetDict(dataset_dict)

    def canonicalize_dataset(self, dataset_name: str) -> datasets.DatasetDict:
        """Canonicalize a dataset into a suitable text-to-text format."""
        configs = datasets.get_dataset_config_names(dataset_name)
        chosen_config = None
        if len(configs) == 1:
            chosen_config = configs[0]
        else:
            self.print_divider()
            print(f"Multiple dataset configs available: {configs}")
            while chosen_config is None:
                print("Which dataset config would you like to use for this?")
                user_response = input_string()
                if user_response in configs:
                    chosen_config = user_response
                else:
                    print(
                        f"Invalid config provided: {user_response}. Please choose "
                        + "from {configs}\n\n"
                    )
            self.print_divider()

        dataset = datasets.load_dataset(dataset_name, chosen_config)
        assert "train" in dataset
        train_columns = dataset["train"].column_names
        train_columns_formatted = ", ".join(train_columns)

        assert len(dataset["train"]) > 0
        example_rows = json.dumps(dataset["train"][0], indent=4)

        self.print_divider()
        print(f"Loaded dataset. Example row:\n{example_rows}\n")

        print(
            "Which column(s) should we use as input? Provide a comma-separated "
            + f"list from: {train_columns_formatted}."
        )
        user_response = input_string()
        input_columns = [c.strip() for c in user_response.split(",")]
        print(f"Will use the columns {json.dumps(input_columns)} as input.\n")

        output_column = None
        while output_column is None:
            print(
                "Which column(s) should we use as the target? Choose a single "
                + f"value from: {train_columns_formatted}."
            )
            user_response = input_string()
            if user_response in train_columns:
                output_column = user_response
            else:
                print(
                    "Invalid column provided: {user_response}. Please choose "
                    + f"from {train_columns}\n\n"
                )
        print(f'Will use the column "{output_column}" as our target.\n')
        self.print_divider()

        canonicalized_dataset = self.canonicalize_dataset_using_columns(
            dataset, input_columns, output_column
        )
        return canonicalized_dataset

    def retrieve_dataset_dict(
        self,
        prompt_spec: PromptSpec,
    ) -> datasets.DatasetDict | None:
        """Select a dataset from a prompt using a dual-encoder retriever.

        Args:
            prompt_spec: A prompt whose instruction field we use to retrieve datasets.

        Return:
            A list of relevant datasets dictionaries.
        """
        if not os.path.exists(self.search_index_path):
            print("Creating dataset descriptions")
            self.encode_dataset_descriptions(self.dataset_infos, self.search_index_path)

        query_text = prompt_spec.instruction

        query_vector = encode_text(
            self.encoder_model_name,
            text_to_encode=query_text,
            device=self.device,
        )
        dataset_names = [dataset_info.name for dataset_info in self.dataset_infos]
        ranked_list = retrieve_objects(
            query_vector,
            self.search_index_path,
            dataset_names,
            self.first_stage_search_depth,
        )
        top_dataset_infos = []
        dataset_name_to_dataset_idx = {
            d.name: i for i, d in enumerate(self.dataset_infos)
        }
        for dataset_name, dataset_score in ranked_list:
            dataset_idx = dataset_name_to_dataset_idx[dataset_name]
            self.dataset_infos[dataset_idx].score = dataset_score
            top_dataset_infos.append(self.dataset_infos[dataset_idx])

        ranked_list = sorted(top_dataset_infos, key=lambda x: x.score, reverse=True)[
            : self.max_search_depth
        ]
        assert len(ranked_list) > 0, "No datasets retrieved from search index."
        top_dataset_name = self.choose_dataset(ranked_list)
        if top_dataset_name is None:
            return None
        return self.canonicalize_dataset(top_dataset_name)
