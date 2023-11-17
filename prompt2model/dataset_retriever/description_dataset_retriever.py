"""An dual-encoder dataset retriever using HuggingFace dataset descriptions."""

from __future__ import annotations  # noqa FI58

import json
import logging
import os
import random
import urllib.request
from collections.abc import MutableMapping

import datasets
import torch

from prompt2model.dataset_retriever.base import DatasetInfo, DatasetRetriever
from prompt2model.dataset_retriever.column_selection_prompt import (
    construct_prompt_for_column_selection,
)
from prompt2model.dataset_retriever.reranking_prompt import (
    build_input,
    construct_prompt_for_dataset_reranking,
)

# FIXME
from prompt2model.dataset_retriever.retrieve_dataset_mp import (
    fetch_first_row_with_timeout,
    get_dataset_validity,
)
from prompt2model.prompt_parser import PromptSpec
from prompt2model.utils import encode_text, retrieve_objects
from prompt2model.utils.dataset_utils import get_dataset_size
from prompt2model.utils.parse_responses import parse_prompt_to_fields

datasets.utils.logging.disable_progress_bar()
logger = logging.getLogger(__name__)


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
        self.device = device
        self.dataset_info_file = dataset_info_file
        self.initialize_search_index()

    def initialize_search_index(self) -> None:
        """Initialize the search index."""
        self.dataset_infos: list[DatasetInfo] = []
        if not os.path.exists(self.dataset_info_file):
            # Download the dataset search index if one is not on disk already.
            logger.info("Downloading the dataset search index")
            os.makedirs(os.path.dirname(self.dataset_info_file), exist_ok=True)
            urllib.request.urlretrieve(
                "http://phontron.com/data/prompt2model/dataset_index.json",
                self.dataset_info_file,
            )
        self.full_dataset_metadata = json.load(open(self.dataset_info_file, "r"))
        for dataset_name in sorted(self.full_dataset_metadata.keys()):
            self.dataset_infos.append(
                DatasetInfo(
                    name=dataset_name,
                    description=self.full_dataset_metadata[dataset_name]["description"],
                    score=0.0,
                )
            )
        if os.path.isdir(self.search_index_path):
            raise ValueError(
                "Search index must either be a valid file or not exist yet. "
                "But {self.search_index_path} is provided."
            )
        if not os.path.exists(self.search_index_path):
            logger.info("Creating dataset descriptions")
            encode_text(
                self.encoder_model_name,
                text_to_encode=[x.description for x in self.dataset_infos],
                encoding_file=self.search_index_path,
                device=self.device,
            )

    # ---------------------------- Utility Functions ----------------------------
    @staticmethod
    def _input_string():
        """Utility function to read a string from stdin."""
        description = str(input())
        return description

    @staticmethod
    def _input_y_n() -> bool:
        """Utility function to get a yes/no answer from the user via stdin."""
        y_n = str(input())
        return not (y_n.strip() == "" or y_n.strip().lower() == "n")

    @staticmethod
    def _print_divider():
        """Utility function to assist with the retriever's command line interface."""
        print("\n-------------------------------------------------\n")

    def flatten_dict(
        self, d: MutableMapping, parent_key: str = "", sep: str = "."
    ) -> MutableMapping:
        items = []
        for k, v in d.items():
            new_key = parent_key + sep + k if parent_key else k
            if isinstance(v, MutableMapping):
                items.extend(self.flatten_dict(v, new_key, sep=sep).items())
            else:
                items.append((new_key, v))
        return dict(items)

    def replace_duplicate_columns(self, original_dataset_columns):
        columns_mapping: dict[str, str] = {}
        new_columns = []
        counter: dict[str, int] = {}
        # convert flattened columns like answer.text -> answer_text
        for col in original_dataset_columns:
            new_col = col.replace(".", "_")
            if new_col in columns_mapping.values():
                counter[new_col] = counter.get(new_col, 0) + 1
                new_col = f"{new_col}_{counter[new_col]}"
            columns_mapping[col] = new_col
            new_columns.append(new_col)
        return new_columns, columns_mapping

    def choose_dataset_by_cli(self, top_datasets: list[DatasetInfo]) -> str | None:
        """Have the user choose an appropriate dataset from a list of top datasets.

        Args:
            top_datasets: A list of top datasets to choose from.

        Returns:
            The name of the chosen dataset, or None if no dataset is chosen as relevant.
        """
        self._print_divider()
        print("Here are the datasets I've retrieved for you:")
        print("#\tName\tSize[MB]\tDescription")
        for i, d in enumerate(top_datasets):
            description_no_space = d.description.replace("\n", " ")
            print(
                f"{i+1}):\t{d.name}\t{get_dataset_size(d.name)}\t{description_no_space}"
            )

        self._print_divider()
        print(
            "If none of these are relevant to your prompt, we'll only use "
            + "generated data. Are any of these datasets relevant? (y/N)"
        )
        any_datasets_relevant = self._input_y_n()
        if any_datasets_relevant:
            print(
                "Which dataset would you like to use? Give the number between "
                + f"1 and {len(top_datasets)}."
            )
            dataset_idx = int(input())
            chosen_dataset_name = top_datasets[dataset_idx - 1].name
        else:
            chosen_dataset_name = None
        self._print_divider()
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
            curr_string = ""
            for col in input_columns:
                curr_string += f"{col}: {dataset_split[i][col]}\n"
            curr_string = curr_string.strip()
            input_col.append(curr_string)
            output_col.append(dataset_split[i][output_column])
        return datasets.Dataset.from_dict(
            {"input_col": input_col, "output_col": output_col}
        )

    def get_configs_info(self, dataset_name):
        config_names = datasets.get_dataset_config_names(dataset_name)
        if len(config_names) > 5:
            config_names = random.sample(
                config_names, 5
            )  # Loading more configs would take really long

        all_config_infos = []
        for config_name in config_names:

            if "train" not in datasets.get_dataset_split_names(
                dataset_name, config_name
            ):
                continue
            dataset = datasets.load_dataset(
                dataset_name, config_name, split="train", streaming=True
            )
            sample_rows = fetch_first_row_with_timeout(dataset)
            if sample_rows is None:
                continue
            sample_rows = self.flatten_dict(sample_rows)
            if any(
                sample_rows[key].__class__.__name__ == "PngImageFile"
                for key in sample_rows
            ):
                continue  # We dont want to handle image datasets.
            # FUTURE TODO: This can be put into flatten_dict if required
            columns, columns_mapping = self.replace_duplicate_columns(
                sample_rows.keys()
            )

            columns = ", ".join(columns)
            all_config_infos.append(
                {
                    "config_name": config_name,
                    "sample_row": sample_rows,
                    "columns": columns,
                    "columns_mapping": columns_mapping,
                    "dataset_description": dataset.info.description,
                    "dataset_name": dataset_name,
                }
            )
            print("compelted config: ", config_name)
            del dataset

        return all_config_infos

    def get_dataset_info(self, dataset_name):
        print("Now doing..", dataset_name)
        try:
            if not get_dataset_validity(dataset_name):
                return None
            configs = self.get_configs_info(dataset_name)
            if len(configs) == 0:
                return None
            dataset_information = {
                "dataset_name": dataset_name,
                "configs": configs,
                "dataset_description": configs[0][
                    "dataset_description"
                ],  # Configs dont have different descriptions from the original dataset, even for datasets like glue
            }
        except Exception as e:
            print(f"Error processing {dataset_name}: {e}")
            return None
        return dataset_information

    def automatic_column_selection(
        self,
        instruction: str,
        dataset_info: str,
    ) -> tuple[list[str], str]:
        """Find appropriate input and output columns for a given dataset and tasks."""
        prompt = construct_prompt_for_column_selection(
            instruction,
            dataset_info["dataset_name"],
            dataset_info["dataset_description"],
            dataset_info["columns"],
            dataset_info["sample_row"],
        )
        required_keys = ["input", "output"]
        optional_keys = ["ambiguous", "irrelevant"]

        response = parse_prompt_to_fields(prompt, required_keys, optional_keys)
        input_columns = response["input"]
        output_column = response["output"]
        if len(input_columns) < 1 or len(output_column) != 1:
            raise RuntimeError(
                "Input columns length was less than 1 or output column length was not 1"
            )

        dataset_columns = dataset_info["columns"]
        incorrect_columns = [
            col for col in input_columns + output_column if col not in dataset_columns
        ]
        if len(incorrect_columns) > 0:
            raise RuntimeError(
                f"One or more columns ({incorrect_columns}) were output that were "
                f"not in the list of columns in the dataset ({dataset_columns})."
            )

        return input_columns, output_column[0]

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

    def canonicalize_dataset_by_cli(
        self, dataset_name: str, prompt_spec
    ) -> datasets.DatasetDict:
        """Canonicalize a dataset into a suitable text-to-text format.

        Args:
            dataset_name: The name of the dataset to canonicalize.

        Returns:
            A canonicalized dataset.
        """
        configs = datasets.get_dataset_config_names(dataset_name)
        chosen_config = None
        if len(configs) == 1:
            chosen_config = configs[0]
        else:
            self._print_divider()
            print(f"Multiple dataset configs available: {configs}")
            while chosen_config is None:
                print("Which dataset config would you like to use for this?")
                user_response = self._input_string()
                if user_response in configs:
                    chosen_config = user_response
                else:
                    print(
                        f"Invalid config provided: {user_response}. Please choose "
                        + "from {configs}\n\n"
                    )
            self._print_divider()

        dataset = datasets.load_dataset(dataset_name, chosen_config).flatten()

        if "train" not in dataset:
            raise ValueError("The dataset must contain a `train` split.")
        columns_mapping: dict[str, str] = {}
        counter: dict[str, int] = {}
        # convert flattened columns like answer.text -> answer_text
        for col in dataset["train"].column_names:
            new_col = col.replace(".", "_")
            if new_col in columns_mapping.values():
                counter[new_col] = counter.get(new_col, 0) + 1
                new_col = f"{new_col}_{counter[new_col]}"
            columns_mapping[col] = new_col
        dataset = dataset.rename_columns(columns_mapping)

        train_columns = dataset["train"].column_names
        train_columns_formatted = ", ".join(train_columns)
        dataset_description = dataset["train"].info.description

        if len(dataset["train"]) == 0:
            raise ValueError("train split is empty.")
        example_rows = json.dumps(dataset["train"][0], indent=4)

        self._print_divider()
        print(f"Loaded dataset. Example row:\n{example_rows}\n")

        try:
            input_columns, output_column = self.automatic_column_selection(
                prompt_spec.instruction,
                dataset_name,
                dataset_description,
                train_columns_formatted,
                dataset["train"][0],
            )
        except RuntimeError:
            logger.error(f"{dataset_name} did not work. Try another!")
            return None  # Returning None means that the dataset chosen didn't work,
            # and we would rather generate a dataset.

        print(f"Will use the columns {json.dumps(input_columns)} as input.\n")
        print(f'Will use the column "{output_column}" as our target.\n')
        self._print_divider()

        canonicalized_dataset = self.canonicalize_dataset_using_columns(
            dataset, input_columns, output_column
        )
        return canonicalized_dataset

    def retrieve_top_datasets(
        self,
        prompt_spec: PromptSpec,
    ) -> list[DatasetInfo]:
        """Retrieve the top datasets for a prompt.

        Specifically, the datasets are scored using a dual-encoder retriever model
        and the datasets with the highest similarity scores with the query are returned.

        Args:
            prompt_spec: A prompt whose instruction field we use to retrieve datasets.

        Returns:
            A list of the top datasets for the prompt according to retriever score.
        """
        query_vector = encode_text(
            self.encoder_model_name,
            text_to_encode=prompt_spec.instruction,
            device=self.device,
        )
        ranked_list = retrieve_objects(
            query_vector,
            self.search_index_path,
            [x.name for x in self.dataset_infos],
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

        sorted_list = sorted(top_dataset_infos, key=lambda x: x.score, reverse=True)[
            : self.max_search_depth
        ]
        if len(sorted_list) == 0:
            raise ValueError("No datasets retrieved from search index.")
        return sorted_list

    def dataset_reranking(self, dataset_list, prompt_spec):
        dataset_info_list = []
        import time

        start_time = time.time()
        for dataset_name in dataset_list:
            info = self.get_dataset_info(dataset_name.name)
            if info is None:
                continue
            dataset_info_list.append(info)
        end_time = time.time()
        total_time = end_time - start_time
        print("Time taken to retrieve datastet retriever is:", total_time, "seconds")
        if len(dataset_info_list) == 0:
            return None  # All datasets private/took too long to be retrieved

        prompt = construct_prompt_for_dataset_reranking(
            prompt_spec.instruction, prompt_spec.examples, dataset_info_list
        )
        print(prompt)
        # TODO: Is this an overkill?
        try:
            (
                dataset_index,
                dataset_name,
                config_index,
                config_name,
                confidence_level,
            ) = parse_prompt_to_fields(prompt=prompt, response_type="rerank")

            print()
            print(
                "Rernaking results: ",
                dataset_index,
                dataset_name,
                config_index,
                config_name,
                confidence_level,
            )
            if (
                dataset_index == -1
                or dataset_info_list[dataset_index - 1]["dataset_name"] != dataset_name
                or dataset_info_list[dataset_index - 1]["configs"][config_index - 1][
                    "config_name"
                ]
                != config_name
                or confidence_level == "low"
            ):
                return None  # None of the datasets are relevant or there is hallucination or reranker is not confident

            return (
                dataset_info_list[dataset_index - 1]["configs"][config_index - 1],
                prompt,
            )
        except Exception as e:
            print(
                "dataset reranker failed probably because of output being in incorrect format."
            )
            return None, prompt

    def canocalize_dataset_automatically(self, top_dataset_info, task_instruction):
        if top_dataset_info is None:
            return None
        try:
            input_columns, output_column = self.automatic_column_selection(
                task_instruction, top_dataset_info
            )
        except Exception as e:
            print("Column selection failed: ", e)
            return None
        full_dataset = datasets.load_dataset(
            top_dataset_info["dataset_name"], top_dataset_info["config_name"]
        ).flatten()
        full_dataset = full_dataset.rename_columns(top_dataset_info["columns_mapping"])

        canonicalized_dataset = self.canonicalize_dataset_using_columns(
            full_dataset, input_columns, output_column
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
        sorted_list = self.retrieve_top_datasets(prompt_spec)

        top_dataset_info, reranking_prompt = self.dataset_reranking(
            sorted_list, prompt_spec
        )

        dataset_name, config_name = None, None
        if top_dataset_info:
            dataset_name, config_name = (
                top_dataset_info["dataset_name"],
                top_dataset_info["config_name"],
            )

        return (
            self.canocalize_dataset_automatically(
                top_dataset_info, prompt_spec.instruction
            ),
            reranking_prompt,
            dataset_name,
            config_name,
        )
