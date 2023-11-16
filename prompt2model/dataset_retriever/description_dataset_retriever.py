"""An dual-encoder dataset retriever using HuggingFace dataset descriptions."""

from __future__ import annotations  # noqa FI58

import json
import logging
import os
import urllib.request

import datasets
import torch

from prompt2model.dataset_retriever.base import DatasetInfo, DatasetRetriever
from prompt2model.dataset_retriever.column_selection_prompt import (
    construct_prompt_for_column_selection,
)
from prompt2model.dataset_retriever.data_transform_prompt import (
    construct_prompt_for_plan,
    construct_prompt_for_transform_data,
)
from prompt2model.prompt_parser import PromptSpec
from prompt2model.utils import encode_text, retrieve_objects
from prompt2model.utils.dataset_utils import get_dataset_size
from prompt2model.utils.parse_json_responses import (
    make_request_from_prompt,
    parse_prompt_to_fields,
)


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

    @staticmethod
    def automatic_column_selection(
        instruction: str,
        dataset_name: str,
        dataset_description: str,
        dataset_columns: str,
        example_rows: dict,
    ) -> tuple[list[str], str]:
        """Find appropriate input and output columns for a given dataset and tasks."""
        prompt = construct_prompt_for_column_selection(
            instruction,
            dataset_name,
            dataset_description,
            dataset_columns,
            example_rows,
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

        incorrect_columns = [
            col for col in input_columns + output_column if col not in dataset_columns
        ]
        if len(incorrect_columns) > 0:
            raise RuntimeError(
                f"One or more columns ({incorrect_columns}) were output that were "
                f"not in the list of columns in the dataset ({dataset_columns})."
            )

        return input_columns, output_column[0]

    def canonicalize_dataset_using_samples(
        self,
        inputs: list[str],
        outputs: list[str],
    ) -> datasets.DatasetDict:
        """Canonicalize a dataset into a suitable text-to-text format."""
        dataset_dict = {}
        dataset_dict["train"] = datasets.Dataset.from_dict(
            {"input_col": inputs, "output_col": outputs}
        )
        return datasets.DatasetDict(dataset_dict)

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

    def data_transform(
        self, prompt_spec: PromptSpec, dataset: datasets.Dataset, num_transform: int
    ) -> tuple[list[str], list[str]]:
        """Transform the dataset into the required format according to the prompt_spec.
        1. Use the prompt_spec and an example row from the dataset to create a "plan" for the data transformation.
        2. Use the prompt_spec and the plan to transform each row of the dataset into the required format.
        3. Return the transformed inputs and outputs.
        """
        # 1. Use the prompt_spec and an example row from the dataset to create a "plan" for the data transformation.
        plan_prompt = construct_prompt_for_plan(
            task_description=prompt_spec.instruction,
            dataset_row=dataset[0],
            example=prompt_spec.examples,
        )
        plan = make_request_from_prompt(plan_prompt)

        print(f"plan_prompt: {plan_prompt}")

        print(f"plan: {plan}")

        # 2. Use the prompt_spec and the plan to transform each row of the dataset into the required format.
        inputs = []
        outputs = []

        required_keys = ["input", "output"]

        max_len = min(num_transform, len(dataset))
        len_count = 0
        for row in dataset:
            print(f"row: {row}")
            transform_prompt = construct_prompt_for_transform_data(
                task_description=prompt_spec.instruction,
                dataset_row=row,
                example=prompt_spec.examples,
                plan=plan,
            )
            response = parse_prompt_to_fields(transform_prompt, required_keys, [])
            inputs.append(str(response["input"]))
            outputs.append(str(response["output"]))
            print(f"transformed_input: {response['input']}")
            print(f"transformed_output: {response['output']}")
            len_count += 1
            if len_count >= max_len:
                break

        # 3. Return the transformed inputs and outputs.
        return inputs, outputs

    def canonicalize_dataset_by_cli_data_transform(
        self, dataset_name: str, prompt_spec, num_transform: int = 10
    ) -> datasets.DatasetDict:
        """Canonicalize a dataset into a suitable text-to-text format.

        Args:
            dataset_name: The name of the dataset to canonicalize.

        Returns:
            A canonicalized dataset.
        """
        configs = datasets.get_dataset_config_names(dataset_name)
        chosen_config = configs[0]
        # chosen_config = None
        # if len(configs) == 1:
        #     chosen_config = configs[0]
        # else:
        #     self._print_divider()
        #     print(f"Multiple dataset configs available: {configs}")
        #     while chosen_config is None:
        #         print("Which dataset config would you like to use for this?")
        #         user_response = self._input_string()
        #         if user_response in configs:
        #             chosen_config = user_response
        #         else:
        #             print(
        #                 f"Invalid config provided: {user_response}. Please choose "
        #                 + "from {configs}\n\n"
        #             )
        #     self._print_divider()

        dataset = datasets.load_dataset(dataset_name, chosen_config).flatten()

        if "train" not in dataset:
            # raise ValueError("The dataset must contain a `train` split.")
            logger.error(f"{dataset_name} must contain a `train` split.")
            return None

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
            # raise ValueError("train split is empty.")
            logger.error("train split is empty.")
            return None

        example_rows = json.dumps(dataset["train"][0], indent=4)

        self._print_divider()
        print(f"Loaded dataset. Example row:\n{example_rows}\n")

        try:
            inputs, outputs = self.data_transform(
                prompt_spec=prompt_spec,
                dataset=dataset["train"],
                num_transform=num_transform,
            )
        except RuntimeError:
            logger.error(f"{dataset_name} did not work. Try another!")
            return None  # Returning None means that the dataset chosen didn't work,
            # and we would rather generate a dataset.

        print(f"Data transformation completed\n")
        self._print_divider()

        canonicalized_dataset = self.canonicalize_dataset_using_samples(inputs, outputs)
        return canonicalized_dataset

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
        chosen_config = configs[0]
        # chosen_config = None
        # if len(configs) == 1:
        #     chosen_config = configs[0]
        # else:
        #     self._print_divider()
        #     print(f"Multiple dataset configs available: {configs}")
        #     while chosen_config is None:
        #         print("Which dataset config would you like to use for this?")
        #         user_response = self._input_string()
        #         if user_response in configs:
        #             chosen_config = user_response
        #         else:
        #             print(
        #                 f"Invalid config provided: {user_response}. Please choose "
        #                 + "from {configs}\n\n"
        #             )
        #     self._print_divider()

        dataset = datasets.load_dataset(dataset_name, chosen_config).flatten()

        if "train" not in dataset:
            # raise ValueError("The dataset must contain a `train` split.")
            logger.error(f"{dataset_name} must contain a `train` split.")
            return None

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
            # raise ValueError("train split is empty.")
            logger.error("train split is empty.")
            return None

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
        top_dataset_name = self.choose_dataset_by_cli(sorted_list)
        if top_dataset_name is None:
            return None
        return self.canonicalize_dataset_by_cli(top_dataset_name, prompt_spec)


if __name__ == "__main__":
    from prompt2model.utils import api_tools
    from prompt2model.prompt_parser import MockPromptSpec
    api_tools.default_api_agent = api_tools.APIAgent(model_name="azure/vijay-gpt-4", max_tokens=8000)
    retriever = DescriptionDatasetRetriever()
    retrieved_dataset_name = "Fraser/python-state-changes"
    prompt_spec = MockPromptSpec(
        1,
        instruction="Answer questions about a Python 3.7 program's intermediate state",
        examples="""input=
```
while True
	print('hello world')
```
What type of exception does this program produce?

output=SyntaxError: invalid syntax

input=
```
def test(x):
	for i in range(2, x**(0.5)):
		if x % int(i) == 0:
			return False
	return True
```
What is test(101)?

output=True

input=
```
x = [i for i in range(10)]
for i, x_elem in enumerate(x):
	x_elem *= 5
	x[i] *= 3
```
What is x at the end of this program?

output=[0, 3, 6, 9, 12, 15, 18, 21, 24, 27]""",
    )
    num_transform = 10
    retrieved_dataset_dict = retriever.canonicalize_dataset_by_cli_data_transform(
        retrieved_dataset_name, prompt_spec, num_transform=num_transform
    )
