"""Utilities to construct an LLM "metaprompt" for our column selection."""

from __future__ import annotations  # noqa FI58

import json

import datasets

# Questions:
# 1. In the prompt template, should I mention that I will be using the dataset for training a machine learning model?
# 2. Aligning dataset columns with task's input/output would require that the model already knows about what column selection needs to be done. (How else would it know that it fits the required task's input output.) -- this is a cyclic problem
# 3. [current caveat] Which config to load for a dataset. How to provide description of each? There is a get_dataset_infos for loading info of each config, but it is a pretty slow method, and so running it for 25 datasets might take time. Also not the most relevant. Will probably just mention the config names and hope that pretraining knows enough?? Not super happy with this approach
# 3b. If a dataset with multiple configs is chosen, how to load the appropriate config? Write a sep prompt for that? return it in dataset reranking itself?
# (1) select top config and load that.
# 4a. To load rows, we need config chosen.

# 4b Loading rows might be slow, even with streaming [~1s]. Should time it.
#  5. probably delete things actively because some datasets formats cannot be streamed
# 6. Probably refactor code for flatten. and maybe just remove cli code
# 7. Not included popularity for now.
# 8. Should we update dataset_info_file with column names, rows etc instead of fetching from HF?

# TODO: Add in-context learning
# TODO: Add configs
METAPROMPT_BASE = """Your objective is to rerank datasets given a task (and few examples of the task) based on relevancy. Relevant factors involve how suited the dataset is for the task, whether the input/output formats of the task and the dataset match.  Each dataset is indicated by number identifier []. For each dataset, you will be provided with the dataset description, and the configurations available for each dataset. Each config will be represented with the config name, the columns in the dataset, and a few example rows. Please return the combination of the most relevant dataset, with the best suited configs using their identifiers and name, with the most relevant passages listed first, along with a confidence level ranging from [low, medium, high] . The output format should be a tuple of form (dataset_index,dataset_name,config_index,config_name,confidence_level). e.g., (1,squad,a,"plain_text",low). """  # noqa: E501

INPUT_PROMPT_TEMPLATE = """The following is the task \n {instruction} and these are some examples of the same: {examples} \n
There are {num} datasets available for this task, each indicated by number identifier []. \n
{datasets}
The reranking results of the {num} datasets in (dataset_index,dataset_name,config_index,config_name,confidence_level) format is:"""  # noqa: E501
# SINGLE_DEMONSTRATION_TEMPLATE = (
#     'Task: """\n{prompt}\n"""\n\nRequired Columns :\n{columns}'
# )
# ENDING_LINE = "After seeing these examples with the required columns, please provide the relevant columns for this context:"  # noqa: E501
ENDING_LINE = ""
DATASET_TEMPLATE = """[{counter}] {dataset_name}\n: Description-{dataset_description}.\n. This dataset has the following configs:\n  """
CONFIG_TEMPLATE = """\t[{counter}] {config_name}\n: The columns in this config are {dataset_columns}.\n An example row from this config is {sample_row}.\n\n """


def truncate_row(example_row: dict, max_length=50) -> str:
    """Truncate the row before displaying if it is too long."""
    truncated_row = {}
    for key in example_row.keys():
        curr_row = json.dumps(example_row[key])
        truncated_row[key] = (
            curr_row
            if len(curr_row) <= max_length - 3
            else curr_row[:max_length] + "..."
        )
    return json.dumps(truncated_row)


def build_input(instruction: str, examples: str, datasets_infos) -> str:
    """Template function to build input based on arguments."""
    dataset_string = ""
    for i in range(len(datasets_infos)):
        dataset_info = datasets_infos[i]
        curr_dataset = f"""{DATASET_TEMPLATE.format(
                                                    counter = i+1,
                                                    dataset_name=dataset_info["dataset_name"],
                                                    dataset_description=dataset_info["dataset_description"]
                                                  )}\n\n"""

        for j in range(len(dataset_info["configs"])):
            config = dataset_info["configs"][j]
            curr_dataset += f"""{CONFIG_TEMPLATE.format(
                                                    counter = chr(ord('a')+j+1),
                                                    config_name = config["config_name"],
                                                    dataset_columns = config["columns"],
                                                    sample_row = truncate_row(config["sample_row"])
                                                    )}\n\n"""

        dataset_string += curr_dataset + "\n\n\n"

    input_prompt = INPUT_PROMPT_TEMPLATE.format(
        instruction=instruction,
        examples=examples,
        datasets=dataset_string,
        num=len(datasets_infos),
    )
    return input_prompt


def construct_prompt_for_dataset_reranking(
    instruction: str, examples: str, datasets_infos
) -> str:
    """Generate prompt for column selection."""
    prompt_sections = [METAPROMPT_BASE]
    # for prompt, columns in METAPROMPT_EXAMPLES:
    #     prompt_sections.append(
    #         SINGLE_DEMONSTRATION_TEMPLATE.format(prompt=prompt, columns=columns)
    #     )
    all_prompts = "\n\n------\n\n".join(prompt_sections) + "\n\n------\n\n"
    input_prompt = build_input(instruction, examples, datasets_infos)
    all_prompts += ENDING_LINE + input_prompt

    return all_prompts
