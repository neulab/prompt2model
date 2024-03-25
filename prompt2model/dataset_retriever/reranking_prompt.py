from __future__ import annotations  # noqa FI58

METAPROMPT_BASE_DATASET = """Your objective is to choose the most relevant dataset for a given a task (and few examples of the task). For each dataset, you will be provided with the dataset description, and tags related to the dataset. Please return the most relevant dataset, e.g., squad """  # noqa: E501

METAPROMPT_BASE_CONFIG = """Your objective is to choose the most relevant config of a dataset for a given a task (and few examples of the task). For each config, you will be provided with the config name, and columns and rows of that config. The columns of the config could be useful at understanding whether this config is relevant to the given task. Another relevant factor is the config_name, this would give information on a high level about what each config represents. """  # noqa: E501

INPUT_PROMPT_DATASET_TEMPLATE = """The following is the task \n {instruction} \n and these are some examples of the same: \n{examples} \n
There are {num} datasets available for this task. \n
{datasets}.
The name of the most relevant dataset for this task is:"""  # noqa: E501

INPUT_PROMPT_CONFIG_TEMPLATE = """ The following is the task: \n {instruction} \n
The following is the dataset selected: {dataset_name}: {dataset_description}
There are {num} configs available in this dataset for this task. \n
{configs}
The name of the most relevant config from these for this task is:
"""

DATASET_TEMPLATE = """[{counter}] **{dataset_name}**:\nDescription-{dataset_description}.\nThis dataset has the following tags:\n {tags} """  # noqa: E501
CONFIG_TEMPLATE = """\t[{counter}] **{config_name}**\n: The columns in this config are {dataset_columns}.\n An example row from this config is {sample_row}.\n """  # noqa: E501


INCONTEXT_EXAMPLE_DATASET = """

"""

INCONTEXT_EXAMPLE_CONFIG = """

"""
ENDING_LINE_DATASET = "After seeing this example, please provide the most relevant dataset for this task:"  # noqa: E501
ENDING_LINE_CONFIG = "After seeing this example, please provide the most relevant config from this dataset for this task:"  # noqa: E501






def build_datasets_prompt(instruction: str, examples: str, datasets_infos):
    """Constructs a prompt that describes each dataset.

    Args:
        datasets_infos (dict): Dictionary with dataset information.

    Returns:
        str: A string that lists each dataset with its description and tags.
    """
    dataset_string = ""
    for i, (_, dataset_info) in enumerate(datasets_infos.items(), start=1):
        dataset_string += f"""{DATASET_TEMPLATE.format(
                                                    counter = i,
                                                    dataset_name=dataset_info["dataset_name"],
                                                    dataset_description=dataset_info["description"],
                                                    tags = dataset_info["tags"]
                                                  )}\n\n"""

    input_prompt = INPUT_PROMPT_DATASET_TEMPLATE.format(
        instruction=instruction,
        examples=examples,
        datasets=dataset_string,
        num=len(datasets_infos),
    )
    return input_prompt


def build_configs_prompt(instruction: str, examples: str, dataset_info):
    """Constructs a prompt for selecting relevant configurations from a given dataset.

    Args:
        dataset_name (str): The name of the dataset for which to select configurations.
        configs (dict): Dictionary of configurations for the dataset.

    Returns:
        str: A string that lists each configuration with its details for the specified dataset.
    """
    configs_string = ""
    j=0
    for _, config in dataset_info["configs"].items():
        configs_string += f"""{CONFIG_TEMPLATE.format(
                                            counter = chr(ord('a')+j),
                                            config_name = config["config_name"],
                                            dataset_columns = config["columns"],
                                            sample_row = config["sample_row"]
                                            )}\n"""  # noqa: E501
        j += 1
    
    input_prompt = INPUT_PROMPT_CONFIG_TEMPLATE.format(
        instruction=instruction,
        examples=examples,
        dataset_name = config["dataset_name"],
        dataset_description = config["dataset_description"],
        configs = configs_string,
        num=len(dataset_info["configs"]),
    )
    return input_prompt
    

def construct_prompt_for_dataset_reranking(instruction: str, examples: str, datasets_infos,config=False):
    """Generate the full prompt for dataset reranking based on the given parameters.

    Args:
        instruction (str): Instruction of the task.
        examples (str): Examples of the task.
        datasets_infos (dict): Dictionary with dataset information. Each dataset_info
                               object also has a configs object representing the various
                               configs of that dataset..

    Returns:
        str: Builds a comprehensive prompt for dataset reranking. This prompt includes
             the base instructions, incontext example and the prompt returned by the
             build_input function.
    """
    if config:
        metaprompt_base = METAPROMPT_BASE_CONFIG
        incontext_example = INCONTEXT_EXAMPLE_CONFIG
        input_prompt = build_configs_prompt(instruction, examples, datasets_infos)
        ending_line = ENDING_LINE_CONFIG
    else:
        metaprompt_base = METAPROMPT_BASE_DATASET
        incontext_example = INCONTEXT_EXAMPLE_DATASET
        input_prompt = build_datasets_prompt(instruction, examples, datasets_infos)
        ending_line = ENDING_LINE_DATASET

    prompt_sections = [metaprompt_base]
    all_prompts = "\n\n------\n\n".join(prompt_sections) + "\n\n------\n\n"
    all_prompts +=  input_prompt

    return all_prompts


