METAPROMPT_BASE = """
Your task is to analyse a given task description and a dataset, and determine the confidence of whether that dataset is usable, with some amount of preprocessing. 
"""
TASK = """
Task Description: {task_description}

Task Examples: {task_examples}

Dataset Chosen: {dataset_name}

Config Name: {config_name}

Dataset Description: {dataset_description}

Dataset Sample: {sample}

Dataset Tags: {tags}

Number of Downloads: {num_downloads}

"""
ENDING_LINE = """ Think step by step and finally return a json with confidence key in ["low", "medium", "high"], eg {confidence: low} if this dataset can be used with some processing."""

def construct_prompt_for_dataset_confidence(instruction, demonstrations, dataset_info, config_name):
    
    task = TASK.format(
        task_description = instruction,
        task_examples = demonstrations,
        dataset_name = dataset_info["dataset_name"],
        dataset_description = dataset_info["description"],
        config_name = config_name,
        sample = dataset_info["configs"][config_name]["sample_row"],
        tags = dataset_info["tags"],
        num_downloads = dataset_info["downloads"]
    )

    prompt = "\n------------\n".join([METAPROMPT_BASE, task, ENDING_LINE])

    return prompt