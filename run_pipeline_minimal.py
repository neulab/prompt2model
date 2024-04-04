from prompt2model.prompt_parser import PromptSpec, PromptBasedInstructionParser, TaskType
from prompt2model.dataset_retriever.description_dataset_retriever_v2 import DescriptionDatasetRetriever_V2
from prompt2model.utils import api_tools
from prompt2model.utils.api_tools import APIAgent
import wandb
import os
import json


def get_tasks(path, task_list = []):
    task_dict = {}
    if task_list == []:
        task_list = os.listdir(path)
        # load task json as a dictionary
        for task in task_list:
            with open(os.path.join(path, task), "r") as f:
                curr_task_dict = json.load(f)
                task_name = task.replace(".json", "")
                task_dict[task_name] = curr_task_dict['prompt_instruction'], curr_task_dict['prompt_examples']
    else:
        for task in task_list:
            task = task + ".json"
            with open(os.path.join(path, task), "r") as f:
                curr_task_dict = json.load(f)
                task_name = task.replace(".json", "")
                task_dict[task_name] = curr_task_dict['prompt_instruction'], curr_task_dict['prompt_examples']
    
    return task_dict
        

if __name__=="__main__":
    os.environ["AZURE_API_BASE"] = "https://vijay-gpt-4-sweden.openai.azure.com/"
    os.environ["AZURE_API_VERSION"] = '2023-05-15'
    api_tools.default_api_agent = APIAgent(model_name="azure/vijay-gpt-4-turbo-sweden", max_tokens=2000)
    
    task_to_dataset_config_chosen = {
        "cause_and_effect": ("race","all"),
        "code_line_description": ("codeparrot/apps","all"),
        "elementary_math": ("lighteval/MATH","all"),
        "implicatures": ("pragmeval","persuasiveness-relevance"),
        "medical_questions_russian": ("blinoff/medical_qa_ru_data","default"),
        "temporal_sequences": ("squad","plain_text"),
    }
    
    prompt_version = "new_planning_wincontext_test_"
    # task_name = "implicatures"
    # task_description = "Predict whether Speaker 2's answer to Speaker 1 counts as a yes or as a no"
    # task_examples = "input=\n\nQ: Speaker 1: 'Have you found him yet? ' Speaker 2: 'We're still looking.' \nA: \noutput=no\n\ninput=\n\nQ: Speaker 1: 'You want to do this to the whole world?' Speaker 2: 'So the whole world will be exactly how I want.' \nA: \noutput=yes\n\ninput=\n\nQ: Speaker 1: 'Would he fire me?' Speaker 2: 'He's all bark and no bite.' \nA: \noutput=no"
    
    task_dict = get_tasks(path="task_jsons", task_list=[])
    
    for task_name, (task_description, task_examples) in task_dict.items():    
        wandb.init(
            project="bigbench_p2m",
            name=f"{task_name}_{prompt_version}",
        )
        
        wandb.log({"task_description": task_description, "task_examples": task_examples})
        
        chosen_ds_config = task_to_dataset_config_chosen.get(task_name, None)
        chosen_ds_config = [chosen_ds_config] if chosen_ds_config is not None else None
        
        prompt_spec = PromptBasedInstructionParser(task_type=TaskType.TEXT_GENERATION)
        prompt_spec.set_instruction_and_examples(task_description, task_examples)
        d = DescriptionDatasetRetriever_V2()
        dataset = d.retrieve_dataset_dict_minimal(prompt_spec, True, 10, num_votes = 5, num_datasets_to_choose=1, chosen_ds_config=chosen_ds_config)
        # end wandb run
        wandb.finish()