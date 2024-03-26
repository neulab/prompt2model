METAPROMPT_BASE = "Carefully analyse the  task description and examples of the task, and explain the task to give a clearer description. Do not explain each example, but rather capture the general trends. Also place special focus on the format of the input/output examples."

TASK = """
Task Description: {task_description}

Task Examples: {examples}
"""

def construct_prompt_for_task_explanation(instruction, demonstrations):
    task = TASK.format(task_description=instruction, examples=demonstrations)
    prompt = "\n--------\n".join([METAPROMPT_BASE, task])
    return prompt
