from prompt2model.prompt_parser import MockPromptSpec, TaskType
from prompt2model.dataset_retriever import DescriptionDatasetRetriever

if __name__ == "__main__":
    prompt_spec = MockPromptSpec(TaskType.TEXT_GENERATION)
    # prompt_spec._instruction = """Verify scientific claims automatically as fact or fiction. Provide supporting evidence with your decision.."""
    prompt = """Generate one line of code in Python to solve a Japanese question on StackOverflow. Do not include comments or expressions. No import statements are required.

For this task, the input is text in Japanese, describing variable names and operations. The output is a single line of Python code to accomplish the task. Do not include comments or expressions. Import statements are also not required.
"""
    prompt_spec._instruction = prompt

    retriever = DescriptionDatasetRetriever()
    # retriever.encode_dataset_descriptions(retriever.search_index_path)
    retriever.retrieve_dataset_dict(prompt_spec)
