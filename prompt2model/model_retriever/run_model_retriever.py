from prompt2model.prompt_parser import MockPromptSpec, TaskType
from prompt2model.model_retriever import DescriptionModelRetriever


if __name__ == "__main__":
    prompt_spec = MockPromptSpec(TaskType.TEXT_GENERATION)
    # prompt_spec._instruction = """Verify scientific claims automatically as fact or fiction. Provide supporting evidence with your decision.."""
    prompt = 'Generate one line of code in Python to solve a Japanese question on StackOverflow. Do not include comments or expressions. No import statements are required.\n\nFor this task, the input is text in Japanese, describing variable names and operations. The output is a single line of Python code to accomplish the task. Do not include comments or expressions. Import statements are also not required.\n'
    prompt_spec._instruction = prompt


    retriever = DescriptionModelRetriever(
        search_index_path="/tmp",
        model_descriptions_index_path="huggingface_data/huggingface_models/model_info/",
        use_bm25=True,
        use_HyDE=True
    )
    top_model_name = retriever.retrieve(prompt_spec)
