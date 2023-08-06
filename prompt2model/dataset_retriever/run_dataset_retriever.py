from prompt2model.dataset_retriever import DescriptionDatasetRetriever
from prompt2model.prompt_parser import MockPromptSpec, TaskType

if __name__ == "__main__":
    prompt_spec = MockPromptSpec(TaskType.TEXT_GENERATION)
    # prompt_spec._instruction = """Verify scientific claims automatically as fact or fiction. Provide supporting evidence with your decision.."""
    prompt = """Please generate code that satisfies a Japanese StackOverflow question. Please generate only code - no supporting text or explanations. For most queries, the correct code will be a single line of Python code. Do not import any libraries. Do not return any variables or store the answer to a variable; just having a line that evaluates to the correct answer is sufficient."""
    prompt_spec._instruction = prompt

    retriever = DescriptionDatasetRetriever()
    # retriever.encode_dataset_descriptions(retriever.search_index_path)
    retrieved_dataset_dict = retriever.retrieve_dataset_dict(
        prompt_spec, blocklist=["squad", "stanford question answering"]
    )
    retrieved_dataset_dict.save_to_disk("retrived_dataset_dict")