"""Script to run the dataset retriever in isolation."""
from prompt2model.dataset_retriever import DescriptionDatasetRetriever
from prompt2model.prompt_parser import MockPromptSpec, TaskType

if __name__ == "__main__":
    prompt_spec = MockPromptSpec(TaskType.TEXT_GENERATION)
    prompt = """Your task is to generate an answer to a natural question. In this task, the input is a string that consists of both a question and a context passage. The context is a descriptive passage related to the question and contains the answer. And the question can range from Math, Cultural, Social, Geometry, Biology, History, Sports, Technology, Science, and so on."""  # noqa E501
    prompt_spec._instruction = prompt

    retriever = DescriptionDatasetRetriever()
    retriever.retrieve_dataset_dict(prompt_spec)
