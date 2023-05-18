"""An interface for prompt parsing."""

from prompt2model.prompt_parser.base import PromptSpec


class MockPromptSpec(PromptSpec):
    """Mock the bebavior of PromptSpec."""

    def __init__(self):
        """Mock the elements of PromptSpec."""
        self.instruction = (
            "Give me some translation from Chinese to English."
            " Input Chinese and output English."
        )
        self.examples = [
            "input: '人生苦短，我用 Python', output: 'Life is short, I use Python.'",
            "input: '明天是周末', output: 'Tomorrow is weekend.'",
        ]
        self.prompt_template = (
            "Requirement: {instruction} \n"
            "Few-Shot Examples: {examples} \n"
            "sample: \n"
            "annotation: \n"
            "Please answer me in JSON format, with `sample` and `annotation` keys."
        )

    def parse_from_prompt(self, prompt: str) -> None:
        """Don't parse anything."""
        return None

    def get_instruction(self, prompt: str) -> str:
        """Return the prompt itself, since we do not parse it."""
        return self.instruction
