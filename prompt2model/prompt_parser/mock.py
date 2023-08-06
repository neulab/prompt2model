"""An interface for prompt parsing."""

from prompt2model.prompt_parser.base import PromptSpec, TaskType


class MockPromptSpec(PromptSpec):
    """Mock the bebavior of PromptSpec."""

    def __init__(
        self, task_type: TaskType, instruction: str = None, examples: str = None
    ):
        """Mock the elements of PromptSpec."""
        self.task_type = task_type
        if instruction is None:
            self._instruction = (
                "Give me some translation from Chinese to English."
                " Input Chinese and output English."
            )
        else:
            self._instruction = instruction
        if examples is None:
            self._examples = (
                "input: '人生苦短，我用 Python', output: 'Life is short, I use Python. '"
                "input: '明天是周末', output: 'Tomorrow is weekend.'"
            )
        else:
            self._examples = examples

    def parse_from_prompt(self, prompt: str) -> None:
        """Don't parse anything."""
        self._instruction = prompt
        return None
