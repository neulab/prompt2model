"""BIGBENCH main script."""

import json
import os

BBLITE_TASKS = [
    "auto_debugging",
    "bbq_lite_json",
    "code_line_description",
    "conceptual_combinations",
    "conlang_translation",
    "emoji_movie",
    "formal_fallacies_syllogisms_negation",
    "hindu_knowledge",
    "known_unknowns",
    "language_identification",
    "linguistics_puzzles",
    "logic_grid_puzzle",
    "logical_deduction",
    "misconceptions_russian",
    "novel_concepts",
    "operators",
    "parsinlu_reading_comprehension",
    "play_dialog_same_or_different",
    "repeat_copy_logic",
    "strange_stories",
    "strategyqa",
    "symbol_interpretation",
    "vitaminc_fact_verification",
    "winowhy",
]
TASKS_WITH_PREFERRED_SCORE = [
    "auto_debugging",
    "code_line_description",
    "emoji_movie",
    "formal_fallacies_syllogisms_negation",
    "hindu_knowledge",
    "known_unknowns",
    "language_identification",
    "linguistics_puzzles",
    "logic_grid_puzzle",
    "misconceptions_russian",
    "novel_concepts",
    "operators",
    "parsinlu_reading_comprehension",
    "play_dialog_same_or_different",
    "repeat_copy_logic",
    "strategyqa",
    "vitaminc_fact_verification",
    "winowhy",
]
TASKS_WITH_EXACT_STR_MATCH = [
    "auto_debugging",
    "linguistics_puzzles",
    "operators",
    "parsinlu_reading_comprehension",
    "repeat_copy_logic",
]


class BigBenchTask:
    """Define BigBench Task object."""

    def __init__(self, bb_path, task_name, type="JSON") -> None:
        """Initialize BigBench Task object."""
        if type == "JSON":
            self.path = os.path.join(bb_path, task_name, "task.json")
        elif type == "programmatic":
            raise NotImplementedError

        with open(self.path, "r") as f:
            self.task_json = json.load(f)

        try:
            self.name = self.task_json["name"]
            self.description = self.task_json["description"]
            self.preferred_score = self.task_json["preferred_score"]
            self.metrics = self.task_json["metrics"]
            self.examples = self.task_json["examples"]
            if (
                "example_input_prefix" in self.task_json
                and "example_output_prefix" in self.task_json
            ):
                self.example_input_prefix = self.task_json["example_input_prefix"]
                self.example_output_prefix = self.task_json["example_output_prefix"]
            else:
                self.example_input_prefix = ""
                self.example_output_prefix = ""
        except KeyError as e:
            print(f"Key {e} not found in task.json")
            self.name = None
            self.description = None
            self.preferred_score = None
            self.metrics = None
            self.examples = []
            self.example_input_prefix = None
            self.example_output_prefix = None

    def get_fewshot_examples(self, num_examples=3, random=False):
        """Get fewshot examples from task."""
        if random:
            return random.sample(self.examples, num_examples)

        return self.examples[:num_examples]


if __name__ == "__main__":
    bb_path = "./BIG-bench/bigbench/benchmark_tasks"
    working_tasks = []
    exact_str_match_tasks = []

    for task_name in BBLITE_TASKS:
        task = BigBenchTask(bb_path, task_name)
        if task.name:
            working_tasks.append(task.name)

    for task in working_tasks:
        task = BigBenchTask(bb_path, task)
        print(
            f"Task: {task.name}\tPreferred Score: {task.preferred_score}\t\
                num_examples: {len(task.examples)}"
        )

        if task.preferred_score == "exact_str_match":
            exact_str_match_tasks.append(task.name)

    print(
        f"Number of working tasks:{len(working_tasks)}\tWorking tasks: {working_tasks}"
    )
    print(
        f"Number of exact_str_match tasks:{len(exact_str_match_tasks)}\t\
            Exact str match tasks: {exact_str_match_tasks}"
    )
    print(BBLITE_TASKS)
