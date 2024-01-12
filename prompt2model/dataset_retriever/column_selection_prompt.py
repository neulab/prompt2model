"""Utilities to construct an LLM "metaprompt" for our column selection."""

from __future__ import annotations  # noqa FI58

METAPROMPT_BASE = """Your objective is to carefully analyze the task and the dataset mentioned, and decide whether the columns are relevant input, relevant output, irrelevant for the given task, or if it is ambiguous. There should be at most one output column. It is possible to have no relevant columns, in which case return the input and output column as empty lists.  Answer in a json format, with the following keys: input, output, irrelevant, ambiguous"""  # noqa: E501
METAPROMPT_EXAMPLES = [
    (
        """You are tasked with the following process. In this task, you will generate summaries for given texts. For this task, you will use the Scientific Papers dataset from HuggingFace. Dataset_description: Scientific papers datasets contains two sets of long and structured documents. The datasets are obtained from ArXiv and PubMed OpenAccess repositories.
        A sample data instance from this dataset is as follows.
        {
            "abstract": "\" we have studied the leptonic decay @xmath0 , via the decay channel @xmath1 , using a sample of tagged @xmath2 decays collected...",
            "article": "\"the leptonic decays of a charged pseudoscalar meson @xmath7 are processes of the type @xmath8 , where @xmath9 , @xmath10 , or @...",
            "section_names": "[sec:introduction]introduction\n[sec:detector]data and the cleo- detector\n[sec:analysys]analysis method\n[sec:conclusion]summary"
        }
        This dataset has the following columns: [abstract, article, section_names].
        """,  # noqa: E501
        """
        {
            "input": ["article"],
            "output": ["abstract"],
            "irrelevant": ["section_names"],
            "ambiguous": []
        }""",
    ),
    (
        """
        You are tasked with the following process. In this task, you will detect whether some given text uses hateful speech or not. For this task you will use the hate_speech_offensive dataset from HuggingFace. Dataset_description: An annotated dataset for hate speech and offensive language detection on tweets.
        A sample data instance from this is as follows:
        {
            "count": 3,
            "hate_speech_count": 0,
            "offensive_language_count": 0,
            "neither_count": 3,
            "label": 2,  # "neither"
            "tweet": "!!! RT @mayasolovely: As a woman you shouldn't complain about cleaning up your house. &amp; as a man you should always take the trash out...")
        }.
        This dataset has the following columns: [count, hate_speech_count, offensive_language_count, neither_count, class, tweet]""",  # noqa: E501
        """
        {
            "input": ["tweet"],
            "output": ["label"],
            "irrelevant": [],
            "ambiguous": ["hate_speech_count", "offensive_language_count", "neither_count", "count"]
        }""",  # noqa: E501
    ),
    (
        """You are tasked with the following process. Your job is to be able to translate between languages. For this task, you will use a custom dataset. Dataset_description: This dataset is meant to translate between languages.
        A sample data instance from this is as follows:
        {
            "translation": ["ca: "El department de bombers té el seu propi equip d'investigació.", "en": "Well, the fire department has its own investigative unit."]
        }
        This dataset has the following columns: [translation]. """,  # noqa: E501
        """
        {
            "input": [],
            "output": [],
            "irrelevant": []
            "ambiguous": ["translation"]
        }""",
    ),
    (
        """You are tasked with the following process. Your job is to be able to summarize a given text. For this task, you will use the math_qa dataset from HuggingFace. Dataset_description: Our dataset is gathered by using a new representation language to annotate over the AQuA-RAT dataset with fully-specified operational programs.
        A sample data instance from this is as follows:
        {
            "Problem": "a multiple choice test consists of 4 questions , and each question has 5 answer choices . in how many r ways can the test be completed if every question is unanswered ?",
            "Rationale": "\"5 choices for each of the 4 questions , thus total r of 5 * 5 * 5 * 5 = 5 ^ 4 = 625 ways to answer all of them . answer : c .\"",
            "annotated_formula": "power(5, 4)",
            "category": "general",
            "correct": "c",
            "linear_formula": "power(n1,n0)|",
            "options": "a ) 24 , b ) 120 , c ) 625 , d ) 720 , e ) 1024"
        }
        This dataset has the following columns: [problem, rationale, options, correct, annotated_formula]. """,  # noqa: E501
        """
        {
            "input": [],
            "output": [],
            "irrelevant": ["problem", "rationale", "options", "correct", "annotated_formula"],
            "ambiguous": []
        }""",  # noqa: E501
    ),
]

INPUT_PROMPT_TEMPLATE = """You are tasked with the following process. {instruction} For this task, you will use the {dataset_name} dataset from HuggingFace. Dataset Description: {dataset_description} \nA sample data instance from this is as follows. {sample_row}.\nThis dataset has the following columns: [{dataset_columns} ]."""  # noqa: E501
SINGLE_DEMONSTRATION_TEMPLATE = (
    'Task: """\n{prompt}\n"""\n\nRequired Columns :\n{columns}'
)
ENDING_LINE = "After seeing these examples with the required columns, please provide the relevant columns for this context:"  # noqa: E501


def build_input(
    instruction: str,
    dataset_name: str,
    dataset_description: str,
    dataset_columns: str,
    sample_row: dict,
) -> str:
    """Template function to build input based on arguments."""
    input_prompt = INPUT_PROMPT_TEMPLATE.format(
        instruction=instruction,
        dataset_name=dataset_name,
        dataset_description=dataset_description,
        dataset_columns=dataset_columns,
        sample_row=sample_row,
    )
    input_prompt = SINGLE_DEMONSTRATION_TEMPLATE.format(
        prompt=input_prompt, columns=""
    )  # columns="" because that is what we are trying to predict
    return input_prompt


def construct_prompt_for_column_selection(
    instruction: str,
    dataset_name: str,
    dataset_description: str,
    dataset_columns: str,
    sample_row: dict,
) -> str:
    """Generate prompt for column selection."""
    prompt_sections = [METAPROMPT_BASE]
    for prompt, columns in METAPROMPT_EXAMPLES:
        prompt_sections.append(
            SINGLE_DEMONSTRATION_TEMPLATE.format(prompt=prompt, columns=columns)
        )
    all_prompts = "\n\n------\n\n".join(prompt_sections) + "\n\n------\n\n"
    input_prompt = build_input(
        instruction, dataset_name, dataset_description, dataset_columns, sample_row
    )
    all_prompts += ENDING_LINE + input_prompt

    return all_prompts
