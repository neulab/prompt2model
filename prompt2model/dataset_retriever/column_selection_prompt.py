"""Utilities to construct an LLM "metaprompt" for our column selection."""

from __future__ import annotations  # noqa FI58

METAPROMPT_BASE = """Your objective is to carefully analyze the task, with the dataset mentioned and assign the columns into relevant input, output, ambigous and irrelevant columns for the given task. There should be atmost one output column. It is possible to have no relevant columns, in which case return the input and output column as empty lists.  Answer in a json format, with the following keys: input, output, irrelevant, ambiguous"""  # noqa: E501
METAPROMPT_EXAMPLES = [
    (
        """Your task is as follows. In this task, you will generate summaries for given texts. You will use the Scientific Papers dataset from HuggingFace. A sample data instance from this dataset is as follows.
        {
        "abstract": "\" we have studied the leptonic decay @xmath0 , via the decay channel @xmath1 , using a sample of tagged @xmath2 decays collected...",
        "article": "\"the leptonic decays of a charged pseudoscalar meson @xmath7 are processes of the type @xmath8 , where @xmath9 , @xmath10 , or @...",
        "section_names": "[sec:introduction]introduction\n[sec:detector]data and the cleo- detector\n[sec:analysys]analysis method\n[sec:conclusion]summary"
        }. This dataset has the following columns: [abstract, article, section_names].
        """,  # noqa: E501
        """ {
            "input": ["article"],
            "output": ["abstract"],
            "irrelevant": ["section_names"],
            "ambiguous": []
        }""",  # noqa: E501
    ),
    (
        """
        Your task is the following. In this task, the input is a string that consists of both a question and a context passage. The context is a descriptive passage related to the question and contains the answer. And the question can range from Math, Cultural, Social, Geometry, Biology, History, Sports, Technology, Science, and so on.
        You will use the Children's Book Test dataset from HuggingFace. A sample data instance from this dataset is as follows: 
        {'answer': 'said', 'options': ['christening', 'existed', 'hear', 'knows', 'read', 'remarked', 'said', 'sitting', 'talking', 'wearing'], 'question': "`` They are very kind old ladies in their way , '' XXXXX the king ; `` and were nice to me when I was a boy . ''", 'sentences': ['This vexed the king even more than the queen , who was very clever and learned , and who had hated dolls when she was a child .', 'However , she , too in spite of all the books she read and all the pictures she painted , would have been glad enough to be the mother of a little prince .', 'The king was anxious to consult the fairies , but the queen would not hear of such a thing .', 'She did not believe in fairies : she said that they had never existed ; and that she maintained , though The History of the Royal Family was full of chapters about nothing else .', 'Well , at long and at last they had a little boy , who was generally regarded as the finest baby that had ever been seen .', 'Even her majesty herself remarked that , though she could never believe all the courtiers told her , yet he certainly was a fine child -- a very fine child .', 'Now , the time drew near for the christening party , and the king and queen were sitting at breakfast in their summer parlour talking over it .', 'It was a splendid room , hung with portraits of the royal ancestors .', 'There was Cinderella , the grandmother of the reigning monarch , with her little foot in her glass slipper thrust out before her .'"]}
        This dataset has the following columns: [sentences, questions, answers, options]""",  # noqa: E501
        """{
        "input": ["context", "question"],
        "output": ["answer"],
        "irrelevant": [],
        "ambiguous": ["options"]
        }""",  # noqa: E501
    ),
    (
        """Your job is to be able to translate between languages. You will use the Opus100 dataset from HuggingFace. A sample data instance from this is as follows: 
        {"translation":{ "ca": "El department de bombers té el seu propi equip d'investigació.", "en": "Well, the fire department has its own investigative unit." }}. This dataset has the following columns: [translation]. 
       """,  # noqa: E501
        """{
        "input": [],
        "output": [],
        "irrelevant": []
        "ambiguous": ["translation"]
        }""",  # noqa: E501
    ),
]

# TODO
def truncate_row(example_row):
    """Add truncation if the example row is too big"""
    pass


def build_input(instruction, dataset_name, dataset_columns, sample_row):

    input_prompt_template = f"""You are tasked with the following process. {instruction}  

For this task, you will use the {dataset_name} dataset from HuggingFace.  The following is a sample data instance from this is as follows. {sample_row}.
This dataset has the following columns: [{dataset_columns} ].
"""  # noqa: E501

    input_prompt_template = construct_single_demonstration(input_prompt_template)
    return input_prompt_template


def construct_single_demonstration(
    prompt: str,
    cols: str = "",
) -> str:
    """Format a demonstration or prediction example to give to an LLM.

    Args:
        user_prompt: A textual prompt describing the task.
        parse_dict: A parsing dictionary containing the correct prompt parse
                    corresponding to the given `user_prompt`.
        input_only: Whether the returned string should only contain
                    the input part. Defaults to False. This should be set to
                    true when constructing the final prediction template to be
                    completed by the LLM.
    """
    return f'''Task: """\n{prompt}\n"""\n\nRequired Columns :\n{cols}'''


def construct_prompt_for_column_selection(
    instruction, dataset_name, dataset_columns, sample_row
):
    prompt_sections = [METAPROMPT_BASE]
    for prompt, correct_parse in METAPROMPT_EXAMPLES:
        prompt_sections.append(construct_single_demonstration(prompt, correct_parse))
    all_prompts = "\n\n------\n\n".join(prompt_sections) + "\n\n------\n\n"
    input_prompt = build_input(instruction, dataset_name, dataset_columns, sample_row)
    all_prompts += (
        "After seeing these examples with the required columns, please provide the relevant columns for this context:"
        + input_prompt  # noqa: E501
    )

    return all_prompts
