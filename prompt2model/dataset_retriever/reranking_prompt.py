"""Utilities to construct an LLM "metaprompt" for our column selection."""

from __future__ import annotations  # noqa FI58

METAPROMPT_BASE = """Your objective is to rerank datasets given a task (and few examples of the task) based on relevancy. Relevant factors involve how suited the dataset is for the task and whether the input/output formats of the task and the dataset data match. For each dataset, you will be provided with the dataset description, and the configurations available. Each configuration of the dataset will be represented with the config_name, the columns in that configuration, and an example row. Please return a SINGLE tuple of the combination of the most relevant dataset, with the best suited config using their name, along with a confidence level ranging from [low, medium, high] representing how relevant the dataset is to the task. The output format should be a tuple of form (dataset_name,config_name,confidence_level). e.g., (squad,plain_text,low). """  # noqa: E501

INPUT_PROMPT_TEMPLATE = """The following is the task \n {instruction} and these are some examples of the same: {examples} \n
There are {num} datasets available for this task. \n
{datasets}.
The reranking results of the {num} datasets in (dataset_name,config_name,confidence_level) format is: \n{reranking}"""  # noqa: E501

DATASET_TEMPLATE = """[{counter}] **{dataset_name}**\n: Description-{dataset_description}.\n. This dataset has the following configs:\n  """  # noqa: E501
CONFIG_TEMPLATE = """\t[{counter}] **{config_name}**\n: The columns in this config are {dataset_columns}.\n An example row from this config is {sample_row}.\n """  # noqa: E501

INCONTEXT_EXAMPLE = """
An example is as follows:

The following is the task
 In this task, you're given passages that contain mentions of names of people, places, or things. Some of these mentions refer to the same person, place, or thing. Your job is to write questions that evaluate one's understanding of such references. Good questions are expected to link pronouns (she, her, him, his, their, etc.) or other mentions to people, places, or things to which they may refer. Do not ask questions that can be answered correctly without understanding the paragraph or having multiple answers. Avoid questions that do not link phrases referring to the same entity. For each of your questions, the answer should be one or more phrases in the paragraph, and it should be unambiguous. and these are some examples of the same: Passage: Nearing London, Oliver encounters Jack Dawkins, a pickpocket more commonly known by the nickname the "Artful Dodger", and his sidekick, a boy of a humorous nature named Charley Bates, but Oliver's innocent and trusting nature fails to see any dishonesty in their actions. The Dodger provides Oliver with a free meal and tells him of a gentleman in London who will "give him lodgings for nothing, and never ask for change". Grateful for the unexpected assistance, Oliver follows the Dodger to the "old gentleman's" residence. In this way Oliver unwittingly falls in with an infamous Jewish criminal known as Fagin, the gentleman of whom the Artful Dodger spoke. Ensnared, Oliver lives with Fagin and his gang of juvenile pickpockets in their lair at Saffron Hill for some time, unaware of their criminal occupations. He believes they make wallets and handkerchiefs.

Passage: Nearing London, Oliver encounters Jack Dawkins, a pickpocket more commonly known by the nickname the "Artful Dodger", and his sidekick, a boy of a humorous nature named Charley Bates, but Oliver's innocent and trusting nature fails to see any dishonesty in their actions. The Dodger provides Oliver with a free meal and tells him of a gentleman in London who will "give him lodgings for nothing, and never ask for change". Grateful for the unexpected assistance, Oliver follows the Dodger to the "old gentleman's" residence. In this way Oliver unwittingly falls in with an infamous Jewish criminal known as Fagin, the gentleman of whom the Artful Dodger spoke. Ensnared, Oliver lives with Fagin and his gang of juvenile pickpockets in their lair at Saffron Hill for some time, unaware of their criminal occupations. He believes they make wallets and handkerchiefs.
Output: Who believes Fagin's gang make wallets and handkerchiefs?.
Explanation: This question is based on the following sentence in the passage "He believes they make wallets and handkerchiefs". It evaluates the understanding that the pronoun "he" refers to name "Oliver". You can ask questions like this one about most pronouns in a paragraph.

Output: What is the alias of the person whose sidekick had a humorous nature?.
Explanation: This question is based on the following sentence in the passage "Nearing London, Oliver encounters Jack Dawkins, a pickpocket more commonly known by the nickname the "Artful Dodger", and his sidekick, a boy of a humorous nature named Charley Bates". The pronoun "his" refers to a person with multiple names. But since the question explicitly asks for the alias, the answer is unambiguous.

There are 3 datasets available for this task.

[1] **facebook/babi_qa**
: Description-The (20) QA bAbI tasks are a set of proxy tasks that evaluate reading
comprehension via question answering. Our tasks measure understanding
in several ways: whether a system is able to answer questions via chaining facts,
simple induction, deduction and many more. The tasks are designed to be prerequisites
for any system that aims to be capable of conversing with a human.
The aim is to classify these tasks into skill sets,so that researchers
can identify (and then rectify)the failings of their systems.
.
. This dataset has the following configs:

        [a] **shuffled-10k-qa1**
: The columns in this config are story_id, story_type, story_text, story_supporting_ids, story_answer.
 An example row from this config is {"story.id": "[\"1\", \"2\", \"3\", \"4\", \"5\", \"6\", \"7\", \"8\", \"9\", \"10\"...", "story.type": "[0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1]", "story.text": "[\"Utxi ybnha qb qzh ptqzxbby.\", \"Hbzm jhmq qb qzh ...", "story.supporting_ids": "[[], [], [\"1\"], [], [], [\"4\"], [], [], [\"4\"], [], ...", "story.answer": "[\"\", \"\", \"ptqzxbby\", \"\", \"\", \"ztuujti\", \"\", \"\", \"z..."}.


        [b] **en-valid-10k-qa1**
: The columns in this config are story_id, story_type, story_text, story_supporting_ids, story_answer.
 An example row from this config is {"story.id": "[\"1\", \"2\", \"3\", \"4\", \"5\", \"6\", \"7\", \"8\", \"9\", \"10\"...", "story.type": "[0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1]", "story.text": "[\"Mary moved to the bathroom.\", \"John went to the ...", "story.supporting_ids": "[[], [], [\"1\"], [], [], [\"4\"], [], [], [\"4\"], [], ...", "story.answer": "[\"\", \"\", \"bathroom\", \"\", \"\", \"hallway\", \"\", \"\", \"h..."}.


        [c] **hn-10k-qa1**
: The columns in this config are story_id, story_type, story_text, story_supporting_ids, story_answer.
 An example row from this config is {"story.id": "[\"1\", \"2\", \"3\", \"4\", \"5\", \"6\", \"7\", \"8\", \"9\", \"10\"...", "story.type": "[0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1]", "story.text": "[\"Sita gusalkhaney mein gayi.\", \"Priya sayanakaksh...", "story.supporting_ids": "[[], [], [\"2\"], [], [], [\"5\"], [], [], [\"7\"], [], ...", "story.answer": "[\"\", \"\", \"sayanakaksh\", \"\", \"\", \"rasoi ghar\", \"\", ..."}.


        [d] **en-valid-qa1**
: The columns in this config are story_id, story_type, story_text, story_supporting_ids, story_answer.
 An example row from this config is {"story.id": "[\"1\", \"2\", \"3\", \"4\", \"5\", \"6\", \"7\", \"8\", \"9\", \"10\"...", "story.type": "[0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1]", "story.text": "[\"Mary moved to the bathroom.\", \"John went to the ...", "story.supporting_ids": "[[], [], [\"1\"], [], [], [\"4\"], [], [], [\"4\"], [], ...", "story.answer": "[\"\", \"\", \"bathroom\", \"\", \"\", \"hallway\", \"\", \"\", \"h..."}.




[2] **drop**
: Description-DROP: A Reading Comprehension Benchmark Requiring Discrete Reasoning Over Paragraphs.
. DROP is a crowdsourced, adversarially-created, 96k-question benchmark, in which a system must resolve references in a
question, perhaps to multiple input positions, and perform discrete operations over them (such as addition, counting, or
 sorting). These operations require a much more comprehensive understanding of the content of paragraphs than what was
 necessary for prior datasets.
.
. This dataset has the following configs:

        [a] **default**
: The columns in this config are section_id, query_id, passage, question, answers_spans_spans, answers_spans_types.
 An example row from this config is {"section_id": "\"nfl_2201\"", "query_id": "\"f16c0ee7-f131-4a8b-a6ac-4d275ea68066\"", "passage": "\"To start the season, the Lions traveled south to ...", "question": "\"How many points did the buccaneers need to tie in...", "answers_spans.spans": "[\"3\"]", "answers_spans.types": "[\"number\"]"}.



[3] **SajjadAyoubi/persian_qa**
: Description-\\\\
Persian Question Answering (PersianQA) Dataset is a reading comprehension dataset on Persian Wikipedia.
The crowd-sourced dataset consists of more than 9,000 entries. Each entry can be either an impossible to answer or a question with one or more answers spanning in the passage (the context) from which the questioner proposed the question. Much like the SQuAD2.0 dataset, the impossible or unanswerable questions can be utilized to create a system which "knows that it doesn't know the answer".
.
. This dataset has the following configs:

        [a] **persian_qa**
: The columns in this config are id, title, context, question, answers_text, answers_answer_start.
 An example row from this config is {"id": "1", "title": "\"\\u0634\\u0631\\u06a9\\u062a \\u0641\\u0648\\u0644\\u0627...", "context": "\"\\u0634\\u0631\\u06a9\\u062a \\u0641\\u0648\\u0644\\u0627...", "question": "\"\\u0634\\u0631\\u06a9\\u062a \\u0641\\u0648\\u0644\\u0627...", "answers.text": "[\"\\u062f\\u0631 \\u0634\\u0631\\u0642 \\u0634\\u0647\\u06...", "answers.answer_start": "[114]"}.


The reranking results of the 3 datasets in (dataset_name,config_name,confidence_level) format is:


(drop,default,medium)

"""  # noqa: E501
ENDING_LINE = "After seeing this example, please provide the reranking of the datasets for this task:"  # noqa: E501


def build_input(instruction: str, examples: str, datasets_infos) -> str:
    """Template function to build user specific part of thr prompt.

    Args:
        instruction (str): Instruction of the task.
        examples (str): Examples of the task.
        datasets_infos (dict): Dictionary with dataset information. Each dataset_info
                               object also has a configs object representing the various
                               configs of that dataset.

    Returns:
        str: Builds the reranking prompt by combining instruction, examples, and dataset
            information into a structured input prompt. Iterates over datasets_infos,
            and its configurations.
    """
    dataset_string = ""
    i = 0
    for _, dataset_info in datasets_infos.items():
        curr_dataset = f"""{DATASET_TEMPLATE.format(
                                                    counter = i+1,
                                                    dataset_name=dataset_info["dataset_name"],
                                                    dataset_description=dataset_info["description"]
                                                  )}\n\n"""
        j = 0
        for _, config in dataset_info["configs"].items():
            curr_dataset += f"""{CONFIG_TEMPLATE.format(
                                                    counter = chr(ord('a')+j),
                                                    config_name = config["config_name"],
                                                    dataset_columns = config["columns"],
                                                    sample_row = config["sample_row"]
                                                    )}\n"""  # noqa: E501
            j += 1

        dataset_string += curr_dataset + "\n\n\n"
        i += 1

    input_prompt = INPUT_PROMPT_TEMPLATE.format(
        instruction=instruction,
        examples=examples,
        datasets=dataset_string,
        num=len(datasets_infos),
        reranking="",
    )
    return input_prompt


def construct_prompt_for_dataset_reranking(
    instruction: str, examples: str, datasets_infos
) -> str:
    """Generate the full prompt for dataset reranking based on the given parameters.

    Args:
        instruction (str): Instruction of the task.
        examples (str): Examples of the task.
        datasets_infos (dict): Dictionary with dataset information. Each dataset_info
                               object also has a configs object representing the various
                               configs of that dataset..

    Returns:
        str: Builds a comprehensive prompt for dataset reranking. This prompt includes
             the base instructions, incontext example and the prompt returned by the
             build_input function.
    """
    prompt_sections = [METAPROMPT_BASE, INCONTEXT_EXAMPLE]
    all_prompts = "\n\n------\n\n".join(prompt_sections) + "\n\n------\n\n"
    input_prompt = build_input(instruction, examples, datasets_infos)
    all_prompts += ENDING_LINE + input_prompt

    return all_prompts
