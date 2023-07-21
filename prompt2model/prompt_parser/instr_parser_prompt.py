"""Utilities to construct an LLM "metaprompt" for our instruction parser."""

from __future__ import annotations  # noqa FI58

import json

METAPROMPT_INSTRUCTION = """
As a PromptParser, your objective is to carefully analyze prompts and divide them into two distinct components: an 'Instruction' that provides the primary description of the task, and 'Demonstrations' which are optional examples showcasing the task. Your aim is to generate a JSON dictionary response containing the `Instruction` and `Demonstrations` fields, corresponding to these two components. In case there are no demonstrations provided, the 'Demonstrations' field should be marked as 'N/A'. When including demonstrations, only consider complete examples that consist of both input and output pairs, disregarding any incomplete ones. It is crucial to maintain the precise formatting, word choice, and punctuation exactly as presented in the original prompt. Here are some parsed output you can refer to.
"""  # noqa: E501

METAPROMPT_EXAMPLES = [
    (
        """I am trying to cluster entity strings on Wikipedia according to the Wikipedia article title they refer to. To help me with this, for a given entity name, please provide me with a comprehensive set of alternative names that could refer to the same entity. Entities may be weirdly truncated or ambiguous - e.g. "Wind" may refer to the band "Earth, Wind, and Fire" or to "rescue service". For each entity, I will provide you with a sentence where this entity is used to help you understand what this entity refers to. Generate a comprehensive set of alternate entity names as a JSON-formatted list.

Entity: "fictional character"
Context Sentence: "Jenna Marshall is a fictional character created by Sara Shepard for the `` Pretty Little Liars '' book series , and later developed for the Freeform television series adaptation by I. Marlene King and portrayed by Tammin Sursok ."
Alternate Entity Names: ["fictional characters", "characters", "character"]

Entity: "Catholicism"
Context Sentence: "At home , significantly more electorate residents spoke Italian , Cantonese , Mandarin and Greek at home , and whilst the top three religions (Catholicism , no religion and Anglicanism) differed little from other parts of Perth , Buddhism and Eastern Orthodox adherents outnumbered those of the Uniting Church ."
Alternate Entity Names: ["Catholic Church", "Roman Catholic", "Catholic"]

Entity: "Wind"
Context Sentence: "Illinois musicians with a # 1 Billboard Hot 100 hit include artists from the 1950s : Sam Cooke (d. 1964) ; from the 1960s : The Buckinghams ; from the 1970s : Earth , Wind & Fire , The Chi-Lites , The Staple Singers , Minnie Riperton , Styx ; from the 1980s : Chicago , Cheap Trick , REO Speedwagon , Survivor , Richard Marx ; from the 1990s : R. Kelly ; from the 2000s : Kanye West , Twista , Plain White T 's ."
""",  # noqa: E501
        {
            "Instruction": """I am trying to cluster entity strings on Wikipedia according to the Wikipedia article title they refer to. To help me with this, for a given entity name, please provide me with a comprehensive set of alternative names that could refer to the same entity. Entities may be weirdly truncated or ambiguous - e.g. "Wind" may refer to the band "Earth, Wind, and Fire" or to "rescue service". For each entity, I will provide you with a sentence where this entity is used to help you understand what this entity refers to. Generate a comprehensive set of alternate entity names as a JSON-formatted list.""",  # noqa: E501
            "Demonstrations": """Entity: "fictional character"
Context Sentence: "Jenna Marshall is a fictional character created by Sara Shepard for the `` Pretty Little Liars '' book series , and later developed for the Freeform television series adaptation by I. Marlene King and portrayed by Tammin Sursok ."
Alternate Entity Names: ["fictional characters", "characters", "character"]

Entity: "Catholicism"
Context Sentence: "At home , significantly more electorate residents spoke Italian , Cantonese , Mandarin and Greek at home , and whilst the top three religions (Catholicism , no religion and Anglicanism) differed little from other parts of Perth , Buddhism and Eastern Orthodox adherents outnumbered those of the Uniting Church ."
Alternate Entity Names: ["Catholic Church", "Roman Catholic", "Catholic"]""",  # noqa: E501
        },
    ),
    (
        """You are an expert baker answering users' questions. Reply as agent.

Example conversation:

User: Hey can you help me with something

Agent: Sure! What do you need help with?

User: I want to bake a cake but don't know what temperature to set the oven to.

Agent: For most cakes, the oven should be preheated to 350°F (177°C).

Current conversation:

User: [Insert user's question]

Agent:""",
        {
            "Instruction": (
                "You are an expert baker answering users' "
                + "questions. Reply as agent."
            ),
            "Demonstrations": """User: Hey can you help me with something

Agent: Sure! What do you need help with?

User: I want to bake a cake but don't know what temperature to set the oven to.

Agent: For most cakes, the oven should be preheated to 350°F (177°C).""",
        },
    ),
    (
        "You are given a list of integers. A list is shown by comma-separated numbers between two brackets. For example, [7,3,6] is a list. The number in location one is 7, the number in location two is 3, and the number in location three is 6. You should answer with a list such that every element at each location is equal to the product of elements at every other location in the input array. For example, if a list has four numbers, the answer you give should be created like this: First element of your list = product of second, third, and fourth elements in the given list. Second element of your list = product of First, third and fourth elements in the given list, etc.",  # noqa: E501
        {
            "Instruction": "You are given a list of integers. A list is shown by comma-separated numbers between two brackets. For example, [7,3,6] is a list. The number in location one is 7, the number in location two is 3, and the number in location three is 6. You should answer with a list such that every element at each location is equal to the product of elements at every other location in the input array. For example, if a list has four numbers, the answer you give should be created like this: First element of your list = product of second, third, and fourth elements in the given list. Second element of your list = product of First, third and fourth elements in the given list, etc.",  # noqa: E501
            "Demonstrations": "N/A",
        },
    ),
    (
        "I am learning Japanese. Please translate some Japanese sentences to English. For example, Japanese: その日、人類は思い出した。ヤツらに支配されていた恐怖を鳥籠の中に囚われていた屈辱を English: On that day, humanity remembered the fear of being dominated by them and the humiliation of being trapped in a birdcage.",  # noqa: E501
        {
            "Instruction": "I am learning Japanese. Please translate some Japanese sentences to English.",  # noqa: E501
            "Demonstrations": "Japanese: その日、人類は思い出した。ヤツらに支配されていた恐怖を鳥籠の中に囚われていた屈辱を English: On that day, humanity remembered the fear of being dominated by them and the humiliation of being trapped in a birdcage.",  # noqa: E501
        },
    ),
    (
        "来到美国后，我需要学习如何自己做饭。你能告诉我一些菜需要准备的原料么？这里有一些例子：1. 菜名：西红柿炒蛋。原料：2. 菜名：青椒肉丝炒肉。原料：瘦肉、青椒、调味料（如大蒜、姜、料酒、生抽、盐、糖、鸡精或味精、胡椒粉）、植物油。",  # noqa: E501
        {
            "Instruction": "来到美国后，我需要学习如何自己做饭。你能告诉我一些菜需要准备的原料么？",  # noqa: E501
            "Demonstrations": "2. 菜名：青椒肉丝炒肉。原料：瘦肉、青椒、调味料（如大蒜、姜、料酒、生抽、盐、糖、鸡精或味精、胡椒粉）、植物油。",  # noqa: E501
        },
    ),
    (
        "As a programer, I am learning software development. Here are some of my problems. Input: What is CI/CD? Output: CI/CD is a way to automate and speed up software development by continuously integrating code changes and deploying them quickly and reliably. Input: What is Git? Output:",  # noqa: E501
        {
            "Instruction": "As a programer, I am learning software development. Here are some of my problems.",  # noqa: E501
            "Demonstrations": " Input: What is CI/CD? Output: CI/CD is a way to automate and speed up software development by continuously integrating code changes and deploying them quickly and reliably.",  # noqa: E501
        },
    ),
]


def construct_single_demonstration(
    user_prompt: str,
    parse_dict: dict[str, str] | None,
    input_only: bool = False,
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
    input_part = f'''Prompt: """\n{user_prompt}\n"""\n\nParsed Outputs:\n'''
    if input_only:
        return input_part
    output_part = json.dumps(parse_dict, ensure_ascii=False)
    return input_part + output_part


def construct_prompt_for_instruction_parsing(user_prompt: str) -> str:
    """A (GPT-3) prompt for separating instructions from demonstrations.

    Args:
        user_prompt: A user-generated prompt asking for a response.

    Returns:
        A prompt to instruct GPT-3 to parse the user's provided prompt.
    """
    prompt_sections = [METAPROMPT_INSTRUCTION]
    for prompt, correct_parse in METAPROMPT_EXAMPLES:
        prompt_sections.append(
            construct_single_demonstration(prompt, correct_parse, input_only=False)
        )
    all_prompts = "\n\n------\n\n".join(prompt_sections) + "\n\n------\n\n"
    user_input = construct_single_demonstration(user_prompt, None, input_only=True)
    all_prompts += (
        "After seeing these parsed output, please parse this prompt:\n\n" + user_input
    )
    return all_prompts
