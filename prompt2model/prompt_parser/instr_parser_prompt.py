"""Utilities to construct an LLM "metaprompt" for our instruction parser."""

from __future__ import annotations  # noqa FI58

METAPROMPT_INSTRUCTION = (
    '"Prompts" are a description of a task provided to an AI language'
    " model to guide its performance. Prompts typically consist of two"
    ' components: a task "instruction" and, optionally, a few'
    ' "demonstrations" (examples to illustrate the task). I want to'
    " segment prompts into these two components. For each prompt, return"
    ' a line starting with "1) Instruction: " and a line starting with "2)'
    ' Demonstrations: ". If no demonstration is provided, write'
    ' "NO DEMONSTRATION.". When demonstrations are provided, only'
    " include examples where the full input-output pair is given; ignore"
    " partial examples written with the intent of being completed by the"
    " AI language model. Otherwise, match the formatting, word selection,"
    " and punctuation used in the original prompt."
)

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
Alternate Entity Names: ["Earth & Fire", "Earth", "Wind & Fire"]""",  # noqa: E501
        {
            "Instruction": """I am trying to cluster entity strings on Wikipedia according to the Wikipedia article title they refer to. To help me with this, for a given entity name, please provide me with a comprehensive set of alternative names that could refer to the same entity. Entities may be weirdly truncated or ambiguous - e.g. "Wind" may refer to the band "Earth, Wind, and Fire" or to "rescue service". For each entity, I will provide you with a sentence where this entity is used to help you understand what this entity refers to. Generate a comprehensive set of alternate entity names as a JSON-formatted list.""",  # noqa: E501
            "Demonstrations": """Entity: "fictional character"
Context Sentence: "Jenna Marshall is a fictional character created by Sara Shepard for the `` Pretty Little Liars '' book series , and later developed for the Freeform television series adaptation by I. Marlene King and portrayed by Tammin Sursok ."
Alternate Entity Names: ["fictional characters", "characters", "character"]

Entity: "Catholicism"
Context Sentence: "At home , significantly more electorate residents spoke Italian , Cantonese , Mandarin and Greek at home , and whilst the top three religions (Catholicism , no religion and Anglicanism) differed little from other parts of Perth , Buddhism and Eastern Orthodox adherents outnumbered those of the Uniting Church ."
Alternate Entity Names: ["Catholic Church", "Roman Catholic", "Catholic"]

Entity: "Wind"
Context Sentence: "Illinois musicians with a # 1 Billboard Hot 100 hit include artists from the 1950s : Sam Cooke (d. 1964) ; from the 1960s : The Buckinghams ; from the 1970s : Earth , Wind & Fire , The Chi-Lites , The Staple Singers , Minnie Riperton , Styx ; from the 1980s : Chicago , Cheap Trick , REO Speedwagon , Survivor , Richard Marx ; from the 1990s : R. Kelly ; from the 2000s : Kanye West , Twista , Plain White T 's ."
Alternate Entity Names: ["Earth & Fire", "Earth", "Wind & Fire"]""",  # noqa: E501
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
            "Demonstrations": "NO DEMONSTRATION.",
        },
    ),
]


def construct_single_demonstration(
    user_prompt: str,
    instruction: str | None,
    demonstrations: str | None,
    input_only: bool = False,
) -> str:
    """Format a demonstration or prediction example to give to an LLM."""
    if demonstrations is None:
        formatted_demonstrations = "NO DEMONSTRATIONS."
    else:
        formatted_demonstrations = demonstrations
    input_part = f'''Prompt: """\n{user_prompt}\n"""\n\nParsed Outputs:\n'''
    output_part = (
        f"""\n1) Instruction:\n{instruction}\n"""
        + f"""\n2) Demonstrations:\n{formatted_demonstrations}\n"""
    )
    if input_only:
        return input_part
    else:
        return input_part + output_part


def construct_full_parsing_prompt():
    """Construct a full prompt for the instruction parsing task."""
    prompt_sections = [METAPROMPT_INSTRUCTION]
    for prompt, correct_parse in METAPROMPT_EXAMPLES:
        instruction = correct_parse["Instruction"]
        demonstration = correct_parse["Demonstrations"]
        prompt_sections.append(
            construct_single_demonstration(prompt, instruction, demonstration)
        )
    return "\n\n------\n\n".join(prompt_sections)
