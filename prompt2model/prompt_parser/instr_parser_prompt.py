"""Utilities to construct an LLM "metaprompt" for our instruction parser."""

from __future__ import annotations  # noqa FI58

import json

METAPROMPT_INSTRUCTION = (
    "Prompts are task descriptions given to an AI language"
    + " model to guide its responses. They usually consist of"
    + " an 'instruction' detailing the task and, optionally, a few"
    + " 'demonstrations' that serve as examples of the task. I"
    + " want to break down prompts into these two components."
    + "For each prompt, the response should be a JSON dictionary"
    + " with two fields: the 'Instruction' and the 'Demonstrations'."
    + " If there are no demonstrations, return 'N/A' for the demonstrations"
    + " field. When demonstrations are available, include only those"
    + " examples that provide a complete input-output pair, and ignore"
    + " those that are incomplete and intended to be finished by the AI."
    + " The formatting, word choice, and punctuation should match that"
    + " of the original prompt."
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
        "我正在学习计算机网络空间安全这门课程，我希望你能帮我解释一些概念。比如 IP 分片污染攻击的基本原理：IP 分片是一种位于网络层的机制，其主要目的是解决 IP 分组在不同最大传输单元（MTU）网络中的传输问题。然而，在某些情况下，网络层的 IP 分片机制可能会被攻击者利用来破坏和污染原始的网络数据流。如果攻击者能够被动地监视，或者主动触发源主机和目标主机之间的IP分片，那么攻击者就有可能伪装成源主机，制造恶意的 IP 分片，并将其注入源主机和目标主机之间的数据流中，从而污染原始流量，对目标主机进行攻击。DNS请求洪水攻击：DNS请求洪水攻击是一种攻击手段，其中黑客通过控制僵尸网络向DNS服务器发送大量不存在的域名解析请求，最终导致服务器因处理大量DNS请求而超载，无法继续响应正常用户的DNS请求，从而实现攻击目标。在这种攻击中，攻击源可能是虚假的，也可能是真实的；攻击目标可能是DNS授权服务器，也可能是DNS缓存服务器。因此，针对不同类型的攻击源，需要采取不同的防御策略。",  # noqa: E501
        {
            "Instruction": "我正在学习计算机网络空间安全这门课程，我希望你能帮我解释一些概念。",  # noqa: E501
            "Demonstrations": "IP 分片污染攻击的基本原理：IP 分片是一种位于网络层的机制，其主要目的是解决 IP 分组在不同最大传输单元（MTU）网络中的传输问题。然而，在某些情况下，网络层的 IP 分片机制可能会被攻击者利用来破坏和污染原始的网络数据流。如果攻击者能够被动地监视，或者主动触发源主机和目标主机之间的IP分片，那么攻击者就有可能伪装成源主机，制造恶意的 IP 分片，并将其注入源主机和目标主机之间的数据流中，从而污染原始流量，对目标主机进行攻击。DNS请求洪水攻击：DNS请求洪水攻击是一种攻击手段，其中黑客通过控制僵尸网络向DNS服务器发送大量不存在的域名解析请求，最终导致服务器因处理大量DNS请求而超载，无法继续响应正常用户的DNS请求，从而实现攻击目标。在这种攻击中，攻击源可能是虚假的，也可能是真实的；攻击目标可能是DNS授权服务器，也可能是DNS缓存服务器。因此，针对不同类型的攻击源，需要采取不同的防御策略。",  # noqa: E501
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
    prediction_template = construct_single_demonstration(
        user_prompt, None, input_only=True
    )
    prompt_sections.append(prediction_template)
    return "\n\n------\n\n".join(prompt_sections)
