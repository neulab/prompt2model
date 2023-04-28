"""An interface for prompt parsing."""

from __future__ import annotations  # noqa FI58

from enum import Enum

import openai

from prompt2model.prompt_parser.base import PromptSpec


class TaskType(Enum):
    """High-level taxonomy of possible NLP model outputs."""

    TEXT_GENERATION = 1
    CLASSIFICATION = 2
    SEQUENCE_TAGGING = 3
    SPAN_EXTRACTION = 4


# pylint: disable=too-few-public-methods
class OpenAIInstructionParser(PromptSpec):
    """Parse the prompt to separate instructions from task demonstrations."""

    def __init__(self, task_type: TaskType):
        """By default, assume that every task is a text generation task."""
        self.task_type = task_type
        self.instruction: str | None = None
        self.demonstration: str | None = None

    def get_prompt_for_instruction_parsing(self, prompt: str) -> tuple[str, str]:
        """A (GPT-3) prompt for separating instructions from demonstrations."""
        prefix = '''"Prompts" are a description of a task provided to an AI language model to guide its performance. Prompts typically consist of two components: a task "instruction" and, optionally, a few "demonstrations" (examples to illustrate the task). I want to segment prompts into these two components. For each prompt, return a line starting with "1) Instruction: " and a line starting with "2) Demonstrations: ". If no demonstration is provided, write "NO DEMONSTRATION.". When demonstrations are provided, only include examples where the full input-output pair is given; ignore partial examples written with the intent of being completed by the AI language model. Otherwise, match the formatting, word selection, and punctuation used in the original prompt.

------

Prompt: """
I am trying to cluster entity strings on Wikipedia according to the Wikipedia article title they refer to. To help me with this, for a given entity name, please provide me with a comprehensive set of alternative names that could refer to the same entity. Entities may be weirdly truncated or ambiguous - e.g. "Wind" may refer to the band "Earth, Wind, and Fire" or to "rescue service". For each entity, I will provide you with a sentence where this entity is used to help you understand what this entity refers to. Generate a comprehensive set of alternate entity names as a JSON-formatted list.

Entity: "fictional character"
Context Sentence: "Jenna Marshall is a fictional character created by Sara Shepard for the `` Pretty Little Liars '' book series , and later developed for the Freeform television series adaptation by I. Marlene King and portrayed by Tammin Sursok ."
Alternate Entity Names: ["fictional characters", "characters", "character"]

Entity: "Catholicism"
Context Sentence: "At home , significantly more electorate residents spoke Italian , Cantonese , Mandarin and Greek at home , and whilst the top three religions (Catholicism , no religion and Anglicanism) differed little from other parts of Perth , Buddhism and Eastern Orthodox adherents outnumbered those of the Uniting Church ."
Alternate Entity Names: ["Catholic Church", "Roman Catholic", "Catholic"]

Entity: "Wind"
Context Sentence: "Illinois musicians with a # 1 Billboard Hot 100 hit include artists from the 1950s : Sam Cooke (d. 1964) ; from the 1960s : The Buckinghams ; from the 1970s : Earth , Wind & Fire , The Chi-Lites , The Staple Singers , Minnie Riperton , Styx ; from the 1980s : Chicago , Cheap Trick , REO Speedwagon , Survivor , Richard Marx ; from the 1990s : R. Kelly ; from the 2000s : Kanye West , Twista , Plain White T 's ."
Alternate Entity Names: ["Earth & Fire", "Earth", "Wind & Fire"]
"""

Parsed Outputs:

1) Instruction:
I am trying to cluster entity strings on Wikipedia according to the Wikipedia article title they refer to. To help me with this, for a given entity name, please provide me with a comprehensive set of alternative names that could refer to the same entity. Entities may be weirdly truncated or ambiguous - e.g. "Wind" may refer to the band "Earth, Wind, and Fire" or to "rescue service". For each entity, I will provide you with a sentence where this entity is used to help you understand what this entity refers to. Generate a comprehensive set of alternate entity names as a JSON-formatted list.

2) Demonstrations:
Entity: "fictional character"
Context Sentence: "Jenna Marshall is a fictional character created by Sara Shepard for the `` Pretty Little Liars '' book series , and later developed for the Freeform television series adaptation by I. Marlene King and portrayed by Tammin Sursok ."
Alternate Entity Names: ["fictional characters", "characters", "character"]

Entity: "Catholicism"
Context Sentence: "At home , significantly more electorate residents spoke Italian , Cantonese , Mandarin and Greek at home , and whilst the top three religions (Catholicism , no religion and Anglicanism) differed little from other parts of Perth , Buddhism and Eastern Orthodox adherents outnumbered those of the Uniting Church ."
Alternate Entity Names: ["Catholic Church", "Roman Catholic", "Catholic"]

Entity: "Wind"
Context Sentence: "Illinois musicians with a # 1 Billboard Hot 100 hit include artists from the 1950s : Sam Cooke (d. 1964) ; from the 1960s : The Buckinghams ; from the 1970s : Earth , Wind & Fire , The Chi-Lites , The Staple Singers , Minnie Riperton , Styx ; from the 1980s : Chicago , Cheap Trick , REO Speedwagon , Survivor , Richard Marx ; from the 1990s : R. Kelly ; from the 2000s : Kanye West , Twista , Plain White T 's ."
Alternate Entity Names: ["Earth & Fire", "Earth", "Wind & Fire"]

-----

Prompt: """
You are an expert baker answering users' questions. Reply as agent.

Example conversation:

User: Hey can you help me with something

Agent: Sure! What do you need help with?

User: I want to bake a cake but don't know what temperature to set the oven to.

Agent: For most cakes, the oven should be preheated to 350°F (177°C).

Current conversation:

User: [Insert user's question]

Agent:
"""

Parsed Outputs:

1) Instruction:
You are an expert baker answering users' questions. Reply as agent.

2) Demonstrations:
User: Hey can you help me with something

Agent: Sure! What do you need help with?

User: I want to bake a cake but don't know what temperature to set the oven to.

Agent: For most cakes, the oven should be preheated to 350°F (177°C).

------

Prompt: """
You are given a list of integers. A list is shown by comma-separated numbers between two brackets. For example, [7,3,6] is a list. The number in location one is 7, the number in location two is 3, and the number in location three is 6. You should answer with a list such that every element at each location is equal to the product of elements at every other location in the input array. For example, if a list has four numbers, the answer you give should be created like this: First element of your list = product of second, third, and fourth elements in the given list. Second element of your list = product of First, third and fourth elements in the given list, etc.
"""

Parsed Outputs:

1) Instruction:
You are given a list of integers. A list is shown by comma-separated numbers between two brackets. For example, [7,3,6] is a list. The number in location one is 7, the number in location two is 3, and the number in location three is 6. You should answer with a list such that every element at each location is equal to the product of elements at every other location in the input array. For example, if a list has four numbers, the answer you give should be created like this: First element of your list = product of second, third, and fourth elements in the given list. Second element of your list = product of First, third and fourth elements in the given list, etc.

2) Demonstrations:
NO DEMONSTRATION.

------

'''  # noqa: E501
        filled_template = f'''Prompt: """\n{prompt}\n"""\n\nParsed Outputs: '''
        return prefix, filled_template

    def extract_response(self, response: openai.Completion) -> tuple[str, str | None]:
        """Parse stuctured fields from the OpenAI API response.

        Args:
            response (openai.Completion): OpenAI API response.

        Returns:
            tuple[str, str | None]: Tuple consisting of:
                1) Instruction: The instruction parsed from the API response.
                2) Demonstrations: (Optional) demonstrations parsed from the
                   API response.
        """
        response_text = ""
        return response_text, response_text

    def parse_from_prompt(self, prompt: str) -> None:
        """Parse the prompt into an instruction and demonstrations."""
        prefix, filled_template = self.get_prompt_for_instruction_parsing(prompt)

        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": f"{prefix}"},
                {"role": "user", "content": filled_template},
            ],
            temperature=0,
        )
        self.instruction, self.demonstration = self.extract_response(response)
