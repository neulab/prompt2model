"""Utilities to construct an LLM "metaprompt" for our dataset generator."""

import random

LONG_PROMPT_TEMPLATE = """
As a DatasetGenerator, your task is to generate a new example (input and output) based on the [new instruction] and [few-shot examples]. Please provide a JSON dictionary response that includes the new input and its corresponding output. Use the `input` and `output` keys in the dictionary. The 'input' field should be marked as 'N/A' if the instruction doesn't require additional input.
Try you best to ensure that the input and output you generate are distinct from the provided examples while maintaining a diverse, detailed, precise, comprehensive, and high-quality response.

--------------------------------------------------------------------------------------------
Here are some exmaples you can refer to:

- Example 1

{example_1}

- Example 2

{example_2}

- Example 3

{example_3}

- Example 4

{example_4}
--------------------------------------------------------------------------------------------

[new instruction]:
{instruction}

[few-shot examples]:
{few_shot_example_string}

[new example (in JSON)]:"""  # noqa: E501

SHORT_PROMPT_TEMPLATE = """
As a DatasetGenerator, your task is to generate a new example (input and output) based on the [new instruction] and [few-shot examples]. Please provide a JSON dictionary response that includes the new input and its corresponding output. Use the `input` and `output` keys in the dictionary. The 'input' field should be marked as 'N/A' if the instruction doesn't require additional input. It is important that the input and output you generate are distinct from the examples provided. Please ensure that your response is diverse, detailed, precise, comprehensive, and of high-quality.

--------------------------------------------------------------------------------------------
Here are some exmaples you can refer to:

- Example 1

{example_1}

- Example 2

{example_2}
--------------------------------------------------------------------------------------------

[new instruction]:
{instruction}

[few-shot examples]:
{few_shot_example_string}

[new example (in JSON)]:"""  # noqa: E501

# String templates for the prompt. Can be modified by the users.
# Prompt_template must contains `instruction` and `examples` fields.
# The LONG_PROMPT_TEMPLATE is used when random_example_num < 5.
# The SHORT_PROMPT_TEMPLATE is used when random_example_num >= 5.
# To save the price of calling OPENAI's API.

META_EXAMPLES = [
    """instruction: Which exercises are best for reducing belly fat at home?
input: N/A
output:
- Lying Leg Raises
- Leg In And Out
- Plank
- Side Plank
- Sit-ups""",
    """instruction: Extract all the country names in the paragraph, and list them separated by commas.
input: Dr. No is the sixth novel by the English author Ian Fleming to feature his British Secret Service agent James Bond. Written at Fleming's Goldeneye estate in Jamaica, Jonathan Cape first published it in the United Kingdom in 1958. In the novel Bond looks into the disappearance in Jamaica of two fellow MI6 operatives who had been investigating Doctor No. Bond travels to No's Caribbean island and meets Honeychile Rider, who is there to collect shells. They are captured and taken to a luxurious facility carved into a mountain. The character of Doctor No, the son of a German missionary and a Chinese woman, was influenced by Sax Rohmer's Fu Manchu stories. Dr. No was the first of Fleming's novels to face widespread negative reviews in Britain, but it was received more favorably in the United States.
output: English, British, Jamaica, the United Kingdom, German, Chinese, Britain, the United States.""",  # noqa: E501
    """instruction: Converting 85 F to Celsius.
input: N/A
output: 85°F = 29.44°C""",  # noqa: E501
    """instruction: Sort the given list ascendingly.
input: [10, 92, 2, 5, -4, 92, 5, 101]
output: [-4, 2, 5, 5, 10, 92, 92, 101]""",  # noqa: E501
    """instruction: Suggest a better and more professional rephrasing of the following sentence.
input: This house is surprisingly not constructed very well, and you probably need more money to fix it after you buy it. If you ask me, I would suggest you consider other candidates.
output: This house does not seem to be constructed well, so you may need to spend more money to fix it after you purchase it. I would suggest that you look at other properties.""",  # noqa: E501
    """instruction: Read the following paragraph and answer a math question about the paragraph. You need to write out the calculation to get the final answer.
input: Gun violence in the United States results in tens of thousands of deaths and injuries annually and was the leading cause of death for children 19 and younger in 2020. In 2018, the most recent year for which data are available as of 2021, the Centers for Disease Control and Prevention's (CDC) National Center for Health Statistics reports 38,390 deaths by firearm, of which 24,432 were by suicide. The rate of firearm deaths per 100,000 people rose from 10.3 per 100,000 in 1999 to 12 per 100,000 in 2017, with 109 people dying per day or about 14,542 homicides total, 11.9 per 100,000 in 2018. In 2010, there were 19,392 firearm-related suicides and 11,078 firearm-related homicides in the U.S. In 2010, 358 murders were reported involving a rifle, while 6,009 were reported involving a handgun; another 1,939 were reported with an unspecified type of firearm. In 2011, a total of 478,400 fatal and nonfatal violent crimes were committed with a firearm. How many more firearm-related deaths were there in 2018 compared to 2010?
output:
38390 - (19392 + 11078) = 38390 - 30470 = 7920. So, in 2018, there were 7920 more deaths by firearm than in 2010.""",  # noqa: E501
    """instruction: Write Python code to solve this leet code problem.
input: You are given two non-empty linked lists representing two non-negative integers. The digits are stored in reverse order, and each of their nodes contains a single digit. Add the two numbers and return the sum as a linked list. You may assume the two numbers do not contain any leading zero except the number 0 itself.
output:
class Solution(object):
    def addTwoNumbers(self, l1, l2):
        carry = 0
        root = n = ListNode(0)
        while l1 or l2 or carry:
            v1 = v2 = 0
            if l1:
                v1 = l1.val
                l1 = l1.next
            if l2:
                v2 = l2.val
                l2 = l2.next
            carry, val = divmod(v1+v2+carry, 10)
            n.next = ListNode(val)
            n = n.next
        return root.next
""",  # noqa: E501
    """instruction: Solve the equation and find the value of X. Show your steps.
input: 10X + 5 = 10
output: 10X = 5,  X = 0.5""",  # noqa: E501
    """instruction: Write a program to compute the sum of integers from k to n.
input: N/A
output:
def sum(k, n):
    sum = 0
    for i in range(k, n+1):
        sum += i
    return sum""",  # noqa: E501
    """instruction: Select the oldest person from the given list.
input: George Washington, Confucius, Michael Jordan, Michelangelo
output: Confucious""",  # noqa: E501
    """instruction: Turn down a job offer by sending an email to a recruiter explaining the reason.
input: N/A
output: Hi  Recruiter,
Thank you so much for the generous offer to join your team. As we discussed, I've admired the company for a number of years, and am a proud endorser of its products. However, after further consideration of where I currently am in my career, I've decided to accept an offer at another company.
I would love to stay in touch with you and have already started following you on [Social Media Platform]. Again, thank you so much for your time and consideration.
Thanks again,
Your Name""",  # noqa: E501
]


def construct_meta_prompt(
    instruction: str = None,
    few_shot_example_string: str = None,
    use_long_template: bool = True,
) -> str:
    """Constructs a prompt template for the dataset generator.

    Args:
        instruction: The natural language instruction for the prompt.
        examples_string: A string representing the few-shot examples.
        use_long_template: If True, uses the LONG_PROMPT_TEMPLATE,
            otherwise uses the SHORT_PROMPT_TEMPLATE.

    Returns:
        str: A prompt template, where the `instruction` and `examples` fields
            are filled in.
    """
    if use_long_template:
        meta_examples = random.sample(META_EXAMPLES, 4)
        example_1, example_2, example_3, example_4 = meta_examples
        return LONG_PROMPT_TEMPLATE.format(
            example_1=example_1,
            example_2=example_2,
            example_3=example_3,
            example_4=example_4,
            instruction=instruction,
            few_shot_example_string=few_shot_example_string,
        )
    else:
        meta_examples = random.sample(META_EXAMPLES, 2)
        example_1, example_2 = meta_examples
        return SHORT_PROMPT_TEMPLATE.format(
            example_1=example_1,
            example_2=example_2,
            instruction=instruction,
            few_shot_example_string=few_shot_example_string,
        )
