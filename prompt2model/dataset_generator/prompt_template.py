"""Utilities to construct an LLM "metaprompt" for our dataset generator."""

import random

COMPLEX_PROMPT_TEMPLATE = """
{META_PROMPT}
--------------------------------------------------------------------------------------------
Here are some examples you can refer to:

- Example 1

{example_1}

- Example 2

{example_2}

- Example 3

{example_3}

- Example 4

{example_4}
--------------------------------------------------------------------------------------------
Here is the requirement for the generation of a new example:

[new instruction]:
{instruction}
---------------------------------------------------------------------------------------------
Here are some [high-quality examples] for the [new instruction]. These examples can provide you with very strict format requirements. You should pay extreme attention to them!!!

[high-quality examples]:
{high_quality_example_string}
---------------------------------------------------------------------------------------------
These are some [low-quality examples]. Their formats and contents may not be accurate. Please strictly follow the format of the [high-quality examples], but you may also refer to the content of the [low-quality examples].

[low-quality examples]:
{low_quality_example_string}
---------------------------------------------------------------------------------------------
Before generating a new example, ensure that you strictly adhere to the rules mentioned in the [new instruction] and follow the format of the [high-quality examples]. Even if there are conflicts between [low-quality examples] and [new instruction], prioritize the [new instruction] guidelines to maintain consistency and quality. Think twice before generating a new example.

[new example (in JSON)]:"""  # noqa E501

MIDDLE_PROMPT_TEMPLATE = """
{META_PROMPT}
--------------------------------------------------------------------------------------------
Here are some examples you can refer to:

- Example 1

{example_1}

- Example 2

{example_2}

- Example 3

{example_3}
--------------------------------------------------------------------------------------------
Here is the requirement for the generation of a new example:

[new instruction]:
{instruction}
---------------------------------------------------------------------------------------------
Here are some [high-quality examples] for the [new instruction]. These examples can provide you with very strict format requirements. You should pay extreme attention to them!!!

[high-quality examples]:
{high_quality_example_string}
---------------------------------------------------------------------------------------------
These are some [low-quality examples]. Their formats and contents may not be accurate. Please strictly follow the format of the [high-quality examples], but you may also refer to the content of the [low-quality examples].

[low-quality examples]:
{low_quality_example_string}
---------------------------------------------------------------------------------------------
Before generating a new example, ensure that you strictly adhere to the rules mentioned in the [new instruction] and follow the format of the [high-quality examples]. Even if there are conflicts between [low-quality examples] and [new instruction], prioritize the [new instruction] guidelines to maintain consistency and quality. Think twice before generating a new example.

[new example (in JSON)]:"""  # noqa E501

SIMPLE_PROMPT_TEMPLATE = """
{META_PROMPT}
--------------------------------------------------------------------------------------------
Here are some examples you can refer to:

- Example 1

{example_1}

- Example 2

{example_2}
--------------------------------------------------------------------------------------------
Here is the requirement for the generation of a new example:

[new instruction]:
{instruction}
---------------------------------------------------------------------------------------------
Here are some [high-quality examples] for the [new instruction]. These examples can provide you with very strict format requirements. You should pay extreme attention to them!!!

[high-quality examples]:
{high_quality_example_string}
---------------------------------------------------------------------------------------------
These are some [low-quality examples]. Their formats and contents may not be accurate. Please strictly follow the format of the [high-quality examples], but you may also refer to the content of the [low-quality examples].

[low-quality examples]:
{low_quality_example_string}
---------------------------------------------------------------------------------------------
Before generating a new example, ensure that you strictly adhere to the rules mentioned in the [new instruction] and follow the format of the [high-quality examples]. Even if there are conflicts between [low-quality examples] and [new instruction], prioritize the [new instruction] guidelines to maintain consistency and quality. Think twice before generating a new example.

[new example (in JSON)]:"""  # noqa E501

# String templates for the prompt. Can be modified by the users.
# Prompt_template must contains `instruction` and `examples` fields.
# The COMPLEX_PROMPT_TEMPLATE is used when random_example_num < 5.
# The SIMPLE_PROMPT_TEMPLATE is used when random_example_num >= 5.
# To save the price of making API calls.

META_PROMPT = """
As a DatasetGenerator, your task is to generate a new example (`input`, 'explanation' and `output`) based on the [new instruction] and [few-shot examples]. Please provide a JSON dictionary response that includes the new `input` and its corresponding 'explanation' and `output`. Use the `input`,'explanation' and `output` keys in the dictionary. The 'input' field should be marked as 'N/A' if the instruction doesn't require additional input.

Try you best to ensure that the input, explanation and output you generate are distinct from the provided examples while maintaining a diverse, detailed, precise, comprehensive, and high-quality response.

Avoid generating examples that are the same as the provided examples.
"""  # noqa E501


META_EXAMPLES = [
    """instruction: I am learning Japanese. Please translate some Japanese sentences to English.
input=\"その日、人類は思い出した。ヤツらに支配されていた恐怖を鳥籠の中に囚われていた屈辱を\"
explanation=\"The input is a Japanese sentence which is conveying that on that day, humanity remembered the fear of being dominated by them and the humiliation of being trapped in a birdcage.\"
output=\"On that day, humanity remembered the fear of being dominated by them and the humiliation of being trapped in a birdcage.\"""",  # noqa E501
    """instruction: As a programer, I am learning software development. Here are some of my problems.
input=\"What is CI/CD?\"
explanation=\"The input is a question asking about what the term CI/CD mean. So the output should be the xplanation of CI/CD, which is way to automate and speed up the sofwatre devolopment by efficient integration and deployment of the code changes\"
output=\"CI/CD is a way to automate and speed up software development by continuously integrating code changes and deploying them quickly and reliably.\"""",  # noqa E501
    """instruction: 来到美国后，我需要学习如何自己做饭。你能告诉我一些菜需要准备的原料么？
input=\"青椒肉丝炒肉\"
explanation=\"The instruction is to provide the ingredients for the input dish, "青椒肉丝炒肉" which appears to be a Chinese dish, commonly known as "Stir-Fried Pork with Green Peppers. Thus the output should be a list of ingredients used in preparing this dish: "Lean meat, green peppers, seasonings (such as garlic, ginger, cooking wine, light soy sauce, salt, sugar, chicken bouillon or monosodium glutamate, pepper), vegetable oil."\"
output=\"瘦肉、青椒、调味料（如大蒜、姜、料酒、生抽、盐、糖、鸡精或味精、胡椒粉）、植物油。\"""",  # noqa E501
    """instruction: Classify the sentiment of the sentence into positive, negative, or mixed.
input=\"I enjoy the flavor of the restaurant but their service is too slow.\"
explanation=\"Since the input indicates that the person enjoys flavor of the restaurant, but does not like the slow service, the sentiment of the sentence should be mixed \"
output=\"mixed\"""",  # noqa E501
    """instruction: Given a dialogue, classify whether the user is satisfied with the service. You should respond with "Satisfied" or "Unsatisfied".
input=\"
- Agent: Thank you for your feedback. We will work to improve our service in the future.
- Customer: I am happy with the service you provided. Thank you for your help.
\"
explanation=\"Since the customer is happy with the service provided by the agent and thanks them for the help, the user/customer is satisfied with the service\"
output=\"Satisfied\"""",  # noqa E501
    """instruction: Tell me if the following email is a promotion email or not. If the email is a promotion email, output Promotion. Otherwise, output Not Promotion.
input=\"We hope you are doing well. Let us know if you need any help..\"
explanation=\"Since the email is not promoting anything and merely checking if the user is doing well, it is not a promotional email\"
output=\"Not Promotion\"""",  # noqa E501
    """instruction: Detect if the Reddit thread contains hate speech. If the thread contains hate speech, output True. Otherwise, output False.
input=\"All people of color are stupid and should not be allowed to vote.\"
exlanation=\"The input is a clear indication of hate speech since it conveys a negative connotation that people of certain race are stupid and should not be allowed to vote"
output=\"True\"""",  # noqa E501
    """instruction: Does the information in the document supports the claim? You can answer "Support" or "Oppose".
input=\"Document: After a record-breaking run that saw mortgage rates plunge to all-time lows and home prices soar to new highs, the U.S. housing market finally is slowing. While demand and price gains are cooling, any correction is likely to be a modest one, housing economists and analysts say. No one expects price drops on the scale of the declines experienced during the Great Recession. Claim: The US housing market is going to crash soon.\"
explanation=\"The document suggests that the housing market has cooled down after seeing a record drop in mortgage rates and the home prices to new highs. It also notes that despite this any correction to the market will be a modest one and the prices will not crash as much as during the Great Recession. Hence, the document does not support the claim that US housing market is going to crash soon\"
output=\"Oppose\"""",  # noqa E501
    """instruction: You need to read a code and detect if there is a syntax error or not. Output true if there is an error, output false if there is not.
input=\"
def calculate_average(numbers):
    total = 0
    for number in numbers:
        total += number
    return total / len(numbers)
\"
explanation=\"Since there are no syntax error in the input the output should be false\"
output=\"false\"""",  # noqa E501
    """instruction: You are provided with a news article, and you need to identify all the categories that this article belongs to. Possible categories include Sports and Politics. Output its categories one by one, separated by a comma.
input=\"The Golden State Warriors have won the NBA championship for the second year in a row.\"
explanation=\"The input suggests that the team Golden State Warriors won the NBA championship which is a basketball league, hence falling under the Sports category\"
output=\"Sports\"""",  # noqa E501
    """instruction: Tell me what's the second largest city by population in Canada.
input=\"N/A\"
explanation=\"This is a fact based question, asking for second largest city by popoulation in Canada and hence the answer is Montreal\"
output=\"Montreal\"""",  # noqa E501
    """instruction: Classifying different types of mathematical equations, such as linear, and quadratic equations, based on the coefficients and terms in the equation.
input=\"y = x^2 - 4x + 3\"
explanation=\"The highest degree of x in the equation given is 2 and hence it is a quadratic equation\"
output=\"Quadratic equation\"""",  # noqa E501
    """instruction: Tell me the first number of the given list.
input=\"[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]\"
explanation=\"The first number on the list is 1 and hence the output is 1\"
output=\"1\"""",  # noqa E501
    """instruction: Which exercises are best for reducing belly fat at home?
input=\"N/A\"
explanation=\"The instruction asks for set of excercises that are best for reducing belly fat and hence the answer could be plank, sit-ups etc\"
output=\"
- Lying Leg Raises
- Leg In And Out
- Plank
- Side Plank
- Sit-ups
\"""",  # noqa E501
    """instruction: Extract all the country names in the paragraph, and list them separated by commas.
input=\"Dr. No is the sixth novel by the English author Ian Fleming to feature his British Secret Service agent James Bond. Written at Fleming's Goldeneye estate in Jamaica, Jonathan Cape first published it in the United Kingdom in 1958. In the novel Bond looks into the disappearance in Jamaica of two fellow MI6 operatives who had been investigating Doctor No. Bond travels to No's Caribbean island and meets Honeychile Rider, who is there to collect shells. They are captured and taken to a luxurious facility carved into a mountain. The character of Doctor No, the son of a German missionary and a Chinese woman, was influenced by Sax Rohmer's Fu Manchu stories. Dr. No was the first of Fleming's novels to face widespread negative reviews in Britain, but it was received more favorably in the United States.\"
output=\"English, British, Jamaica, the United Kingdom, German, Chinese, Britain, the United States.\"""",  # noqa: E501
    """instruction: Converting 85 F to Celsius.
input=\"N/A\"
explanation=\"The formula for converting Fahrenheit to Celcius is (°F − 32) × 5/9 = (85-32)*5/9 = 29.44°C\"
output=\"29.44°C\"""",  # noqa: E501
    """instruction: Sort the given list ascendingly.
input=\"[10, 92, 2, 5, -4, 92, 5, 101]\"
explanation=\"The instruction is to sort the list in ascending order, meaning lowest to highest. \"
output=\"[-4, 2, 5, 5, 10, 92, 92, 101]\"""",  # noqa: E501
    """instruction: Suggest a better and more professional rephrasing of the following sentence.
input=\"This house is surprisingly not constructed very well, and you probably need more money to fix it after you buy it. If you ask me, I would suggest you consider other candidates.\"
explanation=\"For a formal construction of the sentece, phrases like 'surprisingly' should be replaced with 'does not seem to be', etc\"
output=\"This house does not seem to be constructed well, so you may need to spend more money to fix it after you purchase it. I would suggest that you look at other properties.\"""",  # noqa: E501
    """instruction: Read the following paragraph and answer a math question about the paragraph. You need to write out the calculation to get the final answer.
input=\"Gun violence in the United States results in tens of thousands of deaths and injuries annually and was the leading cause of death for children 19 and younger in 2020. In 2018, the most recent year for which data are available as of 2021, the Centers for Disease Control and Prevention's (CDC) National Center for Health Statistics reports 38,390 deaths by firearm, of which 24,432 were by suicide. The rate of firearm deaths per 100,000 people rose from 10.3 per 100,000 in 1999 to 12 per 100,000 in 2017, with 109 people dying per day or about 14,542 homicides total, 11.9 per 100,000 in 2018. In 2010, there were 19,392 firearm-related suicides and 11,078 firearm-related homicides in the U.S. In 2010, 358 murders were reported involving a rifle, while 6,009 were reported involving a handgun; another 1,939 were reported with an unspecified type of firearm. In 2011, a total of 478,400 fatal and nonfatal violent crimes were committed with a firearm. How many more firearm-related deaths were there in 2018 compared to 2010?\"
explanation=\"38390 - (19392 + 11078) = 38390 - 30470 = 7920. So, in 2018, there were 7920 more deaths by firearm than in 2010.\"
output=\"7920\"""",  # noqa: E501
    """instruction: Write Python code to solve this leet code problem.
input=\"You are given two non-empty linked lists representing two non-negative integers. The digits are stored in reverse order, and each of their nodes contains a single digit. Add the two numbers and return the sum as a linked list. You may assume the two numbers do not contain any leading zero except the number 0 itself.\"
explanation=\"To add two numbers whose digits are stored in reverse order we need to iterate over each node of the 2 lists and add them, by updating the carry over and the value at the current digit using divmod.\"
output=\"
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
\"""",  # noqa: E501
    """instruction: Solve the equation and find the value of X. Show your steps.
input=\"10X + 5 = 10\"
explanation=\"10X+5=10, implies 10X=5, which implies X=0.5\"
output=\"0.5\"""",  # noqa: E501
    """instruction: Write a program to compute the sum of integers from k to n.
input=\"N/A\"
explanation=\"To find the sum of integers from k to n, we have to loop through each number starting from k and ending at n and add each of those numbers along the way.\"
output=\"
def sum(k, n):
    sum = 0
    for i in range(k, n+1):
        sum += i
    return sum
    \"""",  # noqa: E501
    """instruction: Select the oldest person from the given list.
input=\"George Washington, Confucius, Michael Jordan, Michelangelo\"
explanation=\"This is a fact-based question asking to choose the oldest person from the list and hence the answer should be Confucious, since other people from the list are born after him\"
output=\"Confucious\"""",  # noqa: E501
    """instruction: Turn down a job offer by sending an email to a recruiter explaining the reason.
input=\"N/A\"
explanation=\"The email to be sent to the recruiter should be a professional one, thanking them for the offer and saying a few good things about the company. Then there should be a clear explanation as to why the candidate is turning doen the job offer, so as to not burn bridges for the future.\"
output=\"Hi  Recruiter,
Thank you so much for the generous offer to join your team. As we discussed, I've admired the company for a number of years, and am a proud endorser of its products. However, after further consideration of where I currently am in my career, I've decided to accept an offer at another company.
I would love to stay in touch with you and have already started following you on [Social Media Platform]. Again, thank you so much for your time and consideration.
Thanks again,
Your Name\"""",  # noqa: E501
]


def construct_meta_prompt(
    instruction: str = None,
    low_quality_example_string: str = None,
    high_quality_example_string: str = None,
    template_type: str = "SIMPLE",
) -> str:
    """Constructs a prompt template for the dataset generator.

    Args:
        instruction: The natural language instruction for the prompt.
        low_quality_example_string: A string representing the low quality examples.
        high_quality_example_string: A string representing the high quality examples.
        template_type: If template_type is COMPLEX, uses the
        COMPLEX_PROMPT_TEMPLATE, if template_type is MIDDLE, uses the
        MIDDLE_PROMPT_TEMPLATE, and if template_type is SIMPLE,
        uses the SIMPLE_PROMPT_TEMPLATE.

    Returns:
        str: A prompt template, where the `instruction` and `examples` fields
            are filled in.
    """
    if template_type not in [
        "SIMPLE",
        "MIDDLE",
        "COMPLEX",
    ]:
        raise ValueError("template_type must be SIMPLE, MIDDLE, or COMPLEX")
    meta_examples = random.sample(META_EXAMPLES, 4)
    example_1, example_2, example_3, example_4 = meta_examples
    if template_type == "COMPLEX":
        return COMPLEX_PROMPT_TEMPLATE.format(
            # The META_PROMPT variable is a shared global variable.
            META_PROMPT=META_PROMPT,
            example_1=example_1,
            example_2=example_2,
            example_3=example_3,
            example_4=example_4,
            instruction=instruction,
            high_quality_example_string=high_quality_example_string,
            low_quality_example_string=low_quality_example_string,
        )
    elif template_type == "MIDDLE":
        return MIDDLE_PROMPT_TEMPLATE.format(
            META_PROMPT=META_PROMPT,
            example_1=example_1,
            example_2=example_2,
            example_3=example_3,
            instruction=instruction,
            high_quality_example_string=high_quality_example_string,
            low_quality_example_string=low_quality_example_string,
        )
    else:
        return SIMPLE_PROMPT_TEMPLATE.format(
            META_PROMPT=META_PROMPT,
            example_1=example_1,
            example_2=example_2,
            instruction=instruction,
            high_quality_example_string=high_quality_example_string,
            low_quality_example_string=low_quality_example_string,
        )
