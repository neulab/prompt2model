"""Prompt template for dataset transformer."""

import json

CREATE_PLAN_PROMPT = """You are a Data Transforming Agent. You create a plan to transform data samples from their existing format into the required format for a given task.

Here is an example of your job as a Data Transforming Agent:

Task Description: Identify whether a claim is True or False or Neither based on a given context

input=Westlife: According to the British Phonographic Industry ( BPI ) , Westlife has been certified for 13 million albums and 9.8 million singles , with a total of more than 23 million combined sales in the UK . Claim: Westlife made under 23.5 million sales in the UK
output=Neither

Here is a sample from a potentially relevant dataset for the task above. Notice how the format below is not as required by the task above.

{{
    \"abstract\": \"[\"Alterations of the architecture of cerebral white matter in the developing human brain can affect cortical development and result in functional disabilities...\",
    \"doc_id\": 4983,
    \"structured\": false,
    \"title\": \"Microstructural development of human newborn cerebral white matter assessed in vivo by diffusion tensor magnetic resonance imaging.\"
}}

Propose a higher level plan to convert data from the potentially relevant dataset to data in the required format of the original task. Your plan should be a list of sequential steps that can be taken to perform the data transformation. Each step in the plan can take the following actions:
1. Expand on a particular data field.
2. Combine multiple data fields
3. Generate new data fields as relevant and required.
4. Choose data fields that will form "input" and data fields that will form "output"

Plan:
1. Combine "abstract" and "title" to create "context".
2. Create a "claim" field based on "context" using GPT-4 that is either supported or not supported by the "context".
3. Create an "output" field based on whether "claim" is supported or not by the "context". The value of "output" must be "True", "False", or "Neither".
4. Combine "context" and "claim" to create "input"

Perform the same job of a Data Transforming agent with the following task and dataset:

Task Description: {task_description}

{example}

Here is a sample from a potentially relevant dataset for the task above. Notice how the format below is not as required by the task above.

{dataset_row}

Propose a higher level plan to convert data from the potentially relevant dataset to data in the required format of the original task. Your plan should be a list of sequential steps that can be taken to perform the data transformation. Each step in the plan can take the following actions:
1. Expand on a particular data field.
2. Combine multiple data fields
3. Generate new data fields as relevant and required.
4. Choose data fields that will form "input" and data fields that will form "output"
"""  # noqa E501

TRANSFORM_DATA_PROMPT = """You are a Data Transforming Agent. Your job is to:
1. Read the task description.
2. Read an exemplar of what an input and output looks like for the task.
3. Read a particular data sample carefully that needs to be transformed such that it is relevant to the task described.
4. Read the data transformation plan carefully that will help you convert the particular data sample into a relevant format.
5. Respond with the transformed sample as a JSON response with exactly 2 fields: "input" and "output".

Here is one full example of the entire process:

Task Description: Identify whether a claim is True or False or Neither based on a given context

Exemplar:
input=Westlife: According to the British Phonographic Industry ( BPI ) , Westlife has been certified for 13 million albums and 9.8 million singles , with a total of more than 23 million combined sales in the UK . Claim: Westlife made under 23.5 million sales in the UK
output=Neither

Data Sample:
{{
    \"abstract\": \"[\"Alterations of the architecture of cerebral white matter in the developing human brain can affect cortical development and result in functional disabilities. A line scan diffusion-weighted magnetic resonance imaging (MRI) sequence with diffusion tensor analysis was applied to measure the apparent diffusion coefficient, to calculate relative anisotropy, and to delineate three-dimensional fiber architecture in cerebral white matter in preterm (n = 17) and full-term infants (n = 7). To assess effects of prematurity on cerebral white matter development, early gestation preterm infants (n = 10) were studied a second time at term. In the central white matter the mean apparent diffusion coefficient at 28 wk was high, 1.8 microm2/ms, and decreased toward term to 1.2 microm2/ms.\",
    \"doc_id\": 4983,
    \"structured\": false,
    \"title\": \"Microstructural development of human newborn cerebral white matter assessed in vivo by diffusion tensor magnetic resonance imaging.\"
}}

Plan:
1. Combine "abstract" and "title" to create "context".
2. Create a "claim" field based on "context" using GPT-4 that is either supported or not supported by the "context".
3. Create an "output" field based on whether "claim" is supported or not by the "context". The value of "output" must be "True", "False", or "Neither".
4. Combine "context" and "claim" to create "input"

Response:
{{
\"input\" : \"Microstructural development of human newborn cerebral white matter assessed in vivo by diffusion tensor magnetic resonance imaging. Alterations of the architecture of cerebral white matter in the developing human brain can affect cortical development and result in functional disabilities. A line scan diffusion-weighted magnetic resonance imaging (MRI) sequence with diffusion tensor analysis was applied to measure the apparent diffusion coefficient, to calculate relative anisotropy, and to delineate three-dimensional fiber architecture in cerebral white matter in preterm (n = 17) and full-term infants (n = 7). To assess effects of prematurity on cerebral white matter development, early gestation preterm infants (n = 10) were studied a second time at term. In the central white matter the mean apparent diffusion coefficient at 28 wk was high, 1.8 microm2/ms, and decreased toward term to 1.2 microm2/ms. Claim: The study found that preterm infants at term showed lower mean diffusion coefficients in the central white matter compared to full-term infants.\",
\"output\" : \"False\"
}}

Task Description: {task_description}

Exemplar:
{example}

Data Sample:
{dataset_row}

{plan}

Your response MUST be a JSON with exactly 2 fields: "input" and "output".

Response:
"""  # noqa E501


def truncate_row(example_row: dict, max_length=200) -> str:
    """Truncate the row before displaying if it is too long."""
    truncated_row = {}
    for key in example_row.keys():
        curr_row = json.dumps(example_row[key])
        truncated_row[key] = (
            curr_row
            if len(curr_row) <= max_length - 3
            else curr_row[:max_length] + "..."
        )
    return json.dumps(truncated_row)


def construct_prompt_for_plan(
    task_description: str, dataset: list[dict], example: str
) -> str:
    """Construct prompt for plan."""
    return CREATE_PLAN_PROMPT.format(
        task_description=task_description,
        dataset_row=truncate_row(dataset[0]),
        example=example,
    )


def construct_prompt_for_transform_data(
    task_description: str, dataset_row: dict, plan: str, example: str
) -> str:
    """Construct prompt for transform data."""
    return TRANSFORM_DATA_PROMPT.format(
        task_description=task_description,
        dataset_row=truncate_row(dataset_row),
        plan=plan,
        example=example,
    )
