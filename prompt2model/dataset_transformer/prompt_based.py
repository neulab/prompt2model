"""A simple dataset transformer that uses a plan prompt and transform prompt."""

from __future__ import annotations

import asyncio

import datasets

from prompt2model.dataset_transformer.base import DatasetTransformer
from prompt2model.dataset_transformer.prompt_template import (
    construct_prompt_for_plan,
    construct_prompt_for_transform_data,
)
from prompt2model.prompt_parser import PromptSpec
from prompt2model.utils import (
    API_ERRORS,
    api_tools,
    get_formatted_logger,
    handle_api_error,
)
from prompt2model.utils.parse_json_responses import (
    extract_response,
    make_request_from_prompt,
)

logger = get_formatted_logger("DatasetTransformer")


class PromptBasedDatasetTransformer(DatasetTransformer):
    """A class for transforming given dataset to required format using a plan and a transform prompt."""  # noqa E501

    def __init__(
        self,
        plan_prompt_fn=construct_prompt_for_plan,
        transform_prompt_fn=construct_prompt_for_transform_data,
    ):
        """Initialize the class."""
        self.plan_prompt_fn = plan_prompt_fn
        self.transform_prompt_fn = transform_prompt_fn
        self.plan = None

    def canonicalize_dataset_using_samples(
        self,
        inputs: list[str],
        outputs: list[str],
    ) -> datasets.DatasetDict:
        """Canonicalize a dataset into a suitable text-to-text format."""
        dataset_dict = {}
        dataset_dict["train"] = datasets.Dataset.from_dict(
            {"input_col": inputs, "output_col": outputs}
        )
        return datasets.DatasetDict(dataset_dict)

    def transform_data(
        self,
        prompt_spec: PromptSpec,
        dataset: datasets.Dataset,
        num_transform: int,
    ) -> datasets.DatasetDict:
        """Transform the dataset into the required format according to the prompt_spec and dataset."""  # noqa E501
        # 1. Use the prompt_spec and an example row from the dataset to create a "plan" for the data transformation. # noqa E501
        plan_prompt = self.plan_prompt_fn(
            task_description=prompt_spec.instruction,
            dataset=dataset,
            example=prompt_spec.examples,
        )
        self.plan = make_request_from_prompt(plan_prompt)

        print(f"plan: {self.plan}")

        # 2. Use the prompt_spec and the plan to transform each row of the dataset into the required format. # noqa E501
        inputs = []
        outputs = []

        required_keys = ["input", "output"]

        max_len = min(num_transform, len(dataset))
        len_count = 0
        transform_prompts = []
        for row in dataset:
            transform_prompt = self.transform_prompt_fn(
                task_description=prompt_spec.instruction,
                dataset_row=row,
                example=prompt_spec.examples,
                plan=self.plan,
            )
            transform_prompts.append(transform_prompt)

            len_count += 1
            if len_count >= max_len:
                break

        async def generate_responses(transform_prompts):
            responses = await api_tools.default_api_agent.generate_batch_completion(
                transform_prompts,
                temperature=0,
                responses_per_request=1,
                requests_per_minute=15,
            )
            return responses

        try:
            loop = asyncio.get_event_loop()
            responses = loop.run_until_complete(generate_responses(transform_prompts))
        except API_ERRORS as e:
            handle_api_error(e)

        for response in responses:
            try:
                extraction = extract_response(response, required_keys, [])
                if extraction is not None:
                    inputs.append(str(extraction["input"]))
                    outputs.append(str(extraction["output"]))
            except Exception as e:
                logger.error(f"Error extracting from response: {response}\nError: {e}")
                continue

        logger.info(f"Requested length: {max_len}\nActual length: {len(inputs)}\n")

        return self.canonicalize_dataset_using_samples(inputs, outputs)
