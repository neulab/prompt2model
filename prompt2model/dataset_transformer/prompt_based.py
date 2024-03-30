"""A simple dataset transformer that uses a plan prompt and transform prompt."""
from __future__ import annotations

import asyncio
from collections.abc import Callable

import datasets
from typing import List, Tuple

from prompt2model.dataset_transformer.base import DatasetTransformer
from prompt2model.dataset_transformer.prompt_template import (
    construct_prompt_for_plan,
    construct_prompt_for_transform_data,
)
from prompt2model.dataset_retriever.task_expansion_prompt import (
    construct_prompt_for_task_explanation
)
from prompt2model.prompt_parser import PromptSpec
from prompt2model.utils import (
    API_ERRORS,
    api_tools,
    get_formatted_logger,
    handle_api_error,
)
from prompt2model.utils.parse_responses import (
    find_and_parse_json,
    make_single_api_request,
)

logger = get_formatted_logger("DatasetTransformer")


class PromptBasedDatasetTransformer(DatasetTransformer):
    """Transform data based on a transform prompt."""

    def __init__(
            self,
            num_points_to_transform: int ,
            max_allowed_failed_transforms: int,
            plan_prompt_fn: Callable[
                [str, list[dict], str], str
            ] = construct_prompt_for_plan,
            transform_prompt_fn: Callable[
                [str, dict, str, str], str
            ] = construct_prompt_for_transform_data,

        ):
            """
            Initializes an instance of the PromptBasedDatasetTransformer class.

            Args:
                num_points_to_transform (int): The number of points to transform.
                max_allowed_failed_transforms (int): The maximum number of failed transforms allowed.
                plan_prompt_fn (Callable[[str, list[dict], str], str], optional): The function to construct the prompt for plan data. Defaults to construct_prompt_for_plan.
                transform_prompt_fn (Callable[[str, dict, str, str], str], optional): The function to construct the prompt for transform data. Defaults to construct_prompt_for_transform_data.
            """
            
            self.plan_prompt_fn = plan_prompt_fn
            self.transform_prompt_fn = transform_prompt_fn
            self.plan: str = ""
            self.num_points_to_transform = num_points_to_transform
            self.curr_failed_transforms = 0
            self.max_allowed_failed_transforms = max_allowed_failed_transforms


    def generate_task_explanation(self, prompt_spec: PromptSpec) -> str:
        """ Generate task explanation"""
        task_explanation_prompt = construct_prompt_for_task_explanation(prompt_spec.instruction, prompt_spec.examples)
        return make_single_api_request(task_explanation_prompt, max_api_calls=10)

    def generate_plan(self, task_explanation:str, dataset:datasets.Dataset, prompt_spec: PromptSpec) -> str:
        """ Generate plan for the task"""
        plan_prompt = self.plan_prompt_fn(task_explanation, dataset, prompt_spec.examples)
        return make_single_api_request(plan_prompt, max_api_calls=10)

    
    def generate_transform_prompts(self, task_explanation:str,  dataset:datasets.Dataset, prompt_spec:PromptSpec,) -> List[str]:
        """ Get transform prompts for each row in the dataset."""
        transform_prompts = []
        for i in range(min(self.num_points_to_transform, len(dataset))):
            row = dataset[i]
            transform_prompt = self.transform_prompt_fn(task_explanation, row, self.plan, prompt_spec.examples)
            transform_prompts.append(transform_prompt)
        return transform_prompts

    
    def generate_responses(self, transform_prompts_batch:List[str]) -> List[str]:
        """ Generate responses for the transform prompts."""
        async def generate_responses_async(transform_prompts):
            """
            Generate responses asynchronously using the specified model.
            """
            responses = await api_tools.APIAgent(model_name="azure/GPT-3-5-turbo-chat", max_tokens=4000).generate_batch_completion(
                transform_prompts,
                temperature=0,
                responses_per_request=1,
                requests_per_minute=15,
            )
            return responses
        
        try:
            loop = asyncio.get_event_loop()
            responses = loop.run_until_complete(generate_responses_async(transform_prompts_batch))
        except API_ERRORS as e:
            handle_api_error(e)
            #TODO: What to return here?
        return responses


    def process_responses(self, responses:list, prompt_spec: PromptSpec) -> Tuple[List[str], List[str]]:
        """
        Process the responses received from the API. Also write the current set of inputs and outputs to a file.

        Args:
            responses: A list of response strings from the API.
            prompt_spec: The PromptSpec object containing the instruction and examples.

        Returns:
            A tuple containing two lists: inputs and outputs.
            - inputs: A list of transformed input strings.
            - outputs: A list of transformed output strings.
        """
        inputs, outputs = [], []
        counter=0
        for response in responses:
            try:
                extraction = find_and_parse_json(response, ["input", "output"], [])
                if extraction is not None:
                    if extraction["input"] is None or extraction["output"] is None:
                        raise ValueError("Input or output is None")
                    input = str(extraction["input"]).strip()
                    if input in prompt_spec.examples:
                        raise ValueError("Repeated Task Examples from prompt")

                    str1 = str("Q: " + input + "\nA:")
                    str2 = str(extraction["output"]).strip()

                    inputs.append(str1)
                    outputs.append(str2)
                    if counter < 2:
                        logger.info(f"inputs\n{str1}\n\nouputs\n{str2}")
                        counter += 1
                    counter+=1

            except Exception as e:
                logger.error(f"Error extracting from response: {e}")
                self.curr_failed_transforms += 1
                if self.curr_failed_transforms > self.max_allowed_failed_transforms:
                    break

        with open('dump.txt', 'a') as file:
            file.write('Input: ' + ', '.join(map(str, inputs)) + '\n')
            file.write('Output: ' + ', '.join(map(str, outputs)) + '\n')

        return inputs, outputs
    
    def transform_data(self, prompt_spec:PromptSpec, dataset: datasets.Dataset) -> tuple[list[str], list[str]]:
        """
        Transforms the given dataset based on the provided prompt specification.

        Args:
            prompt_spec (PromptSpec): The prompt specification object that defines the transformation rules.
            dataset (datasets.Dataset): The dataset to be transformed.

        Returns:
            A tuple containing two lists: inputs and outputs.
        """
        task_explanation = self.generate_task_explanation(prompt_spec)
        self.plan = self.generate_plan(task_explanation, dataset, prompt_spec)
        logger.info(f"Plan created. Plan: {self.plan}")

        transform_prompts = self.generate_transform_prompts(task_explanation, dataset, prompt_spec)
        inputs, outputs = [], []
        for batch_indices in range(0,len(transform_prompts), 100):
            transform_prompt_batch = transform_prompts[batch_indices:batch_indices+100]
            responses = self.generate_responses(transform_prompt_batch)
            curr_inputs, curr_outputs = self.process_responses(responses, prompt_spec)
            inputs.extend(curr_inputs)
            outputs.extend(curr_outputs)
            if self.curr_failed_transforms > self.max_allowed_failed_transforms:
                logger.error(f"Exceeded max allowed failed transforms: {self.curr_failed_transforms}")
                break

        logger.info(f"Requested length: {self.num_points_to_transform}\nActual length: {len(inputs)}\n")
        return inputs,outputs