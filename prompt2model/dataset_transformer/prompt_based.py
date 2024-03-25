"""A simple dataset transformer that uses a plan prompt and transform prompt."""
from __future__ import annotations

import asyncio
from collections.abc import Callable

import datasets
import wandb

from prompt2model.dataset_transformer.base import DatasetTransformer
from prompt2model.dataset_transformer.prompt_template import (
    construct_prompt_for_plan,
    construct_prompt_for_transform_data,
)
from prompt2model.dataset_retriever.task_expansion_prompt import (
    construct_propmt_for_task_explanation
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
        plan_prompt_fn: Callable[
            [str, list[dict], str], str
        ] = construct_prompt_for_plan,
        transform_prompt_fn: Callable[
            [str, dict, str, str], str
        ] = construct_prompt_for_transform_data,
    ):
        """Initialize the class."""
        self.plan_prompt_fn = plan_prompt_fn
        self.transform_prompt_fn = transform_prompt_fn
        self.plan: str = ""

    def make_dataset_from_samples(
        self,
        inputs: list[str],
        outputs: list[str],
    ) -> datasets.DatasetDict:
        """Given a list of inputs and outputs, make a dataset.

        This function takes in inputs and outputs, both as list of strings,
        and returns a DatasetDict object with a single split, "train". It has
        two columns, "input_col" and "output_col".


        Args:
            inputs: A list of inputs, each input is a string.
            outputs: A list of outputs, each output is a string.

        Returns:
            A DatasetDict object with a single split, "train". It has two
            columns, "input_col" and "output_col".
        """
        if len(inputs) < 0 or len(inputs) != len(outputs):
            raise ValueError("Length of inputs and outputs must be >0 and equal.")
        updated_inputs, updated_outputs = [],[]
        for i,o in zip(inputs, outputs):
            if i is not None and o is not None:
                updated_inputs.append(i)
                updated_outputs.append(o)
            else:
                print("Input/output was None")
                
            

        dataset_dict = {}
        dataset_dict["train"] = datasets.Dataset.from_dict(
            {"input_col": updated_inputs, "output_col": updated_outputs}
        )
        return datasets.DatasetDict(dataset_dict)

    def transform_data(
        self,
        prompt_spec,
        dataset: datasets.Dataset,
        num_points_to_transform: int,
    ) -> datasets.DatasetDict:
        """Transform the dataset according to the prompt_spec and dataset."""
        task_explanation_prompt = construct_propmt_for_task_explanation(prompt_spec.instruction, prompt_spec.examples)
        task_explanation = make_single_api_request(task_explanation_prompt, max_api_calls=10)
        plan_prompt = self.plan_prompt_fn(
            task_explanation,
            dataset,
            prompt_spec.examples,
        )
        self.plan = make_single_api_request(plan_prompt, max_api_calls=100)

        logger.info(f"Plan created. Plan: {self.plan}")

        inputs = []
        outputs = []

        max_len = min(num_points_to_transform, len(dataset))
        len_count = 0
        transform_prompts = []
        flag = False
        for row in dataset:
            transform_prompt = self.transform_prompt_fn(
                task_explanation,
                row,
                self.plan,
                prompt_spec.examples,
           
            )
            if not flag:

                print("Transformation prompt!!")
                print(transform_prompt)
                flag = True
            transform_prompts.append(transform_prompt)

            len_count += 1
            if len_count >= max_len:
                break
        
        max_allowed_failed_transforms = 1000
        curr_failed_transforms = 0
        counter = 0

        # for prompt in transform_prompts:
        #     try:
        #         response = make_single_api_request(prompt)
        #     except API_ERRORS as e:
        #         handle_api_error(e)

        #     temp1 = []
        #     temp2 = []

        #     try:
        #         extraction = find_and_parse_json(response, ["input", "output"], [])
        #         if extraction is not None:
        #             if extraction["input"] is None or  extraction["output"] is  None:
        #                 raise ValueError("Input or output is None")
                    
        #             input = str(extraction["input"]).strip()

        #             if input in prompt_spec.examples:
        #                 raise ValueError("Repeated Task Examples from prompt")
                    
        #             str1 = str("Q: " + input + "\nA:")
        #             str2 = str(extraction["output"]).strip()

        #             temp1.append(str1)
        #             temp2.append(str2)
        #             if counter%50==0:
        #                 #Just for printing some input/output examples
        #                 print(f"inputs\n{str1}\n\nouputs\n{str2}")
        #                 counter +=1

        #     except Exception as e:
        #         logger.error(f"Error extracting from response: {e}")
        #         curr_failed_transforms +=1
        #         if curr_failed_transforms > max_allowed_failed_transforms:
        #             dataset_dict = {}
        #             dataset_dict["train"] = datasets.Dataset.from_dict({"input_col":[], "output_col":[]})#Dont bother with this dataset, just skip it.
        #             return datasets.DatasetDict(dataset_dict)
        #         continue
        # if len(temp1)!=len(temp2): logger.error(f"input and output arrays are not same length: {len(temp1)} {len(temp2)}")

        # elif counter%50==0:
        #     inputs += temp1
        #     outputs += temp2
            
        #     with open('temp_dump_gpt_4_cause_effect.txt', 'a') as file:
        #         file.write('Input: ' + ', '.join(map(str, temp1)) + '\n')
        #         file.write('Output: ' + ', '.join(map(str, temp2)) + '\n')
            

                    

        # logger.info(f"Requested length: {max_len}\nActual length: {len(inputs)}\n")
        # return self.make_dataset_from_samples(inputs, outputs)




        async def generate_responses(transform_prompts):
            responses = await api_tools.APIAgent(model_name="azure/GPT-3-5-turbo-sweden", max_tokens=4000).generate_batch_completion(
                transform_prompts,
                temperature=0,
                responses_per_request=1,
                requests_per_minute=15,
            )
            return responses
        
        max_allowed_failed_transforms = 1000
        curr_failed_transforms = 0
        for batch_indices in range(0,len(transform_prompts), 100):
            transform_prompt_batch = transform_prompts[batch_indices:batch_indices+100]
            try:
                loop = asyncio.get_event_loop()
                responses = loop.run_until_complete(generate_responses(transform_prompt_batch))
            except API_ERRORS as e:
                handle_api_error(e)

            counter = 0
            temp1 = []
            temp2 = []
            for response in responses:

                try:
                    extraction = find_and_parse_json(response, ["input", "output"], [])
                    if extraction is not None:
                        if extraction["input"] is None or  extraction["output"] is  None:
                            raise ValueError("Input or output is None")
                        
                        input = str(extraction["input"]).strip()

                        if input in prompt_spec.examples:
                            raise ValueError("Repeated Task Examples from prompt")
                        
                        str1 = str("Q: " + input + "\nA:")
                        str2 = str(extraction["output"]).strip()

                        temp1.append(str1)
                        temp2.append(str2)
                        if counter <2:
                            #Just for printing some input/output examples
                            print(f"inputs\n{str1}\n\nouputs\n{str2}")
                            counter +=1

                except Exception as e:
                    logger.error(f"Error extracting from response: {e}")
                    curr_failed_transforms +=1
                    if curr_failed_transforms > max_allowed_failed_transforms:
                        dataset_dict = {}
                        dataset_dict["train"] = datasets.Dataset.from_dict({"input_col":[], "output_col":[]})#Dont bother with this dataset, just skip it.
                        return datasets.DatasetDict(dataset_dict)

                    continue
            if len(temp1)!=len(temp2): logger.error(f"input and output arrays are not same length: {len(temp1)} {len(temp2)}")
            else:
                inputs += temp1
                outputs += temp2
                with open('temp_dump_arithmetic_simple_q.txt', 'a') as file:
                    file.write('Input: ' + ', '.join(map(str, temp1)) + '\n')
                    file.write('Output: ' + ', '.join(map(str, temp2)) + '\n')

        logger.info(f"Requested length: {max_len}\nActual length: {len(inputs)}\n")
        return self.make_dataset_from_samples(inputs, outputs)
