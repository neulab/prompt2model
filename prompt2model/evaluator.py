"""An interface for automatic model evaluation.

Input:
   1) Trained model
   2) Test set
   3) Metrics to use (currently, inferred from PromptSpec)

Output:
   Dictionary of metric values
"""

from typing import Any

import datasets
import transformers

from prompt_parser import PromptSpec


def evaluate_model(
    model: transformers.PreTrainedModel,
    test_data: datasets.Dataset,
    prompt_spec: PromptSpec,
) -> dict[str, Any]:
    """Evaluate a model on a test set. The specific metrics to use are
    specified or inferred from the PromptSpec"""
    # raise NotImplementedError
    return {"accuracy": -1.0}
