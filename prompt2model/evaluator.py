import datasets
from typing import Any, Dict, List

from prompt_parser import PromptSpec

# Input:
#    1) Path to trained model
#    2) Metrics to use (how to specify???)
#
# Output:
#    Dictionary of metric values


def evaluate_model(
    model_path: str, prompt_spec: PromptSpec, test_data: datasets.Dataset
) -> Dict[str, Any]:
    # raise NotImplementedError
    return {"accuracy": -1.0}
