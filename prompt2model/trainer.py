"""An interface for trainers.
"""

import datasets
import transformers
from typing import Dict, Any
import wandb

from prompt_parser import PromptSpec

# Input:
#    1) training dataset (datasets.Dataset)
#    2) Dictionary consisting of hyperparameter values to use
#       (e.g. base model, optimizer, LR, etc)
#
# Output:
#    transformers.PreTrainedModel


class Trainer:
    def __init__(
        self,
        training: datasets.Dataset,
        hyperparameter_choices: Dict[str, Any],
        prompt_spec: PromptSpec,
    ):
        self.training = training
        self.hyperparameter_choices = hyperparameter_choices
        self.wandb = None

    def set_up_weights_and_biases(self) -> None:
        self.wandb = None
        raise NotImplementedError

    def train_model(self) -> transformers.PreTrainedModel:
        # raise NotImplementedError
        model = transformers.BertModel.from_pretrained("bert-base-uncased")
        return model
