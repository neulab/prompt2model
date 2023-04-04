import datasets
import transformers
from typing import Dict, Any

# Input:
#    1) training dataset (datasets.Dataset)
#    2) Dictionary consisting of hyperparameter values to use
#       (e.g. base model, optimizer, LR, etc)
#
# Output:
#    transformers.PreTrainedModel


class Trainer:
    def __init__(
        self, training: datasets.Dataset, hyperparameter_choices: Dict[str, Any]
    ):
        self.training = training
        self.hyperparameter_choices = hyperparameter_choices

    def train_model(self) -> transformers.PreTrainedModel:
        # raise NotImplementedError
        model = transformers.PreTrainedModel.from_pretrained("bert-base-uncased")
        return model
