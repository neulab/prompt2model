"""An interface for model selection.
"""

import datasets
import transformers
from typing import Any, Dict, List

from prompt_parser import PromptSpec
from trainer import Trainer

# Input:
#    1) training dataset (datasets.Dataset)
#    2) validation dataset (datasets.Dataset)
#    3) Dictionary-of-lists consisting of hyperparameter
#       values to consider (e.g. different base models to
#       consider, different optimizers, different LRs, etc)
#
# Output:
#    transformers.PreTrainedModel


class ModelSelector:
    def __init__(
        self,
        training: datasets.Dataset,
        validation: datasets.Dataset,
        prompt_spec: PromptSpec,
    ):
        self.training = training
        self.validation = validation
        self.prompt_spec = prompt_spec
        self.hyperparameter_choices = self._extract_hyperparameter_choices()

    def _extract_hyperparameter_choices(self) -> Dict[str, List[Any]]:
        # TODO: Extract or infer from self.prompt_spec.
        # raise NotImplementedError
        return {
            "model": ["bert-base-uncased", "t5-base"],
            "optimizer": ["adam", "sgd+m"],
            "learning_rate": [0.0001, 0.001, 0.01],
        }

    def select_model(
        self,
    ) -> transformers.PreTrainedModel:
        # raise NotImplementedError
        t = Trainer(self.training, self.hyperparameter_choices, self.prompt_spec)
        single_model = t.train_model()
        return single_model
