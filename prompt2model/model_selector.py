import datasets
import transformers
from typing import Dict, List

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


def select_model(
    training: datasets.Dataset,
    validation: datasets.Dataset,
    hyperparameter_choices=Dict[str, List],
) -> transformers.PreTrainedModel:
    # raise NotImplementedError
    t = Trainer(training, hyperparameter_choices)
    single_model = t.train_model()
    return single_model
