"""Testing hyperparameter optimization with different configurations."""

import gc
import logging
import os

import datasets

from prompt2model.model_trainer.generate import GenerationModelTrainer
from prompt2model.param_selector.search_with_optuna import OptunaParamSelector

os.environ["WANDB_MODE"] = "dryrun"
logger = logging.getLogger("AutoHyperparamOptimization")


def test_optimize_hyperparameters():
    """Tests whether the hyperparameter optimization is working correctly or not."""
    # Create a simple training dataset.
    training_datasets = [
        datasets.Dataset.from_dict(
            {
                "model_input": [
                    "<task 0>Given a product review, predict the sentiment score associated with it.\nExample:\nIt isn’t my fav lip balm, but it’s up there. It moisturises really well and the lemon isn’t strong or over powering.\nLabel:\n",  # noqa E501
                ],
                "model_output": ["4"],
            }
        ),
    ]

    validation_datasets = [
        datasets.Dataset.from_dict(
            {
                "model_input": [
                    "<task 0>Given a product review, predict the sentiment score associated with it.\nExample:\nBroke me out and gave me awful texture all over my face. I typically have clear skin and after using this product my skin HATED it. Could work for you though.\nLabel:\n",  # noqa E501
                ],
                "model_output": ["2"],
            }
        ),
    ]

    param_selector = OptunaParamSelector(
        n_trials=1,
        trainer=GenerationModelTrainer(
            "patrickvonplaten/t5-tiny-random", has_encoder=True
        ),
    )
    best_hyperparameters = param_selector.optimize_hyperparameters(
        training_datasets=training_datasets,
        validation=validation_datasets,
        hyperparameters={
            "min_num_train_epochs": 1,
            "max_num_train_epochs": 1,
            "save_strategy": ["epoch"],
            "evaluation_strategy": ["epoch"],
            "per_device_train_batch_size": [2],
            "min_weight_decay": 4e-5,
            "max_weight_decay": 1e-1,
            "min_learning_rate": 4e-5,
            "max_learning_rate": 1e-1,
        },
    )
    assert isinstance(best_hyperparameters, dict)
    gc.collect()
