"""The real evaluation will be conduted after each mock evaluation of Trainer."""

import logging
import tempfile
from pathlib import Path

from transformers import TrainerCallback

from prompt2model.model_evaluator import Seq2SeqEvaluator
from prompt2model.model_executor import GenerationModelExecutor


class RealEvaluation(TrainerCallback):
    """The real evaluation will be conduted after each mock evaluation of Trainer."""

    def __init__(self, trainer, tokenizer, val_dataset) -> None:
        """Initializes a new instance of Model Trainer Callback.

        Args:
            trainer: Trainer instance.
                After each epoch of Training, this callback will be called.
            tokenizer: Tokenizer to initialize model executor.
            val_dataset: Validation dataset to be evaluated on.
        """
        super().__init__()
        self.trainer = trainer
        self.tokenizer = tokenizer
        self.val_dataset = val_dataset

    def on_epoch_end(self, args, state, control, **kwargs):
        """After each  evaluation, this function will be called."""
        _ = (args, state, control, kwargs)
        # Pass the unused paramerters warning.
        logging.info("Coduct real evaluation on each epoch's ending.")

        with tempfile.TemporaryDirectory() as cache_dir:
            # Save the original model's weights to a file
            temp_model_location = Path(cache_dir) / "temp_model"
            self.trainer.model.save_pretrained(temp_model_location)
            # Load the weights into a new model
            cpu_model = type(self.trainer.model).from_pretrained(temp_model_location)
            # Move the model to CPU
            cpu_model = cpu_model.to("cpu")

        model_executor = GenerationModelExecutor(
            cpu_model,
            self.tokenizer,
            self.val_dataset,
            "model_input",
            10,
        )
        model_outputs = model_executor.make_prediction()
        evaluator = Seq2SeqEvaluator()
        metric_values = evaluator.evaluate_model(
            self.val_dataset,
            "output_col",
            model_outputs,
            encoder_model_name="xlm-roberta-base",
        )
        logging.info(metric_values)
