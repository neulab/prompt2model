"""The real evaluation will be conduted after each mock evaluation of Trainer."""


from transformers import TrainerCallback

from prompt2model.model_evaluator import Seq2SeqEvaluator
from prompt2model.model_executor import GenerationModelExecutor
from prompt2model.utils import get_formatted_logger

logger = get_formatted_logger("ModelTrainer")


class ValidationCallback(TrainerCallback):
    """The real evaluation will be conduted after each mock evaluation of Trainer."""

    def __init__(
        self,
        trainer,
        tokenizer,
        val_dataset,
        executor_batch_size=10,
        tokenizer_max_length=256,
        sequence_max_length=512,
    ) -> None:
        """Initializes a new instance of Model Trainer Callback.

        Args:
            trainer: Trainer instance.
                After each epoch of Training, this callback will be called.
            tokenizer: Tokenizer to initialize model executor.
            val_dataset: Validation dataset to be evaluated on.
            executor_batch_size: The batch size for model executor to
                make predictions.
            tokenizer_max_length: The maximum number of tokens that
                tokenizer is allowed to generate.
            sequence_max_length: The maximum number of tokens in
                the input and output.
        """
        super().__init__()
        self.trainer = trainer
        self.tokenizer = tokenizer
        self.val_dataset = val_dataset
        self.epoch_count = 0
        self.val_dataset_size = len(self.val_dataset)
        self.executor_batch_size = executor_batch_size
        self.tokenizer_max_length = tokenizer_max_length
        self.sequence_max_length = sequence_max_length

    def on_epoch_end(self, args, state, control, **kwargs):
        """After each  evaluation, this function will be called."""
        _ = (args, state, control, kwargs)
        # Suppress the unused parameters warning.
        self.epoch_count += 1
        logger.info(
            f"Epoch: {self.epoch_count}. Evaluate on { self.val_dataset_size} examples."
        )
        # For multi-GPU training, the training processor will be segmented
        # into multi-threads with data paralyzation, so the validation dataset
        # used in the callback is also segmented.
        model_executor = GenerationModelExecutor(
            model=self.trainer.model,
            tokenizer=self.tokenizer,
            batch_size=self.executor_batch_size,
            tokenizer_max_length=self.tokenizer_max_length,
            sequence_max_length=self.sequence_max_length,
        )
        model_outputs = model_executor.make_prediction(
            self.val_dataset,
            "model_input",
        )
        evaluator = Seq2SeqEvaluator()
        metric_values = evaluator.evaluate_model(
            self.val_dataset,
            "model_output",
            model_outputs,
            encoder_model_name="xlm-roberta-base",
        )
        logger.info(metric_values)
