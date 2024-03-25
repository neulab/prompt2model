"""A trainer class for fine-tuning a model using QLoRA."""

import gc
import os

import datasets
import torch
import transformers
from accelerate import Accelerator, FullyShardedDataParallelPlugin
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from torch.distributed.fsdp.fully_sharded_data_parallel import (
    FullOptimStateDictConfig,
    FullStateDictConfig,
)
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from prompt2model.utils import get_formatted_logger

logger = get_formatted_logger("QLoRATrainer")

fsdp_plugin = FullyShardedDataParallelPlugin(
    state_dict_config=FullStateDictConfig(offload_to_cpu=True, rank0_only=False),
    optim_state_dict_config=FullOptimStateDictConfig(
        offload_to_cpu=True, rank0_only=False
    ),
)

accelerator = Accelerator(fsdp_plugin=fsdp_plugin)


class EvalAccuracyCallback(transformers.TrainerCallback):
    """A callback to evaluate the model on the validation set and log the accuracy."""

    def __init__(self, val_data: datasets.Dataset, eval_tokenizer) -> None:
        """Initialize the callback with the validation data and tokenizer.

        Args:
            val_data: The val dataset. Has 2 columns: "input_col" and "output_col".
            eval_tokenizer: The tokenizer to use for evaluation.

        Note: The val_data is shuffled and a max of 100 samples are used for evaluation.
        """
        range_len = min(100, len(val_data))
        self.val_data = val_data.shuffle(seed=42).select(range(range_len))
        self.eval_tokenizer = eval_tokenizer
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def on_evaluate(self, args, state, control, model, metrics, **kwargs):
        """Evaluate the model on the validation set and log the accuracy."""
        # Evaluate the model on the validation set
        # 1. use the val data
        # 2. use model and tokenizer
        # 3. get accuracy
        val_acc = 0
        with torch.no_grad():
            for row in self.val_data:
                input_text = row["input_col"]
                target = row["output_col"]

                max_gen_len = (
                    len(self.eval_tokenizer(target, return_tensors="pt").input_ids[0])
                    + 5
                )
                input_ids = self.eval_tokenizer(
                    input_text, return_tensors="pt"
                ).input_ids.to(self.device)
                generated_ids = model.generate(
                    input_ids=input_ids,
                    max_new_tokens=max_gen_len,
                    do_sample=False,
                    temperature=0,
                )
                text = self.eval_tokenizer.decode(
                    generated_ids[0], skip_special_tokens=True
                )
                text = text.replace(input_text, "").lower().strip()
                target = target.lower().strip()
                if target in text:
                    val_acc += 1
        val_acc = val_acc / len(self.val_data)
        print(f"Validation accuracy: {val_acc}")
        if metrics is not None:
            metrics["eval_accuracy"] = val_acc


class QLoRATrainer:
    """A class for fine-tuning a model using QLoRA."""

    def __init__(self, model_name: str, model_max_length: int) -> None:
        """Initialize the QLoRATrainer with a model name and evaluation size."""
        self.model_name = model_name
        self.model_max_length = model_max_length
        self.bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name, device_map="auto", quantization_config=self.bnb_config
        )
        logger.info("Model loaded")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            model_max_length=self.model_max_length,
            padding_side="left",
            add_eos_token=True,
        )
        logger.info("Tokenizer loaded")

        self.tokenizer.pad_token = self.tokenizer.eos_token

    def qlora_tokenize(self, prompt):
        """Tokenize the prompt for QLoRA training."""
        result = self.tokenizer(
            prompt["text"],
            truncation=True,
            max_length=512,
            padding="max_length",
        )
        result["labels"] = result["input_ids"].copy()
        return result

    def train_model(
        self,
        train_dataset: datasets.Dataset,  # columns: "text"
        eval_dataset: datasets.Dataset,  # columns: "text"
        original_eval_dataset: datasets.Dataset,  # columns: "input_col", "output_col"
        train_batch_size: int = 1,
        num_epochs=1,
        alpha=16,
        r=8,
        lr=5e-5,
        save_folder_path="./",
        load_best_model_at_end=True,
    ):
        """Train the model using QLoRA and return the trained model and tokenizer."""
        QLORA_MODEL_DIRECTORY = os.path.join(save_folder_path, "qlora")
        train_dataset = train_dataset.map(self.qlora_tokenize)
        eval_dataset = eval_dataset.map(self.qlora_tokenize)
        self.model.gradient_checkpointing_enable()
        self.model = prepare_model_for_kbit_training(self.model)

        config = LoraConfig(
            r=r,
            lora_alpha=alpha,
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
                "lm_head",
            ],
            bias="none",
            lora_dropout=0.05,  # Conventional
            task_type="CAUSAL_LM",
        )

        self.model = get_peft_model(self.model, config)
        self.model = accelerator.prepare_model(self.model)

        if torch.cuda.device_count() > 1:  # If more than 1 GPU
            self.model.is_parallelizable = True
            self.model.model_parallel = True

        self.eval_tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, add_bos_token=True, trust_remote_code=True
        )

        trainer = transformers.Trainer(
            model=self.model,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            args=transformers.TrainingArguments(
                output_dir=QLORA_MODEL_DIRECTORY,
                num_train_epochs=num_epochs,
                warmup_steps=5,
                per_device_train_batch_size=train_batch_size,
                gradient_checkpointing=True,
                gradient_accumulation_steps=2,
                weight_decay=0.001,
                max_steps=-1,
                learning_rate=lr,
                logging_steps=100,
                fp16=True,
                optim="paged_adamw_8bit",
                logging_dir="./logs",
                save_strategy="steps",
                save_steps=200,
                evaluation_strategy="steps",
                eval_steps=100,
                do_eval=True,
                load_best_model_at_end=load_best_model_at_end,
                metric_for_best_model="eval_loss",
                greater_is_better=False,
            ),
            data_collator=transformers.DataCollatorForLanguageModeling(
                self.tokenizer, mlm=False
            ),
            callbacks=[
                EvalAccuracyCallback(original_eval_dataset, self.eval_tokenizer)
            ],
        )

        self.model.config.use_cache = (
            False  # silence the warnings. Please re-enable for inference!
        )
        trainer.train()

        trainer.model.save_pretrained(
            os.path.join(QLORA_MODEL_DIRECTORY, "qlora_model")
        )

        del self.model
        del trainer
        del self.tokenizer
        gc.collect()
        torch.cuda.empty_cache()

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,  # Mistral, same as before
            quantization_config=self.bnb_config,  # Same quantization config as before
            device_map="auto",
            trust_remote_code=True,
        )

        self.model = PeftModel.from_pretrained(
            self.model, os.path.join(QLORA_MODEL_DIRECTORY, "qlora_model")
        )
        self.model = self.model.merge_and_unload()
        self.model.save_pretrained(os.path.join(QLORA_MODEL_DIRECTORY, "final_model"))

        del self.model
        gc.collect()
        torch.cuda.empty_cache()

        self.model = AutoModelForCausalLM.from_pretrained(
            os.path.join(QLORA_MODEL_DIRECTORY, "final_model"),
            quantization_config=self.bnb_config,
            device_map="auto",
            trust_remote_code=True,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, add_bos_token=True, trust_remote_code=True
        )
        self.model.config.use_cache = True
        return self.model, self.tokenizer
