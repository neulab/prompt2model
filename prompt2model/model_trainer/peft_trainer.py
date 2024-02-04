import gc
import os

import bitsandbytes
import datasets
import torch
import transformers
import wandb
from accelerate import Accelerator, FullyShardedDataParallelPlugin
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from torch.distributed.fsdp.fully_sharded_data_parallel import (
    FullOptimStateDictConfig,
    FullStateDictConfig,
)
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from prompt2model.utils.dataset_utils import make_combined_datasets

fsdp_plugin = FullyShardedDataParallelPlugin(
    state_dict_config=FullStateDictConfig(offload_to_cpu=True, rank0_only=False),
    optim_state_dict_config=FullOptimStateDictConfig(
        offload_to_cpu=True, rank0_only=False
    ),
)

accelerator = Accelerator(fsdp_plugin=fsdp_plugin)


class EvalAccuracyCallback(transformers.TrainerCallback):
    def __init__(self, val_data: datasets.Dataset, eval_tokenizer) -> None:
        range_len = min(100, len(val_data))
        self.val_data = val_data.shuffle(seed=42).select(range(range_len))
        self.eval_tokenizer = eval_tokenizer
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def on_evaluate(self, args, state, control, **kwargs):
        # Evaluate the model on the validation set
        # 1. use the val data
        # 2. use model and tokenizer
        # 3. get accuracy
        # 4. log to wandb
        model = kwargs.get("model")
        metrics = kwargs.get("metrics")
        val_acc = 0
        for row in self.val_data:
            input_text = row["input_col"]
            target = row["output_col"]

            max_gen_len = (
                len(self.eval_tokenizer(target, return_tensors="pt").input_ids[0]) + 5
            )
            inpt_ids = self.eval_tokenizer(
                input_text, return_tensors="pt"
            ).input_ids.to(self.device)
            generated_ids = model.generate(
                input_ids=inpt_ids,
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
        wandb.log({"val_acc": val_acc})
        if metrics is not None:
            metrics["eval_accuracy"] = val_acc


class QLoraTrainer:
    def __init__(self, model_name="mistralai/Mistral-7B-v0.1", eval_size=50) -> None:
        print("QLoraTrainer init")
        self.model_name = model_name
        self.eval_size = eval_size
        self.bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        print("configs fine")
        print(f"Attempting to load model {self.model_name}")
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name, device_map="auto", quantization_config=self.bnb_config
        )
        print("Model loaded")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            model_max_length=512,
            padding_side="left",
            add_eos_token=True,
        )
        print("Tokenizer loaded")

        self.tokenizer.pad_token = self.tokenizer.eos_token

    def qlora_tokenize(self, prompt):
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

        output_dir = os.path.join(save_folder_path, "qlora")

        self.eval_tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, add_bos_token=True, trust_remote_code=True
        )

        trainer = transformers.Trainer(
            model=self.model,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            args=transformers.TrainingArguments(
                output_dir=output_dir,
                num_train_epochs=num_epochs,
                warmup_steps=5,
                per_device_train_batch_size=train_batch_size,
                gradient_checkpointing=True,
                gradient_accumulation_steps=2,
                weight_decay=0.001,
                max_steps=-1,
                learning_rate=lr,  # Want about 10x smaller than the Mistral learning rate
                logging_steps=50,
                fp16=True,
                optim="paged_adamw_8bit",
                logging_dir="./logs",  # Directory for storing logs
                save_strategy="steps",  # Save the model checkpoint every logging step
                save_steps=200,  # Save checkpoints every 50 steps
                evaluation_strategy="steps",  # Evaluate the model every logging step
                eval_steps=50,  # Evaluate and save checkpoints every 50 steps
                do_eval=True,  # Perform evaluation at the end of training
                report_to="wandb",  # Enable WandB logging
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

        trainer.model.save_pretrained(os.path.join(output_dir, "qlora_model"))

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
            self.model, os.path.join(output_dir, "qlora_model")
        )
        self.model = self.model.merge_and_unload()
        self.model.save_pretrained(os.path.join(output_dir, "final_model"))

        del self.model
        gc.collect()
        torch.cuda.empty_cache()

        self.model = AutoModelForCausalLM.from_pretrained(
            os.path.join(output_dir, "final_model"),
            quantization_config=self.bnb_config,
            device_map="auto",
            trust_remote_code=True,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, add_bos_token=True, trust_remote_code=True
        )
        self.model.config.use_cache = True
        return self.model, self.tokenizer


class LoraTrainer:
    def __init__(self, model_name="mistralai/Mistral-7B-v0.1", eval_size=50) -> None:
        print("LoraTrainer init")
        self.model_name = model_name
        self.eval_size = eval_size
        print("configs fine")
        print(f"Attempting to load model {self.model_name}")
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name, device_map="auto", trust_remote_code=True
        )
        print("Model loaded")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            model_max_length=512,
            padding_side="left",
            add_eos_token=True,
        )
        print("Tokenizer loaded")

        self.tokenizer.pad_token = self.tokenizer.eos_token

    def qlora_tokenize(self, prompt):
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
        dataset: datasets.Dataset,
        train_batch_size: int = 1,
        num_epochs=1,
        alpha=16,
        r=8,
        lr=5e-5,
        save_folder_path="./",
        eval_dataset=None,
        load_best_model_at_end=True,
    ):
        if eval_dataset is None:
            # split hf dataset into train and test
            splits = dataset.train_test_split(test_size=0.1)
            train_dataset = splits["train"]
            eval_dataset = splits["test"]
        else:
            eval_len = len(eval_dataset)
            wandb.log({"eval_original_size": eval_len})
            if eval_len < self.eval_size:
                required_len = self.eval_size - eval_len
                splits = dataset.train_test_split(
                    test_size=min(required_len / len(dataset), 0.1)
                )
                train_dataset = splits["train"]
                eval_dataset = make_combined_datasets(
                    [splits["test"], eval_dataset], dataset_type="text"
                )
            else:
                train_dataset = dataset

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

        output_dir = os.path.join(save_folder_path, "lora")

        trainer = transformers.Trainer(
            model=self.model,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            args=transformers.TrainingArguments(
                output_dir=output_dir,
                num_train_epochs=num_epochs,
                warmup_steps=5,
                per_device_train_batch_size=train_batch_size,
                gradient_checkpointing=True,
                gradient_accumulation_steps=2,
                weight_decay=0.001,
                max_steps=-1,
                learning_rate=lr,  # Want about 10x smaller than the Mistral learning rate
                logging_steps=50,
                optim="paged_adamw_8bit",
                logging_dir="./logs",  # Directory for storing logs
                save_strategy="steps",  # Save the model checkpoint every logging step
                save_steps=200,  # Save checkpoints every 50 steps
                evaluation_strategy="steps",  # Evaluate the model every logging step
                eval_steps=50,  # Evaluate and save checkpoints every 50 steps
                do_eval=True,  # Perform evaluation at the end of training
                report_to="wandb",  # Enable WandB logging
                load_best_model_at_end=load_best_model_at_end,
                metric_for_best_model="eval_loss",
                greater_is_better=False,
            ),
            data_collator=transformers.DataCollatorForLanguageModeling(
                self.tokenizer, mlm=False
            ),
        )

        self.model.config.use_cache = (
            False  # silence the warnings. Please re-enable for inference!
        )
        trainer.train()

        trainer.model.save_pretrained(os.path.join(output_dir, "lora_model"))

        del self.model
        del trainer
        del self.tokenizer
        gc.collect()
        torch.cuda.empty_cache()

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,  # Mistral, same as before
            device_map="auto",
            trust_remote_code=True,
        )

        self.model = PeftModel.from_pretrained(
            self.model, os.path.join(output_dir, "lora_model")
        )
        self.model = self.model.merge_and_unload()
        self.model.save_pretrained(os.path.join(output_dir, "final_model"))

        del self.model
        gc.collect()
        torch.cuda.empty_cache()

        self.model = AutoModelForCausalLM.from_pretrained(
            os.path.join(output_dir, "final_model"),
            device_map="auto",
            trust_remote_code=True,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, add_bos_token=True, trust_remote_code=True
        )
        self.model.config.use_cache = True
        return self.model, self.tokenizer
