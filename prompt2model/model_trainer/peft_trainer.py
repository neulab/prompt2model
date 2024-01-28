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

fsdp_plugin = FullyShardedDataParallelPlugin(
    state_dict_config=FullStateDictConfig(offload_to_cpu=True, rank0_only=False),
    optim_state_dict_config=FullOptimStateDictConfig(
        offload_to_cpu=True, rank0_only=False
    ),
)

accelerator = Accelerator(fsdp_plugin=fsdp_plugin)


class QLoraTrainer:
    def __init__(self, model_name="mistralai/Mistral-7B-v0.1") -> None:
        self.model_name = model_name
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
        self, dataset: datasets.Dataset, train_batch_size: int = 1, num_epochs=1, alpha=16, r=8, lr=5e-5
    ):
        # split hf dataset into train and test
        splits = dataset.train_test_split(test_size=0.1)
        train_dataset = splits["train"]
        eval_dataset = splits["test"]

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

        output_dir = f"./peft_{self.model_name}"

        trainer = transformers.Trainer(
            model=self.model,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            args=transformers.TrainingArguments(
                output_dir=output_dir,
                num_train_epochs=3,
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
            ),
            data_collator=transformers.DataCollatorForLanguageModeling(
                self.tokenizer, mlm=False
            ),
        )

        self.model.config.use_cache = (
            False  # silence the warnings. Please re-enable for inference!
        )
        trainer.train()

        trainer.model.save_pretrained(os.path.join(output_dir, "final_model"))

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
            self.model, os.path.join(output_dir, "final_model")
        )
        self.model = self.model.merge_and_unload()
        self.model.save_pretrained(f"./final_{self.model_name}/final_tuned_model")

        del self.model
        gc.collect()
        torch.cuda.empty_cache()

        self.model = AutoModelForCausalLM.from_pretrained(
            f"./final_{self.model_name}/final_tuned_model",
            quantization_config=self.bnb_config,
            device_map="auto",
            trust_remote_code=True,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, add_bos_token=True, trust_remote_code=True
        )
        self.model.config.use_cache = True
        return self.model, self.tokenizer
