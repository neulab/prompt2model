"""Example of how to fine-tune a model using the QLoraTrainer class."""

import os

from datasets import load_from_disk

from prompt2model.model_trainer.qlora_trainer import QLoraTrainer
from prompt2model.utils.dataset_utils import format_train_data, make_combined_datasets

if __name__ == "__main__":
    # load in datasets
    retrieved_dataset_dict = load_from_disk("demo_retrieved_dataset_dict")
    retrieved_dataset = retrieved_dataset_dict["train"]
    generated_dataset = load_from_disk("demo_generated_dataset")
    dataset_list = [retrieved_dataset, generated_dataset]

    # combine datasets and create train and eval splits
    train_dataset = make_combined_datasets(dataset_list)
    splits = train_dataset.train_test_split(test_size=0.1)
    train_dataset = splits["train"]  # has 2 cols: "input_col" and "output_col"
    eval_dataset = splits["test"]  # has 2 cols: "input_col" and "output_col"
    formatted_train_dataset = format_train_data(
        train_dataset
    )  # combined into one col: "text"
    formatted_eval_dataset = format_train_data(
        eval_dataset
    )  # combined into one col: "text"

    # set hyperparams
    num_epochs = 1
    qlora_alpha = 8
    qlora_r = 16
    qlora_lr = 1e-5
    save_folder_path = "qlora_finetuned_model"
    load_best_model_at_end = False

    trainer = QLoraTrainer(model_name="mistralai/Mistral-7B-v0.1", model_max_length=512)

    trained_model, trained_tokenizer = trainer.train_model(
        formatted_train_dataset,  # passed for fine-tuning the model
        formatted_eval_dataset,  # passed for calculating eval "loss" over time
        eval_dataset,  # passed for calculating eval "accuracy" over time
        # (whether generated output matches expected output)
        num_epochs=1,
        alpha=qlora_alpha,
        r=qlora_r,
        lr=qlora_lr,
        save_folder_path=save_folder_path,
        load_best_model_at_end=load_best_model_at_end,
    )
    trained_model.save_pretrained(os.path.join(save_folder_path, "demo_final_model"))
    trained_tokenizer.save_pretrained(
        os.path.join(save_folder_path, "demo_final_tokenizer")
    )
