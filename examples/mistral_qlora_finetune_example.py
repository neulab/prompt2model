"""Example of how to fine-tune a model using the QLoRATrainer class."""

import os

from datasets import load_from_disk

from prompt2model.model_trainer.qlora_trainer import QLoRATrainer
from prompt2model.utils.dataset_utils import format_train_data, make_combined_datasets

if __name__ == "__main__":
    # First, we load in the datasets we want to fine-tune on.
    retrieved_dataset_dict = load_from_disk("demo_retrieved_dataset_dict")
    retrieved_dataset = retrieved_dataset_dict["train"]
    generated_dataset = load_from_disk("demo_generated_dataset")
    dataset_list = [retrieved_dataset, generated_dataset]

    # Next, we combine datasets and create train and eval splits.
    train_dataset = make_combined_datasets(dataset_list)
    splits = train_dataset.train_test_split(test_size=0.1)
    train_dataset = splits["train"]
    eval_dataset = splits["test"]

    # At this point, both train_dataset and eval_dataset are datasets with two
    # columns: "input_col" and "output_col".
    # We need to format them into a single column, "text", for the QLoRATrainer to use.
    formatted_train_dataset = format_train_data(train_dataset)
    formatted_eval_dataset = format_train_data(eval_dataset)

    # Next, we define the hyperparameters for the QLoRATrainer.
    num_epochs = 1
    qlora_alpha = 8
    qlora_r = 16
    qlora_lr = 1e-5
    save_folder_path = "qlora_finetuned_model"
    load_best_model_at_end = False

    # Next, we create a QLoRATrainer object and call the train_model method.
    trainer = QLoRATrainer(model_name="mistralai/Mistral-7B-v0.1", model_max_length=512)

    # `formatted_eval_dataset` contains just one column: "text",
    # and is used to calculate eval loss, by checking loss for each next token.
    # `eval_dataset` contains two columns: "input_col" and "output_col",
    # and is used to calculate eval accuracy, by checking whether the generated output
    # exactly matches the expected output.
    trained_model, trained_tokenizer = trainer.train_model(
        formatted_train_dataset,
        formatted_eval_dataset,
        eval_dataset,
        num_epochs=1,
        alpha=qlora_alpha,
        r=qlora_r,
        lr=qlora_lr,
        save_folder_path=save_folder_path,
        load_best_model_at_end=load_best_model_at_end,
    )

    # Finally, we save the trained model and tokenizer to disk.
    trained_model.save_pretrained(os.path.join(save_folder_path, "demo_final_model"))
    trained_tokenizer.save_pretrained(
        os.path.join(save_folder_path, "demo_final_tokenizer")
    )
