from transformers import GPT2LMHeadModel, GPT2Tokenizer, Trainer, TrainingArguments
import torch
from datasets import load_dataset

def fine_tune():
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2")

    dataset = load_dataset("csv", data_files="data/fine_tune_data.csv")["train"]

    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=512)

    tokenized_datasets = dataset.map(tokenize_function, batched=True)

    training_args = TrainingArguments(
        output_dir="models/fine_tuned_gpt2",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        num_train_epochs=3,
        logging_dir="logs",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets
    )

    trainer.train()
    model.save_pretrained("models/fine_tuned_gpt2")
    tokenizer.save_pretrained("models/fine_tuned_gpt2")
    print("Fine-tuning complete.")
