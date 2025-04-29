import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer
)
from peft import LoraConfig, get_peft_model, TaskType
import pandas as pd
import json

with open("lora_config.json", "r") as f:
  lora_params = json.load(f)


model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
base_model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

lora_config = LoraConfig(
  r = lora_params["r"],
  lora_alpha= lora_params["lora_alpha"],
  target_modules = lora_params["target_modules"],
  lora_dropout=lora_params["lora_dropout"],
  bias = lora_params["bias"],
  task_type=TaskType.SEQ_CLS
)

model = get_peft_model(base_model, lora_config)
print("the model peft:", model.print_trainable_parameters())
dataset = load_dataset("csv", data_files={"train": "datasets/train.csv"})

# Optional: split 80% train, 20% test
dataset = dataset["train"].train_test_split(test_size=0.2)

#tokenize the dataset


def tokenize_function(example):
  return tokenizer(example["text"], truncation = True, padding = "max_length")

tokenized_dataset = dataset.map(tokenize_function, batched = True)

print("Using device:", torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"))

args_custom = TrainingArguments(
  output_dir="./results",
  evaluation_strategy="epoch",
  save_strategy="epoch",
  learning_rate=2e-4,
  per_device_train_batch_size=2,
  per_device_eval_batch_size=2,
  gradient_accumulation_steps=2,
  num_train_epochs=3,
  weight_decay=0.01,
  logging_dir="./logs",
  logging_steps=10,
  no_cuda=False
)


# 7. Create Trainer
trainer = Trainer(
    model=model,
    args=args_custom,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"],
)



print("the datasets:", dataset)
print("the tokenized dataset", tokenized_dataset)


trainer.train()

# 9. Save LoRA adapter only (not full model)
model.save_pretrained("./lora_adapter")
tokenizer.save_pretrained("./lora_adapter")