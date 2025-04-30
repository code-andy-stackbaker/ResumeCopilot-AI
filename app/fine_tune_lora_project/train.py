import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer
)
from peft import LoraConfig, get_peft_model, TaskType
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import pandas as pd
import json
from datasets import DatasetDict
from datasets import Dataset
from sklearn.model_selection import train_test_split

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
# print("the model peft:", model.print_trainable_parameters())
# dataset = load_dataset("csv", data_files={"train": "datasets/train.csv"})

# # using seed=42 to make sure shuffling should be predictable
# dataset = dataset.shuffle(seed = 42)
# print("the dataset::::", dataset)
# # Optional: split 80% train, 20% test
# dataset = dataset["train"].train_test_split(data, test_size=0.2, shuffle=True, stratify=labels)




# Load CSV and manually split using sklearn for stratification
df = pd.read_csv("datasets/train.csv")
train_df, test_df = train_test_split(
    df,
    test_size=0.2,
    random_state=42,
    shuffle=True,
    stratify=df["label"]
)

# Convert pandas DataFrames to Hugging Face Datasets
train_dataset = DatasetDict({"train": Dataset.from_pandas(train_df)})
test_dataset = DatasetDict({"test": Dataset.from_pandas(test_df)})

# Combine into one DatasetDict
dataset = DatasetDict({
    "train": train_dataset["train"],
    "test": test_dataset["test"]
})

# Tokenize the dataset
def tokenize_function(example):
    return tokenizer(example["text"], truncation=True, padding="max_length")

tokenized_dataset = dataset.map(tokenize_function, batched=True)

# Compute metrics function
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = torch.argmax(torch.tensor(logits), dim=-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="binary")
    acc = accuracy_score(labels, preds)
    return {
        "accuracy": acc,
        "f1": f1,
        "precision": precision,
        "recall": recall
    }

# Training arguments
args_custom = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-4,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=2,
    num_train_epochs=5,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=10,
    no_cuda=False
)

# Trainer setup
trainer = Trainer(
    model=model,
    args=args_custom,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"],
    compute_metrics=compute_metrics
)

print("Training and Evaluation Start...")
trainer.train()
eval_results = trainer.evaluate()
print("Final Evaluation Results:", eval_results)

# Save LoRA adapter
model.save_pretrained("./lora_adapter")
tokenizer.save_pretrained("./lora_adapter")