import torch
from datasets import load_dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments
from peft import LoraConfig, get_peft_model, TaskType
import json

with open("lora_config.json", "r") as f:
  lora_params = json.load(f)


model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
base_model = AutoModelForSequenceClassification.from_pretrained(model_name, num_label=2)

lora_config = LoraConfig(
  r = lora_params["r"],
  lora_alpha= lora_params["lora_alpha"],
  target_modules = lora_params["target_modules"],
  lora_dropout=lora_params["lora_dropout"],
  bias = lora_params["bias"],
  task_type=TaskType.SEQ_CLS
)

