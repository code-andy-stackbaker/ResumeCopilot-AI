from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel
import torch
import logging

class ClassifierReranker:
  def __init__(self, base_model_name="bert-base-uncased", lora_model_path="./lora_adapter"):
    try:
      self.tokenizer = AutoTokenizer.from_pretrained(base_model_name)
      base_model = AutoModelForSequenceClassification.from_pretrained(base_model_name)
      self.model = PeftModel.from_pretrained(base_model, lora_model_path)
      self.model.eval()
    except Exception as e:
      logging.error(f"Failed to load LoRA classifier model: {e}")
      raise

  def predict_match_score(self, resume_text: str, job_text: str) -> float:
    try:
      input_text = f"{resume_text} [SEP] {job_text}"
      # input_text = f"{resume_text} [SEP] {job_text}"
      inputs = self.tokenizer(input_text, return_tensors="pt", truncation=True, padding=True)

      with torch.no_grad():
        outputs = self.model(**inputs)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=1).squeeze()

        # Return probability of class 1 (match)
        return float(probs[1].item())
    except Exception as e:
      logging.error(f"Error during prediction: {e}")
      return 0.0