import os
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel, PeftConfig
import logging

class ClassifierReranker:
    def __init__(self, lora_model_path=None):
      try:
        # Dynamically resolve adapter path
        self.lora_model_path = lora_model_path or os.path.join(os.path.dirname(__file__), "lora_adapter")
        self.device = torch.device("cpu")  # ✅ Always use CPU for compatibility (especially on Mac M1/M2)
        print("📦 Loading LoRA adapter config...")
        # Load LoRA adapter config
        config = PeftConfig.from_pretrained(self.lora_model_path)

        # Load base model
        print("📥 Loading base model...")
        base_model = AutoModelForSequenceClassification.from_pretrained(config.base_model_name_or_path, num_labels=2)
        print("🔗 Applying LoRA adapter...")
        # Apply LoRA adapter
        self.model = PeftModel.from_pretrained(base_model, self.lora_model_path)
        self.model.to(self.device)
        self.model.eval()
        print("📝 Loading tokenizer...")
        
        # Load tokenizer (from adapter if saved, else base model)
        self.tokenizer = AutoTokenizer.from_pretrained(self.lora_model_path)
        print("✅ ClassifierReranker initialized successfully.\n")
      except Exception as e:
          logging.error(f"[ClassifierReranker] Failed to initialize model: {e}")
          raise

    def predict_match_score(self, resume_text: str, job_text: str) -> float:
      try:
        if not resume_text or not job_text:
          raise ValueError("Both resume_text and job_text must be provided.")
        input_text = f"{resume_text} [SEP] {job_text}"
        print("🔠 Combined text:", input_text)
        inputs = self.tokenizer(
            input_text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        )
        print("📦 Tokenized inputs.")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        print("⚙️ Moved inputs to device.")
        with torch.no_grad():
          outputs = self.model(**inputs)
          probs = torch.softmax(outputs.logits, dim=1).squeeze()
          print("✅ Softmax computed:", probs)
        return float(probs[1].item())  # Probability of class 1 (semantic match)
      except ValueError as ve:
        logging.error(f"[ClassifierReranker] Input validation error: {ve}")
        return 0.0
      except Exception as e:
        logging.error(f"[ClassifierReranker] Prediction failed: {e}")
        return 0.0