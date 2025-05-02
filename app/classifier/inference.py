from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel
import torch
from collections import Counter

# Load base model and LoRA adapter
base_model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)
model = PeftModel.from_pretrained(base_model, "./lora_adapter")
tokenizer = AutoTokenizer.from_pretrained("./lora_adapter")

# Set model to evaluation mode
model.eval()

# Define sample input texts
# sample_texts = [
#     "Backend engineer skilled in Django and PostgreSQL. [SEP] Looking for a frontend engineer skilled in Vue.js.",
#     "Remote Product manager. [SEP] Looking for product manager skilled in sql and related technologies. ",
#     "Intern Software tester skilled in bug tracking. Looking for a Software tester skilled in.",
#     "Full-stack developer skilled in MERN stack. [SEP] Hiring a UI/UX designer experienced in Adobe XD.",
#     "Frontend engineer proficient in Angular and RxJS. [SEP] Hiring a backend developer experienced in Firebase.",
#     "Cloud architect experienced in AWS and Terraform. [SEP] Looking for a DevOps engineer skilled in GitHub Actions.",
#     "AI researcher focused on NLP and Transformers. [SEP] Hiring a data scientist experienced in BERT models.",
#     "Software tester skilled in Cypress and Playwright. [SEP] Looking for a QA automation engineer.",
#     "Junior Node.js developer skilled in MongoDB. [SEP] Hiring a senior backend engineer with GraphQL experience.",
#     "Senior product manager experienced in Agile and Scrum. [SEP] Looking for a UX researcher with mobile design skills.",
#     "Frontend React.js developer skilled in Material UI. [SEP] Hiring a backend engineer with NestJS knowledge.",
#     "Cloud engineer skilled in Azure and Kubernetes. [SEP] Looking for a security engineer with DevSecOps knowledge.",
#     "Junior Flutter developer with Firebase experience. [SEP] Hiring a mobile QA tester for Android apps.",
#     "System admin skilled in Linux and Docker. [SEP] Looking for a DevOps engineer skilled in CI/CD pipelines.",
#     "Data engineer experienced with Airflow and ETL. [SEP] Hiring a data scientist skilled in feature engineering.",
#     "UI designer skilled in Figma and prototyping. [SEP] Looking for a frontend engineer with animation experience.",
#     "Senior backend developer with Java Spring Boot. [SEP] Hiring a frontend developer skilled in TypeScript.",
#     "React Native engineer skilled in Expo and Redux. [SEP] Looking for a mobile developer with Firebase experience.",
#     "DevOps specialist with Jenkins and Ansible. [SEP] Hiring a cloud engineer with multi-cloud experience.",
#     "Cybersecurity expert with penetration testing skills. [SEP] Looking for a backend developer skilled in OAuth2.",
#     "Business analyst experienced in Jira and Confluence. [SEP] Hiring a product manager for Agile SaaS product.",
#     "Senior machine learning engineer with LoRA finetuning experience. [SEP] Looking for a Python backend engineer.",
#     "Data analyst skilled in Tableau and Excel. [SEP] Looking for a business intelligence expert with Power BI experience.",
#     "Next.js developer with SEO experience. [SEP] Hiring a content strategist for a headless CMS project.",
#     "Remote UX researcher with usability testing skills. [SEP] Looking for a UI designer skilled in micro-interactions.",
#     "AI engineer skilled in LangChain and RAG. [SEP] Hiring a backend developer to integrate LLMs with APIs.",
#     "Backend developer skilled in FastAPI and SQLAlchemy. [SEP] Looking for a frontend developer with React Query experience.",
#     "Freelance developer skilled in Gatsby and Contentful. [SEP] Hiring a CMS integrator for a blog platform.",
#     "Product manager experienced in B2B SaaS. [SEP] Looking for a data engineer to build usage tracking pipelines.",
#     "Junior software engineer with JavaScript skills. [SEP] Hiring a mentor to lead code reviews and pair programming.",
#     "Cloud architect familiar with cost optimization on GCP. [SEP] Looking for a DevOps consultant to automate billing alerts."
# ]

# Tokenize input
inputs = tokenizer(sample_texts, return_tensors="pt", padding=True, truncation=True)

# Run inference
with torch.no_grad():
    outputs = model(**inputs)
    predictions = outputs.logits.argmax(dim=-1)

# Print results
for i, (text, pred) in enumerate(zip(sample_texts, predictions)):
    print(f"{i+1}. Text: {text}")
    print(f"   Predicted class: {pred.item()}\n")

# Distribution of predictions
print("Prediction distribution:", Counter(predictions.tolist()))