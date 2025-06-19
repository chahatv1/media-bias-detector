import pandas as pd
from datasets import Dataset
from transformers import RobertaTokenizer, RobertaForSequenceClassification, Trainer, TrainingArguments
from sklearn.preprocessing import LabelEncoder
import torch

# Load and prepare data
df = pd.read_csv("data/clean_labeled_news.csv")
df = df.dropna(subset=["clean_title", "bias"])

le = LabelEncoder()
df["label"] = le.fit_transform(df["bias"])  # Convert bias to 0,1,2

# HuggingFace Dataset
dataset = Dataset.from_pandas(df[["clean_title", "label"]].rename(columns={"clean_title": "text"}))
dataset = dataset.train_test_split(test_size=0.2)

# Tokenize
tokenizer = RobertaTokenizer.from_pretrained("roberta-base")

def tokenize(batch):
    return tokenizer(batch["text"], padding=True, truncation=True)

dataset = dataset.map(tokenize, batched=True)

# Model
model = RobertaForSequenceClassification.from_pretrained("roberta-base", num_labels=3)

# Training settings
training_args = TrainingArguments(
    output_dir="./models/roberta-bias",
    eval_strategy="epoch",
    save_strategy="epoch",
    num_train_epochs=4,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    logging_dir="./logs",
    load_best_model_at_end=True,
    logging_steps=10,
)

# Accuracy metric
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = torch.argmax(torch.tensor(logits), dim=1)
    accuracy = (preds == torch.tensor(labels)).float().mean()
    return {"accuracy": accuracy.item()}

# Train
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

trainer.train()
trainer.save_model("models/roberta-bias")
tokenizer.save_pretrained("models/roberta-bias")
print("✅ RoBERTa model and tokenizer saved.")