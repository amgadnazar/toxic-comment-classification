"""
DistilBERT-based toxic comment classification.


NOTE:
The training dataset is a combined dataset that includes:
- Jigsaw Toxic Comment dataset
- Additional hate speech dataset

This combination is used to improve model robustness and generalization.
Results should not be directly compared with models trained on Jigsaw alone.


Pipeline:
Text -> Tokenization -> Transformer -> Classification Head
"""

import numpy as np
import pandas as pd
import torch

from datasets import Dataset
from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
    TrainingArguments,
    Trainer
)

from sklearn.metrics import classification_report, roc_auc_score


# =========================
# Configuration
# =========================

TRAIN_CSV       = "jigsaw-dataset/final_combined_dataset.csv"
TEST_CSV        = "jigsaw-dataset/test.csv"
TEST_LABELS_CSV = "jigsaw-dataset/test_labels.csv"

OUTPUT_DIR = "./final_distilbert"

MAX_LEN   = 256
BATCH_SIZE = 128
EPOCHS     = 3
LR         = 2e-5

TOXIC_COLS = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]


# =========================
# Data Loading
# =========================

train_df   = pd.read_csv(TRAIN_CSV).fillna("")
test_df    = pd.read_csv(TEST_CSV).fillna("")
tlabels_df = pd.read_csv(TEST_LABELS_CSV)

# Keep only valid test samples (Jigsaw uses -1 for missing labels)
tlabels_df = tlabels_df[tlabels_df["toxic"] != -1]

test_df    = test_df[test_df["id"].isin(tlabels_df["id"])].reset_index(drop=True)
tlabels_df = tlabels_df.set_index("id").loc[test_df["id"]].reset_index()


# =========================
# Label Creation
# =========================

def create_label(df, toxic_cols):
    """
    Convert multi-label toxicity columns into a single binary label.
    """
    df["label"] = (df[toxic_cols].max(axis=1) > 0).astype(int)
    return df


# IMPORTANT: apply to BOTH train and test
train_df = create_label(train_df, TOXIC_COLS)
test_df["label"] = (tlabels_df[TOXIC_COLS].max(axis=1) > 0).values


# =========================
# Basic Data Stats
# =========================

def print_stats(name, df):
    total = len(df)
    toxic = df["label"].sum()
    ratio = df["label"].mean() * 100

    print(f"{name}: {total:,} samples | Toxic: {toxic:,} ({ratio:.2f}%)")


print_stats("Train", train_df)
print_stats("Test ", test_df)


# =========================
# Tokenization
# =========================

tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")


def tokenize(batch):
    """
    Tokenize text using DistilBERT tokenizer.
    """
    return tokenizer(
        batch["comment_text"],
        truncation=True,
        padding="max_length",
        max_length=MAX_LEN
    )


# Convert pandas -> HuggingFace Dataset
train_dataset = Dataset.from_dict({
    "comment_text": train_df["comment_text"].tolist(),
    "label": train_df["label"].tolist()
})

test_dataset = Dataset.from_dict({
    "comment_text": test_df["comment_text"].tolist(),
    "label": test_df["label"].tolist()
})

# Apply tokenization
train_dataset = train_dataset.map(tokenize, batched=True)
test_dataset  = test_dataset.map(tokenize, batched=True)


# =========================
# Model
# =========================

model = DistilBertForSequenceClassification.from_pretrained(
    "distilbert-base-uncased",
    num_labels=2
)


# =========================
# Training Configuration
# =========================

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,

    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE * 2,

    learning_rate=LR,
    weight_decay=0.01,
    warmup_ratio=0.1,

    evaluation_strategy="no",   # disable eval during training
    save_strategy="no",         # disable checkpoints

    logging_steps=200,
    fp16=torch.cuda.is_available(),
)


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
)


# =========================
# Training
# =========================

trainer.train()

trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print(f"\nModel saved to {OUTPUT_DIR}")


# =========================
# Evaluation
# =========================

def evaluate(predictions):
    """
    Evaluate model predictions using classification metrics.
    """
    logits = predictions.predictions
    labels = predictions.label_ids

    probs = torch.softmax(torch.tensor(logits), dim=-1)[:, 1].numpy()
    preds = (probs >= 0.5).astype(int)

    print("\n" + "=" * 50)
    print("FINAL TEST RESULTS")
    print("=" * 50)

    print(classification_report(labels, preds, target_names=["clean", "toxic"]))
    print(f"ROC-AUC: {roc_auc_score(labels, probs):.4f}")


print("\nRunning evaluation on test set...")
predictions = trainer.predict(test_dataset)
evaluate(predictions)