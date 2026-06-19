import torch
import random
import numpy as np
import pandas as pd
import os
from transformers import BertTokenizer
from torch.utils.data import Dataset, DataLoader
from transformers import BertForSequenceClassification
from transformers import get_linear_schedule_with_warmup
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix

# Load train and test datasets from data_preperation.py
train_data = pd.read_csv("data/train_set.csv")
test_data = pd.read_csv("data/test_set.csv")

# Load the bert-base-uncased tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# Set seeds for reproducibility
SEED = 11
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# Detect device
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

# Hyperparameters
EPOCHS = 3
LEARNING_RATE = 2e-5
BATCH_SIZE = 16
WEIGHT_DECAY = 0.01
MAX_LENGTH = 512


class ReviewDataset(Dataset):
    def __init__(self, encodings, labels):
        '''Creates a custom dataset class that holds all the reviews and labels in a format PyTorch understands.'''
        self.encodings = encodings
        self.labels    = labels

    def __len__(self):
        '''Returns the total number of reviews in the dataset'''
        return len(self.labels)

    def __getitem__(self, idx):
        '''Fetches one single review at a time by its index.'''
        return {
            "input_ids":      self.encodings["input_ids"][idx],
            "attention_mask": self.encodings["attention_mask"][idx],
            "token_type_ids": self.encodings["token_type_ids"][idx],
            "labels":         torch.tensor(self.labels[idx], dtype=torch.long),
        }


def tokenize(texts):
    return tokenizer(
        texts,
        truncation=True,
        padding="max_length",
        max_length=MAX_LENGTH,
        return_tensors="pt",
    )


def build_model():
    model = BertForSequenceClassification.from_pretrained(
        "bert-base-uncased",
        num_labels=2,
    )
    return model.to(device)


def train_model(model, train_loader):
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    total_steps  = len(train_loader) * EPOCHS
    warmup_steps = int(0.1 * total_steps)

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0

        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad()
            outputs = model(**batch)
            loss    = outputs.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"  Epoch {epoch}/{EPOCHS}  |  avg loss: {avg_loss:.4f}")

    return model


def evaluate_model(model, loader):
    model.eval()
    all_preds       = []
    all_labels      = []
    all_confidences = []

    with torch.no_grad():
        for batch in loader:
            labels = batch.pop("labels").to(device)
            batch  = {k: v.to(device) for k, v in batch.items()}

            logits     = model(**batch).logits
            probs      = torch.softmax(logits, dim=-1)
            preds      = torch.argmax(probs, dim=-1)
            confidence = probs[torch.arange(len(preds)), preds]

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_confidences.extend(confidence.cpu().numpy())

    return all_preds, all_labels, all_confidences


## 5-fold Stratified Cross Validation
print("\n5-Fold Stratified Cross-Validation:")

texts  = train_data["text"].tolist()
labels = train_data["label"].tolist()

skf          = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
fold_results = []

for fold, (train_idx, val_idx) in enumerate(skf.split(texts, labels), 1):
    print(f"\nFold {fold}/5")

    fold_train_texts  = [texts[i] for i in train_idx]
    fold_train_labels = [labels[i] for i in train_idx]
    fold_val_texts    = [texts[i] for i in val_idx]
    fold_val_labels   = [labels[i] for i in val_idx]

    fold_train_enc = tokenize(fold_train_texts)
    fold_val_enc   = tokenize(fold_val_texts)

    fold_train_dataset = ReviewDataset(fold_train_enc, fold_train_labels)
    fold_val_dataset   = ReviewDataset(fold_val_enc,   fold_val_labels)

    fold_train_loader = DataLoader(fold_train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    fold_val_loader   = DataLoader(fold_val_dataset,   batch_size=BATCH_SIZE, shuffle=False)

    fold_model = build_model()
    fold_model = train_model(fold_model, fold_train_loader)

    preds, true_labels, _ = evaluate_model(fold_model, fold_val_loader)

    report = classification_report(
        true_labels, preds,
        target_names=["Truthful", "Deceptive"],
        output_dict=True
    )
    fold_results.append(report)
    print(classification_report(true_labels, preds, target_names=["Truthful", "Deceptive"]))

    # Free memory between folds
    del fold_model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

# Summarize the CV results
print("\nCross-Validation Results")
metrics = ["precision", "recall", "f1-score"]
classes = ["Truthful", "Deceptive", "macro avg"]

cv_summary = {}
for cls in classes:
    cv_summary[cls] = {}
    for metric in metrics:
        scores = [fold[cls][metric] for fold in fold_results]
        mean, std = np.mean(scores), np.std(scores)
        cv_summary[cls][metric] = (mean, std)
        print(f"{cls:12s} {metric:12s}: {mean:.4f} ± {std:.4f}")

## Save CV summary to CSV
cv_rows = []
for cls in classes:
    for metric in metrics:
        mean, std = cv_summary[cls][metric]
        cv_rows.append({"class": cls, "metric": metric, "mean": mean, "std": std})

pd.DataFrame(cv_rows).to_csv("data/cv_summary.csv", index=False)


## Final Model Training
print("\Training Final Model on Full Training Set:")

train_encodings = tokenize(train_data["text"].tolist())
test_encodings  = tokenize(test_data["text"].tolist())

train_dataset = ReviewDataset(train_encodings, train_data["label"].tolist())
test_dataset  = ReviewDataset(test_encodings,  test_data["label"].tolist())

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE, shuffle=False)

model = build_model()
print(f"Number of parameters: {sum(p.numel() for p in model.parameters()):,}")
model = train_model(model, train_loader)

# Evaluate final model on held-out test set
all_preds, all_labels, all_confidences = evaluate_model(model, test_loader)

print("\nFinal Model Test Set Performance")
print(classification_report(all_labels, all_preds, target_names=["Truthful", "Deceptive"]))
print(confusion_matrix(all_labels, all_preds))

test_data["pred_label"] = all_preds
test_data["confidence"] = all_confidences
test_data.to_csv("data/test_set_with_predictions.csv", index=False)

# Save the final model
os.makedirs("models/bert_deception_classifier", exist_ok=True)
model.save_pretrained("models/bert_deception_classifier")
tokenizer.save_pretrained("models/bert_deception_classifier")
print("\nFinal model saved to models/bert_deception_classifier")