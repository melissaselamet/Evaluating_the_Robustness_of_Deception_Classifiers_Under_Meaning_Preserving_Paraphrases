import torch
import random
import numpy as np
import pandas as pd
import os
from transformers import BertTokenizer
from torch.utils.data import Dataset, DataLoader
from transformers import BertForSequenceClassification
from transformers import get_linear_schedule_with_warmup
from sklearn.metrics import classification_report, confusion_matrix

# Load the train and test datasets from data_preperation.py
train_data = pd.read_csv("data/train_set.csv")
test_data = pd.read_csv("data/test_set.csv")

## Tokenizing the text
# Load the bert-base-uncased tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# Tokenize the train and test datasets
train_encodings = tokenizer(
    train_data["text"].tolist(),
    truncation=True,
    padding="max_length",
    max_length=512,
    return_tensors="pt",
)

test_encodings = tokenizer(
    test_data["text"].tolist(),
    truncation=True,
    padding="max_length",
    max_length=512,
    return_tensors="pt",
)

# Create the pytorch datasets and the dataloade4rs
class ReviewDataset(Dataset):
    def __init__(self, encodings, labels):
        '''Creates a custom dataset class that holds all the reviews
         and labels in a format PyTorch understands.'''
        self.encodings = encodings
        self.labels    = labels

    def __len__(self):
        '''Returns the total number of reviews in the dataset and
        gives these values to PyTorch'''
        return len(self.labels)

    def __getitem__(self, idx):
        '''Fetches one single review at a time by its index.
        It returns a dictionary with everything BERT needs for that review:'''
        return {
            "input_ids":      self.encodings["input_ids"][idx],
            "attention_mask": self.encodings["attention_mask"][idx],
            "token_type_ids": self.encodings["token_type_ids"][idx],
            "labels":         torch.tensor(self.labels[idx], dtype=torch.long),
        }


# Create dataset objects for train and test
train_dataset = ReviewDataset(train_encodings, train_data["label"].tolist())
test_dataset  = ReviewDataset(test_encodings,  test_data["label"].tolist())

# Wrap datasets in DataLoaders (feed data to the model in batches)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader  = DataLoader(test_dataset,  batch_size=16, shuffle=False)

## Loading the bert model
# Set seeds for reproducibility
SEED = 11
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# To speed up the process detect device and use it
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

# Load bert-base-uncased with a classification head on top
model = BertForSequenceClassification.from_pretrained(
    "bert-base-uncased",
    num_labels=2,
)

model = model.to(device)
print(f"Number of parameters: {sum(p.numel() for p in model.parameters()):,}")

## Fine-tune the model
epochs = 3
learning_rate = 2e-5


optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
total_steps  = len(train_loader) * epochs
warmup_steps = int(0.1 * total_steps)

scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=warmup_steps,
    num_training_steps=total_steps,
)

# The training loop
for epoch in range(1, epochs + 1):
    model.train()
    total_loss = 0.0

    for batch in train_loader:
        batch = {k: v.to(device) for k, v in batch.items()}

        optimizer.zero_grad()
        outputs = model(**batch)
        loss    = outputs.loss
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()   # update model weights
        scheduler.step()   # update learning rate
        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch}/{epochs}  |  avg loss: {avg_loss:.4f}")

## Evaluate model
model.eval()

all_preds       = []
all_labels      = []
all_confidences = []

with torch.no_grad():
    for batch in test_loader:
        labels = batch.pop("labels").to(device)
        batch = {k: v.to(device) for k, v in batch.items()}

        logits = model(**batch).logits   # raw output values
        probs = torch.softmax(logits, dim=-1)   # converted the logits to probabilities
        preds = torch.argmax(probs, dim=-1)   # picking the highest probability class
        confidence = probs[torch.arange(len(preds)), preds]   # confidence rate of the predicted class

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_confidences.extend(confidence.cpu().numpy())

print(classification_report(all_labels, all_preds, target_names=["Truthful", "Deceptive"]))
print(confusion_matrix(all_labels, all_preds))

test_data["pred_label"]  = all_preds
test_data["confidence"]  = all_confidences
test_data.to_csv("data/test_set_with_predictions.csv", index=False)

## Save the results
os.makedirs("models/bert_deception_classifier", exist_ok=True)

model.save_pretrained("models/bert_deception_classifier")
tokenizer.save_pretrained("models/bert_deception_classifier")