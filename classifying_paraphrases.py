import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification

# Configurations to run the model
model_path  = "models/bert_deception_classifier"
data_path   = "data/paraphrases.csv"
batch_size  = 16
device      = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# Load the saved bert model and its tokenizer (from BERT-classifier.py)
tokenizer = BertTokenizer.from_pretrained(model_path)
model     = BertForSequenceClassification.from_pretrained(model_path).to(device)
model.eval()

# Dataset class
class ReviewDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length=512):
        self.encodings = tokenizer(
            texts,
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt",
        )

    def __len__(self):
        return len(self.encodings["input_ids"])

    def __getitem__(self, idx):
        return {
            "input_ids":      self.encodings["input_ids"][idx],
            "attention_mask": self.encodings["attention_mask"][idx],
            "token_type_ids": self.encodings["token_type_ids"][idx],
        }

# Function to classify a list of texts
@torch.no_grad()
def classify(texts):
    dataset = ReviewDataset(texts, tokenizer)
    loader  = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    all_preds       = []
    all_confidences = []

    for batch in loader:
        batch   = {k: v.to(device) for k, v in batch.items()}
        logits  = model(**batch).logits
        probs   = torch.softmax(logits, dim=-1)
        preds   = torch.argmax(probs, dim=-1)
        confidence = probs[torch.arange(len(preds)), preds]

        all_preds.extend(preds.cpu().numpy())
        all_confidences.extend(confidence.cpu().numpy())

    return all_preds, all_confidences

# Load paraphrases
df = pd.read_csv(data_path)

# Classify all three versions
print("classify the original reviews")
df["pred_original"],        df["conf_original"]        = classify(df["text"].tolist())

print("classify the non-strategic paraphrases")
df["pred_non_strategic"],   df["conf_non_strategic"]   = classify(df["non_strategic"].tolist())

print("classify the strategic paraphrases")
df["pred_strategic"],       df["conf_strategic"]       = classify(df["strategic"].tolist())

# Save results
df.to_csv("data/results.csv", index=False)

# Final summary of flip-rates
ns_flips = (df["pred_original"] != df["pred_non_strategic"]).sum()
s_flips  = (df["pred_original"] != df["pred_strategic"]).sum()

print(f"  non-strategic flips: {ns_flips}/{len(df)} ({ns_flips/len(df)*100:.1f}%)")
print(f"  strategic flips:     {s_flips}/{len(df)} ({s_flips/len(df)*100:.1f}%)")