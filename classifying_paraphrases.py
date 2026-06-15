import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.metrics import roc_auc_score, classification_report

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

    all_preds  = []
    all_prob_0 = []  # probability of truthful class (label=0)
    all_prob_1 = []  # probability of deceptive class (label=1)

    for batch in loader:
        batch  = {k: v.to(device) for k, v in batch.items()}
        logits = model(**batch).logits
        probs  = torch.softmax(logits, dim=-1)
        preds  = torch.argmax(probs, dim=-1)

        all_preds.extend(preds.cpu().numpy())
        all_prob_0.extend(probs[:, 0].cpu().numpy())
        all_prob_1.extend(probs[:, 1].cpu().numpy())

    return all_preds, all_prob_0, all_prob_1

# Load paraphrases
df = pd.read_csv(data_path)
true_labels = df["label"].tolist()

# Classify original reviews
print("classifying original reviews...")
df["pred_original"], df["prob0_original"], df["prob1_original"] = classify(df["text"].tolist())

# Classify 8B conditions
print("classifying 8B non-strategic paraphrases...")
df["pred_non_strategic_8b"], df["prob0_ns_8b"], df["prob1_ns_8b"] = classify(df["non_strategic_8b"].tolist())

print("classifying 8B strategic paraphrases...")
df["pred_strategic_8b"], df["prob0_s_8b"], df["prob1_s_8b"] = classify(df["strategic_8b"].tolist())

# Classify 70B conditions
print("classifying 70B non-strategic paraphrases...")
df["pred_non_strategic_70b"], df["prob0_ns_70b"], df["prob1_ns_70b"] = classify(df["non_strategic_70b"].tolist())

print("classifying 70B strategic paraphrases...")
df["pred_strategic_70b"], df["prob0_s_70b"], df["prob1_s_70b"] = classify(df["strategic_70b"].tolist())

# Save results
df.to_csv("data/results.csv", index=False)

# Baseline performance on original reviews
print("\nBaseline Classifier Performance:")
print(classification_report(true_labels, df["pred_original"], target_names=["Truthful", "Deceptive"]))
auc_original = roc_auc_score(true_labels, df["prob1_original"])
print(f"AUC (original reviews): {auc_original:.4f}")

# Flip rate summary
ns_8b_flips  = (df["pred_original"] != df["pred_non_strategic_8b"]).sum()
s_8b_flips   = (df["pred_original"] != df["pred_strategic_8b"]).sum()
ns_70b_flips = (df["pred_original"] != df["pred_non_strategic_70b"]).sum()
s_70b_flips  = (df["pred_original"] != df["pred_strategic_70b"]).sum()

print(f"\nFlip Rate Summary:")
print(f"  8B  non-strategic flips: {ns_8b_flips}/{len(df)} ({ns_8b_flips/len(df)*100:.1f}%)")
print(f"  8B  strategic flips:     {s_8b_flips}/{len(df)} ({s_8b_flips/len(df)*100:.1f}%)")
print(f"  70B non-strategic flips: {ns_70b_flips}/{len(df)} ({ns_70b_flips/len(df)*100:.1f}%)")
print(f"  70B strategic flips:     {s_70b_flips}/{len(df)} ({s_70b_flips/len(df)*100:.1f}%)")

# AUC for each condition
print(f"\nAUC per Condition:")
print(f"  8B  non-strategic AUC: {roc_auc_score(true_labels, df['prob1_ns_8b']):.4f}")
print(f"  8B  strategic AUC:     {roc_auc_score(true_labels, df['prob1_s_8b']):.4f}")
print(f"  70B non-strategic AUC: {roc_auc_score(true_labels, df['prob1_ns_70b']):.4f}")
print(f"  70B strategic AUC:     {roc_auc_score(true_labels, df['prob1_s_70b']):.4f}")

# Confidence shifts (always relative to original predicted class)
print(f"\nConfidence Shifts (M, SD):")
for prob0_col, prob1_col, name in [
    ("prob0_ns_8b",  "prob1_ns_8b",  "8B  non-strategic"),
    ("prob0_s_8b",   "prob1_s_8b",   "8B  strategic    "),
    ("prob0_ns_70b", "prob1_ns_70b", "70B non-strategic"),
    ("prob0_s_70b",  "prob1_s_70b",  "70B strategic    "),
]:
    orig_class_prob_original = df.apply(
        lambda row: row["prob0_original"] if row["pred_original"] == 0 else row["prob1_original"], axis=1
    )
    orig_class_prob_para = df.apply(
        lambda row: row[prob0_col] if row["pred_original"] == 0 else row[prob1_col], axis=1
    )
    shift = orig_class_prob_para - orig_class_prob_original
    print(f"  {name}: M = {shift.mean():.4f}, SD = {shift.std():.4f}")