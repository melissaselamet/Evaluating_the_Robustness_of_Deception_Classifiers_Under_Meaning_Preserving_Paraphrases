import pandas as pd
import spacy
import nltk
from nltk.corpus import opinion_lexicon
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
import numpy as np

nltk.download("opinion_lexicon")
nltk.download("punkt")

# Load spacy model
nlp = spacy.load("en_core_web_sm")

# Load positive/negative word lists from nltk
negative_words = set(opinion_lexicon.negative())

# Feature extraction function
def extract_features(text: str) -> dict:
    doc = nlp(text)
    words = [token.text.lower() for token in doc if token.is_alpha]
    sentences = list(doc.sents)

    # Personal involvement: the first-person pronouns
    first_person = ["i", "me", "my", "myself", "mine", "we", "our", "us"]
    pronoun_count = sum(1 for w in words if w in first_person)
    pronoun_rate  = pronoun_count / len(words) if words else 0

    # Negative emotion words
    neg_count = sum(1 for w in words if w in negative_words)
    neg_rate  = neg_count / len(words) if words else 0

    # Sentence length
    avg_sent_length = np.mean([len(sent.text.split()) for sent in sentences]) if sentences else 0

    # Type-token ratio (lexical diversity)
    ttr = len(set(words)) / len(words) if words else 0

    # Pos distributions
    pos_counts = {}
    for token in doc:
        pos_counts[token.pos_] = pos_counts.get(token.pos_, 0) + 1
    total_tokens = len(doc)
    noun_rate  = pos_counts.get("NOUN", 0)  / total_tokens if total_tokens else 0
    verb_rate  = pos_counts.get("VERB", 0)  / total_tokens if total_tokens else 0
    adj_rate   = pos_counts.get("ADJ", 0)   / total_tokens if total_tokens else 0
    adv_rate   = pos_counts.get("ADV", 0)   / total_tokens if total_tokens else 0

    return {
        "pronoun_rate":     pronoun_rate,
        "neg_rate":         neg_rate,
        "avg_sent_length":  avg_sent_length,
        "ttr":              ttr,
        "noun_rate":        noun_rate,
        "verb_rate":        verb_rate,
        "adj_rate":         adj_rate,
        "adv_rate":         adv_rate,
    }

## Load the previous resultts
df = pd.read_csv("../data/results.csv")
df["flip_non_strategic"] = (df["pred_original"] != df["pred_non_strategic"]).astype(int)
df["flip_strategic"]     = (df["pred_original"] != df["pred_strategic"]).astype(int)

## Extract features
orig_features = pd.DataFrame([extract_features(t) for t in df["text"]])

ns_features = pd.DataFrame([extract_features(t) for t in df["non_strategic"]])

s_features = pd.DataFrame([extract_features(t) for t in df["strategic"]])

## Compute feature differences
ns_diff = ns_features - orig_features
s_diff  = s_features  - orig_features

ns_diff.columns = [f"ns_diff_{c}" for c in ns_diff.columns]
s_diff.columns  = [f"s_diff_{c}"  for c in s_diff.columns]

# Print mean feature differences between non-strategic vs original
print("mean feature differences (non-strategic vs original):")
for col in ns_diff.columns:
    print(f"  {col}: {ns_diff[col].mean():.4f}")

print("mean feature differences (strategic vs original):")
for col in s_diff.columns:
    print(f"  {col}: {s_diff[col].mean():.4f}")

## Logistic regression
print("logistic regression: (non-strategic flips)")
X_ns = StandardScaler().fit_transform(ns_diff)
y_ns = df["flip_non_strategic"]
lr_ns = LogisticRegression(max_iter=1000)
lr_ns.fit(X_ns, y_ns)
coef_ns = pd.Series(lr_ns.coef_[0], index=ns_diff.columns).sort_values(key=abs, ascending=False)
print("top features that predict a flip:")
print(coef_ns.head(5))

print("logistic regression: (strategic flips)")
X_s = StandardScaler().fit_transform(s_diff)
y_s = df["flip_strategic"]
lr_s = LogisticRegression(max_iter=1000)
lr_s.fit(X_s, y_s)
coef_s = pd.Series(lr_s.coef_[0], index=s_diff.columns).sort_values(key=abs, ascending=False)
print("top features that predict a flip:")
print(coef_s.head(5))

## Save feature differences
features_df = pd.concat([df[["label", "polarity", "flip_non_strategic", "flip_strategic"]], ns_diff, s_diff], axis=1)
features_df.to_csv("../data/linguistic_features.csv", index=False)