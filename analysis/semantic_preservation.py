import pandas as pd
from bert_score import BERTScorer

# Load results
df = pd.read_csv("../data/results.csv")

# Load scorer once
scorer = BERTScorer(lang="en", device="mps")

# Extract the bert scores
paraphrases_df = pd.read_csv("../data/paraphrases.csv")

ns_scores = paraphrases_df["non_strategic_bertscore"].tolist()
s_scores  = paraphrases_df["strategic_bertscore"].tolist()

# Summary
print(f"non-strategic avg bertscore: {sum(ns_scores)/len(ns_scores):.4f}")
print(f"non-strategic min bertscore: {min(ns_scores):.4f}")
print(f"non-strategic max bertscore: {max(ns_scores):.4f}")
print(f"non-strategic below threshold (0.85): {sum(s < 0.85 for s in ns_scores)}/{len(ns_scores)}")

print(f"strategic avg bertscore: {sum(s_scores)/len(s_scores):.4f}")
print(f"strategic min bertscore:  {min(s_scores):.4f}")
print(f"strategic max bertscore:  {max(s_scores):.4f}")
print(f"strategic below threshold (0.85): {sum(s < 0.85 for s in s_scores)}/{len(s_scores)}")

# Compare bertscores between flipped and non-flipped reviews
df["flip_non_strategic"] = (df["pred_original"] != df["pred_non_strategic"]).astype(int)
df["flip_strategic"] = (df["pred_original"] != df["pred_strategic"]).astype(int)
df["ns_bertscore"] = ns_scores
df["s_bertscore"] = s_scores

print("bertscores for flipped vs non-flipped reviews:")
print(f"non-strategic flipped avg bertscore: {df[df['flip_non_strategic']==1]['ns_bertscore'].mean():.4f}")
print(f"non-strategic non-flipped avg bertscore: {df[df['flip_non_strategic']==0]['ns_bertscore'].mean():.4f}")
print(f"strategic flipped avg bertscore: {df[df['flip_strategic']==1]['s_bertscore'].mean():.4f}")
print(f"strategic non-flipped avg bertscore: {df[df['flip_strategic']==0]['s_bertscore'].mean():.4f}")

# Check if lower bertscore correlates with higher flip rate
from scipy.stats import pointbiserialr

corr_ns, p_ns = pointbiserialr(df["ns_bertscore"], df["flip_non_strategic"])
corr_s,  p_s  = pointbiserialr(df["s_bertscore"],  df["flip_strategic"])

print(f"correlation between bertscore and flip:")
print(f" non-strategic: r={corr_ns:.4f}, p={p_ns:.4f}")
print(f" strategic: r={corr_s:.4f},  p={p_s:.4f}")

# Save results
df.to_csv("../data/semantic_preservation.csv", index=False)