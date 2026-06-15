import pandas as pd
from scipy.stats import pointbiserialr, ttest_ind

# Load results
df = pd.read_csv("../data/results.csv")

## Compute flip indicators
df["flip_ns_8b"]  = (df["pred_original"] != df["pred_non_strategic_8b"]).astype(int)
df["flip_s_8b"]   = (df["pred_original"] != df["pred_strategic_8b"]).astype(int)
df["flip_ns_70b"] = (df["pred_original"] != df["pred_non_strategic_70b"]).astype(int)
df["flip_s_70b"]  = (df["pred_original"] != df["pred_strategic_70b"]).astype(int)

## Load bertscores from paraphrases file
paraphrases_df = pd.read_csv("../data/paraphrases.csv")

df["ns_8b_bertscore"]  = paraphrases_df["non_strategic_8b_bertscore"].tolist()
df["s_8b_bertscore"]   = paraphrases_df["strategic_8b_bertscore"].tolist()
df["ns_70b_bertscore"] = paraphrases_df["non_strategic_70b_bertscore"].tolist()
df["s_70b_bertscore"]  = paraphrases_df["strategic_70b_bertscore"].tolist()

## BERTScore summary statistics for each condition
print("bertscore summary statistics:")
for score_col, name in [
    ("ns_8b_bertscore",  "8B  non-strategic"),
    ("s_8b_bertscore",   "8B  strategic    "),
    ("ns_70b_bertscore", "70B non-strategic"),
    ("s_70b_bertscore",  "70B strategic    "),
]:
    scores = df[score_col]
    below  = (scores < 0.85).sum()
    print(f"\n  {name}:")
    print(f"    avg: {scores.mean():.4f}")
    print(f"    min: {scores.min():.4f}")
    print(f"    max: {scores.max():.4f}")
    print(f"    below threshold (0.85): {below}/{len(scores)}")

## BERTScores for flipped vs non-flipped reviews
print("\nbertscores for flipped vs non-flipped reviews:")
for score_col, flip_col, name in [
    ("ns_8b_bertscore",  "flip_ns_8b",  "8B  non-strategic"),
    ("s_8b_bertscore",   "flip_s_8b",   "8B  strategic    "),
    ("ns_70b_bertscore", "flip_ns_70b", "70B non-strategic"),
    ("s_70b_bertscore",  "flip_s_70b",  "70B strategic    "),
]:
    flipped     = df[df[flip_col] == 1][score_col]
    non_flipped = df[df[flip_col] == 0][score_col]
    print(f"  {name}: flipped = {flipped.mean():.4f}, non-flipped = {non_flipped.mean():.4f}")


# Independent samples t-test with Cohen's d as effect size
print("\nt-tests comparing BERTScores between flipped and non-flipped reviews (APA 7):")
for score_col, flip_col, name in [
    ("ns_8b_bertscore",  "flip_ns_8b",  "8B  non-strategic"),
    ("s_8b_bertscore",   "flip_s_8b",   "8B  strategic    "),
    ("ns_70b_bertscore", "flip_ns_70b", "70B non-strategic"),
    ("s_70b_bertscore",  "flip_s_70b",  "70B strategic    "),
]:
    flipped     = df[df[flip_col] == 1][score_col]
    non_flipped = df[df[flip_col] == 0][score_col]
    t, p = ttest_ind(flipped, non_flipped)
    # Cohen's d as effect size
    pooled_sd = ((flipped.std()**2 + non_flipped.std()**2) / 2) ** 0.5
    d = (flipped.mean() - non_flipped.mean()) / pooled_sd
    print(f"  {name}: t = {t:.3f}, p = {p:.4f}, d = {d:.3f}")

## Point-biserial correlation between bertscore and flip likelihood
print("\ncorrelation between bertscore and flip likelihood:")
for score_col, flip_col, name in [
    ("ns_8b_bertscore",  "flip_ns_8b",  "8B  non-strategic"),
    ("s_8b_bertscore",   "flip_s_8b",   "8B  strategic    "),
    ("ns_70b_bertscore", "flip_ns_70b", "70B non-strategic"),
    ("s_70b_bertscore",  "flip_s_70b",  "70B strategic    "),
]:
    corr, p = pointbiserialr(df[score_col], df[flip_col])
    print(f"  {name}: r = {corr:.4f}, p = {p:.4f}")

## Save results
df.to_csv("../data/semantic_preservation.csv", index=False)