import pandas as pd
import numpy as np
from statsmodels.stats.contingency_tables import mcnemar
from scipy.stats import chi2_contingency
import os

# Load the results
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
df = pd.read_csv(os.path.join(BASE_DIR, "..", "data", "results.csv"))

## Flip rates
df["flip_ns_8b"]  = (df["pred_original"] != df["pred_non_strategic_8b"]).astype(int)
df["flip_s_8b"]   = (df["pred_original"] != df["pred_strategic_8b"]).astype(int)
df["flip_ns_70b"] = (df["pred_original"] != df["pred_non_strategic_70b"]).astype(int)
df["flip_s_70b"]  = (df["pred_original"] != df["pred_strategic_70b"]).astype(int)

print(f"overall flip rates:")
print(f"  8B  non-strategic: {df['flip_ns_8b'].mean():.3f} ({df['flip_ns_8b'].sum()}/{len(df)})")
print(f"  8B  strategic:     {df['flip_s_8b'].mean():.3f} ({df['flip_s_8b'].sum()}/{len(df)})")
print(f"  70B non-strategic: {df['flip_ns_70b'].mean():.3f} ({df['flip_ns_70b'].sum()}/{len(df)})")
print(f"  70B strategic:     {df['flip_s_70b'].mean():.3f} ({df['flip_s_70b'].sum()}/{len(df)})")

## Flip rates by true label
print(f"\nflip rates by true label:")
for label, name in [(0, "truthful"), (1, "deceptive")]:
    subset = df[df["label"] == label]
    print(f"  {name}:")
    print(f"    8B  non-strategic: {subset['flip_ns_8b'].mean():.3f} ({subset['flip_ns_8b'].sum()}/{len(subset)})")
    print(f"    8B  strategic:     {subset['flip_s_8b'].mean():.3f} ({subset['flip_s_8b'].sum()}/{len(subset)})")
    print(f"    70B non-strategic: {subset['flip_ns_70b'].mean():.3f} ({subset['flip_ns_70b'].sum()}/{len(subset)})")
    print(f"    70B strategic:     {subset['flip_s_70b'].mean():.3f} ({subset['flip_s_70b'].sum()}/{len(subset)})")

## Flip rates by polarity
print(f"\nflip rates by polarity:")
for polarity in df["polarity"].unique():
    subset = df[df["polarity"] == polarity]
    print(f"  {polarity}:")
    print(f"    8B  non-strategic: {subset['flip_ns_8b'].mean():.3f} ({subset['flip_ns_8b'].sum()}/{len(subset)})")
    print(f"    8B  strategic:     {subset['flip_s_8b'].mean():.3f} ({subset['flip_s_8b'].sum()}/{len(subset)})")
    print(f"    70B non-strategic: {subset['flip_ns_70b'].mean():.3f} ({subset['flip_ns_70b'].sum()}/{len(subset)})")
    print(f"    70B strategic:     {subset['flip_s_70b'].mean():.3f} ({subset['flip_s_70b'].sum()}/{len(subset)})")

## Chi-square tests for polarity differences
print(f"\nchi-square tests for polarity differences (positive vs negative):")
for flip_col, name in [
    ("flip_ns_8b",  "8B  non-strategic"),
    ("flip_s_8b",   "8B  strategic    "),
    ("flip_ns_70b", "70B non-strategic"),
    ("flip_s_70b",  "70B strategic    "),
]:
    table = pd.crosstab(df["polarity"], df[flip_col])
    chi2, p, dof, _ = chi2_contingency(table)
    # Cramer's V as effect size
    n = len(df)
    cramers_v = np.sqrt(chi2 / (n * (min(table.shape) - 1)))
    print(f"  {name}: chi2({dof}) = {chi2:.3f}, p = {p:.3f}, V = {cramers_v:.3f}")

## Confidence shifts
print(f"\nconfidence shifts (M, SD):")
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

## McNemar Tests
def run_mcnemar(flip_a, flip_b, label_a, label_b):
    """Run McNemar test and report chi-square statistic, p-value, and odds ratio effect size in APA 7 format.
    b = only A flipped, c = only B flipped
    Odds ratio = b/c: how many times more likely A flips without B than B flips without A."""
    a = ((flip_a == 1) & (flip_b == 1)).sum()
    b = ((flip_a == 1) & (flip_b == 0)).sum()
    c = ((flip_a == 0) & (flip_b == 1)).sum()
    d = ((flip_a == 0) & (flip_b == 0)).sum()

    table  = np.array([[a, b], [c, d]])
    result = mcnemar(table, exact=False)

    # Odds ratio as effect size for McNemar (b/c)
    odds_ratio = b / c if c > 0 else float('inf')

    print(f"\n  {label_a} vs {label_b}:")
    print(f"    contingency table: a={a}, b={b}, c={c}, d={d}")
    print(f"    chi2(1) = {result.statistic:.3f}, p = {result.pvalue:.3f}, OR = {odds_ratio:.3f}")
    if result.pvalue < 0.05:
        print(f"    significant difference between conditions")
    else:
        print(f"    no significant difference between conditions")

print(f"\nmcnemar tests:")

# Within 8B: non-strategic vs strategic
run_mcnemar(df["flip_ns_8b"], df["flip_s_8b"],
            "8B non-strategic", "8B strategic")

# Within 70B: non-strategic vs strategic
run_mcnemar(df["flip_ns_70b"], df["flip_s_70b"],
            "70B non-strategic", "70B strategic")

# Between models: non-strategic 8B vs non-strategic 70B
run_mcnemar(df["flip_ns_8b"], df["flip_ns_70b"],
            "8B non-strategic", "70B non-strategic")

# Between models: strategic 8B vs strategic 70B
run_mcnemar(df["flip_s_8b"], df["flip_s_70b"],
            "8B strategic", "70B strategic")