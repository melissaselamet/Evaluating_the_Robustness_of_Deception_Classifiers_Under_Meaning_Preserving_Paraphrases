import pandas as pd
import numpy as np
from statsmodels.stats.contingency_tables import mcnemar

# Load the results
df = pd.read_csv("../data/results.csv")

## Flip rates
df["flip_non_strategic"] = (df["pred_original"] != df["pred_non_strategic"]).astype(int)
df["flip_strategic"]     = (df["pred_original"] != df["pred_strategic"]).astype(int)

ns_flip_rate = df["flip_non_strategic"].mean()
s_flip_rate  = df["flip_strategic"].mean()

print(f"overall flip rates:")
print(f"  non-strategic: {ns_flip_rate:.3f} ({df['flip_non_strategic'].sum()}/{len(df)})")
print(f"  strategic:     {s_flip_rate:.3f} ({df['flip_strategic'].sum()}/{len(df)})")

## Flip rates by true label
print(f"\nflip rates by true label:")
for label, name in [(0, "truthful"), (1, "deceptive")]:
    subset = df[df["label"] == label]
    ns = subset["flip_non_strategic"].mean()
    s  = subset["flip_strategic"].mean()
    print(f"  {name}:")
    print(f"    non-strategic: {ns:.3f} ({subset['flip_non_strategic'].sum()}/{len(subset)})")
    print(f"    strategic:     {s:.3f} ({subset['flip_strategic'].sum()}/{len(subset)})")

## Flip rates by polarity
print(f"\nflip rates by polarity:")
for polarity in df["polarity"].unique():
    subset = df[df["polarity"] == polarity]
    ns = subset["flip_non_strategic"].mean()
    s  = subset["flip_strategic"].mean()
    print(f"  {polarity}:")
    print(f"    non-strategic: {ns:.3f} ({subset['flip_non_strategic'].sum()}/{len(subset)})")
    print(f"    strategic:     {s:.3f} ({subset['flip_strategic'].sum()}/{len(subset)})")

## Confidence shifts
print(f"\nconfidence shifts:")
df["conf_shift_non_strategic"] = df["conf_non_strategic"] - df["conf_original"]
df["conf_shift_strategic"]     = df["conf_strategic"]     - df["conf_original"]

print(f"  non-strategic avg confidence shift: {df['conf_shift_non_strategic'].mean():.4f}")
print(f"  strategic avg confidence shift:     {df['conf_shift_strategic'].mean():.4f}")

## Mcnemear Test
# build a contingency table: a = both flipped, b = only non-strategic flipped, c = only strategic flipped, d = neither flipped
a = ((df["flip_non_strategic"] == 1) & (df["flip_strategic"] == 1)).sum()
b = ((df["flip_non_strategic"] == 1) & (df["flip_strategic"] == 0)).sum()
c = ((df["flip_non_strategic"] == 0) & (df["flip_strategic"] == 1)).sum()
d = ((df["flip_non_strategic"] == 0) & (df["flip_strategic"] == 0)).sum()

print(f"mcnemar test:")
print(f"  contingency table: a={a}, b={b}, c={c}, d={d}")
table  = np.array([[a, b], [c, d]])
result = mcnemar(table, exact=True)
print(f"  p-value: {result.pvalue:.4f}")
if result.pvalue < 0.05:
    print("significant difference between conditions")
else:
    print("no significant difference between conditions")